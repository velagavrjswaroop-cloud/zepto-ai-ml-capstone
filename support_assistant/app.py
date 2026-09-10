import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
PROMPT_FILE = BASE_DIR / "prompt_template.txt"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection("zepto_policies")

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

prompt_template = PROMPT_FILE.read_text(encoding="utf-8")


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AskRequest(BaseModel):
    question: str


class GraphState(TypedDict, total=False):
    question: str
    intent: str
    context: list[str]
    sources: list[str]
    answer: str
    confidence: float


def call_real_llm(messages):
    if not GROQ_API_KEY:
        raise ValueError("ERROR: GROQ_API_KEY is not configured.")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_real_response(raw_output):
    cleaned = raw_output.strip()

    if cleaned.startswith("`"):
        cleaned = cleaned.replace("`json", "", 1).replace("`", "", 1).strip()

    parsed = json.loads(cleaned)
    return AssistantResponse.model_validate(parsed)


def generate_real_answer(question, context):
    context_text = "\n\n".join(context)
    base_prompt = prompt_template.format(
        context=context_text,
        question=question
    )

    messages = [
        {
            "role": "system",
            "content": "Return only valid JSON matching the requested schema."
        },
        {
            "role": "user",
            "content": base_prompt
        }
    ]

    last_error = ""

    for attempt in range(3):
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": f"Corrective instruction: The previous output failed structured validation because {last_error}. Return only a valid JSON object with exactly answer, sources, and confidence."
                }
            )

        try:
            raw_output = call_real_llm(messages)
            return parse_real_response(raw_output)
        except Exception as exc:
            last_error = str(exc)

    return AssistantResponse(
        answer="ERROR: Structured output validation failed after 3 attempts.",
        sources=[],
        confidence=0.0
    )


def classify_intent_with_real_llm(question):
    messages = [
        {
            "role": "system",
            "content": "Classify the user question as exactly one of policy_question or general_question. Return only JSON with the field intent."
        },
        {
            "role": "user",
            "content": f'Question: {question}'
        }
    ]

    last_error = ""

    for attempt in range(3):
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": f"Corrective instruction: {last_error}. Return only JSON such as {{\"intent\":\"policy_question\"}} or {{\"intent\":\"general_question\"}}."
                }
            )

        try:
            raw_output = call_real_llm(messages)
            cleaned = raw_output.strip()

            if cleaned.startswith("`"):
                cleaned = cleaned.replace("`json", "", 1).replace("`", "", 1).strip()

            data = json.loads(cleaned)
            intent = data.get("intent")

            if intent in {"policy_question", "general_question"}:
                return intent

            last_error = "intent must be policy_question or general_question"
        except Exception as exc:
            last_error = str(exc)

    return "general_question"


def classify_intent(state: GraphState):
    question = state["question"].lower()

    if MOCK_LLM:
        intent = "general_question"

        if any(keyword in question for keyword in POLICY_KEYWORDS):
            intent = "policy_question"

        return {"intent": intent}

    return {"intent": classify_intent_with_real_llm(state["question"])}


def retrieve_context(question):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = result["documents"][0]
    ids = result["ids"][0]

    return documents, ids


def retrieve_and_answer(state: GraphState):
    documents, ids = retrieve_context(state["question"])

    if MOCK_LLM:
        snippet = documents[0][:200]
        return {
            "context": documents,
            "sources": ids,
            "answer": f"Based on the retrieved context: {snippet}",
            "confidence": 1.0
        }

    response = generate_real_answer(
        state["question"],
        documents
    )

    return {
        "context": documents,
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence
    }


def direct_answer(state: GraphState):
    if MOCK_LLM:
        return {
            "context": [],
            "sources": [],
            "answer": "I can only answer questions about Zepto policies right now.",
            "confidence": 1.0
        }

    response = generate_real_answer(
        state["question"],
        []
    )

    return {
        "context": [],
        "sources": response.sources,
        "answer": response.answer,
        "confidence": response.confidence
    }


def route_question(state: GraphState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph_builder = StateGraph(GraphState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)

graph_builder.add_edge(START, "classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

graph = graph_builder.compile()

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": MOCK_LLM
    }


@app.post("/ask", response_model=AssistantResponse)
def ask(request: AskRequest):
    result = graph.invoke({
        "question": request.question
    })

    response = AssistantResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )

    return response
