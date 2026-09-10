# Zepto Support Assistant

## Overview

This module implements a small offline GenAI-style customer support service for Zepto. It loads a local policy corpus, creates local embeddings, stores them in ChromaDB, retrieves grounded context, routes questions with LangGraph, and exposes the service through FastAPI.

The default mode is deterministic and offline. No API key or external LLM service is required for the graded baseline.

## Architecture

The flow is:

Ingestion → Embedding → Retrieval → Generation

- Ingestion: `docs/doc_01.txt` through `docs/doc_08.txt` contain the Zepto policy corpus.
- Embedding: `ingest.py` uses Sentence Transformers with `all-MiniLM-L6-v2`.
- Vector store: ChromaDB stores the document embeddings in the `zepto_policies` collection.
- Routing: `app.py` uses a LangGraph `StateGraph`.
- `classify_intent`: classifies the question as `policy_question` or `general_question`.
- `retrieve_and_answer`: embeds the policy question, retrieves the top 3 chunks using cosine similarity, and generates a grounded answer.
- `direct_answer`: handles unrelated questions without retrieval.
- Structured output: `AssistantResponse` is a Pydantic model containing `answer`, `sources`, and `confidence`.
- API: FastAPI exposes `POST /ask`.

## LangGraph Flow

The graph contains three named nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

The `classify_intent` node routes the question conditionally:

- `policy_question` → `retrieve_and_answer`
- `general_question` → `direct_answer`

The default mock routing is deterministic and uses policy-related keywords. It does not call an LLM.

## Mock Mode

Mock mode is enabled by default when `MOCK_LLM` is unset or set to `1`.

Policy questions:

- The query is embedded.
- The top 3 matching documents are retrieved from ChromaDB.
- The answer starts with `Based on the retrieved context:`.
- The retrieved document IDs are returned in `sources`.
- Confidence is set to `1.0`.

General questions return the deterministic response:

`I can only answer questions about Zepto policies right now.`

This makes the baseline fully deterministic and offline.

## Optional Real LLM Mode

Set `MOCK_LLM=0` to enable the optional real LLM path.

The implementation uses a Groq-compatible OpenAI API endpoint when `GROQ_API_KEY` is available.

The structured prompt is stored in `prompt_template.txt`. It contains:

- Role
- Context
- Task
- Format
- Length
- Negative grounding constraint
- Few-shot example

The prompt explicitly prevents the model from answering with information that is not present in the retrieved context.

The real generation path validates the model response against the Pydantic `AssistantResponse` schema and retries invalid output up to two additional times with corrective instructions.

The real LLM mode is optional and is not required for the deterministic baseline.

## Installation

From the project root:

```powershell
pip install -r support_assistant/requirements.txt
```

## Build the Vector Store

Run:

```powershell
python support_assistant/ingest.py
```

Expected output:

```text
Documents loaded: 8
Chunks stored: 8
Test retrieval: doc_01
```

The corpus contains 8 policy documents, stored as one chunk per document.

## Run FastAPI

From the project root:

```powershell
uvicorn support_assistant.app:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Example 1: Policy Question

Request:

```json
{
  "question": "How long does delivery take?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: doc_01 — Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current",
  "sources": [
    "doc_01",
    "doc_02",
    "doc_04"
  ],
  "confidence": 1.0
}
```

## Example 2: General Question

Request:

```json
{
  "question": "What is the capital of India?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Pydantic Response Schema

Every `/ask` response follows:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}
```

`confidence` is constrained to the range 0 to 1.

## Docker

Build the image from the project root:

```powershell
docker build -t zepto-support-assistant support_assistant
```

The Dockerfile automatically runs `python ingest.py` during image creation, so the ChromaDB vector store is built inside the image.

Run the container:

```powershell
docker run -p 8000:8000 zepto-support-assistant
```

The API is then available at:

```text
http://127.0.0.1:8000
```

## Project Files

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── app.py
├── ingest.py
├── prompt_template.txt
├── requirements.txt
├── examples.txt
├── Dockerfile
└── README.md
```

## Design Decisions

- The corpus is intentionally small and deterministic so the complete baseline can run locally without paid services.
- One chunk per policy document is sufficient for the small corpus.
- `all-MiniLM-L6-v2` provides local sentence embeddings.
- ChromaDB provides persistent vector storage and cosine-similarity retrieval.
- LangGraph makes the routing and retrieval flow explicit.
- Pydantic guarantees a consistent API response structure.
- The mock branch avoids all LLM calls and provides reproducible outputs.
- The optional real LLM branch supports grounded generation while preserving structured-output validation.
