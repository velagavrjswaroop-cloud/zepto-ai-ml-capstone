from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    client.delete_collection("zepto_policies")
except Exception:
    pass

collection = client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

documents = []
ids = []
metadatas = []

for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()
    documents.append(text)
    ids.append(file_path.stem)
    metadatas.append({
        "document_id": file_path.stem,
        "source": file_path.name
    })

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks stored: {collection.count()}")

result = collection.query(
    query_embeddings=[embeddings[0]],
    n_results=1
)

print(f"Test retrieval: {result['ids'][0][0]}")
