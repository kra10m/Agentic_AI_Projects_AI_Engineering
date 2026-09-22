import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "chroma_storage"

client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_or_create_collection(
    name="engineering_knowledge",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """
    Retrieves the most semantically relevant text chunks from ChromaDB
    and returns them as a structured string for the LLM.
    """
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    if not documents:
        return "No relevant documents found in the knowledge base."

    formatted_chunks = []
    for rank, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        similarity = 1 - dist
        chunk_repr = (
            f"[RESULT {rank}]\n"
            f"Chunk ID: {chunk_id}\n"
            f"Source: {meta.get('source', 'unknown')}\n"
            f"Similarity Score: {similarity:.4f}\n"
            f"Content: {doc}"
        )
        formatted_chunks.append(chunk_repr)

    return "\n\n---\n\n".join(formatted_chunks)

RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Searches the internal engineering knowledge base for documentation, "
            "architecture guides, and best practices. Use this whenever the user "
            "asks technical questions about internal infrastructure, containers, "
            "or databases."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documentation chunks."
                }
            },
            "required": ["query"]
        }
    }
}

if __name__ == "__main__":
    test_query = "What is used for key-value caching?"
    print(f"Testing tool locally with query: '{test_query}'\n")
    
    tool_output = search_knowledge_base(query=test_query, top_k=2)
    print("OUTPUT FORMATTED FOR LLM CONTEXT:")
    print("=" * 60)
    print(tool_output)
    print("=" * 60)