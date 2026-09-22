from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from ingestion import ingest_file

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

db_path = Path("./Agentic_AI_Projects_AI_Engineering/RAG-agent/chroma_storage")
client = chromadb.PersistentClient(path=str(db_path))

collection = client.get_or_create_collection(
    name="engineering_knowledge",
    metadata={"hnsw:space": "cosine"}
)

def populate_vector_db(file_path: str):
    chunks = ingest_file(file_path, chunk_size=40, overlap=10)
    
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source} for c in chunks]
    
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(documents).tolist()
    
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB at: {db_path}\n")

def query_vector_db(query: str, n_results: int = 2):
    """
    Embeds the runtime query and retrieves the Top-N closest matching chunks.
    """

    query_vector = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

if __name__ == "__main__":

    populate_vector_db("Agentic_AI_Projects_AI_Engineering/RAG-agent/cloud_architecture.txt")

    print(f"Total documents currently in collection: {collection.count()}\n")

    query = "How are container deployments scaled and monitored?"
    print(f"SEARCH QUERY: \"{query}\"")
    
    search_output = query_vector_db(query, n_results=2)
    
    docs = search_output["documents"][0]
    meta = search_output["metadatas"][0]
    ids = search_output["ids"][0]
    distances = search_output["distances"][0]

    for rank, (chunk_id, doc, m, dist) in enumerate(zip(ids, docs, meta, distances), start=1):

        similarity = 1 - dist
        print(f"\n--- Match #{rank} (Similarity: {similarity:.4f}) ---")
        print(f"Chunk ID: {chunk_id}")
        print(f"Source:   {m['source']}")
        print(f"Content:  {doc}")