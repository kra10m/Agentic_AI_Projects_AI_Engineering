import os
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class HybridRetriever:
    def __init__(self, chroma_db_dir):
        print("Initializing Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Connecting to ChromaDB...")
        self.vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=self.embeddings)
        
        all_data = self.vectorstore.get()
        self.ids = all_data['ids']
        self.documents = all_data['documents']
        
        print(f"Building BM25 index for {len(self.documents)} chunks...")
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60):
        dense_results = self.vectorstore.similarity_search(query, k=top_k)
        dense_ranked_ids = [doc.metadata.get("chunk_id") for doc in dense_results]

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        sparse_scored = sorted(zip(self.ids, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
        sparse_ranked_ids = [item[0] for item in sparse_scored]

        rrf_scores = {}
        
        for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            
        for rank, chunk_id in enumerate(sparse_ranked_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        

        return [item[0] for item in fused_results[:top_k]]

if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    CHROMA_DB_DIR = os.path.join(PARENT_DIR, "chroma_db_advanced")
    
    retriever = HybridRetriever(CHROMA_DB_DIR)
    results = retriever.retrieve("Which checkpointer should I use for local environments?")
    print(f"\nHybrid Results: {results}")