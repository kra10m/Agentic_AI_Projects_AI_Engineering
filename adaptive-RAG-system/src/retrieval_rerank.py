import os
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(self, chroma_db_dir):
        print("Initializing Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Connecting to ChromaDB...")
        self.vectorstore = Chroma(persist_directory=chroma_db_dir, embedding_function=self.embeddings)
        
        all_data = self.vectorstore.get()
        self.ids = all_data['ids']
        self.documents = all_data['documents']
        
        self.doc_map = dict(zip(self.ids, self.documents))
        
        print(f"Building BM25 index for {len(self.documents)} chunks...")
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Initializing Cross-Encoder Reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, initial_k: int = 10, final_k: int = 3, rrf_k: int = 60):
        dense_results = self.vectorstore.similarity_search(query, k=initial_k)
        dense_ranked_ids = [doc.metadata.get("chunk_id") for doc in dense_results]

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_scored = sorted(zip(self.ids, bm25_scores), key=lambda x: x[1], reverse=True)[:initial_k]
        sparse_ranked_ids = [item[0] for item in sparse_scored]

        rrf_scores = {}
        for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        for rank, chunk_id in enumerate(sparse_ranked_ids, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:initial_k]
        candidate_ids = [item[0] for item in fused_candidates]

        cross_inp = [[query, self.doc_map[chunk_id]] for chunk_id in candidate_ids]
        
        cross_scores = self.reranker.predict(cross_inp)
        
        reranked_results = sorted(zip(candidate_ids, cross_scores), key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in reranked_results[:final_k]]

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    CHROMA_DB_DIR = os.path.join(PARENT_DIR, "chroma_db_advanced")
    
    retriever = AdvancedRetriever(CHROMA_DB_DIR)
    results = retriever.retrieve("What reducer should I use to append messages rather than overwrite them?")
    print(f"\nReranked Results: {results}")