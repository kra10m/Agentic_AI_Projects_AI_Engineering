import os
import json
from retrieval_rerank import AdvancedRetriever

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
CHROMA_DB_DIR = os.path.join(PARENT_DIR, "chroma_db_advanced")
EVAL_DATASET_PATH = os.path.join(DATA_DIR, "eval_dataset.json")

def evaluate_hybrid(k=3):
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    retriever = AdvancedRetriever(CHROMA_DB_DIR)

    total_queries = len(dataset)
    recall_count = 0
    mrr_sum = 0.0

    print(f"\nEvaluating {total_queries} queries using HYBRID SEARCH (Top-{k})...\n")

    for item in dataset:
        query = item["query"]
        expected_ids = item["expected_chunk_ids"]

        retrieved_ids = retriever.retrieve(query, final_k=k)

        found_at_rank = None
        for rank, ret_id in enumerate(retrieved_ids, start=1):
            if ret_id in expected_ids:
                found_at_rank = rank
                break
        
        print(f"Query: {query}")
        print(f"Expected: {expected_ids[0]}")
        print(f"Retrieved: {retrieved_ids}")
        
        if found_at_rank:
            recall_count += 1
            mrr_sum += 1.0 / found_at_rank
            print(f"✅ Found at rank {found_at_rank}\n")
        else:
            print(f"❌ Not found in top {k}\n")

    recall_at_k = recall_count / total_queries
    mrr = mrr_sum / total_queries

    print("-" * 30)
    print("HYBRID EVALUATION RESULTS")
    print("-" * 30)
    print(f"Recall@{k}: {recall_at_k:.2f} ({recall_count}/{total_queries})")
    print(f"MRR:      {mrr:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_hybrid(k=3)