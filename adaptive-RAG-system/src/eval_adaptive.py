import os
import json
from graph_adaptive import AdaptiveRAGGraph

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
EVAL_DATASET_PATH = os.path.join(DATA_DIR, "eval_dataset.json")

def evaluate_adaptive_graph():
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print("Initializing Adaptive RAG Pipeline (This takes a few seconds)...")
    rag = AdaptiveRAGGraph()

    total_queries = len(dataset)
    recall_count = 0
    
    print(f"\nEvaluating {total_queries} queries using FULL ADAPTIVE GRAPH...\n")

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        expected_ids = item["expected_chunk_ids"]

        print(f"--- Query {i}/{total_queries} ---")
        print(f"User Query: {query}")
        
        initial_state = {
            "original_query": query,
            "current_query": query,
            "attempt_count": 0,
            "max_attempts": 2
        }
        
        final_state = rag.app.invoke(initial_state)
        
        retrieved = final_state.get("retrieved_chunk_ids", [])
        
        generation = final_state.get("generation", "")
        citations = generation.split("Sources: ")[-1] if "Sources: " in generation else "None"
        
        found = any(exp_id in retrieved for exp_id in expected_ids)
        if found:
            recall_count += 1
            
        print(f"Attempts taken: {final_state['attempt_count']}")
        print(f"Final Executed Query: {final_state['current_query']}")
        print(f"Expected ID: {expected_ids[0]}")
        print(f"LLM Cited Sources: {citations}")
        print(f"System Retrieval Success: {'✅' if found else '❌'}\n")

    recall = recall_count / total_queries
    print("-" * 30)
    print("ADAPTIVE GRAPH EVALUATION RESULTS")
    print("-" * 30)
    print(f"System Recall (Top-3): {recall:.2f} ({recall_count}/{total_queries})")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_adaptive_graph()