import math
from sentence_transformers import SentenceTransformer

def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(a * a for a in v))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    denom = magnitude(v1) * magnitude(v2)
    if denom == 0.0:
        return 0.0
    return dot_product(v1, v2) / denom

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Docker containers package an application and its dependencies together.",
    "PostgreSQL is an open-source relational database management system.",
    "Kubernetes is an orchestration platform for managing containerized workloads.",
    "Git is a distributed version control system used for tracking code changes.",
    "Python is an interpreted, high-level, general-purpose programming language.",
    "Redis is an in-memory key-value data store often used for caching.",
]

doc_embeddings = model.encode(documents).tolist()

def search(query: str, top_k: int = 2) -> list[dict]:
    """
    Finds the most semantically relevant documents for a given query.
    """
    query_vec = model.encode(query).tolist()

    scored_results = []
    for doc_text, doc_vec in zip(documents, doc_embeddings):
        score = cosine_similarity(query_vec, doc_vec)
        scored_results.append({
            "text": doc_text,
            "score": round(score, 4)
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    return scored_results[:top_k]

test_queries = [
    "How can I manage and scale my containers in production?",
    "Where should I save temporary session data for fast access?",
    "What tool helps teams collaborate on source code history?",
    "A tool to prevent merge conflicts when collaborating with other developers"
]

for q in test_queries:
    print(f"QUERY: '{q}'")
    matches = search(q, top_k=2)
    for rank, match in enumerate(matches, start=1):
        print(f"  [{rank}] Score: {match['score']} | Document: {match['text']}")
    print("-" * 70)