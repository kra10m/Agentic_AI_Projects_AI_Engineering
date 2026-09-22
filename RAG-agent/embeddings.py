import math
from sentence_transformers import SentenceTransformer

def dot_product(v1: list[float], v2: list[float]) -> float:
    """Multiplies matching coordinates and sums them up."""
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v: list[float]) -> float:
    """Computes the length (norm) of a vector using the Pythagorean theorem."""
    return math.sqrt(sum(a * a for a in v))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Measures the cosine of the angle between two vectors:
    - 1.0 means pointing in the exact same direction (identical semantic meaning)
    - 0.0 means completely orthogonal / unrelated
    """
    denom = magnitude(v1) * magnitude(v2)
    if denom == 0.0:
        return 0.0
    return dot_product(v1, v2) / denom

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language.",
    "Python is commonly used for software development.",
    "Chicken is a type of food.",
    "Cats and dogs make great pets."
]

embeddings = model.encode(sentences).tolist()

print(f"Number of sentences embedded: {len(embeddings)}")
print(f"Dimensions per vector: {len(embeddings[0])}\n")

sim_1_vs_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_vs_3 = cosine_similarity(embeddings[0], embeddings[2])
sim_1_vs_4 = cosine_similarity(embeddings[0], embeddings[3])
sim_3_vs_4 = cosine_similarity(embeddings[2], embeddings[3])

print(f"Sentence 0: \"{sentences[0]}\"")
print(f"Sentence 1: \"{sentences[1]}\"")
print(f"Sentence 2: \"{sentences[2]}\"")
print(f"Sentence 3: \"{sentences[3]}\"\n")

print(f"Similarity (Sentence 0 vs Sentence 1): {sim_1_vs_2:.4f}")
print(f"Similarity (Sentence 0 vs Sentence 2): {sim_1_vs_3:.4f}")
print(f"Similarity (Sentence 0 vs Sentence 3): {sim_1_vs_4:.4f}")
print(f"Similarity (Sentence 2 vs Sentence 3): {sim_3_vs_4:.4f}")