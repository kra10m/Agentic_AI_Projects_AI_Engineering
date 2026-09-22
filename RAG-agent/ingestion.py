from pathlib import Path
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    text: str

def chunk_text(
    text: str, 
    source_name: str, 
    chunk_size: int = 50, 
    overlap: int = 15
) -> list[DocumentChunk]:
    """
    Splits text into chunks of `chunk_size` words with an `overlap` of words.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks

    start_idx = 0
    chunk_counter = 0

    while start_idx < len(words):

        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_name}_chunk_{chunk_counter}",
                source=source_name,
                text=chunk_text_str
            )
        )

        chunk_counter += 1
        start_idx += (chunk_size - overlap)

        if end_idx >= len(words):
            break

    return chunks

def ingest_file(file_path: str, chunk_size: int = 50, overlap: int = 15) -> list[DocumentChunk]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    return chunk_text(
        text=raw_content,
        source_name=path.name,
        chunk_size=chunk_size,
        overlap=overlap
    )

if __name__ == "__main__":
    file_to_process = "Agentic_AI_Projects_AI_Engineering/RAG-agent/cloud_architecture.txt"
    chunks = ingest_file(file_to_process, chunk_size=40, overlap=10)

    print(f"Total chunks created: {len(chunks)}\n")

    for c in chunks:
        print(f"ID: {c.chunk_id}")
        print(f"Source: {c.source}")
        print(f"Word Count: {len(c.text.split())}")
        print(f"Content Preview: {c.text[:90]}...")
        print("-" * 60)
        