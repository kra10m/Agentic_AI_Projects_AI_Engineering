import os
import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
CHROMA_DB_DIR = os.path.join(PARENT_DIR, "chroma_db")

def generate_chunk_id(source_name: str, chunk_index: int, text: str) -> str:
    """Generates a unique, deterministic ID for each chunk based on its content and source."""
    hash_obj = hashlib.md5(text.encode("utf-8"))
    return f"{os.path.basename(source_name)}_chunk{chunk_index}_{hash_obj.hexdigest()[:6]}"

def ingest_documents():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    print(f"Loaded {len(docs)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk_id = generate_chunk_id(chunk.metadata.get('source', 'unknown'), i, chunk.page_content)
        chunk.metadata['chunk_id'] = chunk_id

    print(f"Created {len(chunks)} chunks. Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Indexing into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        ids=[chunk.metadata['chunk_id'] for chunk in chunks]
    )
    
    print(f"Successfully indexed {len(chunks)} chunks to {CHROMA_DB_DIR}.")
    
    with open(os.path.join(DATA_DIR, "chunk_reference.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"ID: {chunk.metadata['chunk_id']}\nContent Snippet: {chunk.page_content[:100]}...\n{'-'*50}\n")
    print("Saved chunk references to chunk_reference.txt")

if __name__ == "__main__":
    ingest_documents()