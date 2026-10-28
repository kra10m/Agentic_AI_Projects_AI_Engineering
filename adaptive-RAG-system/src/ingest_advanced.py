import os
import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
CHROMA_DB_DIR = os.path.join(PARENT_DIR, "chroma_db_advanced")

def generate_chunk_id(source_name: str, text: str) -> str:
    hash_obj = hashlib.md5(text.encode("utf-8"))
    return f"{os.path.basename(source_name)}_{hash_obj.hexdigest()[:6]}"

def ingest_advanced_documents():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    advanced_chunks = []
    for doc in docs:
        splits = markdown_splitter.split_text(doc.page_content)
        
        for split in splits:
            source_name = doc.metadata.get('source', 'unknown')
            split.metadata['source'] = source_name
            
            chunk_id = generate_chunk_id(source_name, split.page_content)
            split.metadata['chunk_id'] = chunk_id
            advanced_chunks.append(split)

    print(f"Created {len(advanced_chunks)} structurally-aware chunks.")
    
    with open(os.path.join(DATA_DIR, "chunk_reference_advanced.txt"), "w", encoding="utf-8") as f:
        for chunk in advanced_chunks:
            f.write(f"ID: {chunk.metadata['chunk_id']}\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"Content Snippet: {chunk.page_content[:100]}...\n{'-'*50}\n")
            
    print("Saved advanced chunk references. Ready to embed and index.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Indexing into advanced ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=advanced_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        ids=[chunk.metadata['chunk_id'] for chunk in advanced_chunks]
    )
    print(f"Successfully indexed {len(advanced_chunks)} advanced chunks to {CHROMA_DB_DIR}.")

if __name__ == "__main__":
    ingest_advanced_documents()