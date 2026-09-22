# Agentic RAG Knowledge Agent

A zero-framework, agentic Retrieval-Augmented Generation (RAG) system built from scratch in Python.

Unlike standard naive RAG systems that execute blind vector lookups on every prompt, this agent autonomously decides **whether** to retrieve, **reformulates** queries across multi-turn search steps, and synthesizes answers using a strict **two-phase grounded verification** contract with citations.

---

## Architecture Overview

```text
                  User Query
                      │
                      ▼
        ┌───────────────────────────┐
        │    Agentic Router (LLM)   │
        └─────────────┬─────────────┘
                      │
      ┌───────────────┴───────────────┐
      │                               │
[General Prompt]              [Technical Query]
      │                               │
      ▼                               ▼
Direct Response            ┌─────────────────────────────┐
                           │  search_knowledge_base()   │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                           ChromaDB (HNSW Vector Index)
                                          │
                                          ▼
                              Retrieved Context Chunks
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                        More info needed?             │
                              │                       │
                         ┌────┴────┐                  │
                         │         │                  │
                       YES         NO                 │
                         │         │                  │
                   Refine Query  Proceed              │
                         │         │                  │
                         └────┬────┘                  │
                              │                       │
                              └──────────┬────────────┘
                                         ▼
                            Structured Grounding Phase
                                         │
                                         ▼
                              Pydantic Validated Output:
                              - Grounded Answer
                              - Chunk ID Citations
                              - information_found Flag
```

---

## Core Engineering Features

- **Zero Framework Bloat:** Built purely with Python, ChromaDB, and the OpenAI-compatible SDK (Groq/OpenRouter). No LangChain or LlamaIndex wrappers.
- **Custom Ingestion Pipeline:** Implements a sliding-window text chunker with configurable token overlap to preserve semantic context across boundaries.
- **Persistent Vector Store:** Local on-disk vector database using ChromaDB with an HNSW cosine similarity index.
- **Deterministic Tool Interface:** The database lookup is registered as an OpenAI function calling schema (`search_knowledge_base`).
- **Autonomous Multi-Turn Loop:** The agent evaluates chunk sufficiency, refines search terms if results are incomplete, and stops searching once confident.
- **Strict Grounding & Citations:** Enforces source attribution and handles missing information gracefully (`information_found=False`) without hallucinating.

---

## Project Structure

```text
RAG-agent/
│
├── cloud_architecture.txt   # Raw source text data
├── ingestion.py             # Custom document loader & sliding chunker
├── embeddings.py            # Vector math & similarity foundations
├── vector_search.py         # In-memory top-k semantic search engine
├── vector_db.py             # ChromaDB persistence & HNSW indexer
├── tool_retriever.py        # Retriever packaged as an OpenAI tool contract
├── agent.py                 # Free-form autonomous agent loop
├── grounded_agent.py        # Two-phase loop with Pydantic structured output
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Installation & Setup

### 1. Clone & Navigate

```bash
git clone <your-repo-url>
cd Agentic_AI_Projects_AI_Engineering/RAG-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials

Create a `.env` file in the root of `RAG-agent/`:

```env
GROQ_API_KEY="your-groq-api-key"
# or
OPENROUTER_API_KEY="your-openrouter-key"
```

## Usage

### Step 1: Index the Knowledge Base

Run ingestion and vector database population:

```bash
python vector_db.py
```

This reads `cloud_architecture.txt`, creates sliding-window chunks, embeds them via `all-MiniLM-L6-v2`, and persists the index in `chroma_storage/`.

### Step 2: Run the Grounded Agent

Execute the agent pipeline:

```bash
python grounded_agent.py
```

### Example Output: Grounded Fact Lookup

```json
{
  "answer": "The platform uses PostgreSQL for transactional workloads because it guarantees ACID properties.",
  "citations": ["cloud_architecture.txt_chunk_2"],
  "information_found": true
}
```

### Example Output: Missing Knowledge (Hallucination Fallback)

```json
{
  "answer": "The internal documentation does not specify the team's AWS monthly budget limit.",
  "citations": [],
  "information_found": false
}
```
