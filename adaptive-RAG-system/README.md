# Adaptive RAG System

An evidence-driven Retrieval-Augmented Generation (RAG) system for
answering questions about LangGraph technical documentation. The project
evolves from a semantic-search baseline into a hybrid retrieval pipeline
with cross-encoder reranking and an adaptive LangGraph workflow that
evaluates evidence, rewrites weak queries, and validates citation IDs.

## Overview

Basic semantic retrieval can find conceptually related passages but may
miss exact technical terms, class names, or API identifiers. This
project compares several retrieval approaches and combines dense search,
keyword search, rank fusion, reranking, and an evidence-driven recovery
loop.

### Key Features

-   Markdown structure-aware chunking with `MarkdownHeaderTextSplitter`
-   Deterministic chunk IDs for traceability
-   Dense semantic retrieval with ChromaDB and Hugging Face embeddings
-   Sparse keyword retrieval with BM25
-   Reciprocal Rank Fusion (RRF)
-   Cross-encoder reranking
-   Adaptive query transformation and evidence grading with LangGraph
-   Structured LLM outputs using Pydantic
-   Validation that cited chunk IDs exist in the retrieved candidate set
-   Evaluation scripts for baseline, hybrid, reranked, and adaptive
    pipelines

## Architecture

``` mermaid
flowchart TD
    A[User query] --> B[Adaptive LangGraph workflow]
    B --> C[Retrieve candidates]
    C --> D[Dense search: ChromaDB]
    C --> E[Sparse search: BM25]
    D --> F[Reciprocal Rank Fusion]
    E --> F
    F --> G[Cross-encoder reranking]
    G --> H[Evidence evaluation]
    H -->|Sufficient| I[Grounded generation]
    H -->|Insufficient and retries remain| J[Query transformation]
    J --> C
    I --> K[Pydantic answer and citations]
    K --> L[Validate cited chunk IDs]
    L --> M[Final answer]
```

The adaptive workflow limits retries using `max_attempts`. If evidence
remains insufficient after the configured attempts, the graph uses a
fallback response rather than retrying indefinitely.

## Retrieval Pipeline

### 1. Ingestion and Chunking

The project contains two ingestion approaches:

-   **Baseline ingestion (`ingest_baseline.py`):** Uses
    `RecursiveCharacterTextSplitter` with a chunk size of 1,000
    characters and overlap of 200 characters.
-   **Advanced ingestion (`ingest_advanced.py`):** Uses
    `MarkdownHeaderTextSplitter` with Markdown level-one and level-two
    headings, preserving document structure and associated header
    metadata.

Both approaches use `all-MiniLM-L6-v2` embeddings and index chunks in
ChromaDB. The advanced ingestion script assigns a content-derived chunk
ID and writes a reference file for inspection.

### 2. Dense and Sparse Retrieval

The hybrid retriever combines:

-   **Dense retrieval:** ChromaDB similarity search using
    `all-MiniLM-L6-v2` embeddings.
-   **Sparse retrieval:** `Rank-BM25` over lowercased,
    whitespace-tokenized document text to find exact keyword matches.

### 3. Reciprocal Rank Fusion

The retriever combines the dense and sparse rankings with Reciprocal
Rank Fusion (RRF). RRF uses ranks rather than attempting to compare the
raw scores from different retrieval methods directly.

### 4. Cross-Encoder Reranking

`AdvancedRetriever` sends the fused candidate query-document pairs to
`cross-encoder/ms-marco-MiniLM-L-6-v2`. It sorts the candidates by
cross-encoder score and returns the final top three chunk IDs by
default.

## Adaptive LangGraph Workflow

The adaptive workflow is implemented in `src/graph_adaptive.py` and uses
four nodes:

1.  **Retrieve:** Retrieves candidate chunks using `AdvancedRetriever`,
    with up to 10 initial candidates and the top 3 returned after
    reranking.
2.  **Evaluate evidence:** Uses an LLM with structured Pydantic output
    to judge whether retrieved documents are relevant to the current
    query.
3.  **Transform query:** Rewrites an unclear query to add technical
    context or resolve ambiguous references, then sends it through
    retrieval again.
4.  **Generate:** Produces an answer using only the retrieved context,
    or returns a fallback when the evidence remains insufficient after
    the maximum number of attempts.

The workflow stores the retrieved chunk IDs in its graph state so the
generation stage can validate citations.

## LLM Configuration

The provided Python files use the same model identifier,
`openai/gpt-oss-120b`, through the `ChatGroq` integration for the
LLM-based steps:

-   Query transformation in `src/query_transform.py`
-   Evidence grading in `src/graph_adaptive.py`
-   Grounded answer generation in `src/graph_adaptive.py`

The code sets `temperature=0` for these LLM instances. Despite the model
identifier beginning with `openai/`, the code calls it through
`langchain_groq.ChatGroq`; this README reflects the implementation shown
in the provided files.

## Citation and Grounding Safeguards

The generation prompt instructs the model to answer using only the
retrieved context and to cite the exact chunk IDs used. Pydantic schemas
define the expected structured outputs:

-   `GradeDocuments` contains a binary relevance score (`yes` or `no`).
-   `GroundedAnswer` contains an answer and a list of citation chunk
    IDs.
-   `RewrittenQuery` contains the transformed query.

Before presenting the answer, the graph filters the model-provided
citations against `retrieved_chunk_ids`. This prevents citations to
chunk IDs that were not in the retrieved candidate set. It does **not**
independently prove that each individual claim is entailed by its cited
chunk.

## Evaluation

The repository includes separate scripts for evaluating the retrieval
stages:

  -----------------------------------------------------------------------
  Script                              What it evaluates
  ----------------------------------- -----------------------------------
  `eval_baseline.py`                  Dense semantic retrieval from
                                      ChromaDB

  `eval_hybrid.py`                    Dense + BM25 retrieval with RRF

  `eval_rerank.py`                    Hybrid retrieval followed by
                                      cross-encoder reranking

  `eval_adaptive.py`                  Full adaptive LangGraph workflow
  -----------------------------------------------------------------------

The retrieval evaluation scripts calculate **Recall@k** and **Mean
Reciprocal Rank (MRR)** using `data/eval_dataset.json`, which maps
queries to expected chunk IDs. The adaptive evaluation script reports
system retrieval success at top 3; it also prints the attempts, final
transformed query, and cited sources.

### Metrics

-   **Recall@k:** The proportion of evaluation queries for which at
    least one expected chunk appears in the top *k* retrieved results.
-   **Mean Reciprocal Rank (MRR):** The average reciprocal rank of the
    first relevant result across evaluation queries.

### Results

Add the latest measured results from running the evaluation scripts
here. Results should be copied from the actual run output so the
reported scores and evaluation dataset size remain reproducible.

  -----------------------------------------------------------------------
  Pipeline                             Recall@3                       MRR
  ------------------- ------------------------- -------------------------
  Semantic baseline      Run `eval_baseline.py`    Run `eval_baseline.py`

  Hybrid retrieval         Run `eval_hybrid.py`      Run `eval_hybrid.py`

  Hybrid + reranking       Run `eval_rerank.py`      Run `eval_rerank.py`

  Adaptive graph        Reports top-3 retrieval     Not calculated by the
                                        success          current adaptive
                                                        evaluation script
  -----------------------------------------------------------------------

## Repository Structure

``` text
adaptive-RAG-system/
├── asset/
├── chroma_db/
├── chroma_db_advanced/
├── data/
│   ├── chunk_reference_advanced.txt
│   ├── chunk_reference.txt
│   ├── eval_dataset.json
│   ├── langgraph_checkpoint...   # checkpoint files shown in the workspace
│   ├── langgraph_core.md
│   └── langgraph_multi_age...    # documentation file shown in the workspace
└── src/
    ├── eval_adaptive.py
    ├── eval_baseline.py
    ├── eval_hybrid.py
    ├── eval_rerank.py
    ├── graph_adaptive.py
    ├── ingest_advanced.py
    ├── ingest_baseline.py
    ├── query_transform.py
    ├── retrieval_hybrid.py
    └── retrieval_rerank.py
```

The tree reflects the uploaded workspace screenshot and Python files.
Some filenames in the screenshot are truncated by the editor, so their
full names are not inferred here. The ChromaDB directories contain local
vector-store data and generally should not be committed if they are
large or reproducible from ingestion.

## Tech Stack

  ----------------------------------------------------------------------------
  Component                           Technology
  ----------------------------------- ----------------------------------------
  Workflow orchestration              LangGraph

  LLM integration                     LangChain `ChatGroq`

  LLM model                           `openai/gpt-oss-120b`

  Structured outputs                  Pydantic

  Vector database                     ChromaDB

  Embeddings                          Hugging Face `all-MiniLM-L6-v2`

  Sparse retrieval                    Rank-BM25

  Rank fusion                         Reciprocal Rank Fusion

  Reranker                            Sentence Transformers
                                      `cross-encoder/ms-marco-MiniLM-L-6-v2`

  Source documents                    LangGraph Markdown documentation
  ----------------------------------------------------------------------------

## Setup

### Prerequisites

-   A Python environment compatible with the installed project
    dependencies
-   A Groq API key
-   The LangGraph Markdown source files used by the ingestion scripts

### 1. Clone the repository

``` bash
git clone <repository-url>
cd adaptive-RAG-system
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

**Windows PowerShell**

``` powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the dependencies used by the project:

``` bash
pip install langgraph langchain-core langchain-groq langchain-chroma langchain-huggingface langchain-community langchain-text-splitters chromadb rank-bm25 sentence-transformers pydantic python-dotenv
```

For a reproducible setup, pin the tested versions in a
`requirements.txt` file.

### 4. Configure the API key

Create a `.env` file in the project root:

``` dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit `.env` or expose API keys in source control.

### 5. Build the vector stores

Run the ingestion scripts from the `src` directory or adjust paths to
match your execution environment:

``` bash
python src/ingest_baseline.py
python src/ingest_advanced.py
```

The baseline script writes to `chroma_db/`; the advanced script writes
to `chroma_db_advanced/`.

### 6. Run evaluations

From the repository root:

``` bash
python src/eval_baseline.py
python src/eval_hybrid.py
python src/eval_rerank.py
python src/eval_adaptive.py
```

These commands assume the paths and working directory match the uploaded
scripts. The scripts resolve data and database paths relative to their
own location.

## Limitations and Next Steps

-   Add a pinned dependency file and document tested Python/package
    versions.
-   Record evaluation outputs and dataset size for each pipeline.
-   Add claim-level support checks in addition to checking whether
    citation IDs exist.
-   Expand the evaluation dataset with ambiguous, adversarial, and
    unsupported questions.
-   Add logging, latency measurements, API error handling, and
    retry/backoff behavior.
-   Consider metadata-aware retrieval and incremental re-indexing as the
    documentation corpus grows.
-   Add automated tests for retrieval ranking, graph routing, fallback
    behavior, and citation validation.

## License

Add the license applicable to this repository before publishing or
distributing the project.
