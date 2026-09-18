# Autonomous Web Research Agent (Zero-Framework ReAct API)

A production-grade, autonomous research agent API built from scratch in pure Python, implementing the Reason-Act-Observe loop without relying on black-box agent orchestration frameworks (such as LangChain or CrewAI).

The agent collects multi-source web intelligence, prunes observation context to respect strict Token-Per-Minute (TPM) limits, and synthesizes citation-backed reports validated against strict Pydantic schemas.

---

## Architecture & System Design

```
                  +------------------------+
                  |   User Query (FastAPI) |
                  +----------+------------+
                              |
                     Phase 1: ReAct Collector
                              v
+------------------> Message History <------------------+
|                             |                          |
|                             v                          |
|                      LLM Reasoning                     |
|               (Evaluates state & schemas)              |
|                             |                          |
|                 Does it need more data?                |
|                 /                    \                |
|         [YES: Tool Call]        [NO: Final Answer]     |
|               |                             |         |
|               v                             |         |
|         Execute Tool                        |         |
|     (web_search with pruning)                |         |
|               |                             |         |
|               v                             |         |
|        Append 'tool' msg                     |         |
|               |                             |         |
+--------------+-------------------------------+---------+
                                              |
                                      Phase 2: Structured Output
                                              |
                                              v
                                         Pydantic Schema Synthesis
                                      (Strict JSON Mode Validation)
                                              |
                                              v
                                          HTTP 200: ResearchReport
```

### Core Engineering Highlights
- **Zero Black-Box Orchestration:** Direct integration with native LLM t-calling (OpenAI-compatible function schemas, tool_calls payloads, and tool_call_id synchronization).
- **Context & Token Hygiene:** Observations are trimmed and deduplicated before context injection, preventing token exhaustion and HTTP 413 payload limits.
- **Two-Phase Tool Isolation:** Separates dynamic ReAct gathering from the final synthesis pass, eliminating tool-call recursion conflicts during report generation.
- **Non-blocking Concurrency:** Wrapped in FastAPI with starlette.concurrency.run_in_threadpool to keep the async event loop free during synchronous agent I/O.
-$**Hermetic Testing:** Fully deterministic unit and API contract testing via pytest and unittest.mock (zero token cost in CI/CD).

---

## Tech Stack

- **Language & Runtime:** Python 3.11
- **LLM Engine:** Groq API / OpenAI SDK
- **Search Integration:** Tavily API
-$**Schema Validation:** Pydantic v2
- **Web Framework:** FastAPI, Uvicorn
-$**Testing:** Pytest, HTTPX

---

## Project Structure

```text
research-agent/
|-- .env.example
|-- .gitignore
|-- agent.py          # Core two-phase ReAct execution engine
<-- main.py           # FastAPI service wrapper
<-- requirements.txt  # Project dependencies
<-- schemas.py        # Tool definitions and Pydantic report contracts
-- test_suite.py     # Deterministic unit and API tests
`-- tools.py          # Pruned web search tool integration
```

---

## Getting Started

### 1. Clone & Configure Environment
```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent
cp .env.example .env
```
Fill in your credentials inside `.env`:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
pytest -v test_suite.py
```

### 4. Run the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation will be available at `http://127.0.0.1:8000/docs`.

---

## Sample API Contract

**Request:** `POST /research`
```json
{
  "query": "Recent breakthroughs in solid-state lithium batteries",
  "max_iterations": 2
}
```

**Response (HTTP 200 OK):**
```json
{
  "title": "Recent Advances in Solid-State Lithium Batteries",
  "executive_summary": "Recent milestones show significant movement toward scalable sulfide-based electrolytes...",
  "key_findings": [
    "Dendrite suppression via lithium-indium alloy anodes.",
    "Transition from coin cells to multi-layer pouch cell pilot lines."
  ],
  "detailed_analysis": "Comprehensive multi-source technical synthesis...",
  "sources": [
    {
      "title": "Solid State Battery Commercialization Roadmap",
      "url": 'https://example.com/battery-report'
    }
  ]
}
```