# Adaptive Self-Healing Customer Support RAG

An intelligent **e-commerce customer support system** that dynamically routes queries through cost-optimized LLM paths, self-corrects retrieval quality via automated grading loops, and fact-checks every response before delivery — with zero hallucinations.

---

## Architecture

```
Chainlit UI  →  FastAPI Backend  →  LangGraph State Machine
                                          │
                         ┌────────────────┴───────────────────┐
                         │                                     │
                   [Chitchat Path]                      [RAG Path]
                   Direct 8B LLM                   ChromaDB + BGE-small
                                                         │
                                                   Doc Grader (8B)
                                                   ┌─────┴──────┐
                                                Valid       Invalid
                                                         Query Rewriter
                                                           (max 3×)
                                                         │
                                                 Generator (70B)
                                                         │
                                              Hallucination Grader (8B)
                                              ┌──────────┴──────────┐
                                           Grounded          Not Grounded
                                        Final Response       Regenerate (max 2×)
```

## Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Chainlit (WebSocket streaming + thought-trace steps) |
| **API** | FastAPI + Uvicorn (SSE streaming) |
| **Orchestration** | LangGraph (stateful cyclical graph) |
| **Fast LLM (8B)** | `llama-3.1-8b-instant` via Groq free tier |
| **Power LLM (70B)** | `llama-3.3-70b-versatile` via Groq free tier |
| **Vector Store** | ChromaDB (local persistent, CPU-only) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via sentence-transformers |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Free [Groq API key](https://console.groq.com)
- On Windows: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for ChromaDB)

### 2. Install

```bash
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 4. Ingest Knowledge Base

```bash
python scripts/ingest.py
```

### 5. Run

**Terminal 1 — FastAPI backend:**
```bash
uvicorn src.api.app:app --reload --port 8000
```

**Terminal 2 — Chainlit UI:**
```bash
chainlit run src/ui/app.py --port 8080
```

Open [http://localhost:8080](http://localhost:8080) 🚀

### 6. Test

```bash
pytest tests/ -v
```

---

## Project Structure

```
src/
├── config.py              # Centralized settings (loaded from .env)
├── dependencies.py        # Single dependency-injection factory
├── providers/
│   ├── interfaces.py      # Abstract contracts (ILLMProvider, IVectorStore, IEmbedding)
│   ├── groq_llm.py        # Groq implementation (with 70B daily budget guard)
│   ├── chroma_store.py    # ChromaDB implementation
│   └── bge_embeddings.py  # BGE-small sentence-transformers implementation
├── core/
│   ├── state.py           # LangGraph TypedDict state schema
│   ├── prompts.py         # All LLM prompts in one place
│   ├── nodes/             # Individual graph node handlers
│   ├── edges.py           # Conditional routing logic
│   └── graph_builder.py   # Graph assembly and compilation
├── api/
│   ├── app.py             # FastAPI app factory + lifespan
│   ├── routes.py          # /health, /chat, /chat/stream
│   ├── schemas.py         # Pydantic request/response models
│   └── middleware.py      # CORS, logging, error handling
└── ui/
    └── app.py             # Chainlit UI with thought-trace visualization
```

## Modularity Guarantee

The architecture enforces strict layered dependencies:

```
Presentation → API → Core → Providers (Interfaces)
```

- **Swap Groq → OpenAI**: edit one file (`providers/groq_llm.py`)
- **Swap ChromaDB → Pinecone**: edit one file (`providers/chroma_store.py`)  
- **Swap Chainlit → Streamlit**: edit one file (`ui/app.py`)

No other files change.

## Groq Free Tier Limits

| Model | RPM | TPM | RPD |
|---|---|---|---|
| `llama-3.1-8b-instant` | 30 | 6,000 | 14,400 |
| `llama-3.3-70b-versatile` | 30 | 12,000 | **1,000** |

The system includes a **70B daily budget guard** that warns at 80% and falls back to the 8B model at 95% — the system stays alive all day.
