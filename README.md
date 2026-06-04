# Adaptive Self-Healing Customer Support RAG

> An intelligent **e-commerce customer support system** built for ShopEase.
> Routes queries through cost-optimized LLM tiers, self-corrects retrieval quality
> via automated grading loops, and fact-checks every response before delivery —
> ensuring an enterprise-level customer experience with zero hallucinations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Chainlit UI Dashboard                   │
│          (Async SSE stream + thought-trace steps)        │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP / SSE
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend API                    │
│          /health  /chat  /chat/stream (SSE)             │
└───────────────────────┬─────────────────────────────────┘
                        │  invoke() / astream()
                        ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph State Machine                     │
│                                                          │
│  START → [Router Node]                                   │
│              │                                           │
│    ┌─────────┴──────────┐                               │
│    │ (chitchat)         │ (rag)                         │
│    ▼                    ▼                               │
│ [Direct              [Retriever]                        │
│  Responder]       ChromaDB + BGE-small                  │
│  8B model              │                                │
│    │                   ▼                                │
│    │          [Document Grader]  ←──────────────┐       │
│    │            8B · per-doc                    │       │
│    │              │         │                   │       │
│    │          relevant   irrelevant             │       │
│    │              │         ▼                   │       │
│    │              │   [Query Rewriter]           │       │
│    │              │    8B · max 3×  ────────────┘       │
│    │              │         │ (exhausted)                │
│    │              ▼         ▼                            │
│    │          [Generator]  [Escalate]                   │
│    │           70B model    → human agent               │
│    │              │                                      │
│    │              ▼                                      │
│    │   [Hallucination Grader]  ──(not grounded)──┐      │
│    │       8B · max 2×                           │      │
│    │              │ (grounded)           [Regenerate]   │
│    │              ▼                      max 2× loop    │
│    └──────► Final Response ◄────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Layered Architecture (Modularity Guarantee)

The project enforces a strict one-way dependency chain:

```
Presentation Layer  →  API Layer  →  Core Layer  →  Provider Layer
   (Chainlit)           (FastAPI)     (LangGraph)    (Interfaces)
       ↓                    ↓              ↓               ↓
  Swappable to         Swappable to   Never changes    Implementations
  Streamlit/Gradio     Flask/Django   unless business   swappable via
                                      logic changes     one file each
```

**What this means in practice:**
- Swap Groq → OpenAI: edit **one file** (`src/providers/groq_llm.py`)
- Swap ChromaDB → Pinecone: edit **one file** (`src/providers/chroma_store.py`)
- Swap BGE → OpenAI embeddings: edit **one file** (`src/providers/bge_embeddings.py`)
- Swap Chainlit → Streamlit: edit **one file** (`src/ui/app.py`)

No other files change. The Core layer has **zero imports** from FastAPI, Chainlit, Groq, or ChromaDB.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **UI** | Chainlit ≥ 2.9.4 | Chat interface with collapsible thought-trace steps and real-time SSE token streaming |
| **API** | FastAPI + Uvicorn | `/health`, `/chat`, `/chat/stream` (SSE) endpoints |
| **Orchestration** | LangGraph ≥ 0.4 | Stateful cyclical graph — enables retry loops and conditional routing |
| **Fast LLM** | `llama-3.1-8b-instant` (Groq) | Routing, document grading, query rewriting, fact-checking (14,400 RPD) |
| **Power LLM** | `llama-3.3-70b-versatile` (Groq) | Final response synthesis only (1,000 RPD — budget-guarded) |
| **Vector Store** | ChromaDB (persistent, local) | CPU-only vector search with cosine similarity |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dim local embeddings, ~133 MB disk, ~300-600 MB RAM |

---

## Project Structure

```
adaptive-rag-customer-support/
│
├── .env.example                   # Config template (copy to .env)
├── pyproject.toml                 # Pinned dependencies
├── README.md
│
├── data/
│   └── knowledge_base/            # ShopEase e-commerce source documents
│       ├── policies/              # Refund, shipping, privacy, warranty
│       ├── faqs/                  # Orders, account, products
│       └── troubleshooting/       # Payments, delivery, defects, website
│
├── scripts/
│   └── ingest.py                  # One-time batch ingestion into ChromaDB
│
├── src/
│   ├── config.py                  # Frozen Settings dataclass — single source of truth
│   ├── dependencies.py            # Single factory that wires all providers + graph
│   │
│   ├── providers/                 # ── PROVIDER LAYER ──
│   │   ├── interfaces.py          # ILLMProvider, IEmbeddingProvider, IVectorStore (ABCs)
│   │   ├── groq_llm.py            # Groq 8B/70B two-tier provider + budget guard
│   │   ├── bge_embeddings.py      # BGE-small-en-v1.5 local CPU embeddings
│   │   └── chroma_store.py        # ChromaDB persistent vector store
│   │
│   ├── core/                      # ── CORE LAYER (zero framework imports) ──
│   │   ├── state.py               # GraphState TypedDict (Annotated thought_trace)
│   │   ├── prompts.py             # All LLM prompts centralized in one file
│   │   ├── edges.py               # Conditional routing logic (pure functions)
│   │   ├── graph_builder.py       # LangGraph assembly and compilation
│   │   └── nodes/
│   │       ├── router.py          # Intent classifier (chitchat vs rag)
│   │       ├── retriever.py       # ChromaDB similarity search
│   │       ├── doc_grader.py      # Per-document relevance grader
│   │       ├── query_rewriter.py  # Query optimizer for failed retrievals
│   │       ├── generator.py       # 70B response synthesis
│   │       ├── hallucination_grader.py  # Fact-check against source docs
│   │       └── direct_responder.py      # Chitchat handler (8B, no retrieval)
│   │
│   ├── api/                       # ── API LAYER ──
│   │   ├── app.py                 # FastAPI app factory + lifespan startup
│   │   ├── routes.py              # /health, /chat, /chat/stream
│   │   ├── schemas.py             # Pydantic ChatRequest / ChatResponse models
│   │   └── middleware.py          # CORS, request logging, global error handler
│   │
│   └── ui/                        # ── PRESENTATION LAYER ──
│       └── app.py                 # Chainlit UI with @cl.Step thought-trace display
│
├── tests/
│   ├── conftest.py                # Mock providers (MockLLMProvider, MockVectorStore)
│   ├── test_nodes.py              # Node unit tests (zero API calls)
│   ├── test_graph.py              # End-to-end graph integration tests
│   └── test_api.py                # FastAPI endpoint tests
│
└── chroma_db/                     # ChromaDB persistent storage (gitignored)
```

---

## Key Design Decisions

### 1. `thought_trace` uses `Annotated[List[dict], operator.add]`
Each graph node returns **only its new trace entry** as a list. LangGraph's
`operator.add` reducer automatically accumulates all entries across nodes.
Without this, LangGraph's default last-write-wins would silently drop earlier
trace entries — a hard-to-debug data loss bug.

### 2. Two-tier LLM routing with 70B budget guard
The Groq free tier caps `llama-3.3-70b-versatile` at **1,000 requests/day**.
The system:
- Uses the 8B model for all structured tasks (routing, grading, fact-checking)
- Reserves the 70B model exclusively for final response synthesis
- Logs a warning at 80% daily usage
- Automatically falls back to the 8B model at the budget limit

### 3. Raw ChromaDB over `langchain-chroma`
Using `chromadb` directly (not the LangChain wrapper) gives full control over
embedding injection, distance metric selection, and query result structure.
Explicit `include=["documents", "metadatas", "distances"]` ensures similarity
scores are available for observability and logging.

### 4. All providers implement abstract interfaces
`src/providers/interfaces.py` defines `ILLMProvider`, `IEmbeddingProvider`,
and `IVectorStore`. Graph nodes only import these interfaces — never concrete
implementations. Dependency injection via `src/dependencies.py` wires
everything at startup.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com) (sign up, takes 30 seconds)
- **Windows only**: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) required for ChromaDB's HNSWLIB

### 1. Clone & Install

```bash
git clone https://github.com/Saipramodh033/adaptive-self-healing-rag.git
cd adaptive-self-healing-rag
pip install -e .
```

### 2. Configure

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY=your_key_here`. All other defaults work out of the box.

### 3. Ingest the Knowledge Base

```bash
python scripts/ingest.py
```

Downloads BGE-small model (~133 MB on first run), embeds all documents, stores in ChromaDB.

### 4. Start the FastAPI Backend

```bash
uvicorn src.api.app:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health`

### 5. Start the Chainlit UI

```bash
chainlit run src/ui/app.py --port 8080
```

Open [http://localhost:8080](http://localhost:8080) and start chatting 🚀

### 6. Run Tests

```bash
pytest tests/ -v
```

All tests use mock providers — **zero API calls, runs fully offline**.

---

## Groq Free Tier Rate Limits

| Model | Requests/Min | Tokens/Min | **Requests/Day** |
|---|---|---|---|
| `llama-3.1-8b-instant` | 30 | 6,000 | **14,400** |
| `llama-3.3-70b-versatile` | 30 | 12,000 | **1,000** ⚠️ |

The 70B daily budget guard ensures the system **never crashes** when the limit
is reached — it degrades gracefully to 8B quality responses instead.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status, document count, model names |
| `POST` | `/chat` | Synchronous chat — returns full response + thought trace |
| `POST` | `/chat/stream` | Server-Sent Events stream — real-time tokens + trace events |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your return policy?"}'
```

### Example Response

```json
{
  "answer": "ShopEase offers a 30-day return window...",
  "route": "rag",
  "thought_trace": [
    {"step": "router",    "detail": "Classified as: rag"},
    {"step": "retriever", "detail": {"count": 4, "sources": ["refund_and_returns.md"]}},
    {"step": "doc_grader","detail": {"passed": 3, "filtered": 1}},
    {"step": "generator", "detail": {"attempt": 1, "docs_used": 3}},
    {"step": "hallucination_grader", "detail": {"grounded": true}}
  ],
  "documents_used": 3,
  "retries": {"query_rewrites": 0, "regenerations": 0},
  "is_escalated": false
}
```
