# Adaptive Self-Healing Customer Support RAG

> An intelligent **e-commerce customer support system** for ShopEase.
> Routes queries through cost-optimised LLM tiers, self-corrects retrieval failures
> via automated grading loops, fact-checks every response before delivery —
> and is benchmarked against a Traditional RAG baseline using an LLM-as-a-Judge
> evaluation framework on LangSmith.

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
│  START → [Router Node]  ←── adversarial → [Refusal]    │
│              │                                           │
│    ┌─────────┴──────────┐                               │
│    │(chitchat/OOD)      │ (rag)                         │
│    ▼                    ▼                               │
│ [Direct              [Retriever]                        │
│  Responder]       ChromaDB + BGE-small                  │
│  8B model              │                                │
│    │                   ▼                                │
│    │          [Document Grader]  ←──────────────┐      │
│    │            8B · per-doc                    │      │
│    │              │         │                   │      │
│    │          relevant   irrelevant             │      │
│    │              │         ▼                   │      │
│    │              │   [Query Rewriter]           │      │
│    │              │    8B · max 3×  ────────────┘      │
│    │              │         │ (exhausted)               │
│    │              ▼         ▼                           │
│    │          [Generator]  [Escalate]                  │
│    │           70B model    → human agent              │
│    │              │                                     │
│    │              ▼                                     │
│    │   [Hallucination Grader]  ──(not grounded)──┐     │
│    │       8B · max 2×                           │     │
│    │              │ (grounded)           [Regenerate]  │
│    │              ▼                      max 2× loop   │
│    └──────► Final Response ◄────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Self-Healing Loops

| Loop | Trigger | Max Attempts | Fallback |
|---|---|---|---|
| **Query Rewrite** | DocGrader finds 0 relevant docs | 3 rewrites | Escalate to human |
| **Regeneration** | HallucinationGrader flags hallucination | 2 retries | Escalate to human |

---

## Layered Architecture

The project enforces a strict one-way dependency chain:

```
Presentation  →  API Layer  →  Core Layer  →  Provider Layer
  (Chainlit)     (FastAPI)    (LangGraph)     (Interfaces)
      ↓               ↓            ↓                ↓
 Swappable to    Swappable to  Never changes    Implementations
 Streamlit/      Flask/Django  unless business  swappable via
 Gradio                        logic changes    one file each
```

- Swap Groq → OpenAI: edit **one file** (`src/providers/groq_llm.py`)
- Swap ChromaDB → Pinecone: edit **one file** (`src/providers/chroma_store.py`)
- Swap BGE → OpenAI embeddings: edit **one file** (`src/providers/bge_embeddings.py`)

No other files change. The Core layer has **zero imports** from FastAPI, Chainlit, Groq, or ChromaDB.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **UI** | Chainlit ≥ 2.9.4 | Chat interface with collapsible thought-trace steps and real-time SSE token streaming |
| **API** | FastAPI + Uvicorn | `/health`, `/chat`, `/chat/stream` (SSE) endpoints |
| **Orchestration** | LangGraph ≥ 0.4 | Stateful cyclical graph — enables retry loops and conditional routing |
| **Fast LLM** | `llama-3.1-8b-instant` (Groq) | Routing, document grading, query rewriting, fact-checking (14,400 RPD free tier) |
| **Power LLM** | `llama-3.3-70b-versatile` (Groq) | Final response synthesis only (1,000 RPD — budget-guarded with 8B fallback) |
| **Vector Store** | ChromaDB (persistent, local) | CPU-only vector search with cosine similarity |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | 384-dim local embeddings, ~133 MB disk, ~300–600 MB RAM |
| **Evaluation** | LangSmith + Groq LLM-as-Judge | 6-metric benchmark comparing Adaptive vs Traditional RAG |

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
│   ├── knowledge_base/            # ShopEase e-commerce source documents
│   │   ├── policies/              # Refund, shipping, warranty
│   │   ├── faqs/                  # Orders, account management, payments
│   │   └── troubleshooting/       # Delivery issues, defects
│   └── eval_dataset.json          # 20-question golden dataset (6 categories)
│
├── scripts/
│   ├── ingest.py                  # One-time batch ingestion into ChromaDB
│   ├── run_ls_evals.py            # Main benchmark runner (Traditional vs Adaptive)
│   ├── evaluators.py              # LLM-as-Judge evaluation functions (6 metrics)
│   ├── create_ls_dataset.py       # Uploads golden dataset to LangSmith
│   ├── fetch_comparison.py        # Fetches latest LangSmith results + side-by-side diff
│   ├── fetch_docs.py              # Debug utility: inspect retrieved documents
│   ├── smoke_test_traditional_rag.py  # Quick sanity check for the baseline
│   ├── spot_check.py              # Manual spot check for a single question
│   ├── test_doc_grader.py         # Interactive doc grader testing
│   ├── test_hallucination_grader.py   # Interactive hallucination grader testing
│   └── test_router.py             # Interactive router testing
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
│   │   ├── prompts.py             # All LLM prompts centralised in one file
│   │   ├── edges.py               # Conditional routing logic (pure functions)
│   │   ├── graph_builder.py       # LangGraph assembly and compilation
│   │   ├── naive_rag.py           # Traditional RAG baseline (single-prompt, no loops)
│   │   └── nodes/
│   │       ├── router.py          # Intent classifier (chitchat / rag / adversarial / OOD)
│   │       ├── retriever.py       # ChromaDB similarity search
│   │       ├── doc_grader.py      # Per-document relevance grader (batch JSON)
│   │       ├── query_rewriter.py  # Query optimiser for failed retrievals (3 attempts)
│   │       ├── generator.py       # 70B response synthesis with partial-answer support
│   │       ├── hallucination_grader.py  # Fact-check against source docs
│   │       └── direct_responder.py      # Chitchat / OOD handler (8B, no retrieval)
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
└── tests/
    ├── conftest.py                # Mock providers (MockLLMProvider, MockVectorStore)
    ├── test_nodes.py              # Node unit tests (zero API calls)
    ├── test_graph.py              # End-to-end graph integration tests
    └── test_api.py                # FastAPI endpoint tests
```

---

## Evaluation Framework

The project includes a complete **LLM-as-a-Judge benchmark** comparing the Adaptive RAG system against a Traditional RAG baseline (single-prompt, no self-healing loops).

### Golden Dataset — 20 Questions, 6 Categories

| Category | Count | What it tests |
|---|---|---|
| `standard_easy` | 3 | Basic single-document policy questions |
| `standard_hard` | 2 | Multi-document cross-reference questions |
| `ambiguous` | 4 | Multi-intent questions requiring synthesis |
| `missing_info` | 4 | Questions with no answer in the knowledge base |
| `adversarial` | 4 | Prompt injection, jailbreaks, social engineering |
| `chitchat` | 3 | Greetings, thanks, off-topic deflection |

### 6 Evaluation Metrics

| Metric | Judge | Scale | What it measures |
|---|---|---|---|
| **Faithfulness** | 70B | 0–3 → normalised 0–1 | Are all factual claims grounded in source docs? |
| **Helpfulness** | 70B | 0–1 binary | Did the response correctly address user intent? |
| **Completeness** | 70B | 0–2 → normalised 0–1 | Were all sub-questions in multi-part queries answered? |
| **Escalation Quality** | 8B | 0–2 → normalised 0–1 | Quality of human-handoff messages (channel + context) |
| **Safe Failure Rate** | 70B | 0–1 binary | Did the system correctly refuse unanswerable/adversarial queries? |
| **Retriever Recall@4** | Rule-based | 0–1 | Fraction of ground-truth source docs in top-4 retrieved |

### Benchmark Results (Latest Run)

| Metric | Traditional RAG | Adaptive RAG |
|---|---|---|
| Faithfulness | 2.52 / 3 | 2.40 / 3 |
| Helpfulness | 0.80 | 0.80 |
| Completeness | 0.50 | 0.67 |
| Safe Failure Rate | 1.00 | 1.00 |
| Escalation Quality | 0.25 | 0.63 |
| Retriever Recall@4 | 0.864 | 0.778 |
| **Composite Score** | **1.230** | **1.266** |

> **Note:** Results were produced with a minimal 8-document knowledge base.
> At this KB scale, the self-healing loops have limited room to demonstrate their advantage.
> See [Design Decisions](#key-design-decisions) for the scale context.

### Running the Benchmark

```bash
# Upload the golden dataset to LangSmith (one-time setup)
python scripts/create_ls_dataset.py

# Run the full benchmark (both systems, ~25 min on free tier)
python scripts/run_ls_evals.py

# Or run one system only
python scripts/run_ls_evals.py --target adaptive
python scripts/run_ls_evals.py --target traditional

# Fetch side-by-side output comparison
python scripts/fetch_comparison.py
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com) (sign up, takes 30 seconds)
- Free [LangSmith API key](https://smith.langchain.com) (sign up, for evaluation only)
- **Windows only:** [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) required for ChromaDB's HNSWLIB

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

Edit `.env` and fill in your keys:
```
GROQ_API_KEY=your_groq_key_here
GROQ_API_KEY_JUDGE=your_groq_key_for_evaluation   # can be same as above
LANGSMITH_API_KEY=your_langsmith_key_here
```

### 3. Ingest the Knowledge Base

```bash
python scripts/ingest.py
```

Downloads the BGE-small model (~133 MB on first run), embeds all documents, stores in ChromaDB.

### 4. Start the FastAPI Backend

```bash
uvicorn src.api.app:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health`

### 5. Start the Chainlit UI

```bash
chainlit run src/ui/app.py --port 8080
```

Open [http://localhost:8080](http://localhost:8080) and start chatting.

### 6. Run Unit Tests

```bash
pytest tests/ -v
```

All tests use mock providers — **zero API calls, runs fully offline**.

---

## Key Design Decisions

### 1. `thought_trace` uses `Annotated[List[dict], operator.add]`
Each graph node returns **only its new trace entry** as a list. LangGraph's `operator.add` reducer automatically accumulates all entries across nodes. Without this, the default last-write-wins would silently drop earlier trace entries — a hard-to-debug data loss bug.

### 2. Two-tier LLM routing with 70B budget guard
Groq's free tier caps `llama-3.3-70b-versatile` at **1,000 requests/day**.

| Node | Model | Rationale |
|---|---|---|
| Router | 8B | Binary classification — 8B is sufficient |
| DocGrader | 8B | Per-document yes/no — pattern matching |
| QueryRewriter | 8B | Lexical transformation — no user-facing output |
| HallucinationGrader | 8B | Fact-checking against explicit documents |
| Generator | 70B | User-facing output — quality matters |
| DirectResponder | 8B | Chitchat — low stakes, short output |

The system logs a warning at 80% daily 70B usage and automatically falls back to 8B at the limit.

### 3. Isolated judge API key for evaluation
`GROQ_API_KEY_JUDGE` is separate from the inference key (`GROQ_API_KEY`). This prevents the evaluation framework's LLM-as-Judge calls from consuming the inference budget during benchmark runs — they operate against independent rate limits.

### 4. Traditional RAG baseline for honest comparison
`src/core/naive_rag.py` implements a single-prompt Traditional RAG with identical guardrails. All benchmarks run both systems on the same 20 questions. This prevents the Adaptive system from being evaluated in isolation — every metric is a relative comparison.

### 5. Raw ChromaDB over `langchain-chroma`
Using `chromadb` directly gives full control over embedding injection, distance metric selection, and query result structure. Explicit `include=["documents", "metadatas", "distances"]` ensures similarity scores are available for retrieval observability.

### 6. Scale context for the architecture
The Adaptive self-healing architecture earns its complexity at **200+ document knowledge bases** where:
- Retrieval quality degrades and DocGrader filtering becomes essential
- Multi-document cross-reference queries are the norm
- Query rewriting is required to bridge natural language and policy vocabulary

At the current 8-document demo KB, Traditional RAG with a well-crafted 70B prompt is a credible competitor. The architecture is the correct foundation for production scale — not over-engineering for a demo.

---

## Groq Free Tier Rate Limits

| Model | Requests/Min | Tokens/Min | **Requests/Day** |
|---|---|---|---|
| `llama-3.1-8b-instant` | 30 | 6,000 | **14,400** |
| `llama-3.3-70b-versatile` | 30 | 12,000 | **1,000** ⚠️ |

The 70B daily budget guard ensures the system **never crashes** at the limit — it degrades gracefully to 8B quality responses.

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
  "is_escalated": false,
  "thought_trace": [
    {"step": "router",               "detail": "Classified as: rag"},
    {"step": "retriever",            "detail": {"count": 4, "sources": ["refund_and_returns.md"]}},
    {"step": "doc_grader",           "detail": {"passed": 3, "filtered": 1}},
    {"step": "generator",            "detail": {"attempt": 1, "docs_used": 3}},
    {"step": "hallucination_grader", "detail": {"grounded": true}}
  ]
}
```

---

## Roadmap

- [ ] Expand knowledge base to 60+ documents (digital products, bulk orders, loyalty rewards, regional shipping)
- [ ] Scale golden dataset to 50 questions (5+ samples per category for statistically reliable metrics)
- [ ] Implement blueprint prompt improvements (liberal DocGrader, partial-answer HallucinationGrader carve-out)
- [ ] Add retrieval hybrid search (BM25 + dense vector fusion)
- [ ] Add conversation memory for multi-turn support interactions
