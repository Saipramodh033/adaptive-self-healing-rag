"""
Single dependency-injection factory for the entire application.

This is the ONLY place where concrete implementations are chosen.

Architecture rule: every other module imports INTERFACES.
Only this file imports concrete implementations and wires them together.

Why a single factory?
- One place to change a provider (e.g., swap Groq → OpenAI: edit one line)
- One place to trace startup failures
- One place to add logging/monitoring at startup
- Prevents dependency creep (no other file should import GroqLLMProvider directly)

Called ONCE at FastAPI application startup via lifespan().
The compiled graph and all providers are stored in app.state.deps
and reused for every request.
"""

import logging
from dataclasses import dataclass

from src.config import Settings
from src.core.graph_builder import build_graph
from src.providers.bge_embeddings import BGEEmbeddingProvider
from src.providers.chroma_store import ChromaVectorStore
from src.providers.groq_llm import GroqLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class AppDependencies:
    """
    Container holding all initialized dependencies.
    Stored in app.state.deps and injected into route handlers via request.app.state.deps.

    Fields are typed with their interfaces so route handlers never depend on
    concrete implementations directly.
    """

    graph: object          # Compiled LangGraph (CompiledGraph)
    vectorstore: object    # IVectorStore implementation (ChromaVectorStore)
    settings: Settings     # Frozen config (accessible for health checks)


def create_app_dependencies(settings: Settings) -> AppDependencies:
    """
    Constructs and wires all providers and the compiled LangGraph graph.

    Initialization order matters:
    1. Embedding provider first   — no dependencies
    2. Vector store second        — depends on embedding provider
    3. LLM provider third         — depends only on settings
    4. Graph last                 — depends on LLM + vectorstore + settings

    Raises:
        EnvironmentError: If GROQ_API_KEY or other required settings are missing.
        RuntimeError: If ChromaDB fails to open/create the collection.
        OSError: If the embedding model cannot be downloaded or loaded.
    """
    logger.info("=" * 60)
    logger.info("Initializing application dependencies ...")
    logger.info("=" * 60)

    # ── Step 1: Embedding provider ─────────────────────────────────────────────
    # BGE-small-en-v1.5 is downloaded once from HuggingFace on first run (~133 MB).
    # Subsequent runs load from the local sentence-transformers cache.
    logger.info("[1/4] Loading embedding model ...")
    embedding = BGEEmbeddingProvider(model_name=settings.embedding_model)

    # ── Step 2: Vector store ───────────────────────────────────────────────────
    # Connects to (or creates) a persistent ChromaDB collection on disk.
    # If chroma_db/ directory doesn't exist, ChromaDB creates it automatically.
    logger.info("[2/4] Connecting to ChromaDB ...")
    vectorstore = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding=embedding,
    )

    # Startup health check: warn if knowledge base is empty
    doc_count = vectorstore.get_document_count()
    if doc_count == 0:
        logger.warning(
            "ChromaDB collection is EMPTY. "
            "Run 'python scripts/ingest.py' to populate the knowledge base. "
            "RAG queries will return no results until ingestion is complete."
        )
    else:
        logger.info(f"ChromaDB ready: {doc_count} document chunks loaded")

    # ── Step 3: LLM provider ───────────────────────────────────────────────────
    # Initializes both Groq model clients (8B fast + 70B power).
    # Does NOT make any API calls at this point — clients are lazy.
    logger.info("[3/4] Initializing Groq LLM provider ...")
    llm = GroqLLMProvider(settings=settings)

    # ── Step 4: Compile LangGraph ──────────────────────────────────────────────
    # Wires all nodes and edges into a compiled, executable state machine.
    # After this call, graph.invoke() / graph.astream() are ready to use.
    logger.info("[4/4] Compiling LangGraph state machine ...")
    graph = build_graph(llm=llm, vectorstore=vectorstore, settings=settings)

    logger.info("=" * 60)
    logger.info("All dependencies initialized successfully.")
    logger.info(f"  Fast model  : {settings.fast_model}")
    logger.info(f"  Power model : {settings.power_model} (budget: {settings.power_model_daily_budget}/day)")
    logger.info(f"  Embeddings  : {settings.embedding_model}")
    logger.info(f"  ChromaDB    : {settings.chroma_persist_dir} ({doc_count} chunks)")
    logger.info("=" * 60)

    return AppDependencies(
        graph=graph,
        vectorstore=vectorstore,
        settings=settings,
    )
