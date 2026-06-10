"""
Traditional RAG baseline for benchmarking against the Adaptive Self-Healing RAG.

This module implements a standard RAG pipeline WITHOUT any of the Adaptive RAG's
active self-healing mechanisms:
  - No Router (all queries go through retrieval)
  - No Document Grader (no relevance checking of retrieved docs)
  - No Query Rewriter (no retry on failed retrieval)
  - No Hallucination Grader (no fact-checking of the generated answer)

Instead, all guardrail logic is handled by a single TRADITIONAL_RAG_PROMPT
mega-prompt that instructs the LLM to self-regulate.

PARITY CONSTRAINTS (must match Adaptive RAG exactly):
  - Same ChromaDB collection and embedding model (BAAI/bge-small-en-v1.5)
  - Same Top-K retrieval count (5)
  - Same generator model (llama-3.3-70b-versatile)
  - Same temperature (0.0)
  - Same output schema: {"answer": str, "documents_used": int, "is_escalated": bool}

Used ONLY during Phase 8-9 LangSmith benchmark evaluation.
Do NOT import this module in production application code.
"""

import asyncio
import logging
import time
from typing import Any

from langchain_groq import ChatGroq

from src.config import load_settings
from src.core.prompts import TRADITIONAL_RAG_PROMPT
from src.providers.bge_embeddings import BGEEmbeddingProvider
from src.providers.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def _build_providers():
    """
    Initialize and return the exact same providers used by the Adaptive RAG.
    Called once at module level to avoid re-initializing on every prediction.
    """
    settings = load_settings()

    embedding = BGEEmbeddingProvider(model_name=settings.embedding_model)
    vectorstore = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding=embedding,
    )

    # Strictly locked to temperature=0.0 for parity with Adaptive RAG generator
    # Use the dedicated TRAD key if available to avoid rate limiting during eval
    api_key = settings.groq_api_key_trad or settings.groq_api_key
    llm = ChatGroq(
        model=settings.power_model,   # llama-3.3-70b-versatile — same as Adaptive
        temperature=0.0,              # Parity constraint: no creative variance
        api_key=api_key,
        max_retries=3,
    )

    return vectorstore, llm


# Initialize providers once at import time (mirrors Adaptive RAG startup behavior)
_vectorstore, _llm = _build_providers()


async def predict_traditional_rag(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Standard RAG prediction function compatible with LangSmith evaluate().

    Accepts: {"question": str}
    Returns: {"answer": str, "documents_used": int, "is_escalated": bool}

    The output schema is intentionally identical to the Adaptive RAG's output
    so the LangSmith evaluators receive the same data structure from both systems.

    Args:
        inputs: Dict with key "question" containing the user's query string.

    Returns:
        Dict with keys:
          - answer (str): The generated response text.
          - documents_used (int): Number of docs retrieved (always 4 or fewer).
          - is_escalated (bool): True if the response contains an escalation message.
    """
    question = inputs["question"]
    logger.info(f"[TraditionalRAG] Processing: {question!r}")

    # ── Step 1: Retrieve (Top-K = 5, same as Adaptive RAG) ────────────────────
    settings = load_settings()
    docs = _vectorstore.similarity_search(question, k=settings.retrieval_top_k)
    documents_used = len(docs)

    # ── Step 2: Format context from retrieved docs ─────────────────────────────
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )
    else:
        context = "No relevant documents were found in the knowledge base."

    # ── Step 3: Build the full prompt using the mega-prompt template ───────────
    full_prompt = TRADITIONAL_RAG_PROMPT.format(
        context=context,
        question=question,
    )

    # ── Step 4: Generate answer (70B, temperature=0.0) ────────────────────────
    # Run the synchronous LLM call in a thread pool to keep the function async
    loop = asyncio.get_event_loop()
    t_start = time.perf_counter()
    response = await loop.run_in_executor(
        None,
        lambda: _llm.invoke(full_prompt),
    )
    latency_ms = int((time.perf_counter() - t_start) * 1000)
    answer = response.content.strip()

    # ── Step 5: Detect escalation / safe refusal in the response ──────────────
    # Traditional RAG has no dedicated escalation node — we detect it by
    # checking for any safe refusal pattern in the generated text.
    # This covers all 3 refusal rules from TRADITIONAL_RAG_PROMPT:
    #   RULE 1 — Off-topic refusal
    #   RULE 2 — Adversarial refusal
    #   RULE 3 — Missing knowledge escalation
    escalation_keywords = [
        # RULE 3 — Missing info escalation
        "support@shopease.com",
        "1800-shopease",
        "human support agent",
        "contact our support",
        "connect you to",
        "don't have verified information",
        "don't have information about that",
        # RULE 1 — Off-topic / chitchat refusal
        "only able to assist with shopease",
        "only assist with shopease",
        "only able to help with shopease",
        # RULE 2 — Adversarial refusal
        "unable to do that",
        "i'm unable to do that",
        "cannot comply",
        "can't do that",
    ]
    is_escalated = any(kw.lower() in answer.lower() for kw in escalation_keywords)

    logger.info(
        f"[TraditionalRAG] Done — docs={documents_used}, escalated={is_escalated}"
    )

    return {
        "answer": answer,
        "documents_used": documents_used,
        "is_escalated": is_escalated,
        "route": "rag",           # Traditional RAG always routes to RAG (no router node)
        "latency_ms": latency_ms, # Pure inference time, excludes rate-limit sleep
        "retrieved_doc_sources": [doc.metadata.get("source", "unknown") for doc in docs],
    }
