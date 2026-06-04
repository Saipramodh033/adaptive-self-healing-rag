"""
Conditional edge routing logic — the "traffic controller" of the graph.

All routing decisions live here in one file.
Each function is a pure function of state — no side effects, easy to test.

Routing summary:
  Router → chitchat → direct_responder → END
  Router → rag → retriever → doc_grader
  doc_grader → relevant → generator
  doc_grader → irrelevant + retries left → query_rewriter → retriever (loop)
  doc_grader → irrelevant + exhausted → escalate → END
  generator → hallucination_grader
  hallucination_grader → grounded → END
  hallucination_grader → not grounded + retries left → generator (loop)
  hallucination_grader → not grounded + exhausted → escalate → END
"""

import logging

from src.core.state import GraphState

logger = logging.getLogger(__name__)


def route_after_classification(state: GraphState) -> str:
    """
    After RouterNode: direct chitchat to responder, RAG queries to retriever.

    Returns: 'direct_responder' | 'retriever'
    """
    route = state.get("route_decision", "rag")
    logger.debug(f"[Edge] route_after_classification → {route}")
    return "direct_responder" if route == "chitchat" else "retriever"


def route_after_grading(state: GraphState, max_rewrites: int) -> str:
    """
    After DocGraderNode:
    - At least one relevant doc → proceed to generator
    - No relevant docs + retries available → rewrite query
    - No relevant docs + retries exhausted → escalate to human

    Returns: 'generator' | 'query_rewriter' | 'escalate'
    """
    if state.get("docs_are_relevant", False):
        logger.debug("[Edge] route_after_grading → generator")
        return "generator"

    rewrite_count = state.get("query_rewrite_count", 0)
    if rewrite_count < max_rewrites:
        logger.debug(
            f"[Edge] route_after_grading → query_rewriter "
            f"(attempt {rewrite_count + 1}/{max_rewrites})"
        )
        return "query_rewriter"

    logger.warning(
        f"[Edge] route_after_grading → escalate "
        f"(rewrites exhausted: {rewrite_count}/{max_rewrites})"
    )
    return "escalate"


def route_after_hallucination_check(state: GraphState, max_retries: int) -> str:
    """
    After HallucinationGraderNode:
    - Response is grounded → deliver to user
    - Not grounded + retries available → regenerate
    - Not grounded + retries exhausted → escalate to human

    Returns: 'end' | 'generator' | 'escalate'
    """
    if state.get("generation_is_grounded", True):
        logger.debug("[Edge] route_after_hallucination_check → end")
        return "end"

    retry_count = state.get("generation_retry_count", 0)
    if retry_count < max_retries:
        logger.debug(
            f"[Edge] route_after_hallucination_check → generator "
            f"(retry {retry_count + 1}/{max_retries})"
        )
        return "generator"

    logger.warning(
        f"[Edge] route_after_hallucination_check → escalate "
        f"(retries exhausted: {retry_count}/{max_retries})"
    )
    return "escalate"
