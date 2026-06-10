"""
QueryRewriterNode — rewrites the user's question for better retrieval.

Called when DocGraderNode finds no relevant documents.
Increments the rewrite counter (compared against max_query_rewrites in edges).

Uses the fast 8B model (no power model budget consumed for rewrites).

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import QUERY_REWRITER_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class QueryRewriterNode:
    """
    Rewrites the current question to improve retrieval quality.

    Part of the self-healing rewrite loop:
    [Retriever] → [DocGrader: irrelevant] → [QueryRewriter] → [Retriever] → ...
    (max iterations controlled by Settings.max_query_rewrites)
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        original = state["question"]
        attempt = state["query_rewrite_count"] + 1
        logger.info(f"[QueryRewriter] Rewriting query (attempt {attempt}): '{original[:80]}...'")

        # Progressive decomposition: each attempt uses a different strategy
        attempt_hint = {
            1: f"Original question: {original}",
            2: f"SECOND ATTEMPT — focus on the primary intent only, ignore secondary questions.\nOriginal question: {original}",
            3: f"FINAL ATTEMPT — extract just the core ShopEase policy keyword (2-4 words only).\nOriginal question: {original}",
        }.get(attempt, f"Original question: {original}")

        try:
            rewritten = self._llm.invoke_text(
                prompt=attempt_hint,
                system=QUERY_REWRITER_SYSTEM,
                use_power=False,  # Rewrites use fast 8B (save 70B budget)
            ).strip()

            if not rewritten or rewritten == original:
                # If rewrite is empty or identical, add context to force change
                rewritten = f"e-commerce support: {original}"

            logger.info(f"[QueryRewriter] Rewritten: '{rewritten[:80]}...'")

            return {
                "question": rewritten,
                "query_rewrite_count": attempt,
                "thought_trace": [
                    {
                        "step": "query_rewriter",
                        "detail": {
                            "attempt": attempt,
                            "original": original,
                            "rewritten": rewritten,
                        },
                    }
                ],
            }

        except Exception as e:
            logger.error(f"[QueryRewriter] Error: {e}. Keeping original question.")
            return {
                "question": original,
                "query_rewrite_count": attempt,
                "thought_trace": [
                    {
                        "step": "query_rewriter",
                        "detail": {
                            "attempt": attempt,
                            "error": str(e),
                            "original": original,
                        },
                    }
                ],
            }
