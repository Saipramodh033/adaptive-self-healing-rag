"""
RouterNode — classifies user intent as 'chitchat' or 'rag'.

Uses the fast 8B model (invoke_json) for low-latency classification.
Error boundary: defaults to 'rag' on any failure (safer to over-retrieve).

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import ROUTER_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class RouterNode:
    """
    Classifies user intent to determine the graph execution path.

    - 'chitchat' → DirectResponderNode (no retrieval, fast 8B response)
    - 'rag'      → RetrieverNode (full RAG pipeline with self-healing)
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        question = state["question"]
        logger.info(f"[Router] Classifying: '{question[:80]}...'")

        try:
            result = self._llm.invoke_json(
                prompt=question,
                system=ROUTER_SYSTEM,
            )
            route = result.get("route", "rag")
            # Validate — only accept known routes
            if route not in ("chitchat", "rag", "out_of_domain", "adversarial"):
                logger.warning(f"[Router] Unexpected route '{route}', defaulting to 'rag'")
                route = "rag"

            logger.info(f"[Router] → {route}")
            return {
                "route_decision": route,
                "thought_trace": [
                    {"step": "router", "detail": f"Classified as: {route}"}
                ],
            }

        except Exception as e:
            logger.error(f"[Router] Error: {e}. Defaulting to 'rag'")
            return {
                "route_decision": "rag",
                "thought_trace": [
                    {"step": "router", "detail": f"Classification error — defaulting to RAG"}
                ],
            }
