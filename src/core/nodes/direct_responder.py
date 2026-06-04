"""
DirectResponderNode — handles chitchat/greetings without RAG retrieval.

Uses the fast 8B model (no 70B budget consumed for chitchat).
Called when RouterNode classifies the question as 'chitchat'.

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import DIRECT_RESPONSE_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class DirectResponderNode:
    """
    Handles casual messages, greetings, and off-topic questions.

    Bypasses retrieval entirely — uses the fast 8B model for a warm,
    brief response and an offer to help with shopping-related questions.
    No 70B budget consumed.
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        question = state["question"]
        logger.info(f"[DirectResponder] Handling chitchat: '{question[:60]}...'")

        try:
            response = self._llm.invoke_text(
                prompt=question,
                system=DIRECT_RESPONSE_SYSTEM,
                use_power=False,  # Chitchat uses fast 8B — preserve 70B budget
            )
            return {
                "generation": response,
                "thought_trace": [
                    {
                        "step": "direct_response",
                        "detail": "Handled as chitchat — no retrieval needed",
                    }
                ],
            }

        except Exception as e:
            logger.error(f"[DirectResponder] Error: {e}")
            return {
                "generation": (
                    "Hello! I'm ShopEase's support assistant. "
                    "How can I help you with your shopping today?"
                ),
                "thought_trace": [
                    {"step": "direct_response", "detail": f"Chitchat error: {str(e)}"}
                ],
            }
