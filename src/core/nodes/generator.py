"""
GeneratorNode — synthesizes the final customer support response.

Uses the power 70B model (use_power=True) for high-quality, empathetic responses.
The GroqLLMProvider's budget guard ensures graceful fallback to 8B if daily
limit is reached.

Increments generation_retry_count (compared against max_generation_retries).

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import GENERATOR_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class GeneratorNode:
    """
    Synthesizes a natural, empathetic customer support response.

    Uses ONLY the graded relevant documents as context.
    Invokes the power 70B model for high-quality synthesis.
    Part of the hallucination-check regeneration loop.
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        question = state["question"]
        docs = state["documents"]
        attempt = state["generation_retry_count"] + 1
        logger.info(f"[Generator] Generating response (attempt {attempt}) with {len(docs)} docs ...")

        # Build context from relevant documents
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", f"Document {i}")
                context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "No relevant documents found."

        prompt = (
            f"Context Documents:\n{context}\n\n"
            f"Customer Question: {question}"
        )

        try:
            response = self._llm.invoke_text(
                prompt=prompt,
                system=GENERATOR_SYSTEM,
                use_power=True,  # 70B model — with budget guard fallback to 8B
            )
            logger.info(f"[Generator] Response generated ({len(response)} chars)")

            # Intercept strict escalation instruction
            # Only trigger full escalation if it's the sole response, preserving partial answers.
            if response.strip() == "I require escalation." or ("require escalation" in response.lower() and len(response.strip()) < 50):
                from src.core.prompts import ESCALATION_MESSAGE
                logger.info("[Generator] Triggered strict escalation due to insufficient context.")
                return {
                    "generation": ESCALATION_MESSAGE,
                    "is_escalated": True,
                    "generation_retry_count": attempt,
                    "thought_trace": [
                        {
                            "step": "generator",
                            "detail": {
                                "attempt": attempt,
                                "docs_used": len(docs),
                                "message": "Strict escalation triggered",
                            },
                        }
                    ],
                }

            return {
                "generation": response,
                "generation_retry_count": attempt,
                "thought_trace": [
                    {
                        "step": "generator",
                        "detail": {
                            "attempt": attempt,
                            "docs_used": len(docs),
                            "response_length": len(response),
                        },
                    }
                ],
            }

        except Exception as e:
            logger.error(f"[Generator] Error: {e}")
            fallback = (
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again or contact our support team at support@shopease.com."
            )
            return {
                "generation": fallback,
                "generation_retry_count": attempt,
                "thought_trace": [
                    {
                        "step": "generator",
                        "detail": {"attempt": attempt, "error": str(e)},
                    }
                ],
            }
