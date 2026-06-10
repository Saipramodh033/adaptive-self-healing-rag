"""
GeneratorNode — synthesizes the final customer support response.

Uses the power 70B model (use_power=True) for high-quality, empathetic responses.
The GroqLLMProvider's budget guard ensures graceful fallback to 8B if daily
limit is reached.

Escalation detection (Phase 8):
  The Generator now produces NATURAL LANGUAGE escalations (Scenario B/C) instead
  of magic-string trigger phrases. This file detects escalation INTENT from the
  natural language output — decoupling prompt instructions from Python code.

  Detection signals are phrases the Generator is instructed to use in Scenario B/C.
  If any signal is detected, is_escalated is set True AND the Generator's own text
  is preserved as the user-facing message (not replaced by a generic template).

Increments generation_retry_count (compared against max_generation_retries in edges).
Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import GENERATOR_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)

# ── Escalation intent signals ──────────────────────────────────────────────────
# Phrases the Generator is instructed to use in Scenario B (partial) and C (full).
# Detecting ANY of these signals means is_escalated should be True.
# These match the exact phrases written in GENERATOR_SYSTEM RULE 2.
_ESCALATION_SIGNALS = [
    "wasn't able to find verified information",
    "was not able to find verified information",
    "unable to find verified information",
    "don't have verified information",
    "do not have verified information",
    "contact support@shopease.com",
    "contact our support team",
    "please contact support",
    "connecting you with a human",
    "1-800-shop-ease",
]


def _detect_escalation(response: str) -> bool:
    """
    Returns True if the Generator's response signals a Scenario B or C outcome.

    Checks the response against known escalation phrases the Generator is
    instructed to use. Case-insensitive. Any single match is sufficient.

    This replaces the old magic-string check:
        response.strip() == "I require escalation."
    which the 70B model reliably failed to produce verbatim.
    """
    lowered = response.lower()
    return any(signal in lowered for signal in _ESCALATION_SIGNALS)


class GeneratorNode:
    """
    Synthesizes a natural, empathetic customer support response.

    Uses ONLY the graded relevant documents as context.
    Invokes the power 70B model for high-quality synthesis.
    Part of the hallucination-check regeneration loop.

    Escalation behaviour:
    - If the Generator's natural language signals escalation intent (Scenario B/C),
      is_escalated is set True and the Generator's own text is preserved.
    - The generic ESCALATION_MESSAGE template is NOT used here — that is
      the _escalation_node's job when all retry budgets are exhausted.
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
            context = "No relevant documents found in the knowledge base."

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

            # ── Escalation detection (natural language intent) ──────────────────
            # Replaces the old brittle magic-string check. Detect Scenario B/C
            # from the Generator's natural language output and set is_escalated
            # so evaluation metrics are honest. The Generator's text is preserved
            # as-is — it carries the user-facing escalation message.
            is_escalated = _detect_escalation(response)
            if is_escalated:
                logger.info("[Generator] Escalation intent detected in response. Setting is_escalated=True.")

            result = {
                "generation": response,
                "generation_retry_count": attempt,
                "thought_trace": [
                    {
                        "step": "generator",
                        "detail": {
                            "attempt": attempt,
                            "docs_used": len(docs),
                            "response_length": len(response),
                            "escalation_detected": is_escalated,
                        },
                    }
                ],
            }

            # Only include is_escalated in the returned dict if True —
            # avoids overwriting a previously set True value on retries.
            if is_escalated:
                result["is_escalated"] = True

            return result

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
