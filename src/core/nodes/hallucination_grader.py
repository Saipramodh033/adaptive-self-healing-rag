"""
HallucinationGraderNode — fact-checks the generated response.

Uses the fast 8B model to verify every claim in the response is
supported by the source documents.

Error boundary: defaults to 'grounded' on failure to avoid infinite retry loops.

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.prompts import HALLUCINATION_GRADER_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class HallucinationGraderNode:
    """
    Verifies that the generated response is grounded in retrieved documents.

    Part of the hallucination-prevention loop:
    [Generator] → [HallucinationGrader: not grounded] → [Generator] → ...
    (max iterations controlled by Settings.max_generation_retries)
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        generation = state["generation"]
        docs = state["documents"]
        logger.info("[HallucinationGrader] Fact-checking response ...")

        if not docs:
            # No documents to check against — can't validate, pass through
            logger.warning("[HallucinationGrader] No documents to check against. Passing through.")
            return {
                "generation_is_grounded": True,
                "thought_trace": [
                    {
                        "step": "hallucination_grader",
                        "detail": {
                            "grounded": True,
                            "note": "No documents to validate against",
                        },
                    }
                ],
            }

        # Build context from all source documents
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

        prompt = (
            f"Source Documents:\n{context}\n\n"
            f"Generated Response:\n{generation}"
        )

        try:
            result = self._llm.invoke_json(
                prompt=prompt,
                system=HALLUCINATION_GRADER_SYSTEM,
            )
            is_grounded = result.get("grounded", "yes") == "yes"
            logger.info(f"[HallucinationGrader] Grounded: {is_grounded}")

            return {
                "generation_is_grounded": is_grounded,
                "thought_trace": [
                    {
                        "step": "hallucination_grader",
                        "detail": {"grounded": is_grounded},
                    }
                ],
            }

        except Exception as e:
            logger.error(f"[HallucinationGrader] Error: {e}. Defaulting to grounded.")
            # Error boundary: default to grounded to avoid infinite regeneration loops
            return {
                "generation_is_grounded": True,
                "thought_trace": [
                    {
                        "step": "hallucination_grader",
                        "detail": {
                            "grounded": True,
                            "note": f"Grader error — defaulting to grounded: {str(e)}",
                        },
                    }
                ],
            }
