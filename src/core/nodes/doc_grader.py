"""
DocGraderNode — grades each retrieved document for relevance.

Grades every document individually using the fast 8B model.
Returns only the relevant subset (REPLACE behavior on documents field).

Error boundary: defaults to 'relevant' on parsing failure.
Safer to pass documents through than discard them on model error.

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging
from typing import List

from langchain_core.documents import Document

from src.core.prompts import DOC_GRADER_SYSTEM
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


class DocGraderNode:
    """
    Grades each retrieved document for relevance to the user's question.

    Filters out irrelevant documents before passing context to the generator.
    If all documents are irrelevant, triggers the query rewrite loop.
    """

    def __init__(self, llm: ILLMProvider):
        self._llm = llm

    def __call__(self, state: GraphState) -> dict:
        question = state["question"]
        docs = state["documents"]
        logger.info(f"[DocGrader] Grading {len(docs)} documents ...")

        relevant_docs: List[Document] = []
        grading_details = []

        for i, doc in enumerate(docs):
            prompt = (
                f"User Question: {question}\n\n"
                f"Document:\n{doc.page_content[:1000]}"  # Cap at 1000 chars
            )
            try:
                result = self._llm.invoke_json(
                    prompt=prompt,
                    system=DOC_GRADER_SYSTEM,
                )
                is_relevant = result.get("relevant", "yes") == "yes"
            except Exception as e:
                logger.warning(f"[DocGrader] Grading error on doc {i}: {e}. Defaulting to relevant.")
                is_relevant = True  # Error boundary: safer to keep than discard

            if is_relevant:
                relevant_docs.append(doc)

            grading_details.append({
                "source": doc.metadata.get("source", f"doc_{i}"),
                "relevant": is_relevant,
                "distance": doc.metadata.get("similarity_distance", "N/A"),
            })

        passed = len(relevant_docs)
        filtered = len(docs) - passed
        logger.info(f"[DocGrader] {passed}/{len(docs)} documents relevant")

        return {
            "documents": relevant_docs,          # REPLACE — filtered subset only
            "docs_are_relevant": passed > 0,
            "thought_trace": [
                {
                    "step": "doc_grader",
                    "detail": {
                        "passed": passed,
                        "filtered": filtered,
                        "documents": grading_details,
                    },
                }
            ],
        }
