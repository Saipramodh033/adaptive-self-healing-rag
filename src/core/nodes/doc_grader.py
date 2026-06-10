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
        
        if not docs:
            return {"documents": [], "docs_are_relevant": False, "thought_trace": []}
            
        logger.info(f"[DocGrader] Batch grading {len(docs)} documents ...")

        relevant_docs: List[Document] = []
        grading_details = []

        # Build batched prompt
        prompt_parts = [
            f"User Question: {question}\n\n"
            f"CRITICAL INSTRUCTION: There are EXACTLY {len(docs)} documents below. "
            f"Your JSON array MUST contain EXACTLY {len(docs)} items. "
            f"Stop generating immediately after the {len(docs)}th item.\n\n"
            f"Documents to evaluate:"
        ]
        for i, doc in enumerate(docs):
            prompt_parts.append(f'<document id="{i}">\n{doc.page_content[:1000]}\n</document>')
        prompt = "\n".join(prompt_parts)

        # To trigger the new batch schema, we temporarily append a keyword to the system prompt
        # that our _infer_schema uses to recognize the batch format ("relevance_batch")
        system_prompt = DOC_GRADER_SYSTEM + "\n[relevance_batch]"

        try:
            result = self._llm.invoke_json(prompt=prompt, system=system_prompt)
            batch_results = result.get("results", [])
        except Exception as e:
            logger.warning(f"[DocGrader] Batch grading error: {e}. Defaulting to all relevant.")
            batch_results = [{"relevant": "yes"}] * len(docs)

        # Process results
        for i, doc in enumerate(docs):
            # Safe boundary: if the LLM returned fewer results than docs, default to relevant
            if i < len(batch_results):
                is_relevant = batch_results[i].get("relevant", "yes") == "yes"
            else:
                logger.warning(f"[DocGrader] Result mismatch for doc {i}. Defaulting to relevant.")
                is_relevant = True

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
