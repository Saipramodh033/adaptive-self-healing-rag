"""
RetrieverNode — performs ChromaDB similarity search.

Fetches top-k document chunks most similar to the current question.
The question may have been rewritten by QueryRewriterNode on retry loops.

Returns ONLY the new thought_trace entry — the Annotated reducer auto-appends.
"""

import logging

from src.core.state import GraphState
from src.providers.interfaces import IVectorStore

logger = logging.getLogger(__name__)


class RetrieverNode:
    """
    Performs semantic similarity search against the ChromaDB knowledge base.
    Returns top-k document chunks for downstream grading and generation.
    """

    def __init__(self, vectorstore: IVectorStore, top_k: int = 4):
        self._store = vectorstore
        self._top_k = top_k

    def __call__(self, state: GraphState) -> dict:
        question = state["question"]
        logger.info(f"[Retriever] Searching for: '{question[:80]}...'")

        try:
            docs = self._store.similarity_search(question, k=self._top_k)
            sources = [
                doc.metadata.get("source", "unknown") for doc in docs
            ]
            logger.info(f"[Retriever] Retrieved {len(docs)} documents")

            return {
                "documents": docs,
                "retrieved_doc_sources": sources,  # Full list (pre-grading) for Recall@K eval
                "thought_trace": [
                    {
                        "step": "retriever",
                        "detail": {
                            "count": len(docs),
                            "sources": list(set(sources)),
                        },
                    }
                ],
            }

        except Exception as e:
            logger.error(f"[Retriever] Error: {e}")
            return {
                "documents": [],
                "thought_trace": [
                    {"step": "retriever", "detail": f"Retrieval error: {str(e)}"}
                ],
            }
