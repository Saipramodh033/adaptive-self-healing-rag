"""
LangGraph state schema for the Adaptive Self-Healing RAG system.

Design rules:
1. `thought_trace` uses Annotated[List[dict], operator.add] — LangGraph
   AUTO-APPENDS entries. Nodes return ONLY their new entry.
   Without this, last-write-wins would silently drop earlier trace entries.

2. `documents` uses plain List[Document] (last-write-wins / REPLACE).
   The doc grader returns only the relevant subset — appending would cause
   filtered-out docs to persist (the opposite of what we want).

3. All retry counters live in state (not closures/globals) so the graph
   is fully resumable and inspectable at any point.
"""

import operator
from typing import Annotated, List, Literal

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Complete state for one customer support interaction.

    Passed through every node in the LangGraph state machine.
    Each node returns a partial dict — LangGraph merges it with existing state.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str
    """The user's original question. May be rewritten by QueryRewriterNode."""

    # ── Routing ────────────────────────────────────────────────────────────────
    route_decision: Literal["chitchat", "rag"]
    """
    Routing outcome from RouterNode.
    - 'chitchat': friendly direct response, no retrieval
    - 'rag': full retrieval + grading + synthesis pipeline
    """

    # ── Retrieval & Generation ─────────────────────────────────────────────────
    documents: List[Document]
    """
    Retrieved documents. REPLACE behavior (last-write-wins).
    DocGraderNode overwrites this with only the relevant subset.
    """

    generation: str
    """Final LLM-generated response text."""

    # ── Self-Healing Counters ──────────────────────────────────────────────────
    query_rewrite_count: int
    """
    Number of query rewrites attempted so far.
    Compared against Settings.max_query_rewrites in edge routing.
    """

    generation_retry_count: int
    """
    Number of response regenerations attempted so far.
    Compared against Settings.max_generation_retries in edge routing.
    """

    # ── Grading Outcomes ───────────────────────────────────────────────────────
    docs_are_relevant: bool
    """
    True if DocGraderNode found at least one relevant document.
    False triggers query rewrite loop.
    """

    generation_is_grounded: bool
    """
    True if HallucinationGraderNode confirms generation is document-grounded.
    False triggers response regeneration loop.
    """

    # ── Observability ──────────────────────────────────────────────────────────
    thought_trace: Annotated[List[dict], operator.add]
    """
    Step-by-step reasoning log for the Chainlit UI thought-trace display.

    Uses Annotated[..., operator.add] — LangGraph AUTOMATICALLY APPENDS
    each node's new entry. Nodes return ONLY [{step, detail}] — never
    prepend state['thought_trace'] + [...] (that would double-append).

    Each entry format: {"step": str, "detail": str | dict}
    """

    # ── Terminal ───────────────────────────────────────────────────────────────
    is_escalated: bool
    """
    True if all retry budgets were exhausted.
    Signals to the UI to show a human escalation message.
    """
