"""
LangGraph graph builder — assembles and compiles the full state machine.

All dependencies are injected (llm, vectorstore, settings).
This function has no side effects and produces a fully compilable graph.

Graph structure:
  START → router
  router → [direct_responder | retriever] (conditional)
  retriever → doc_grader
  doc_grader → [generator | query_rewriter | escalate] (conditional)
  query_rewriter → retriever  (self-healing rewrite loop)
  generator → hallucination_grader
  hallucination_grader → [END | generator | escalate] (conditional)
  direct_responder → END
  escalate → END

API notes (validated):
  - Use add_edge(START, ...) — modern API; set_entry_point is legacy
  - Import START from langgraph.graph
  - Lambda closures work in add_conditional_edges
  - Class instances with __call__ work as nodes
"""

import logging

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.core.edges import (
    route_after_classification,
    route_after_grading,
    route_after_hallucination_check,
)
from src.core.nodes.direct_responder import DirectResponderNode
from src.core.nodes.doc_grader import DocGraderNode
from src.core.nodes.generator import GeneratorNode
from src.core.nodes.hallucination_grader import HallucinationGraderNode
from src.core.nodes.query_rewriter import QueryRewriterNode
from src.core.nodes.retriever import RetrieverNode
from src.core.nodes.router import RouterNode
from src.core.prompts import ESCALATION_MESSAGE
from src.core.state import GraphState
from src.providers.interfaces import ILLMProvider, IVectorStore

logger = logging.getLogger(__name__)


def _escalation_node(state: GraphState) -> dict:
    """Terminal node: delivers human escalation message when all retries exhausted."""
    logger.warning("[Escalate] All retry budgets exhausted. Escalating to human agent.")
    return {
        "generation": ESCALATION_MESSAGE,
        "is_escalated": True,
        "thought_trace": [
            {
                "step": "escalate",
                "detail": {
                    "query_rewrites": state.get("query_rewrite_count", 0),
                    "generation_retries": state.get("generation_retry_count", 0),
                    "message": "Escalated to human support agent",
                },
            }
        ],
    }


def build_graph(
    llm: ILLMProvider,
    vectorstore: IVectorStore,
    settings: Settings,
):
    """
    Factory that wires nodes + edges into a compiled LangGraph.

    All dependencies are injected — no globals, no side effects.
    Returns a CompiledGraph with .invoke(), .ainvoke(), .astream() methods.
    """
    logger.info("Building LangGraph state machine ...")

    # ── Instantiate nodes with injected dependencies ────────────────────────────
    router = RouterNode(llm)
    retriever = RetrieverNode(vectorstore, top_k=settings.retrieval_top_k)
    doc_grader = DocGraderNode(llm)
    query_rewriter = QueryRewriterNode(llm)
    generator = GeneratorNode(llm)
    hallucination_grader = HallucinationGraderNode(llm)
    direct_responder = DirectResponderNode(llm)

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("router", router)
    graph.add_node("retriever", retriever)
    graph.add_node("doc_grader", doc_grader)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("generator", generator)
    graph.add_node("hallucination_grader", hallucination_grader)
    graph.add_node("direct_responder", direct_responder)
    graph.add_node("escalate", _escalation_node)

    # ── Wire edges ─────────────────────────────────────────────────────────────
    # Entry point (modern API — add_edge(START, ...) replaces set_entry_point)
    graph.add_edge(START, "router")

    # Router → chitchat or RAG path
    graph.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "direct_responder": "direct_responder",
            "retriever": "retriever",
        },
    )

    # Linear: retriever always goes to doc_grader
    graph.add_edge("retriever", "doc_grader")

    # Doc grader → generate / rewrite / escalate
    graph.add_conditional_edges(
        "doc_grader",
        lambda s: route_after_grading(s, settings.max_query_rewrites),
        {
            "generator": "generator",
            "query_rewriter": "query_rewriter",
            "escalate": "escalate",
        },
    )

    # Self-healing rewrite loop: query_rewriter → retriever
    graph.add_edge("query_rewriter", "retriever")

    # Linear: generator always goes to hallucination check
    graph.add_edge("generator", "hallucination_grader")

    # Hallucination check → end / regenerate / escalate
    graph.add_conditional_edges(
        "hallucination_grader",
        lambda s: route_after_hallucination_check(s, settings.max_generation_retries),
        {
            "end": END,
            "generator": "generator",
            "escalate": "escalate",
        },
    )

    # Terminal edges
    graph.add_edge("direct_responder", END)
    graph.add_edge("escalate", END)

    # ── Compile ────────────────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("LangGraph state machine compiled successfully")
    return compiled
