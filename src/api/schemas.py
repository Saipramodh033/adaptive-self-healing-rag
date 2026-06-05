"""
Pydantic request and response schemas for the FastAPI endpoints.

Pydantic validates every incoming request against these models automatically.
If a field is missing or the wrong type, FastAPI returns a 422 error before
your route handler is ever called.

Schema design notes:
- ChatRequest: strict input validation (min/max length prevents abuse)
- ThoughtStep: mirrors GraphState.thought_trace entry format exactly
- ChatResponse: structured output that matches the full graph result
- HealthResponse: lightweight status check (no LLM calls needed)

All schemas are importable by tests without loading the full FastAPI app.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Incoming chat message from the user (via Chainlit UI or direct API call).
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question or message",
        examples=["What is your return policy?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for future conversation history support",
    )


# ── Outbound ───────────────────────────────────────────────────────────────────

class ThoughtStep(BaseModel):
    """
    A single entry in the thought-trace log.
    Mirrors one dict from GraphState.thought_trace.
    """

    step: str = Field(
        description="Node name: router, retriever, doc_grader, query_rewriter, "
                    "generator, hallucination_grader, direct_response, escalate",
    )
    detail: Any = Field(
        description="Step-specific detail — either a string or a structured dict",
    )


class RetryStats(BaseModel):
    """Tracks self-healing loop iteration counts."""

    query_rewrites: int = Field(
        default=0,
        description="Number of query rewrites performed (max = MAX_QUERY_REWRITES)",
    )
    regenerations: int = Field(
        default=0,
        description="Number of response regenerations performed (max = MAX_GENERATION_RETRIES)",
    )


class ChatResponse(BaseModel):
    """
    Full response from a synchronous /chat call.
    Returned after the entire LangGraph pipeline completes.
    """

    answer: str = Field(description="The final customer support response")
    route: str = Field(description="Routing decision: 'chitchat' or 'rag'")
    thought_trace: List[ThoughtStep] = Field(
        description="Step-by-step reasoning log from the LangGraph pipeline"
    )
    documents_used: int = Field(
        description="Number of knowledge base documents used for generation"
    )
    retries: RetryStats = Field(
        description="Self-healing retry statistics for this interaction"
    )
    is_escalated: bool = Field(
        description="True if all retry budgets were exhausted and escalation occurred"
    )


class HealthResponse(BaseModel):
    """
    Lightweight system status check.
    Used by monitoring tools and the pre-flight check in the Chainlit UI.
    """

    status: str = Field(
        description="'healthy' if system is ready, 'degraded' if ChromaDB is empty"
    )
    document_count: int = Field(
        description="Number of document chunks in ChromaDB (0 means ingest.py not run)"
    )
    models: Dict[str, str] = Field(
        description="Active model names: {'fast': '...', 'power': '...'}"
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional human-readable status note",
    )


class ErrorResponse(BaseModel):
    """
    Standardized error response body.
    Returned by the global exception handler for all unhandled errors.
    Never leaks internal stack traces to the client.
    """

    error: str = Field(description="Short error type identifier")
    detail: str = Field(description="Human-readable error description")
