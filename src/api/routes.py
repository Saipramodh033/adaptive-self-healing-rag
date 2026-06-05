"""
FastAPI route handlers for all API endpoints.

Endpoints:
  GET  /health          - System status, doc count, active models
  POST /chat            - Synchronous chat (full response after pipeline completes)
  POST /chat/stream     - SSE streaming chat (real-time tokens + thought-trace events)

Design notes:
- All handlers are async (non-blocking I/O throughout)
- Dependencies accessed via request.app.state.deps (set at startup in lifespan)
- Initial state construction is explicit and complete (no hidden defaults)
- /chat/stream uses Server-Sent Events (SSE) for real-time streaming
  The Chainlit UI consumes SSE via httpx-sse
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    RetryStats,
    ThoughtStep,
)

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["Customer Support"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_initial_state(question: str) -> dict:
    """
    Build a clean initial GraphState dict for a new conversation.
    Every field must be present — LangGraph does not tolerate missing keys.
    thought_trace starts as an empty list (the Annotated reducer will append to it).
    """
    return {
        "question": question,
        "generation": "",
        "documents": [],
        "route_decision": "",
        "docs_are_relevant": False,
        "generation_is_grounded": False,
        "query_rewrite_count": 0,
        "generation_retry_count": 0,
        "thought_trace": [],
        "is_escalated": False,
    }


def _result_to_response(result: dict) -> ChatResponse:
    """
    Convert a completed LangGraph state dict to a ChatResponse Pydantic model.
    Handles missing or malformed fields gracefully with safe defaults.
    """
    raw_trace = result.get("thought_trace", [])
    thought_trace = [
        ThoughtStep(
            step=entry.get("step", "unknown"),
            detail=entry.get("detail", ""),
        )
        for entry in raw_trace
        if isinstance(entry, dict)
    ]

    return ChatResponse(
        answer=result.get("generation", "I was unable to generate a response."),
        route=result.get("route_decision", "unknown"),
        thought_trace=thought_trace,
        documents_used=len(result.get("documents", [])),
        retries=RetryStats(
            query_rewrites=result.get("query_rewrite_count", 0),
            regenerations=result.get("generation_retry_count", 0),
        ),
        is_escalated=result.get("is_escalated", False),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@chat_router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Returns system status, ChromaDB document count, and active model names.",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Lightweight health check — no LLM calls, just reads from app.state.deps.
    Used by the Chainlit UI on startup and by monitoring tools.
    """
    deps = request.app.state.deps
    doc_count = deps.vectorstore.get_document_count()

    is_healthy = doc_count > 0
    message = (
        None if is_healthy
        else "ChromaDB is empty. Run 'python scripts/ingest.py' to load the knowledge base."
    )

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        document_count=doc_count,
        models={
            "fast": deps.settings.fast_model,
            "power": deps.settings.power_model,
        },
        message=message,
    )


@chat_router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Synchronous chat",
    description=(
        "Runs the full LangGraph pipeline and returns the complete response "
        "after all nodes have executed. Use /chat/stream for real-time streaming."
    ),
)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Synchronous endpoint: blocks until the full pipeline completes.

    Best for: API clients, testing, integrations that don't support SSE.
    Use /chat/stream for the Chainlit UI (better UX with real-time updates).
    """
    deps = request.app.state.deps
    logger.info(f"[/chat] Received: '{body.question[:80]}...'")

    initial_state = _build_initial_state(body.question)

    # ainvoke() is async — does not block the event loop while Groq API calls run
    result = await deps.graph.ainvoke(initial_state)

    response = _result_to_response(result)
    logger.info(
        f"[/chat] Completed - route={response.route}, "
        f"docs={response.documents_used}, "
        f"escalated={response.is_escalated}"
    )
    return response


@chat_router.post(
    "/chat/stream",
    summary="Streaming chat (SSE)",
    description=(
        "Streams the LangGraph pipeline as Server-Sent Events. "
        "Emits 'thought_trace' events for each pipeline step and "
        "'token' events for each response word. "
        "Consumed by the Chainlit UI via httpx-sse."
    ),
)
async def chat_stream(request: Request, body: ChatRequest):
    """
    SSE streaming endpoint — real-time thought-trace + response token streaming.

    SSE format (text/event-stream):
        event: thought_trace
        data: {"step": "router", "detail": "Classified as: rag"}

        event: token
        data: ShopEase

        event: done
        data: {}

    The Chainlit UI parses these events via httpx-sse's aconnect_sse().
    """
    deps = request.app.state.deps
    logger.info(f"[/chat/stream] Received: '{body.question[:80]}...'")

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Yields SSE-formatted strings for each pipeline event.
        SSE format: 'event: <type>\ndata: <json>\n\n'
        """
        try:
            initial_state = _build_initial_state(body.question)
            final_generation = ""

            # astream() with stream_mode="updates" yields {node_name: {changed_fields}}
            # after each node completes — one snapshot per node execution.
            async for state_snapshot in deps.graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                for node_name, node_update in state_snapshot.items():
                    # Stream thought_trace entries as each node produces them.
                    # In stream_mode="updates", node_update contains ONLY the new
                    # entries returned by this node (not the full accumulated list).
                    new_trace = node_update.get("thought_trace", [])
                    for entry in new_trace:
                        event_data = json.dumps(entry, default=str)
                        yield f"event: thought_trace\ndata: {event_data}\n\n"

                    # Capture the generation whenever a node sets it.
                    # The last node to set "generation" wins (generator or escalate).
                    if "generation" in node_update and node_update["generation"]:
                        final_generation = node_update["generation"]

            # Stream the final answer word by word for a typewriter effect.
            # No second ainvoke() needed — we captured it from the stream above.
            if final_generation:
                words = final_generation.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield f"event: token\ndata: {token}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            logger.error(f"[/chat/stream] Stream error: {e}")
            error_event = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )
