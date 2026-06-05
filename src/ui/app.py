"""
Chainlit UI for the ShopEase Adaptive Self-Healing Customer Support system.

What this file does:
- On startup: checks the FastAPI backend is alive and the knowledge base is loaded
- On each message: streams the full LangGraph pipeline via SSE from FastAPI
  - Renders each pipeline step as a collapsible Chainlit Step (thought-trace)
  - Streams response tokens in real-time for a typewriter effect
  - Shows escalation messages with visual alerts when self-healing loops exhaust

SSE consumption:
- Uses httpx.AsyncClient (async HTTP, non-blocking)
- Uses aconnect_sse from httpx-sse (required — httpx alone cannot parse SSE)
- Reads three event types: 'thought_trace', 'token', 'done', 'error'

Start with:
    chainlit run src/ui/app.py --port 8080

FastAPI must be running at localhost:8000 before starting Chainlit.
"""

import json
import logging

import chainlit as cl
import httpx
from httpx_sse import aconnect_sse

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
HTTP_TIMEOUT = 120.0   # seconds — long enough for full pipeline + Groq calls

# ── Step display mapping ───────────────────────────────────────────────────────
# Maps node names to human-readable labels shown in the thought-trace steps

STEP_ICONS = {
    "router":               "🔀 Routing",
    "retriever":            "🔍 Retrieval",
    "doc_grader":           "📝 Grading Documents",
    "query_rewriter":       "✏️ Rewriting Query",
    "generator":            "🤖 Generating Response",
    "hallucination_grader": "✅ Fact-Checking",
    "direct_response":      "💬 Direct Response",
    "escalate":             "🚨 Escalating to Human",
}


# ── Detail formatter ───────────────────────────────────────────────────────────

def _format_detail(step: str, detail) -> str:
    """
    Format a thought-trace detail into a readable multi-line string.
    Handles both string details and structured dict details from nodes.
    """
    if isinstance(detail, str):
        return detail

    if step == "retriever":
        count = detail.get("count", 0)
        sources = detail.get("sources", [])
        return f"Retrieved {count} document(s)\nSources: {', '.join(sources) if sources else 'none'}"

    if step == "doc_grader":
        passed = detail.get("passed", 0)
        filtered = detail.get("filtered", 0)
        docs = detail.get("documents", [])
        lines = [f"{passed} relevant / {filtered} filtered out"]
        for doc in docs:
            icon = "✓" if doc.get("relevant") else "✗"
            source = doc.get("source", "unknown")
            dist = doc.get("distance", "N/A")
            lines.append(f"  {icon} {source} (distance: {dist})")
        return "\n".join(lines)

    if step == "query_rewriter":
        attempt = detail.get("attempt", "?")
        original = detail.get("original", "")
        rewritten = detail.get("rewritten", "")
        return (
            f"Attempt {attempt}\n"
            f"Original:  {original}\n"
            f"Rewritten: {rewritten}"
        )

    if step == "generator":
        attempt = detail.get("attempt", "?")
        docs_used = detail.get("docs_used", 0)
        resp_len = detail.get("response_length", 0)
        return f"Attempt {attempt} | {docs_used} document(s) used | {resp_len} chars generated"

    if step == "hallucination_grader":
        grounded = detail.get("grounded", True)
        note = detail.get("note", "")
        status = "Grounded - response is factually supported" if grounded else "NOT grounded - will regenerate"
        return f"{status}\n{note}" if note else status

    if step == "escalate":
        rewrites = detail.get("query_rewrites", 0)
        retries = detail.get("generation_retries", 0)
        return (
            f"All retry budgets exhausted\n"
            f"Query rewrites attempted: {rewrites}\n"
            f"Generation retries attempted: {retries}\n"
            f"Handing off to human support agent."
        )

    # Fallback: pretty-print any dict
    return json.dumps(detail, indent=2, default=str)


# ── Chainlit lifecycle hooks ───────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    """
    Runs once when a user opens the chat window.

    Checks the FastAPI backend health and greets the user.
    If the backend is down or ChromaDB is empty, shows a warning.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{API_BASE}/health")
            health = resp.json()

        status = health.get("status", "unknown")
        doc_count = health.get("document_count", 0)
        models = health.get("models", {})

        if status == "healthy":
            greeting = (
                f"👋 Welcome to **ShopEase** Customer Support!\n\n"
                f"I'm powered by an Adaptive Self-Healing RAG system.\n\n"
                f"**System Status:** Ready\n"
                f"**Knowledge base:** {doc_count} document chunks loaded\n"
                f"**Models:** {models.get('fast', 'N/A')} (fast) · {models.get('power', 'N/A')} (power)\n\n"
                f"Ask me anything about orders, returns, shipping, payments, or your account!"
            )
        else:
            greeting = (
                f"⚠️ **System Warning: Knowledge base is empty**\n\n"
                f"Please run `python scripts/ingest.py` to load the ShopEase knowledge base.\n\n"
                f"I can still respond, but RAG-based answers will not be grounded in verified documents."
            )

    except httpx.ConnectError:
        greeting = (
            f"❌ **Cannot connect to the API backend**\n\n"
            f"Please start the FastAPI server first:\n"
            f"```\nuvicorn src.api.app:app --reload --port 8000\n```"
        )

    await cl.Message(content=greeting).send()


# ── Message handler ────────────────────────────────────────────────────────────

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handles each incoming user message.

    Flow:
    1. Creates an empty Chainlit message for streaming
    2. Opens SSE connection to FastAPI /chat/stream
    3. For each 'thought_trace' event: renders a collapsible Step
    4. For each 'token' event: streams token into the message
    5. On 'error' event: shows error message
    6. On 'done' event: finalizes the message
    """
    # Create empty message — tokens will be streamed into it
    response_msg = cl.Message(content="")
    await response_msg.send()

    is_escalated = False

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            async with aconnect_sse(
                client,
                "POST",
                f"{API_BASE}/chat/stream",
                json={"question": message.content},
            ) as event_source:

                async for sse in event_source.aiter_sse():

                    # ── Thought-trace step ─────────────────────────────────────
                    if sse.event == "thought_trace":
                        try:
                            trace = json.loads(sse.data)
                            step_name = trace.get("step", "unknown")
                            detail = trace.get("detail", "")

                            label = STEP_ICONS.get(step_name, f"⚙️ {step_name}")
                            formatted = _format_detail(step_name, detail)

                            # Track escalation for post-stream UI
                            if step_name == "escalate":
                                is_escalated = True

                            async with cl.Step(name=label, type="tool") as step:
                                step.output = formatted

                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"[UI] Could not parse thought_trace event: {e}")

                    # ── Response token (typewriter streaming) ──────────────────
                    elif sse.event == "token":
                        await response_msg.stream_token(sse.data)

                    # ── Stream complete ────────────────────────────────────────
                    elif sse.event == "done":
                        break

                    # ── Error from server ──────────────────────────────────────
                    elif sse.event == "error":
                        try:
                            err = json.loads(sse.data)
                            error_msg = err.get("error", "Unknown error")
                        except Exception:
                            error_msg = sse.data

                        await response_msg.stream_token(
                            f"\n\n❌ **Error:** {error_msg}\n"
                            f"Please try again or contact support@shopease.com"
                        )
                        break

        # Finalize the streamed message
        await response_msg.update()

        # Show escalation notice as a follow-up message
        if is_escalated:
            await cl.Message(
                content=(
                    "🚨 **Escalated to Human Agent**\n\n"
                    "Our automated system was unable to resolve your query after "
                    "multiple attempts. A human support agent has been notified.\n\n"
                    "**Contact us directly:**\n"
                    "- Email: support@shopease.com\n"
                    "- Live Chat: [shopease.com/support](http://shopease.com/support)\n"
                    "- Phone: 1-800-SHOP-EASE (Mon–Fri, 9am–6pm)"
                )
            ).send()

    except httpx.ConnectError:
        await response_msg.stream_token(
            "❌ **Cannot reach the API backend.**\n\n"
            "Please ensure the FastAPI server is running:\n"
            "`uvicorn src.api.app:app --reload --port 8000`"
        )
        await response_msg.update()

    except httpx.TimeoutException:
        await response_msg.stream_token(
            "\n\n⏱️ **Request timed out.**\n\n"
            "The pipeline took too long to respond. "
            "This may happen during the first request while the embedding model warms up. "
            "Please try again."
        )
        await response_msg.update()

    except Exception as e:
        logger.error(f"[UI] Unexpected error: {e}")
        await response_msg.stream_token(
            f"\n\n❌ **Unexpected error:** {str(e)}\n"
            f"Please refresh the page and try again."
        )
        await response_msg.update()
