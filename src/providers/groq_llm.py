"""
Groq LLM provider implementation with two-tier model routing and budget guard.

Tier 1 — Fast (llama-3.1-8b-instant):
  - Routing, document grading, hallucination fact-checking
  - 14,400 requests/day on Groq free tier
  - JSON mode via .with_structured_output()

Tier 2 — Power (llama-3.3-70b-versatile):
  - Final response synthesis only
  - 1,000 requests/day on Groq free tier
  - Budget guard: warns at 80%, falls back to 8B at daily limit

Rate limit handling:
  - Groq returns HTTP 429 with Retry-After header
  - Exponential backoff: 1s → 2s → 4s (max 3 retries)

API notes (validated):
  - Use model= (not model_name=) — current langchain-groq API
  - .with_structured_output() for guaranteed JSON responses
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Type

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from groq import RateLimitError

from src.config import Settings
from src.providers.interfaces import ILLMProvider

logger = logging.getLogger(__name__)


# ── Structured output schemas for JSON mode ────────────────────────────────────

class RouteOutput(BaseModel):
    route: str = Field(description="'chitchat', 'rag', 'out_of_domain', or 'adversarial'")


class RelevanceOutput(BaseModel):
    relevant: str = Field(description="'yes' or 'no'")


class BatchRelevanceOutput(BaseModel):
    results: list[RelevanceOutput] = Field(description="A list of relevance results corresponding to each document evaluated in order.")


class GroundednessOutput(BaseModel):
    grounded: str = Field(description="'yes' or 'no'")


# ── Provider implementation ────────────────────────────────────────────────────

class GroqLLMProvider(ILLMProvider):
    """
    Two-tier Groq LLM provider.

    Fast model (8B): all structured JSON tasks — low latency, high daily quota
    Power model (70B): final synthesis only — richer output, tight daily quota

    The budget guard ensures the system degrades gracefully (falls back to 8B)
    rather than crashing when the 70B daily limit (1,000 RPD) is reached.
    """

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry

    def __init__(self, settings: Settings):
        logger.info(
            f"Initializing Groq provider — fast={settings.fast_model}, "
            f"power={settings.power_model}, budget={settings.power_model_daily_budget}"
        )

        # Validated API: use model= (not model_name=)
        self._fast = ChatGroq(
            model=settings.fast_model,
            temperature=0,
            api_key=settings.groq_api_key,
            max_retries=0,  # We handle retries manually for control
        )
        self._power = ChatGroq(
            model=settings.power_model,
            temperature=0.3,
            api_key=settings.groq_api_key,
            max_retries=0,
        )
        self._daily_budget = settings.power_model_daily_budget
        self._power_calls_today = 0  # In-memory counter (resets on restart)

    # ── Public interface ───────────────────────────────────────────────────────

    def invoke_json(self, prompt: str, system: str) -> dict:
        """
        Always uses fast model (8B). Returns parsed dict.
        Infers schema from system prompt content.
        """
        schema = self._infer_schema(system)
        messages = [
            ("system", system),
            ("human", prompt),
        ]
        try:
            structured_llm = self._fast.with_structured_output(schema)
            result = self._retry_sync(structured_llm, messages)
            return result.dict() if hasattr(result, "dict") else result
        except Exception as e:
            logger.error(f"invoke_json failed: {e}")
            return self._json_fallback(system)

    def invoke_text(self, prompt: str, system: str, *, use_power: bool = False) -> str:
        """
        Fast model by default. Power model when use_power=True.
        Falls back to fast model if 70B daily budget is exhausted.
        """
        client = self._select_client(use_power)
        messages = [("system", system), ("human", prompt)]
        try:
            result = self._retry_sync(client, messages)
            return result.content
        except Exception as e:
            logger.error(f"invoke_text failed: {e}")
            return (
                "I'm experiencing technical difficulties. "
                "Please try again or contact support directly."
            )

    async def astream_text(self, prompt: str, system: str) -> AsyncGenerator[str, None]:
        """
        Async generator yielding text tokens for SSE streaming.
        Uses power model with budget guard.
        """
        client = self._select_client(use_power=True)
        messages = [("system", system), ("human", prompt)]
        try:
            async for chunk in client.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"astream_text failed: {e}")
            yield "I'm experiencing technical difficulties. Please try again."

    # ── Private helpers ────────────────────────────────────────────────────────

    def _select_client(self, use_power: bool) -> ChatGroq:
        """Select model tier with budget guard for 70B model."""
        if not use_power:
            return self._fast

        if self._power_calls_today >= self._daily_budget:
            logger.warning(
                f"[BUDGET EXHAUSTED] 70B daily limit reached "
                f"({self._power_calls_today}/{self._daily_budget}). "
                f"Falling back to 8B model."
            )
            return self._fast

        if self._power_calls_today >= int(self._daily_budget * 0.8):
            logger.warning(
                f"[BUDGET WARNING] 70B model at "
                f"{self._power_calls_today}/{self._daily_budget} "
                f"({round(self._power_calls_today/self._daily_budget*100)}%)"
            )

        self._power_calls_today += 1
        return self._power

    def _retry_sync(self, client, messages: list, max_retries: int = _MAX_RETRIES):
        """
        Synchronous retry with exponential backoff.
        Reads Retry-After header on Groq 429 responses.
        """
        delay = self._RETRY_BASE_DELAY
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return client.invoke(messages)
            except RateLimitError as e:
                last_error = e
                # Groq 429: respect Retry-After header if present
                retry_after = getattr(e, "response", None)
                wait = delay
                if retry_after and hasattr(retry_after, "headers"):
                    ra = retry_after.headers.get("Retry-After")
                    if ra:
                        wait = float(ra)
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limited (attempt {attempt+1}/{max_retries+1}). "
                        f"Waiting {wait:.1f}s ..."
                    )
                    time.sleep(wait)
                    delay *= 2  # Exponential backoff
            except Exception as e:
                last_error = e
                # If it's a 400 Bad Request (like tool_use_failed), do not retry. It will just fail again and waste tokens.
                if "400" in str(e) or "invalid_request_error" in str(e):
                    logger.warning(f"Deterministic 400 error caught. Failing fast to trigger fallback...")
                    raise e
                    
                if attempt < max_retries:
                    logger.warning(f"LLM error (attempt {attempt+1}): {e}. Retrying ...")
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(f"All {max_retries+1} attempts failed: {last_error}")

    def _infer_schema(self, system: str) -> type:
        """Infer the appropriate Pydantic schema from the system prompt."""
        s = system.lower()
        if "chitchat" in s or "route" in s:
            return RouteOutput
        if "relevance_batch" in s:
            return BatchRelevanceOutput
        if "relevant" in s:
            return RelevanceOutput
        if "grounded" in s:
            return GroundednessOutput
        return RouteOutput  # Safe default

    def _json_fallback(self, system: str) -> dict:
        """Safe fallback values when JSON parsing fails."""
        s = system.lower()
        if "route" in s:
            return {"route": "rag"}       # Default to RAG (safer)
        if "relevance_batch" in s:
            # Safe default: assume all are relevant if parsing fails so we don't drop context
            return {"results": [{"relevant": "yes"}] * 4}
        if "relevant" in s:
            return {"relevant": "yes"}    # Default to relevant (pass-through)
        if "grounded" in s:
            return {"grounded": "yes"}    # Default to grounded (avoid infinite loop)
        return {}
