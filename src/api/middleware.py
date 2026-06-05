"""
FastAPI middleware: CORS, request logging, and global error handling.

Three responsibilities kept in one file because they are all cross-cutting
concerns that apply to every request, not to any specific route.

1. CORS (Cross-Origin Resource Sharing)
   Allows the Chainlit UI (localhost:8080) to call the FastAPI backend (localhost:8000).
   Without this, the browser blocks cross-origin requests.

2. Request Logging
   Logs method, path, status code, and latency for every request.
   Essential for debugging and performance monitoring.

3. Global Exception Handler
   Catches ALL unhandled exceptions and returns a clean JSON error response.
   NEVER leaks stack traces or internal details to the client.
   Logs the full traceback server-side for debugging.
"""

import logging
import time
import traceback

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """
    Attach all middleware to the FastAPI app.
    Called once during app creation in app.py.
    Order matters: middleware is applied in reverse registration order.
    """
    _add_cors(app)
    _add_request_logging(app)
    _add_exception_handler(app)


# ── CORS ───────────────────────────────────────────────────────────────────────

def _add_cors(app: FastAPI) -> None:
    """
    Allow the Chainlit UI to call the FastAPI backend from a different port.

    allow_origins: Chainlit runs on :8080, FastAPI on :8000.
    In development both are localhost but different ports = different "origins".

    allow_methods: POST for /chat, GET for /health.
    allow_headers: Content-Type is required for JSON bodies.

    For production: replace ["http://localhost:8080"] with your actual domain.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",   # Chainlit default port
            "http://127.0.0.1:8080",  # Alternative localhost format
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )


# ── Request Logging ────────────────────────────────────────────────────────────

def _add_request_logging(app: FastAPI) -> None:
    """
    Log every incoming request with method, path, status, and latency.
    Uses @app.middleware("http") which wraps every request/response cycle.
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} "
            f"[{latency_ms:.1f}ms]"
        )
        return response


# ── Global Exception Handler ───────────────────────────────────────────────────

def _add_exception_handler(app: FastAPI) -> None:
    """
    Catch ALL unhandled exceptions and return a clean JSON response.

    Why this matters:
    - Without this, FastAPI returns an HTML error page (useless for API clients)
    - Stack traces in API responses leak implementation details (security risk)
    - A consistent error format lets clients handle errors programmatically

    The full traceback is logged server-side for debugging.
    Only a safe, generic message is returned to the client.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # Log full traceback server-side
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}:\n"
            f"{traceback.format_exc()}"
        )

        error = ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred. Please try again or contact support.",
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error.model_dump(),
        )
