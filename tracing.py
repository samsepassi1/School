"""OpenTelemetry / Arize Phoenix tracing setup.

Exports a process-wide ``tracer`` that the rest of the app uses to wrap
spans like ``moderate_text``, ``chat_turn``, ``conversation`` and
``feedback``. If Phoenix isn't available we silently fall back to a no-op
tracer so the rest of the app keeps working.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "multimodal-moderation")
_PHOENIX_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
)


def _phoenix_reachable(endpoint: str, timeout: float = 0.25) -> bool:
    """Cheap pre-flight: don't attach the exporter if Phoenix isn't listening.

    The OTLP exporter logs noisy retry messages on every flush when its
    collector is unreachable. We'd rather skip it cleanly.
    """
    try:
        from urllib.parse import urlparse
        import socket

        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _build_provider() -> TracerProvider:
    resource = Resource.create(
        {
            "service.name": _PROJECT_NAME,
            "openinference.project.name": _PROJECT_NAME,
        }
    )
    provider = TracerProvider(resource=resource)

    if not _phoenix_reachable(_PHOENIX_ENDPOINT):
        logger.info(
            "Phoenix not reachable at %s — running with in-memory tracer only.",
            _PHOENIX_ENDPOINT,
        )
        return provider

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=_PHOENIX_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as exc:
        logger.warning("Phoenix OTLP exporter unavailable: %s", exc)

    return provider


@lru_cache(maxsize=1)
def _ensure_provider_installed() -> None:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return
    trace.set_tracer_provider(_build_provider())


def get_tracer(name: str = "multimodal-moderation") -> trace.Tracer:
    """Return a tracer, installing the global provider on first call."""
    _ensure_provider_installed()
    return trace.get_tracer(name)


def instrument_pydantic_ai() -> None:
    """Best-effort: instrument the underlying Gemini SDK so agent spans appear
    in Phoenix automatically. Silently skipped when the package is missing.
    """
    try:
        from openinference.instrumentation.google_genai import (
            GoogleGenAIInstrumentor,
        )

        GoogleGenAIInstrumentor().instrument()
    except Exception as exc:
        logger.info("google-genai instrumentation skipped: %s", exc)


# Module-level tracer the rest of the app imports.
tracer: trace.Tracer = get_tracer("multimodal-moderation")
