"""Langfuse tracing — the one place that owns the Langfuse client.

Nothing else in this codebase imports `langfuse`. The LLM layer
(`src/services/llm_service.py`) reaches tracing through three things here:

  * `traced_task()`   — decorates a task function so its trace is named after the feature
  * `observation()`   — a context manager around one provider call (generation / embedding)
  * `truncate()`      — the payload cap every traced string goes through

Everything degrades to a no-op rather than an error:

  * `LANGFUSE_ENABLED=false` (the default) — `get_langfuse()` returns None, `langfuse` is never
    even imported, `traced_task` returns the undecorated function, and `observation()` yields a
    null object. A deployment that does not configure Langfuse is byte-identical to one built
    before tracing existed.
  * enabled but a key is missing — one warning at startup, then the same no-op path.
  * enabled and configured but the host is unreachable — the OTEL batch exporter retries on its
    own background thread and logs there; request handling never sees it.

`flush()` / `shutdown()` are for the FastAPI lifespan, so spans buffered by background tasks are
not dropped when the process stops.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .configs import settings
from .logger import logger

if TYPE_CHECKING:  # import only for type checkers — never at runtime
    from langfuse import Langfuse


# =============================================================================
# The client
# =============================================================================

@lru_cache(maxsize=1)
def get_langfuse() -> Optional["Langfuse"]:
    """The single Langfuse client, or None when tracing is off or unconfigured.

    `import langfuse` happens inside this function on purpose: it pulls in three
    opentelemetry packages and starts an exporter thread, and none of that should happen in a
    deployment that leaves LANGFUSE_ENABLED at its default.
    """
    if not settings.LANGFUSE_ENABLED:
        return None

    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning(
            "LANGFUSE_ENABLED is true but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not both "
            "set — LLM tracing stays disabled."
        )
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            # `or None` so an unset LANGFUSE_HOST falls back to the SDK default rather than
            # sending an empty base URL.
            host=settings.LANGFUSE_HOST or None,
            environment=settings.ENVIRONMENT.value,
            release=settings.APP_VERSION,
            sample_rate=settings.LANGFUSE_SAMPLE_RATE,
            debug=settings.LANGFUSE_DEBUG,
        )
    except Exception as e:
        # A tracing backend must never keep the service from starting.
        logger.error(f"Langfuse client initialization failed; LLM tracing disabled: {e}")
        return None

    logger.info(
        f"Langfuse tracing enabled (host={settings.LANGFUSE_HOST or 'SDK default'} "
        f"environment={settings.ENVIRONMENT.value} sample_rate={settings.LANGFUSE_SAMPLE_RATE})"
    )
    return client


def tracing_enabled() -> bool:
    """True when LLM calls should be traced. Cheap after the first call (get_langfuse is cached)."""
    return get_langfuse() is not None


# =============================================================================
# Payload handling
# =============================================================================

def truncate(text: str | None) -> str | None:
    """Cap a traced string at LANGFUSE_MAX_PAYLOAD_CHARS.

    Every prompt and response that goes into a trace passes through here. The role-mapping
    prompts inline the entire KCM competency master (hundreds of KB), which would exceed
    Langfuse's per-event size limit and get the whole event rejected.
    """
    if text is None:
        return None
    limit = settings.LANGFUSE_MAX_PAYLOAD_CHARS
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} chars]"


# =============================================================================
# Observations
# =============================================================================

class _NullObservation:
    """Stand-in yielded by `observation()` when tracing is off, so call sites can always
    call `.update(...)` without an `if tracing_enabled()` branch around it."""

    def update(self, **_kwargs: Any) -> "_NullObservation":
        return self


_NULL_OBSERVATION = _NullObservation()


@contextmanager
def observation(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open one Langfuse observation, or yield a no-op stand-in when tracing is off.

    `as_type` is passed straight through to the SDK — "generation" for an LLM call,
    "embedding" for an embedding call, "span" for anything that groups them.
    """
    client = get_langfuse()
    if client is None:
        yield _NULL_OBSERVATION
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
    except Exception as e:
        # Tracing must not be able to fail a request.
        logger.warning(f"Langfuse observation '{name}' could not be started: {e}")
        yield _NULL_OBSERVATION
        return
    with cm as obs:
        yield obs


@contextmanager
def detached_observation(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Like `observation()`, but the observation is NOT made the current one.

    For use around an `async for`/`yield` loop. `observation()` attaches an OpenTelemetry
    context on entry and detaches it on exit; inside an async generator those two halves can run
    in different task contexts (an async generator body runs in whichever task resumed it),
    which produces spurious "failed to detach context" errors. This variant only records the
    parent at creation time, so nesting is still correct and there is no context to mismatch.
    """
    client = get_langfuse()
    if client is None:
        yield _NULL_OBSERVATION
        return
    try:
        obs = client.start_observation(name=name, as_type=as_type, **kwargs)
    except Exception as e:
        logger.warning(f"Langfuse observation '{name}' could not be started: {e}")
        yield _NULL_OBSERVATION
        return
    try:
        yield obs
    finally:
        try:
            obs.end()
        except Exception as e:
            logger.warning(f"Langfuse observation '{name}' could not be ended: {e}")


def traced_task(name: str | None = None, *, as_type: str = "span"):
    """Decorate an LLM task function so its Langfuse trace is named after the feature
    (`generate_frac_batch`) rather than the model that happened to serve it.

    When tracing is off the function is returned unchanged, so a disabled deployment carries no
    wrapper at all. That decision is made at import time, which is also when settings are
    already loaded — enabling tracing therefore requires a restart, as it would anyway for an
    env-var-driven setting.

    IO capture is deliberately off: task arguments include raw PDF bytes and whole organization
    dicts. The nested generation created by the provider wrapper carries the real payload, with
    binary parts replaced by placeholders and text truncated.
    """
    def decorator(func):
        if get_langfuse() is None:
            return func
        from langfuse import observe

        return observe(
            name=name or func.__name__,
            as_type=as_type,
            capture_input=False,
            capture_output=False,
        )(func)

    return decorator


# =============================================================================
# Lifecycle — called from the FastAPI lifespan in src/main.py
# =============================================================================

def flush() -> None:
    """Send anything still buffered. Best-effort: never raises."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed: {e}")


def shutdown() -> None:
    """Flush and stop the exporter. Best-effort: never raises."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.shutdown()
        logger.info("Langfuse tracing shut down")
    except Exception as e:
        logger.warning(f"Langfuse shutdown failed: {e}")
