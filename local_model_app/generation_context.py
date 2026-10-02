from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_GENERATION_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "local_model_generation_context",
    default={},
)


def current_generation_context() -> dict[str, Any]:
    """Return a copy of the identifiers attached to the current generation."""
    return dict(_GENERATION_CONTEXT.get())


@contextmanager
def generation_context(**values: Any) -> Iterator[dict[str, Any]]:
    """Attach durable task/chat provenance across async calls and ``to_thread``."""
    merged = current_generation_context()
    merged.update({key: value for key, value in values.items() if value is not None})
    token = _GENERATION_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _GENERATION_CONTEXT.reset(token)
