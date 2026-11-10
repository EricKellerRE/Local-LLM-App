"""Request-scoped cooperative cancellation, propagated into inference threads."""
from contextvars import ContextVar

stop_event = ContextVar('chat_stop_event', default=None)


class ResponseStopped(RuntimeError):
    pass


def check_stopped():
    event = stop_event.get()
    if event is not None and event.is_set():
        raise ResponseStopped('Stopped by user')


def generation_stop_requested(*args, **kwargs):
    event = stop_event.get()
    return event is not None and event.is_set()
