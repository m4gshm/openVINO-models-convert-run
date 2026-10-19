"""Interrupting of a running request when a newer one arrives.

The IDE autocomplete sends a completion request per keystroke; only the last one matters.
The registry keeps the stop event of the running request and sets it as soon as a newer
request starts, so the older generation stops at the next token and frees the device.
"""
import logging
import threading

log = logging.getLogger(__name__)


class InterruptRegistry:
    """Keeps the stop event of the active request and interrupts it on the next start."""

    def __init__(self):
        """Create an empty registry."""
        self._lock = threading.Lock()
        self._current: threading.Event | None = None

    def start(self) -> threading.Event:
        """Register a new request and interrupt the previously registered one.

        Returns the stop event of the new request: it is set when a newer request starts.
        """
        event = threading.Event()
        with self._lock:
            previous = self._current
            self._current = event
        if previous is not None:
            log.info("interrupt the previous request")
            previous.set()
        return event

    def finish(self, event: threading.Event) -> None:
        """Release the request slot: the given event is no longer the active one."""
        with self._lock:
            if self._current is event:
                self._current = None

    def is_active(self, event: threading.Event) -> bool:
        """Return True when the given event is the one of the active request."""
        with self._lock:
            return self._current is event

    def has_active(self) -> bool:
        """Return True when a request is registered as the active one."""
        with self._lock:
            return self._current is not None
