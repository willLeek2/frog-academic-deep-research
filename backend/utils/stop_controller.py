"""Emergency stop controller using threading.Event."""

from __future__ import annotations

import threading


class StopController:
    """Thread-safe stop signal controller.

    Uses ``threading.Event`` so that any thread can check or request a stop.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        """Set the stop flag."""
        self._event.set()

    def is_stop_requested(self) -> bool:
        """Return True if stop has been requested."""
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the stop flag."""
        self._event.clear()
