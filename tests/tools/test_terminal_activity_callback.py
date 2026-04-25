"""Regression tests for terminal activity callback propagation."""

import threading

from tools.environments.base import _get_activity_callback, set_activity_callback


def test_activity_callback_available_from_helper_thread():
    """Terminal wait loops may run helper threads; they must still heartbeat."""
    seen = []

    def cb(message: str) -> None:
        seen.append(message)

    set_activity_callback(cb)
    try:
        result = []

        def worker():
            result.append(_get_activity_callback())

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2)

        assert result == [cb]
    finally:
        set_activity_callback(None)

    assert _get_activity_callback() is None
