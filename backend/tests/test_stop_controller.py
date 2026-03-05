"""Tests for StopController."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.stop_controller import StopController


def test_initial_state():
    ctrl = StopController()
    assert ctrl.is_stop_requested() is False


def test_request_stop():
    ctrl = StopController()
    ctrl.request_stop()
    assert ctrl.is_stop_requested() is True


def test_reset():
    ctrl = StopController()
    ctrl.request_stop()
    assert ctrl.is_stop_requested() is True
    ctrl.reset()
    assert ctrl.is_stop_requested() is False
