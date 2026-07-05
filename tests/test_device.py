"""Tests for device selection."""

from __future__ import annotations

from hatedetect.device import pick_device


def test_env_override_forces_device(monkeypatch):
    # HATEDETECT_DEVICE wins over autodetection so a flaky backend can be
    # sidestepped without code changes.
    monkeypatch.setenv("HATEDETECT_DEVICE", "cpu")
    assert pick_device() == "cpu"


def test_autodetect_returns_known_device(monkeypatch):
    # With no override, the result is one of the three backends we support.
    monkeypatch.delenv("HATEDETECT_DEVICE", raising=False)
    assert pick_device() in {"cuda", "mps", "cpu"}
