"""Shared pytest configuration for UN Voting Records tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import UN_VOTES_CSV_PATH


def _real_dataset_available() -> bool:
    """True when the full UN voting CSV is present.

    The dataset is gitignored (362MB), so it is absent in CI and on fresh
    checkouts but present in local dev.
    """
    try:
        p = Path(UN_VOTES_CSV_PATH)
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests that require the real (gitignored) dataset when it is absent.

    Without this, a fresh checkout or a CI run — neither of which has the
    362MB CSV — would hit hard failures from the Flask endpoint tests (their
    ``client`` fixture calls ``load_data()``) and the one direct data-loading
    smoke test. Skipping those keeps the logic/unit suite (which runs entirely
    on synthetic fixtures) green everywhere, which is what makes an automated
    CI test workflow viable. When the dataset IS present (local dev), nothing
    is skipped and the full suite runs.
    """
    if _real_dataset_available():
        return
    skip_no_data = pytest.mark.skip(
        reason="real UN dataset not present (gitignored); data-dependent test skipped"
    )
    for item in items:
        needs_dataset = (
            "client" in getattr(item, "fixturenames", ())
            or item.nodeid.endswith("test_quick.py::test_data_loading")
        )
        if needs_dataset:
            item.add_marker(skip_no_data)
