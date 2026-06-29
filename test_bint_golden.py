"""Pytest entry point for the BINT golden-master test.

Run after recording a baseline (`python bint_golden.py capture`):

    PYTHONHASHSEED=0 pytest test_bint_golden.py

The hash seed is pinned for determinism; if it is not already 0 the test is
skipped with an explanatory message rather than risk a misleading result.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bint_golden as bg

BASELINE = Path(__file__).resolve().parent / "bint_golden_baseline"
PROFILE = "fast"


def _hash_seed_ok() -> bool:
    return os.environ.get("PYTHONHASHSEED") == "0"


@pytest.mark.skipif(not _hash_seed_ok(),
                    reason="set PYTHONHASHSEED=0 for a deterministic golden comparison")
@pytest.mark.skipif(not (BASELINE / "manifest.json").exists(),
                    reason="no baseline; run `python bint_golden.py capture` first")
def test_matches_baseline():
    import json
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    profile = bg.PROFILES[manifest.get("profile", PROFILE)]
    ndigits = manifest.get("round_ndigits")

    mismatches = []
    for scenario_name, seed in bg.iter_matrix(profile):
        key = bg._key(scenario_name, seed)
        snaps, _ = bg.run_one(scenario_name, seed, profile)
        got = [bg.hash_snapshot(s, ndigits) for s in snaps]
        want = manifest["hashes"].get(key)
        if got != want:
            first = next((i for i, (g, c) in enumerate(zip(want or [], got)) if g != c), 0)
            mismatches.append(f"{key} (first diverges at checkpoint #{first})")

    assert not mismatches, (
        "Simulation behaviour changed vs baseline:\n  " + "\n  ".join(mismatches)
        + "\nRun `python bint_golden.py verify` for a field-level diff."
    )


@pytest.mark.skipif(not _hash_seed_ok(),
                    reason="set PYTHONHASHSEED=0 for a deterministic comparison")
def test_model_is_deterministic():
    """Independent of any baseline: same seed must reproduce identical runs."""
    profile = bg.PROFILES[PROFILE]
    scenario_name, seed = next(bg.iter_matrix(profile))
    a, _ = bg.run_one(scenario_name, seed, profile)
    b, _ = bg.run_one(scenario_name, seed, profile)
    assert [bg.hash_snapshot(s, None) for s in a] == [bg.hash_snapshot(s, None) for s in b]
