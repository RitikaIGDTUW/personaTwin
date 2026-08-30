"""
Step 0 Audit Script (Stage4_Corrections_Checklist.md, items 1 & 3):

Item 1: Direction-map consistency check — do CES and StudentLife "mobility"
and "screen" actually measure the same underlying construct?
Item 3: Mobility feature / PAM correlation check for leakage auditing.

This reads your REAL feature schema, REAL cached direction maps, and REAL
training data. It never rebuilds or overwrites the live direction-map
caches (CES_DIRECTION_MAP_CACHE / STUDENTLIFE_DIRECTION_MAP_CACHE) — those
are read-only here, since build_sequences.py and run_sensitivity.py both
depend on them staying intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    CES_DIRECTION_MAP_CACHE,
    STUDENTLIFE_DIRECTION_MAP_CACHE,
    STUDENTLIFE_MODEL_DF_CACHE,
)
from src.preprocess_ces import build_ces_model_df


def load_cached_direction_map(cache_path: Path) -> dict[str, list[str]]:
    """Load an existing direction-map cache. Raises if it hasn't been built yet."""
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist yet. Run "
            f"`python -m src.build_sequences <dataset>` first so the real "
            f"direction map is cached before auditing it."
        )
    return json.loads(Path(cache_path).read_text())


def mobility_target_correlations(
    frame: pd.DataFrame,
    direction_map: dict[str, list[str]],
    target: str = "pam",
) -> dict[str, float]:
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} is missing from the dataframe")
    correlations = {}
    for feature in direction_map.get("mobility", []):
        if feature not in frame.columns:
            continue
        correlations[feature] = float(frame[feature].corr(frame[target]))
    return correlations


def run_step0_audit(leakage_threshold: float = 0.6) -> None:
    print("=" * 80)
    print("STEP 0 AUDIT — ITEM 1: DIRECTION-MAP CONSISTENCY CHECK (real cached maps)")
    print("=" * 80)

    ces_map = load_cached_direction_map(CES_DIRECTION_MAP_CACHE)
    sl_map = load_cached_direction_map(STUDENTLIFE_DIRECTION_MAP_CACHE)

    print("\n[Side-by-Side Feature Construction Inspection]")
    for direction in ["mobility", "screen", "activity", "sleep", "social"]:
        ces_features = ces_map.get(direction, [])
        sl_features = sl_map.get(direction, [])
        print(f"\n--- Direction: {direction.upper()} ---")
        print(f"  CES ({len(ces_features)} features):", ces_features)
        print(f"  StudentLife ({len(sl_features)} features):", sl_features)

    print(
        "\n[Item 1 — read this yourself before trusting any verdict]\n"
        "  Look at the mobility and screen lists above. Ask: does the dominant\n"
        "  feature type match between datasets (e.g. both dominated by\n"
        "  distance-traveled, or both by location-cluster-count)? If the\n"
        "  underlying signal differs, do NOT report a direct cross-dataset\n"
        "  comparison for that direction — report each dataset independently\n"
        "  and show both feature lists in an appendix table. This script only\n"
        "  prints the real lists; the construct judgment has to be made by\n"
        "  someone who knows what each raw column actually measures."
    )

    print("\n" + "=" * 80)
    print("STEP 0 AUDIT — ITEM 3: MOBILITY LEAKAGE / CORRELATION CHECK (real training data)")
    print("=" * 80)

    studentlife_df = pd.read_parquet(STUDENTLIFE_MODEL_DF_CACHE)
    ces_df = build_ces_model_df()

    for dataset_name, frame, direction_map in (
        ("CES", ces_df, ces_map),
        ("StudentLife", studentlife_df, sl_map),
    ):
        print(f"\n[{dataset_name} Mobility Feature Correlations with PAM Target]")
        correlations = mobility_target_correlations(frame, direction_map, target="pam")
        if not correlations:
            print("  No mobility features found in this frame — check direction_map/frame alignment.")
            continue
        for feature, corr in correlations.items():
            flag = (
                f"FLAG: SUSPICIOUS (>{leakage_threshold})"
                if abs(corr) > leakage_threshold
                else f"OK (<{leakage_threshold})"
            )
            print(f"  {feature:30s}: corr = {corr:+.4f} [{flag}]")


if __name__ == "__main__":
    run_step0_audit()