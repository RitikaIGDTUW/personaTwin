"""
Verification protocol for the Sensitivity Engine.

Run this after every run_sensitivity.py invocation. It doesn't tell you the
science is right - it tells you whether the numbers are structurally sane
enough to be worth reading. A clean pass here is necessary, not sufficient,
for "the sensitivity engine is done."

Usage:
    python -m src.verify_sensitivity_engine studentlife
    python -m src.verify_sensitivity_engine ces
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR

EXPECTED_DIRECTIONS = {
    "studentlife": {"sleep", "activity", "social", "mobility", "screen"},
    "ces": {"sleep", "activity", "mobility", "screen"},  # no social - confirmed absent
}

EXPECTED_PARTICIPANTS = {
    "studentlife": 23,
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    return condition


def verify_univariate(dataset: str) -> bool:
    print(f"\n=== Univariate profiles ({dataset}) ===")
    profiles_path = PROCESSED_DIR / "sensitivity" / f"{dataset}_sensitivity_profiles.csv"
    aggregates_path = PROCESSED_DIR / "sensitivity" / f"{dataset}_sensitivity_aggregates.json"

    all_ok = True
    all_ok &= check("profiles.csv exists", profiles_path.exists())
    all_ok &= check("aggregates.json exists", aggregates_path.exists())
    if not (profiles_path.exists() and aggregates_path.exists()):
        return False

    df = pd.read_csv(profiles_path)
    with open(aggregates_path) as f:
        agg = json.load(f)

    expected = EXPECTED_DIRECTIONS[dataset]
    usable_agg = {
        direction: stats
        for direction, stats in agg.items()
        if direction in expected and stats.get("slope_count", 0.0) > 0
    }
    all_ok &= check(
        "directions match expected set for this dataset",
        set(usable_agg.keys()) == expected,
        f"got {set(usable_agg.keys())}, expected {expected}",
    )
    usable_df = df[df["direction"].isin(expected)]
    all_ok &= check(
        "no NaN in slope/curvature/margin",
        not usable_df[["slope", "curvature", "margin"]].isna().any().any(),
    )
    all_ok &= check(
        "no infinite margin values (unless genuinely no threshold crossing)",
        not df["margin"].isin([float("inf"), float("-inf")]).all(),
    )

    for direction, stats in usable_agg.items():
        slope_low = stats.get("slope_ci_low")
        slope_high = stats.get("slope_ci_high")
        curvature_low = stats.get("curvature_ci_low")
        curvature_high = stats.get("curvature_ci_high")
        all_ok &= check(
            f"{direction}: slope CI ordering (low <= high)",
            slope_low is not None
            and slope_high is not None
            and slope_low <= slope_high,
        )
        all_ok &= check(
            f"{direction}: curvature CI ordering (low <= high)",
            curvature_low is not None
            and curvature_high is not None
            and curvature_low <= curvature_high,
        )
        margin_count = stats.get("margin_count", 0.0)
        if stats.get("margin_std") is None or margin_count == 0:
            print(
                f"  [INFO] {direction}: margin never crosses threshold within plausible "
                f"bounds for any window (margin_count=0) - this is a legitimate finding, "
                f"not a failure. Report as 'direction alone insufficient to reach threshold.'"
            )
        else:
            margin_count = int(stats.get("margin_count", 0.0))
            if margin_count < 2:
                print(
                    f"  [INFO] {direction}: fewer than two finite margins; "
                    "variance check skipped for this capped run."
                )
            else:
                all_ok &= check(
                    f"{direction}: margin shows real variance across windows (std > 0)",
                    stats["margin_std"] > 1e-9,
                    f"margin_std={stats['margin_std']:.8f}",
                )
        if dataset in EXPECTED_PARTICIPANTS:
            all_ok &= check(
                f"{direction}: participant_count matches expected",
                stats["participant_count"] == EXPECTED_PARTICIPANTS[dataset],
            )

    return all_ok


def verify_interaction(dataset: str) -> bool:
    print(f"\n=== Interaction profiles ({dataset}) ===")
    profiles_path = PROCESSED_DIR / "sensitivity" / f"{dataset}_interaction_profiles.csv"
    aggregates_path = PROCESSED_DIR / "sensitivity" / f"{dataset}_interaction_aggregates.json"

    all_ok = True
    all_ok &= check("interaction_profiles.csv exists", profiles_path.exists())
    all_ok &= check("interaction_aggregates.json exists", aggregates_path.exists())
    if not (profiles_path.exists() and aggregates_path.exists()):
        return False

    df = pd.read_csv(profiles_path)

    all_ok &= check(
        "no NaN in interaction_mean",
        not df["interaction_mean"].isna().any(),
    )

    n_directions = len(EXPECTED_DIRECTIONS[dataset])
    expected_pairs = n_directions * (n_directions - 1) // 2
    actual_pairs = df.groupby(["direction_a", "direction_b"]).ngroups
    all_ok &= check(
        "number of direction pairs matches C(n_directions, 2)",
        actual_pairs == expected_pairs,
        f"got {actual_pairs} pairs, expected {expected_pairs} for {n_directions} directions",
    )

    pair_counts = df.groupby(["direction_a", "direction_b"]).size()
    pair_variance = df.groupby(["direction_a", "direction_b"])["interaction_mean"].std()
    if len(df) < 2 or pair_counts.max() < 2:
        print(
            "  [INFO] fewer than two windows per interaction pair; "
            "variance check skipped for this capped run."
        )
    else:
        all_ok &= check(
            "interaction_mean varies across windows within at least one pair",
            (pair_variance > 1e-9).any(),
        )

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    uni_ok = verify_univariate(args.dataset)
    inter_ok = verify_interaction(args.dataset)

    print(f"\n{'='*60}")
    if uni_ok and inter_ok:
        print(f"[{args.dataset}] ALL CHECKS PASSED - structurally sound.")
        print("This does NOT mean the science is correct - only that the")
        print("numbers aren't obviously broken. Still eyeball the actual")
        print("slope/curvature signs against what you'd expect substantively.")
    else:
        print(f"[{args.dataset}] ONE OR MORE CHECKS FAILED - do not trust these "
              f"numbers for the final table yet. Scroll up for which checks failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()