"""
StudentLife model-data audit
----------------------------

Reads the already-cached:
    data/interim/studentlife_model_df.parquet

and audits:
1. dataset shape / participants / dates
2. target availability
3. feature missingness
4. feature variance
5. target distributions
6. participant coverage
7. candidate model features

This is an AUDIT only.
It does not modify studentlife_model_df.parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR


MODEL_DF_CACHE = INTERIM_DIR / "studentlife_model_df.parquet"
AUDIT_OUTPUT = INTERIM_DIR / "studentlife_audit_summary.txt"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ID_COLUMNS = ["uid", "date"]

# These are the columns currently produced by your StudentLife pipeline.
# We will inspect them rather than assuming all are valid targets.
EXPECTED_TARGETS = [
    "pam",
    "stress",
    "mood",
]

# Raw/sensing-derived columns currently in studentlife_model_df.
# Anything not in ID_COLUMNS or target candidates is treated as a
# candidate sensor feature and audited.
KNOWN_NON_FEATURE_COLUMNS = ID_COLUMNS + EXPECTED_TARGETS


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_missingness(df: pd.DataFrame, columns: list[str]):
    rows = []

    for col in columns:
        missing = df[col].isna().mean()
        observed = df[col].notna().sum()
        unique = df[col].nunique(dropna=True)
        variance = df[col].var()

        rows.append(
            {
                "feature": col,
                "missing_rate": missing,
                "observed_count": observed,
                "unique": unique,
                "variance": variance,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["missing_rate", "variance"],
            ascending=[False, True],
        )

    print(result.to_string(index=False))
    return result


def target_summary(df: pd.DataFrame, target: str):
    print(f"\n--- {target} ---")

    if target not in df.columns:
        print("NOT PRESENT in model dataframe")
        return

    s = df[target]

    print(f"rows              : {len(s)}")
    print(f"observed          : {s.notna().sum()}")
    print(f"missing           : {s.isna().sum()}")
    print(f"missing rate      : {s.isna().mean():.6f}")
    print(f"unique values     : {s.nunique(dropna=True)}")

    if pd.api.types.is_numeric_dtype(s):
        print(f"mean              : {s.mean()}")
        print(f"std               : {s.std()}")
        print(f"min               : {s.min()}")
        print(f"25%               : {s.quantile(.25)}")
        print(f"50%               : {s.quantile(.50)}")
        print(f"75%               : {s.quantile(.75)}")
        print(f"max               : {s.max()}")

        print("\nvalue counts:")
        print(s.value_counts(dropna=True).sort_index().head(30))
    else:
        print("\nvalue counts:")
        print(s.value_counts(dropna=True).head(30))


def participant_target_coverage(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return

    coverage = (
        df.groupby("uid", observed=True)[target]
        .agg(
            days="size",
            observed="count",
        )
    )

    coverage["target_coverage"] = (
        coverage["observed"] / coverage["days"]
    )

    print(f"\nParticipant coverage for {target}:")
    print(coverage["target_coverage"].describe())

    print(
        f"participants with >=1 observed {target}: "
        f"{(coverage['observed'] > 0).sum()} / {len(coverage)}"
    )


def usable_7_day_sequences(df: pd.DataFrame, target: str) -> int:
    """Count seven consecutive calendar-day windows ending with a target."""
    if target not in df.columns:
        return 0

    ordered = df[["uid", "date", target]].dropna(subset=["uid", "date"])
    ordered = ordered.sort_values(["uid", "date"])
    sequence_count = 0

    for _, participant in ordered.groupby("uid", observed=True):
        dates = participant["date"].reset_index(drop=True)
        target_observed = participant[target].notna().reset_index(drop=True)
        consecutive = dates.diff().eq(pd.Timedelta(days=1))
        run_length = 1

        for index in range(1, len(participant)):
            run_length = run_length + 1 if consecutive.iloc[index] else 1
            if run_length >= 7 and target_observed.iloc[index]:
                sequence_count += 1

    return sequence_count


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def main():

    if not MODEL_DF_CACHE.exists():
        raise FileNotFoundError(
            f"StudentLife model dataframe not found:\n"
            f"{MODEL_DF_CACHE}\n\n"
            f"Run preprocess_slife_daily.py first."
        )

    df = pd.read_parquet(MODEL_DF_CACHE)

    # ================================================================
    section("1. STUDENTLIFE MODEL DATASET")
    # ================================================================

    print(f"file          : {MODEL_DF_CACHE}")
    print(f"shape         : {df.shape}")
    print(f"rows          : {len(df)}")
    print(f"columns       : {len(df.columns)}")

    print(
        f"participants  : "
        f"{df['uid'].nunique() if 'uid' in df.columns else 'UNKNOWN'}"
    )

    if "date" in df.columns:
        print(f"date min      : {df['date'].min()}")
        print(f"date max      : {df['date'].max()}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    # ================================================================
    section("2. TARGET AVAILABILITY")
    # ================================================================

    for target in EXPECTED_TARGETS:
        target_summary(df, target)

    # ================================================================
    section("3. TARGET PARTICIPANT COVERAGE")
    # ================================================================

    for target in EXPECTED_TARGETS:
        if target in df.columns:
            participant_target_coverage(df, target)

    # ================================================================
    section("4. CANDIDATE SENSOR FEATURES")
    # ================================================================

    feature_columns = [
        c for c in df.columns
        if c not in KNOWN_NON_FEATURE_COLUMNS
    ]

    print(f"Candidate sensor features: {len(feature_columns)}")

    for col in feature_columns:
        print(f"  {col}")

    # ================================================================
    section("5. FEATURE MISSINGNESS")
    # ================================================================

    missingness = print_missingness(df, feature_columns)

    # ================================================================
    section("6. FEATURE VARIANCE")
    # ================================================================

    variance_rows = []

    for col in feature_columns:
        s = df[col]

        variance_rows.append(
            {
                "feature": col,
                "missing_rate": s.isna().mean(),
                "unique": s.nunique(dropna=True),
                "variance": s.var(),
                "observed": s.notna().sum(),
            }
        )

    variance_df = pd.DataFrame(variance_rows)

    if not variance_df.empty:
        variance_df = variance_df.sort_values(
            "variance",
            ascending=True,
        )

        print(
            variance_df.to_string(index=False)
        )

    # ================================================================
    section("7. ZERO-VARIANCE FEATURES")
    # ================================================================

    zero_variance = []

    for col in feature_columns:
        variance = df[col].var()

        if pd.notna(variance) and variance == 0:
            zero_variance.append(col)

    print(f"Zero-variance features: {len(zero_variance)}")

    for col in zero_variance:
        print(f"  {col}")

    # ================================================================
    section("8. PROPOSED BASIC FEATURE FILTER")
    # ================================================================

    # IMPORTANT:
    # This does NOT save anything.
    # We are only showing what would happen with a 50% missingness rule.
    #
    # We deliberately do not permanently choose the threshold yet.

    proposed_threshold = 0.50

    eligible = missingness[
        missingness["missing_rate"] <= proposed_threshold
    ]

    print(
        f"Missingness threshold shown for audit: "
        f"{proposed_threshold:.0%}"
    )

    print(
        f"Features before filtering : "
        f"{len(feature_columns)}"
    )

    print(
        f"Features <= {proposed_threshold:.0%} missing : "
        f"{len(eligible)}"
    )

    print(
        f"Would remove : "
        f"{len(feature_columns) - len(eligible)}"
    )

    print("\nFeatures that would be retained:")

    for col in eligible["feature"]:
        print(f"  {col}")

    # ================================================================
    section("9. TARGET × SENSOR OBSERVATION OVERLAP")
    # ================================================================

    # This is particularly important because happy/sad are ~93% missing.
    #
    # We calculate how many rows have both:
    #   target observed
    #   sensor feature observed

    for target in EXPECTED_TARGETS:

        if target not in df.columns:
            continue

        print(f"\nTarget: {target}")

        target_observed = df[target].notna()

        for feature in feature_columns:

            both = (
                target_observed
                & df[feature].notna()
            ).sum()

            if both > 0:
                print(
                    f"  {feature:35s} "
                    f"joint_observed={both:5d}"
                )

    # ================================================================
    section("10. DAILY COVERAGE BY PARTICIPANT")
    # ================================================================

    participant_days = (
        df.groupby("uid", observed=True)["date"]
        .nunique()
    )

    print(participant_days.describe())

    print(
        "\nParticipants by number of available days:"
    )

    print(
        participant_days.sort_values().to_string()
    )

    # ================================================================
    section("11. USABLE 7-DAY TARGET SEQUENCES")
    # ================================================================

    print(
        "A sequence is seven consecutive calendar-day rows ending "
        "on an observed target."
    )
    for target in EXPECTED_TARGETS:
        print(
            f"{target:8s}: "
            f"{usable_7_day_sequences(df, target):5d} usable sequences"
        )

    # ================================================================
    section("12. RECOMMENDED NEXT DECISION")
    # ================================================================

    print(
        """
DO NOT create StudentLife sequences yet.

Use this audit to decide:

1. Which StudentLife target(s) are scientifically valid.
2. Whether picture_idx is the intended PAM target.
3. Whether the raw stress table can legitimately become a stress target.
4. Whether happy/sad have enough observations to be usable.
5. Which sensor features survive the missingness rule.
6. Whether any low-variance features should be removed.
7. Whether the same feature-selection philosophy should be used
   for CES and StudentLife.

No filtering decisions are permanently saved by this script.
"""
    )

    # ================================================================
    # Save human-readable audit
    # ================================================================

    print("\nAudit complete.")

    # We intentionally do NOT save a feature list yet.
    # That comes only after we agree on the target/feature policy.


if __name__ == "__main__":
    main()