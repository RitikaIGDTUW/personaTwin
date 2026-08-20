"""
StudentLife daily modeling table construction.

Input:
    data/interim/studentlife_tables.pkl

Output:
    data/interim/studentlife_model_df.parquet

Purpose:
    Convert the cached StudentLife sensing tables into one row per
    participant-day, then merge StudentLife mental-state targets.

This stage is CPU-only and cached. It should not reload raw RDS files
if studentlife_tables.pkl already exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    STUDENTLIFE_MODEL_DF_CACHE,
)
from src.caching import cache_parquet
from src.preprocess_studentlife import (
    get_studentlife_tables,
)

TARGET_COLUMNS = [
    "pam",
    "stress",
    "mood",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_datetime_seconds(series: pd.Series) -> pd.Series:
    """
    Convert Unix timestamps in seconds to pandas datetime.

    StudentLife timestamps in the inspected tables are Unix seconds.
    """
    return pd.to_datetime(series, unit="s", errors="coerce")


def _day_from_timestamp(series: pd.Series) -> pd.Series:
    return _to_datetime_seconds(series).dt.normalize()


def _interval_features(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate interval-based sensing data to participant-day.

    Produces:
        {prefix}_duration
        {prefix}_events
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                f"{prefix}_duration",
                f"{prefix}_events",
            ]
        )

    x = df[["uid", start_col, end_col]].copy()

    x[start_col] = pd.to_numeric(x[start_col], errors="coerce")
    x[end_col] = pd.to_numeric(x[end_col], errors="coerce")

    x["start"] = _to_datetime_seconds(x[start_col])
    x["end"] = _to_datetime_seconds(x[end_col])

    x = x.dropna(subset=["uid", "start", "end"])

    # Protect against malformed negative intervals.
    x["duration_sec"] = (
        x["end"] - x["start"]
    ).dt.total_seconds()

    x.loc[x["duration_sec"] < 0, "duration_sec"] = np.nan

    x["date"] = x["start"].dt.normalize()

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            **{
                f"{prefix}_duration": ("duration_sec", "sum"),
                f"{prefix}_events": ("duration_sec", "count"),
            }
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------

def _build_activity_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate activity inference to participant-day.

    We preserve the original activity_inference values as daily
    summary statistics rather than assuming an undocumented coding
    scheme.

    Outputs:
        activity_mean
        activity_std
        activity_n
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                "activity_mean",
                "activity_std",
                "activity_n",
            ]
        )

    x = df[["uid", "timestamp", "activity_inference"]].copy()

    x["date"] = _day_from_timestamp(x["timestamp"])

    x["activity_inference"] = pd.to_numeric(
        x["activity_inference"],
        errors="coerce",
    )

    x = x.dropna(subset=["uid", "date", "activity_inference"])

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            activity_mean=("activity_inference", "mean"),
            activity_std=("activity_inference", "std"),
            activity_n=("activity_inference", "count"),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# GPS / mobility
# ---------------------------------------------------------------------

def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in kilometres."""

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 6371.0 * 2.0 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def _build_gps_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build conservative daily mobility features.

    Outputs:
        gps_n
        gps_distance_km
        gps_unique_locations

    Unique locations are approximated by rounding latitude/longitude
    to 3 decimal places (~100 m scale), rather than claiming exact
    semantic locations.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                "gps_n",
                "gps_distance_km",
                "gps_unique_locations",
            ]
        )

    cols = [
        "uid",
        "timestamp",
        "latitude",
        "longitude",
    ]

    x = df[cols].copy()

    x["date"] = _day_from_timestamp(x["timestamp"])

    x["latitude"] = pd.to_numeric(x["latitude"], errors="coerce")
    x["longitude"] = pd.to_numeric(x["longitude"], errors="coerce")

    x = x.dropna(
        subset=[
            "uid",
            "date",
            "latitude",
            "longitude",
        ]
    )

    # Remove impossible coordinates.
    x = x[
        x["latitude"].between(-90, 90)
        & x["longitude"].between(-180, 180)
    ]

    # Sort chronologically so consecutive GPS points can be used
    # to estimate travelled distance.
    x = x.sort_values(["uid", "timestamp"])

    x["prev_lat"] = x.groupby(
        "uid",
        observed=True,
    )["latitude"].shift(1)

    x["prev_lon"] = x.groupby(
        "uid",
        observed=True,
    )["longitude"].shift(1)

    # Only count distance when the previous point belongs to the
    # same calendar day.
    previous_date = x.groupby(
        "uid",
        observed=True,
    )["date"].shift(1)

    same_day = x["date"].eq(previous_date)

    valid_previous = (
        same_day
        & x["prev_lat"].notna()
        & x["prev_lon"].notna()
    )

    x["distance_km"] = 0.0

    if valid_previous.any():
        x.loc[valid_previous, "distance_km"] = _haversine_km(
            x.loc[valid_previous, "prev_lat"].to_numpy(),
            x.loc[valid_previous, "prev_lon"].to_numpy(),
            x.loc[valid_previous, "latitude"].to_numpy(),
            x.loc[valid_previous, "longitude"].to_numpy(),
        )

    # Protect against obvious GPS jumps.
    # We don't want a single erroneous coordinate to dominate daily
    # mobility.
    x.loc[
        x["distance_km"] > 10,
        "distance_km",
    ] = np.nan

    # ~100 m spatial bins.
    x["location_lat"] = x["latitude"].round(3)
    x["location_lon"] = x["longitude"].round(3)

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            gps_n=("latitude", "count"),
            gps_distance_km=("distance_km", "sum"),
            gps_unique_locations=(
                "location_lat",
                "nunique",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# App usage
# ---------------------------------------------------------------------

def _build_app_usage_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate app-usage observations to participant-day.

    We do not attempt to classify individual apps into psychological
    categories at this stage.

    Outputs:
        app_usage_events
        app_usage_unique_packages
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                "app_usage_events",
                "app_usage_unique_packages",
            ]
        )

    x = df[
        [
            "uid",
            "timestamp",
            "RUNNING_TASKS_topActivity_mPackage",
        ]
    ].copy()

    x["date"] = _day_from_timestamp(x["timestamp"])

    x = x.dropna(subset=["uid", "date"])

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            app_usage_events=("timestamp", "count"),
            app_usage_unique_packages=(
                "RUNNING_TASKS_topActivity_mPackage",
                "nunique",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------

def _build_call_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate call-log data.

    Outputs:
        call_events
        call_duration_sec
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                "call_events",
                "call_duration_sec",
            ]
        )

    x = df[
        [
            "uid",
            "timestamp",
            "CALLS_duration",
        ]
    ].copy()

    x["date"] = _day_from_timestamp(x["timestamp"])

    x["CALLS_duration"] = pd.to_numeric(
        x["CALLS_duration"],
        errors="coerce",
    )

    x = x.dropna(subset=["uid", "date"])

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            call_events=("timestamp", "count"),
            call_duration_sec=(
                "CALLS_duration",
                "sum",
            ),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------

def _build_sms_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Count SMS records per participant-day."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "uid",
                "date",
                "sms_events",
            ]
        )

    x = df[["uid", "timestamp"]].copy()

    x["date"] = _day_from_timestamp(x["timestamp"])

    x = x.dropna(subset=["uid", "date"])

    result = (
        x.groupby(["uid", "date"], observed=True)
        .agg(
            sms_events=("timestamp", "count"),
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------

def _prepare_studentlife_targets(tables: dict) -> pd.DataFrame:
    """
    Build daily StudentLife EMA targets.

    Targets:
        pam    : daily mean PAM picture index (1-16)
        stress : daily mean stress EMA response (1-5)
        mood   : daily mood score derived from happy/sad EMA responses

    Important:
    The raw StudentLife stress table has a column named 'null' in the
    imported RDS representation. It contains a mixture of actual numeric
    EMA responses and non-response values such as location strings/NA.
    Only numeric stress responses in the valid 1-5 range are retained.

    The source StudentLife stress EMA uses:
        1 = a little stressed
        2 = definitely stressed
        3 = stressed out
        4 = feeling good
        5 = feeling great
    """

    def _timestamp_to_date(df, timestamp_col="timestamp"):
        out = df.copy()

        out["timestamp"] = pd.to_datetime(
            pd.to_numeric(out[timestamp_col], errors="coerce"),
            unit="s",
            errors="coerce",
        )

        out["date"] = out["timestamp"].dt.floor("D")

        return out

    def _numeric_response(series):
        """
        Convert imported EMA response values to numeric.

        Non-numeric values such as GPS-like strings are converted to NaN.
        """
        return pd.to_numeric(series, errors="coerce")

    # ================================================================
    # PAM
    # ================================================================
    pam = tables["pam"].copy()
    pam = _timestamp_to_date(pam)

    pam["picture_idx"] = pd.to_numeric(
        pam["picture_idx"],
        errors="coerce",
    )

    pam = pam[
        pam["uid"].notna()
        & pam["date"].notna()
        & pam["picture_idx"].between(1, 16)
    ]

    pam_daily = (
        pam.groupby(["uid", "date"], observed=True)["picture_idx"]
        .mean()
        .rename("pam")
        .reset_index()
    )

    # ================================================================
    # STRESS
    # ================================================================
    stress = tables["stress"].copy()
    stress = _timestamp_to_date(stress)

    # The imported StudentLife stress response is stored in "null".
    if "null" not in stress.columns:
        raise KeyError(
            "StudentLife stress table does not contain the expected "
            "'null' response column. Available columns: "
            f"{list(stress.columns)}"
        )

    stress["stress_response"] = _numeric_response(stress["null"])

    # Keep ONLY valid StudentLife stress EMA responses.
    stress = stress[
        stress["uid"].notna()
        & stress["date"].notna()
        & stress["stress_response"].between(1, 5)
    ]

    print(
        "[StudentLife] valid stress EMA responses:",
        len(stress),
    )

    print(
        "[StudentLife] stress response distribution:"
    )
    print(
        stress["stress_response"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    stress_daily = (
        stress.groupby(["uid", "date"], observed=True)["stress_response"]
        .mean()
        .rename("stress")
        .reset_index()
    )

    # ================================================================
    # MOOD
    # ================================================================
    mood = tables["mood"].copy()
    mood = _timestamp_to_date(mood)

    mood["happy"] = pd.to_numeric(
        mood["happy"],
        errors="coerce",
    )

    mood["sad"] = pd.to_numeric(
        mood["sad"],
        errors="coerce",
    )

    mood = mood[
        mood["uid"].notna()
        & mood["date"].notna()
    ]

    # Daily means preserve the original happy and sad measurements.
    mood_daily = (
        mood.groupby(["uid", "date"], observed=True)
        .agg(
            happy=("happy", "mean"),
            sad=("sad", "mean"),
        )
        .reset_index()
    )

    # A single mood target is useful for the model.
    #
    # Higher happy and lower sad => better mood.
    #
    # Both are on approximately the same 1-4 scale.
    # We therefore define:
    #
    #     mood = happy - sad
    #
    # This retains directionality without pretending that the two
    # questions are a validated composite psychometric scale.
    mood_daily["mood"] = (
        mood_daily["happy"] - mood_daily["sad"]
    )

    # ================================================================
    # MERGE THREE TARGETS
    # ================================================================
    targets = pam_daily.merge(
        stress_daily,
        on=["uid", "date"],
        how="outer",
    )

    targets = targets.merge(
        mood_daily[
            ["uid", "date", "mood", "happy", "sad"]
        ],
        on=["uid", "date"],
        how="outer",
    )

    targets = targets.sort_values(
        ["uid", "date"]
    ).reset_index(drop=True)

    print(
        "\n[StudentLife] daily target dataset:"
    )

    print(
        f"  rows         : {len(targets)}"
    )

    print(
        f"  participants : {targets['uid'].nunique()}"
    )

    for col in ["pam", "stress", "mood", "happy", "sad"]:
        if col in targets.columns:
            observed = targets[col].notna().sum()
            print(
                f"  {col:8s} : "
                f"{observed:5d} observed "
                f"({observed / len(targets):.2%})"
            )

    return targets

# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_studentlife_model_df(
    force: bool = False,
) -> pd.DataFrame:
    """
    Build and cache the StudentLife participant-day modeling table.

    Output:
        data/interim/studentlife_model_df.parquet
    """

    def _build() -> pd.DataFrame:

        print("[StudentLife] loading cached tables...")
        tables = get_studentlife_tables()

        print(
            "[StudentLife] available tables:",
            list(tables.keys()),
        )

        daily_tables = []

        # -------------------------------------------------------------
        # Activity
        # -------------------------------------------------------------
        if "activity" in tables:
            print("[StudentLife] aggregating activity...")
            daily_tables.append(
                _build_activity_daily(
                    tables["activity"]
                )
            )

        # -------------------------------------------------------------
        # GPS
        # -------------------------------------------------------------
        if "gps" in tables:
            print("[StudentLife] aggregating GPS...")
            daily_tables.append(
                _build_gps_daily(
                    tables["gps"]
                )
            )

        # -------------------------------------------------------------
        # Conversation
        # -------------------------------------------------------------
        if "conversation" in tables:
            print(
                "[StudentLife] aggregating conversation..."
            )
            daily_tables.append(
                _interval_features(
                    tables["conversation"],
                    "start_timestamp",
                    "end_timestamp",
                    "conversation",
                )
            )

        # -------------------------------------------------------------
        # Phone lock
        # -------------------------------------------------------------
        if "phonelock" in tables:
            print(
                "[StudentLife] aggregating phonelock..."
            )
            daily_tables.append(
                _interval_features(
                    tables["phonelock"],
                    "start_timestamp",
                    "end_timestamp",
                    "phonelock",
                )
            )

        # -------------------------------------------------------------
        # Dark
        # -------------------------------------------------------------
        if "dark" in tables:
            print(
                "[StudentLife] aggregating dark intervals..."
            )
            daily_tables.append(
                _interval_features(
                    tables["dark"],
                    "start_timestamp",
                    "end_timestamp",
                    "dark",
                )
            )

        # -------------------------------------------------------------
        # App usage
        # -------------------------------------------------------------
        if "app_usage" in tables:
            print(
                "[StudentLife] aggregating app usage..."
            )
            daily_tables.append(
                _build_app_usage_daily(
                    tables["app_usage"]
                )
            )

        # -------------------------------------------------------------
        # Calls
        # -------------------------------------------------------------
        if "call_log" in tables:
            print(
                "[StudentLife] aggregating call log..."
            )
            daily_tables.append(
                _build_call_daily(
                    tables["call_log"]
                )
            )

        # -------------------------------------------------------------
        # SMS
        # -------------------------------------------------------------
        if "sms" in tables:
            print(
                "[StudentLife] aggregating SMS..."
            )
            daily_tables.append(
                _build_sms_daily(
                    tables["sms"]
                )
            )

        # -------------------------------------------------------------
        # Merge sensing tables
        # -------------------------------------------------------------
        nonempty = [
            x
            for x in daily_tables
            if not x.empty
        ]

        if not nonempty:
            raise ValueError(
                "No StudentLife sensing tables produced "
                "daily features."
            )

        model_df = nonempty[0].copy()

        for daily in nonempty[1:]:
            model_df = model_df.merge(
                daily,
                on=["uid", "date"],
                how="outer",
            )

        # -------------------------------------------------------------
        # Targets
        # -------------------------------------------------------------
        print(
            "[StudentLife] loading/building targets..."
        )

        targets_daily = _prepare_studentlife_targets(tables)

        print(
            "[StudentLife] target daily shape:",
            targets_daily.shape,
        )

        # -------------------------------------------------------------
        # Merge targets with sensing
        # -------------------------------------------------------------
        model_df = model_df.merge(
            targets_daily,
            on=["uid", "date"],
            how="outer",
        )

        # -------------------------------------------------------------
        # Sort
        # -------------------------------------------------------------
        model_df = model_df.sort_values(
            ["uid", "date"]
        ).reset_index(drop=True)

        # -------------------------------------------------------------
        # Remove completely empty feature rows.
        #
        # A row is retained if it has either:
        #   - at least one sensing feature, OR
        #   - at least one target.
        # -------------------------------------------------------------
        non_key_columns = [
            c
            for c in model_df.columns
            if c not in ["uid", "date"]
        ]

        model_df = model_df.loc[
            model_df[non_key_columns]
            .notna()
            .any(axis=1)
        ].copy()

        feature_columns = [
            column
            for column in model_df.columns
            if column not in {"uid", "date", *TARGET_COLUMNS, "happy", "sad"}
        ]
        model_df = model_df[
            ["uid", "date"] + feature_columns + TARGET_COLUMNS
        ]

        # -------------------------------------------------------------
        # Diagnostics
        # -------------------------------------------------------------
        print()
        print(
            "=================================================="
        )
        print(
            "STUDENTLIFE MODEL DATASET"
        )
        print(
            "=================================================="
        )

        print(
            "shape:",
            model_df.shape,
        )

        print(
            "participants:",
            model_df["uid"].nunique(),
        )

        print(
            "date range:",
            model_df["date"].min(),
            "to",
            model_df["date"].max(),
        )

        print()
        print("columns:")
        for c in model_df.columns:
            print("  ", c)

        print()
        print("missingness:")
        print(
            model_df.isna()
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )

        print(
            "=================================================="
        )

        return model_df

    return cache_parquet(
        STUDENTLIFE_MODEL_DF_CACHE,
        build_fn=_build,
        force=force,
    )


if __name__ == "__main__":
    df = build_studentlife_model_df()

    print()
    print("FINAL SHAPE:", df.shape)
    print()
    print(df.head())