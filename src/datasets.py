"""Leakage-safe sequence construction for CES and StudentLife."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.caching import cache_torch
from src.config import (
    MIN_SEQUENCE_DAYS,
    SEQUENCE_LOOKBACK_DAYS,
)


def _split_participant_rows(
    participant: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    participant = participant.sort_values("_date").reset_index(drop=True)
    n_rows = len(participant)
    train_end = max(1, int(n_rows * 0.70))
    val_end = max(train_end + 1, int(n_rows * 0.85))
    val_end = min(val_end, n_rows)
    return {
        "train": participant.iloc[:train_end].copy(),
        "val": participant.iloc[train_end:val_end].copy(),
        "test": participant.iloc[val_end:].copy(),
    }


def _prepare_split(
    frame: pd.DataFrame,
    feature_names: list[str],
    fill_values: pd.Series,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    output = frame.copy()
    output[feature_names] = output.groupby("uid", observed=True)[
        feature_names
    ].ffill()
    output[feature_names] = output[feature_names].fillna(fill_values)
    output[feature_names] = (output[feature_names] - means) / stds
    return output


def _coerce_model_dates(series: pd.Series) -> pd.Series:
    """Normalize StudentLife datetimes and CES integer YYYYMMDD days."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(
            series.astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    return pd.to_datetime(series, errors="coerce")


def _direction_vectors(
    windows: np.ndarray,
    feature_names: list[str],
    direction_map: dict[str, list[str]],
) -> np.ndarray:
    direction_names = list(direction_map)
    feature_indices = {
        feature: index for index, feature in enumerate(feature_names)
    }
    result = np.zeros(
        (len(windows), len(direction_names), len(feature_names)),
        dtype=np.float32,
    )

    for direction_index, direction in enumerate(direction_names):
        indices = [
            feature_indices[feature]
            for feature in direction_map[direction]
            if feature in feature_indices
        ]
        if indices:
            result[:, direction_index, indices] = windows[:, :, indices].mean(axis=1)

    return result


def _build_uncached(
    model_df: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    direction_map: dict[str, list[str]],
) -> dict:
    date_column = "date" if "date" in model_df.columns else "day"
    missing_columns = [
        column
        for column in ["uid", date_column, *feature_names, *target_names]
        if column not in model_df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing sequence input columns: {missing_columns}")

    data = model_df[
        ["uid", date_column, *feature_names, *target_names]
    ].copy()
    data["_date"] = _coerce_model_dates(data[date_column])
    data = data.dropna(subset=["uid", "_date"])
    data = data.sort_values(["uid", "_date"])

    participant_groups = []
    for _, participant in data.groupby("uid", observed=True):
        if len(participant) >= MIN_SEQUENCE_DAYS:
            participant_groups.append(_split_participant_rows(participant))

    splits = {"train": [], "val": [], "test": []}
    for participant_split in participant_groups:
        for split_name, frame in participant_split.items():
            if not frame.empty:
                splits[split_name].append(frame)

    train_frame = pd.concat(splits["train"], ignore_index=True)
    numeric_features = train_frame[feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    )
    fill_values = numeric_features.median()
    means = numeric_features.fillna(fill_values).mean()
    stds = numeric_features.fillna(fill_values).std().replace(0, 1).fillna(1)

    prepared = {
        split_name: [
            _prepare_split(
                frame,
                feature_names,
                fill_values,
                means,
                stds,
            )
            for frame in frames
        ]
        for split_name, frames in splits.items()
    }

    output = {}
    for split_name, frames in prepared.items():
        windows = []
        targets = []
        window_uids = []

        for frame in frames:
            frame = frame.sort_values("_date").reset_index(drop=True)
            feature_array = frame[feature_names].to_numpy(dtype=np.float32)
            target_array = frame[target_names].apply(
                pd.to_numeric,
                errors="coerce",
            ).to_numpy(dtype=np.float32)
            dates = frame["_date"].to_numpy()

            for end_index in range(SEQUENCE_LOOKBACK_DAYS, len(frame)):
                start_index = end_index - SEQUENCE_LOOKBACK_DAYS
                window_dates = dates[start_index:end_index]
                if not np.all(
                    np.diff(window_dates).astype("timedelta64[D]")
                    == np.timedelta64(1, "D")
                ):
                    continue
                if not np.isfinite(target_array[end_index]).all():
                    continue
                windows.append(feature_array[start_index:end_index])
                targets.append(target_array[end_index])
                window_uids.append(frame.loc[end_index, "uid"])

        if windows:
            x_array = np.stack(windows)
            y_array = np.stack(targets)
            direction_array = _direction_vectors(
                x_array,
                feature_names,
                direction_map,
            )
        else:
            x_array = np.empty(
                (0, SEQUENCE_LOOKBACK_DAYS, len(feature_names)),
                dtype=np.float32,
            )
            y_array = np.empty(
                (0, len(target_names)),
                dtype=np.float32,
            )
            direction_array = np.empty(
                (0, len(direction_map), len(feature_names)),
                dtype=np.float32,
            )

        uid_array = (
            torch.tensor(window_uids)
            if all(isinstance(uid, (int, np.integer)) for uid in window_uids)
            else window_uids
        )
        output[split_name] = {
            "X": torch.from_numpy(x_array),
            "y": torch.from_numpy(y_array),
            "uid": uid_array,
            "direction_vectors": torch.from_numpy(direction_array),
        }

    output["metadata"] = {
        "feature_names": feature_names,
        "target_names": target_names,
        "direction_names": list(direction_map),
        "lookback_days": SEQUENCE_LOOKBACK_DAYS,
        "min_sequence_days": MIN_SEQUENCE_DAYS,
        "normalization": "train-only median, mean, and standard deviation",
        "feature_raw_mean": {
            name: float(means[name]) for name in feature_names
        },
        "feature_raw_std": {
            name: float(stds[name]) for name in feature_names
        },
    }
    return output


def build_sequences(
    model_df: pd.DataFrame,
    features_final: list[str],
    target_cols: list[str],
    direction_map: dict[str, list[str]],
    force: bool = False,
    cache_path: Path | None = None,
) -> dict:
    """Build cached next-day sequences for either dataset's model dataframe."""
    feature_names = list(features_final)
    target_names = list(target_cols)
    build_fn = lambda: _build_uncached(
        model_df,
        feature_names,
        target_names,
        direction_map,
    )
    if cache_path is None:
        return build_fn()
    return cache_torch(cache_path, build_fn=build_fn, force=force)
