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


def _add_temporal_features(
    data: pd.DataFrame,
    feature_names: list[str],
    group_col: str = "uid",
) -> tuple[pd.DataFrame, list[str]]:
    """Augment raw per-day features with explicit within-person temporal signal.

    For each base feature this derives, using ONLY past-or-current rows within
    the same participant (never future rows, so this stays leakage-safe the
    same way the rest of this module is):

      {feat}_delta1          day-over-day change
      {feat}_delta3          change vs. 3 days ago
      {feat}_roll_mean7      trailing 7-day mean (recent baseline)
      {feat}_roll_std7       trailing 7-day std (recent volatility)
      {feat}_trend7          (value - value 6 days ago) / 6, a cheap slope
      {feat}_dev_from_own_mean  value minus the participant's own mean-so-far
                                (captures "above/below this person's usual",
                                without hard-coding an embedding for it)

    This directly targets the "model learns the average, not the trajectory"
    failure mode: the GRU no longer has to infer trend from 7 raw points on
    its own, it gets trend/volatility/deviation handed to it as features.
    """
    df = data.copy()
    grouped = df.groupby(group_col, observed=True)
    derived_names: list[str] = []
    derived_columns: dict[str, pd.Series] = {}

    for feat in feature_names:
        g = grouped[feat]

        delta1_col = f"{feat}_delta1"
        delta3_col = f"{feat}_delta3"
        roll_mean_col = f"{feat}_roll_mean7"
        roll_std_col = f"{feat}_roll_std7"
        trend_col = f"{feat}_trend7"
        dev_col = f"{feat}_dev_from_own_mean"

        derived_columns[delta1_col] = g.diff(1)
        derived_columns[delta3_col] = g.diff(3)
        derived_columns[roll_mean_col] = g.transform(
            lambda s: s.rolling(SEQUENCE_LOOKBACK_DAYS, min_periods=2).mean()
        )
        derived_columns[roll_std_col] = g.transform(
            lambda s: s.rolling(SEQUENCE_LOOKBACK_DAYS, min_periods=2).std()
        )
        derived_columns[trend_col] = g.diff(SEQUENCE_LOOKBACK_DAYS - 1) / (
            SEQUENCE_LOOKBACK_DAYS - 1
        )
        own_mean_so_far = g.transform(lambda s: s.shift(1).expanding().mean())
        derived_columns[dev_col] = df[feat] - own_mean_so_far

        derived_names.extend(
            [delta1_col, delta3_col, roll_mean_col, roll_std_col, trend_col, dev_col]
        )

    return pd.concat([df, pd.DataFrame(derived_columns, index=df.index)], axis=1), [
        *feature_names,
        *derived_names,
    ]


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
    predict_delta: bool = False,
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
    data, feature_names = _add_temporal_features(data, feature_names)

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
    filled = numeric_features.fillna(fill_values)
    means = filled.mean()
    stds = filled.std().replace(0, 1).fillna(1)

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
        baselines = []
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
                previous_target = target_array[end_index - 1]
                if predict_delta and not np.isfinite(previous_target).all():
                    continue
                windows.append(feature_array[start_index:end_index])
                if predict_delta:
                    targets.append(target_array[end_index] - previous_target)
                    baselines.append(previous_target)
                else:
                    targets.append(target_array[end_index])
                    baselines.append(previous_target)
                window_uids.append(frame.loc[end_index, "uid"])

        if windows:
            x_array = np.stack(windows)
            y_array = np.stack(targets)
            baseline_array = np.stack(baselines)
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
            baseline_array = np.empty(
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
            # previous day's target value for every window, i.e. PAM_t when
            # y is PAM_{t+1} (predict_delta=False) or PAM_{t+1}-PAM_t
            # (predict_delta=True). Add this back to a delta prediction to
            # recover an absolute-scale PAM estimate: PAM_hat = baseline + y_hat
            "baseline": torch.from_numpy(baseline_array),
            "uid": uid_array,
            "direction_vectors": torch.from_numpy(direction_array),
        }

    output["metadata"] = {
        "feature_names": feature_names,
        "target_names": target_names,
        "direction_names": list(direction_map),
        "lookback_days": SEQUENCE_LOOKBACK_DAYS,
        "min_sequence_days": MIN_SEQUENCE_DAYS,
        "predict_delta": predict_delta,
        "normalization": "train-only median, mean, and standard deviation",
        "feature_raw_mean": {
            name: float(means[name]) for name in feature_names
        },
        "feature_raw_std": {
            name: float(stds[name]) for name in feature_names
        },
        "feature_raw_min": {
            name: float(filled[name].min()) for name in feature_names
        },
        "feature_raw_max": {
            name: float(filled[name].max()) for name in feature_names
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
    predict_delta: bool = False,
) -> dict:
    """Build cached next-day sequences for either dataset's model dataframe.

    predict_delta=True switches the target from absolute PAM_{t+1} to
    PAM_{t+1} - PAM_t, which stops MSE from rewarding "predict the
    participant's usual level" and forces the model to explain the change.
    Each split also carries a "baseline" tensor (PAM_t) so you can recover
    an absolute prediction via baseline + y_hat at eval time.
    """
    feature_names = list(features_final)
    target_names = list(target_cols)
    build_fn = lambda: _build_uncached(
        model_df,
        feature_names,
        target_names,
        direction_map,
        predict_delta=predict_delta,
    )
    if cache_path is None:
        return build_fn()
    return cache_torch(cache_path, build_fn=build_fn, force=force)








# """Leakage-safe sequence construction for CES and StudentLife."""

# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch

# from src.caching import cache_torch
# from src.config import (
#     MIN_SEQUENCE_DAYS,
#     SEQUENCE_LOOKBACK_DAYS,
# )


# def _split_participant_rows(
#     participant: pd.DataFrame,
# ) -> dict[str, pd.DataFrame]:
#     participant = participant.sort_values("_date").reset_index(drop=True)
#     n_rows = len(participant)
#     train_end = max(1, int(n_rows * 0.70))
#     val_end = max(train_end + 1, int(n_rows * 0.85))
#     val_end = min(val_end, n_rows)
#     return {
#         "train": participant.iloc[:train_end].copy(),
#         "val": participant.iloc[train_end:val_end].copy(),
#         "test": participant.iloc[val_end:].copy(),
#     }


# def _prepare_split(
#     frame: pd.DataFrame,
#     feature_names: list[str],
#     fill_values: pd.Series,
#     means: pd.Series,
#     stds: pd.Series,
# ) -> pd.DataFrame:
#     output = frame.copy()
#     output[feature_names] = output.groupby("uid", observed=True)[
#         feature_names
#     ].ffill()
#     output[feature_names] = output[feature_names].fillna(fill_values)
#     output[feature_names] = (output[feature_names] - means) / stds
#     return output


# def _coerce_model_dates(series: pd.Series) -> pd.Series:
#     """Normalize StudentLife datetimes and CES integer YYYYMMDD days."""
#     if pd.api.types.is_numeric_dtype(series):
#         return pd.to_datetime(
#             series.astype("Int64").astype(str),
#             format="%Y%m%d",
#             errors="coerce",
#         )
#     return pd.to_datetime(series, errors="coerce")


# def _direction_vectors(
#     windows: np.ndarray,
#     feature_names: list[str],
#     direction_map: dict[str, list[str]],
# ) -> np.ndarray:
#     direction_names = list(direction_map)
#     feature_indices = {
#         feature: index for index, feature in enumerate(feature_names)
#     }
#     result = np.zeros(
#         (len(windows), len(direction_names), len(feature_names)),
#         dtype=np.float32,
#     )

#     for direction_index, direction in enumerate(direction_names):
#         indices = [
#             feature_indices[feature]
#             for feature in direction_map[direction]
#             if feature in feature_indices
#         ]
#         if indices:
#             result[:, direction_index, indices] = windows[:, :, indices].mean(axis=1)

#     return result


# def _build_uncached(
#     model_df: pd.DataFrame,
#     feature_names: list[str],
#     target_names: list[str],
#     direction_map: dict[str, list[str]],
# ) -> dict:
#     date_column = "date" if "date" in model_df.columns else "day"
#     missing_columns = [
#         column
#         for column in ["uid", date_column, *feature_names, *target_names]
#         if column not in model_df.columns
#     ]
#     if missing_columns:
#         raise KeyError(f"Missing sequence input columns: {missing_columns}")

#     data = model_df[
#         ["uid", date_column, *feature_names, *target_names]
#     ].copy()
#     data["_date"] = _coerce_model_dates(data[date_column])
#     data = data.dropna(subset=["uid", "_date"])
#     data = data.sort_values(["uid", "_date"])

#     participant_groups = []
#     for _, participant in data.groupby("uid", observed=True):
#         if len(participant) >= MIN_SEQUENCE_DAYS:
#             participant_groups.append(_split_participant_rows(participant))

#     splits = {"train": [], "val": [], "test": []}
#     for participant_split in participant_groups:
#         for split_name, frame in participant_split.items():
#             if not frame.empty:
#                 splits[split_name].append(frame)

#     train_frame = pd.concat(splits["train"], ignore_index=True)
#     numeric_features = train_frame[feature_names].apply(
#         pd.to_numeric,
#         errors="coerce",
#     )
#     fill_values = numeric_features.median()
#     means = numeric_features.fillna(fill_values).mean()
#     stds = numeric_features.fillna(fill_values).std().replace(0, 1).fillna(1)

#     prepared = {
#         split_name: [
#             _prepare_split(
#                 frame,
#                 feature_names,
#                 fill_values,
#                 means,
#                 stds,
#             )
#             for frame in frames
#         ]
#         for split_name, frames in splits.items()
#     }

#     output = {}
#     for split_name, frames in prepared.items():
#         windows = []
#         targets = []
#         window_uids = []

#         for frame in frames:
#             frame = frame.sort_values("_date").reset_index(drop=True)
#             feature_array = frame[feature_names].to_numpy(dtype=np.float32)
#             target_array = frame[target_names].apply(
#                 pd.to_numeric,
#                 errors="coerce",
#             ).to_numpy(dtype=np.float32)
#             dates = frame["_date"].to_numpy()

#             for end_index in range(SEQUENCE_LOOKBACK_DAYS, len(frame)):
#                 start_index = end_index - SEQUENCE_LOOKBACK_DAYS
#                 window_dates = dates[start_index:end_index]
#                 if not np.all(
#                     np.diff(window_dates).astype("timedelta64[D]")
#                     == np.timedelta64(1, "D")
#                 ):
#                     continue
#                 if not np.isfinite(target_array[end_index]).all():
#                     continue
#                 windows.append(feature_array[start_index:end_index])
#                 targets.append(target_array[end_index])
#                 window_uids.append(frame.loc[end_index, "uid"])

#         if windows:
#             x_array = np.stack(windows)
#             y_array = np.stack(targets)
#             direction_array = _direction_vectors(
#                 x_array,
#                 feature_names,
#                 direction_map,
#             )
#         else:
#             x_array = np.empty(
#                 (0, SEQUENCE_LOOKBACK_DAYS, len(feature_names)),
#                 dtype=np.float32,
#             )
#             y_array = np.empty(
#                 (0, len(target_names)),
#                 dtype=np.float32,
#             )
#             direction_array = np.empty(
#                 (0, len(direction_map), len(feature_names)),
#                 dtype=np.float32,
#             )

#         uid_array = (
#             torch.tensor(window_uids)
#             if all(isinstance(uid, (int, np.integer)) for uid in window_uids)
#             else window_uids
#         )
#         output[split_name] = {
#             "X": torch.from_numpy(x_array),
#             "y": torch.from_numpy(y_array),
#             "uid": uid_array,
#             "direction_vectors": torch.from_numpy(direction_array),
#         }

#     output["metadata"] = {
#         "feature_names": feature_names,
#         "target_names": target_names,
#         "direction_names": list(direction_map),
#         "lookback_days": SEQUENCE_LOOKBACK_DAYS,
#         "min_sequence_days": MIN_SEQUENCE_DAYS,
#         "normalization": "train-only median, mean, and standard deviation",
#         "feature_raw_mean": {
#             name: float(means[name]) for name in feature_names
#         },
#         "feature_raw_std": {
#             name: float(stds[name]) for name in feature_names
#         },
#         "feature_raw_min": {
#             name: float(filled[name].min()) for name in feature_names
#         },
#         "feature_raw_max": {
#             name: float(filled[name].max()) for name in feature_names
#         },
#     }
#     return output


# def build_sequences(
#     model_df: pd.DataFrame,
#     features_final: list[str],
#     target_cols: list[str],
#     direction_map: dict[str, list[str]],
#     force: bool = False,
#     cache_path: Path | None = None,
# ) -> dict:
#     """Build cached next-day sequences for either dataset's model dataframe."""
#     feature_names = list(features_final)
#     target_names = list(target_cols)
#     build_fn = lambda: _build_uncached(
#         model_df,
#         feature_names,
#         target_names,
#         direction_map,
#     )
#     if cache_path is None:
#         return build_fn()
#     return cache_torch(cache_path, build_fn=build_fn, force=force)
