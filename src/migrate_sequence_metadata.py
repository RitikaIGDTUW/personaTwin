"""Add raw-unit feature statistics to existing sequence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.build_sequences import _ces_inputs, _studentlife_inputs
from src.config import CES_SEQUENCES_CACHE, STUDENTLIFE_SEQUENCES_CACHE
from src.datasets import _coerce_model_dates, _split_participant_rows


def train_feature_statistics(model_df: pd.DataFrame, feature_names: list[str], date_column: str) -> tuple[dict[str, float], dict[str, float]]:
    data = model_df[["uid", date_column, *feature_names]].copy()
    data["_date"] = _coerce_model_dates(data[date_column])
    data = data.dropna(subset=["uid", "_date"]).sort_values(["uid", "_date"])
    train_frames = []
    for _, participant in data.groupby("uid", observed=True):
        if len(participant) >= 14:
            train_frames.append(_split_participant_rows(participant)["train"])
    if not train_frames:
        raise ValueError("No participant training rows found")

    train_frame = pd.concat(train_frames, ignore_index=True)
    numeric = train_frame[feature_names].apply(pd.to_numeric, errors="coerce")
    fill_values = numeric.median()
    filled = numeric.fillna(fill_values)
    means = filled.mean()
    stds = filled.std().replace(0, 1).fillna(1)
    return (
        {name: float(means[name]) for name in feature_names},
        {name: float(stds[name]) for name in feature_names},
    )


def migrate(dataset: str) -> Path:
    if dataset == "studentlife":
        model_df, feature_names, _, _ = _studentlife_inputs()
        artifact_path = STUDENTLIFE_SEQUENCES_CACHE
        date_column = "date"
    else:
        model_df, feature_names, _, _ = _ces_inputs()
        artifact_path = CES_SEQUENCES_CACHE
        date_column = "day"

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    raw_mean, raw_std = train_feature_statistics(model_df, feature_names, date_column)
    metadata = artifact.setdefault("metadata", {})
    metadata["feature_raw_mean"] = raw_mean
    metadata["feature_raw_std"] = raw_std
    torch.save(artifact, artifact_path)
    print(f"updated={artifact_path}")
    print(f"features={len(feature_names)}")
    print(f"raw_statistics={len(raw_mean)}")
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    args = parser.parse_args()
    migrate(args.dataset)


if __name__ == "__main__":
    main()
