"""Build Stage 2.3 sequence artifacts for StudentLife or CES."""

from __future__ import annotations

import argparse

import pandas as pd

from src.behavioral_directions import build_direction_map
from src.config import (
    CES_DIRECTION_MAP_CACHE,
    sequence_cache_path,
    STUDENTLIFE_DIRECTION_MAP_CACHE,
    STUDENTLIFE_MODEL_DF_CACHE,
)
from src.datasets import build_sequences
from src.preprocess_ces import (
    build_ces_features_final,
    build_ces_model_df,
)


def _studentlife_inputs() -> tuple[pd.DataFrame, list[str], list[str], object]:
    model_df = pd.read_parquet(STUDENTLIFE_MODEL_DF_CACHE)
    target_names = ["pam"]
    model_df = model_df.sort_values(["uid", "date"]).copy()
    model_df["pam_history"] = model_df["pam"]
    feature_names = [
        column
        for column in model_df.columns
        if column not in {"uid", "date", "pam", "stress", "mood"}
    ]
    return (
        model_df,
        feature_names,
        target_names,
        STUDENTLIFE_DIRECTION_MAP_CACHE,
    )


def _ces_inputs() -> tuple[pd.DataFrame, list[str], list[str], object]:
    model_df = build_ces_model_df()
    feature_names = build_ces_features_final()
    target_names = ["pam"] if "pam" in model_df else []
    if not target_names:
        raise KeyError("CES model dataframe does not contain the PAM target")
    model_df = model_df.sort_values(["uid", "day"]).copy()
    model_df["pam_history"] = model_df["pam"]
    feature_names = [*feature_names, "pam_history"]
    return (
        model_df,
        feature_names,
        target_names,
        CES_DIRECTION_MAP_CACHE,
    )


def build_dataset_sequences(
    dataset: str,
    force: bool = False,
    predict_delta: bool = False,
) -> dict:
    """Build the default PAM-first sequence artifact for one dataset."""
    if dataset == "studentlife":
        model_df, feature_names, target_names, direction_cache = _studentlife_inputs()
    elif dataset == "ces":
        model_df, feature_names, target_names, direction_cache = _ces_inputs()
    else:
        raise ValueError("dataset must be 'studentlife' or 'ces'")
    cache_path = sequence_cache_path(dataset, predict_delta=predict_delta)

    direction_map = build_direction_map(
        feature_names,
        cache_path=direction_cache,
        force=force,
    )
    print(
        f"[{dataset}] building sequences with "
        f"{len(feature_names)} features and targets={target_names}"
    )
    sequences = build_sequences(
        model_df=model_df,
        features_final=feature_names,
        target_cols=target_names,
        direction_map=direction_map,
        force=force,
        cache_path=cache_path,
        predict_delta=predict_delta,
    )

    for split_name in ("train", "val", "test"):
        split = sequences[split_name]
        print(
            f"[{dataset}] {split_name}: "
            f"X={tuple(split['X'].shape)} y={tuple(split['y'].shape)}"
        )
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["studentlife", "ces"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the direction map and sequence cache",
    )
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        help="predict next-day PAM change instead of next-day PAM",
    )
    args = parser.parse_args()
    build_dataset_sequences(
        args.dataset,
        force=args.force,
        predict_delta=args.predict_delta,
    )


if __name__ == "__main__":
    main()
