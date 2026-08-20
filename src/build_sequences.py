"""Build Stage 2.3 sequence artifacts for StudentLife or CES."""

from __future__ import annotations

import argparse

import pandas as pd

from src.behavioral_directions import build_direction_map
from src.config import (
    CES_DIRECTION_MAP_CACHE,
    CES_SEQUENCES_CACHE,
    STUDENTLIFE_DIRECTION_MAP_CACHE,
    STUDENTLIFE_MODEL_DF_CACHE,
    STUDENTLIFE_SEQUENCES_CACHE,
)
from src.datasets import build_sequences
from src.preprocess_ces import (
    build_ces_features_final,
    build_ces_model_df,
)


def _studentlife_inputs() -> tuple[pd.DataFrame, list[str], list[str], object]:
    model_df = pd.read_parquet(STUDENTLIFE_MODEL_DF_CACHE)
    target_names = ["pam"]
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
    return (
        model_df,
        feature_names,
        target_names,
        CES_DIRECTION_MAP_CACHE,
    )


def build_dataset_sequences(
    dataset: str,
    force: bool = False,
) -> dict:
    """Build the default PAM-first sequence artifact for one dataset."""
    if dataset == "studentlife":
        model_df, feature_names, target_names, direction_cache = _studentlife_inputs()
        cache_path = STUDENTLIFE_SEQUENCES_CACHE
    elif dataset == "ces":
        model_df, feature_names, target_names, direction_cache = _ces_inputs()
        cache_path = CES_SEQUENCES_CACHE
    else:
        raise ValueError("dataset must be 'studentlife' or 'ces'")

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
    args = parser.parse_args()
    build_dataset_sequences(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
