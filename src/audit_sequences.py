"""Audit cached Stage 2.3 sequence artifacts without modifying them."""

from __future__ import annotations

import argparse

import torch

from src.config import (
    CES_SEQUENCES_CACHE,
    STUDENTLIFE_SEQUENCES_CACHE,
)


def audit_sequences(dataset: str) -> dict:
    """Validate one cached sequence artifact and print a compact report."""
    cache_path = (
        STUDENTLIFE_SEQUENCES_CACHE
        if dataset == "studentlife"
        else CES_SEQUENCES_CACHE
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset} sequence cache: {cache_path}. "
            f"Build it with python -m src.build_sequences {dataset}."
        )

    artifact = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )
    metadata = artifact.get("metadata", {})
    report = {
        "dataset": dataset,
        "cache": str(cache_path),
        "features": len(metadata.get("feature_names", [])),
        "targets": metadata.get("target_names", []),
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split = artifact[split_name]
        x = split["X"]
        y = split["y"]
        direction_vectors = split["direction_vectors"]
        if x.ndim != 3:
            raise ValueError(f"{dataset}/{split_name}: X must be 3D")
        if y.ndim != 2:
            raise ValueError(f"{dataset}/{split_name}: y must be 2D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{dataset}/{split_name}: X/y row mismatch")
        if direction_vectors.shape[0] != x.shape[0]:
            raise ValueError(
                f"{dataset}/{split_name}: direction-vector row mismatch"
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{dataset}/{split_name}: X contains non-finite values")
        if not torch.isfinite(y).all():
            raise ValueError(f"{dataset}/{split_name}: y contains non-finite values")

        uids = split["uid"]
        unique_uids = len(set(uids.tolist() if torch.is_tensor(uids) else uids))
        report["splits"][split_name] = {
            "windows": int(x.shape[0]),
            "X": tuple(x.shape),
            "y": tuple(y.shape),
            "direction_vectors": tuple(direction_vectors.shape),
            "participants": unique_uids,
        }

    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    args = parser.parse_args()
    audit_sequences(args.dataset)


if __name__ == "__main__":
    main()
