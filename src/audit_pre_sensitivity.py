"""Audit prerequisites before running Sensitivity Engine analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import (
    CES_SEQUENCES_CACHE,
    MODEL_CHECKPOINT_DIR,
    STUDENTLIFE_SEQUENCES_CACHE,
)
from src.model import UncertaintyPersonalizedGRU, UncertaintyPopulationGRU


def load_artifact(dataset: str) -> dict:
    path = STUDENTLIFE_SEQUENCES_CACHE if dataset == "studentlife" else CES_SEQUENCES_CACHE
    return torch.load(path, map_location="cpu", weights_only=False)


def participant_index(artifact: dict) -> dict[str, int]:
    values = []
    for split_name in ("train", "val", "test"):
        ids = artifact[split_name]["uid"]
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        values.extend(str(value) for value in ids)
    return {value: index for index, value in enumerate(sorted(set(values)))}


def uncertainty_audit(dataset: str, personalized: bool, checkpoint_path: Path) -> dict:
    artifact = load_artifact(dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_targets = artifact["train"]["y"].float()
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0).clamp_min(1e-6)
    feature_count = artifact["test"]["X"].shape[-1]
    target_count = artifact["test"]["y"].shape[-1]
    kwargs = {
        "input_size": feature_count,
        "hidden_size": checkpoint["hidden_size"],
        "projection_size": checkpoint.get("projection_size"),
        "dropout": checkpoint.get("dropout", 0.0),
        "output_size": target_count,
    }
    index = participant_index(artifact) if personalized else None
    if personalized:
        model = UncertaintyPersonalizedGRU(
            num_participants=len(index),
            embedding_size=checkpoint["embedding_size"],
            **kwargs,
        )
    else:
        model = UncertaintyPopulationGRU(**kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    split = artifact["test"]
    features = split["X"].float()
    targets = split["y"].float()
    if personalized:
        ids = split["uid"]
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        participant_ids = torch.tensor([index[str(value)] for value in ids])
    means = []
    logvars = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            batch = features[start : start + 512]
            if personalized:
                mean, logvar = model(batch, participant_ids[start : start + 512])
            else:
                mean, logvar = model(batch)
            means.append(mean)
            logvars.append(logvar)
    mean = torch.cat(means) * target_std + target_mean
    std = torch.exp(0.5 * torch.cat(logvars)) * target_std
    observed = targets
    result = {
        "dataset": dataset,
        "model_type": "uncertainty_personalized" if personalized else "uncertainty_population",
        "checkpoint": str(checkpoint_path),
        "mean_baseline": {
            "mse": torch.mean((observed - target_mean).square()).item(),
            "mae": torch.mean((observed - target_mean).abs()).item(),
            "rmse": torch.sqrt(torch.mean((observed - target_mean).square())).item(),
        },
        "model": {
            "mse": torch.mean((observed - mean).square()).item(),
            "mae": torch.mean((observed - mean).abs()).item(),
            "rmse": torch.sqrt(torch.mean((observed - mean).square())).item(),
            "mean_std": std.mean().item(),
            "coverage_68": torch.mean(((observed >= mean - std) & (observed <= mean + std)).float()).item(),
            "coverage_95": torch.mean(((observed >= mean - 1.96 * std) & (observed <= mean + 1.96 * std)).float()).item(),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--personalized", action="store_true")
    args = parser.parse_args()
    result = uncertainty_audit(args.dataset, args.personalized, args.checkpoint)
    print(result)


if __name__ == "__main__":
    main()
