"""Run the Sensitivity Engine (univariate + interaction) against a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.audit_pre_sensitivity import load_artifact, participant_index
from src.config import (
    CES_DIRECTION_MAP_CACHE,
    PROCESSED_DIR,
    STUDENTLIFE_DIRECTION_MAP_CACHE,
)
from src.model import UncertaintyPersonalizedGRU, UncertaintyPopulationGRU
from src.sensitivity import (
    aggregate_interaction_profiles,
    aggregate_profiles,
    export_continuous_profiles,
    export_interaction_profiles,
    export_profiles,
    participant_counts,
    profile_direction_pairs,
    profile_split,
)


def load_direction_map(dataset: str) -> dict[str, list[str]]:
    path = STUDENTLIFE_DIRECTION_MAP_CACHE if dataset == "studentlife" else CES_DIRECTION_MAP_CACHE
    return json.loads(Path(path).read_text())


def load_model(artifact: dict, checkpoint_path: Path, personalized: bool, device: str = "cpu"):
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] requested device '{device}' is unavailable; falling back to CPU")
        target_device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
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

    state = checkpoint["model_state"]
    if "mean_head.weight" not in state and "head.weight" in state:
        state = dict(state)
        state["mean_head.weight"] = state.pop("head.weight")
        state["mean_head.bias"] = state.pop("head.bias")
        state["logvar_head.weight"] = torch.zeros_like(model.logvar_head.weight)
        state["logvar_head.bias"] = torch.zeros_like(model.logvar_head.bias)
    model.load_state_dict(state, strict=False)
    model.to(target_device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--personalized", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None,
                         help="Cap windows for the interaction sweep (pairwise cost grows fast)")
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Windows per univariate sensitivity batch")
    parser.add_argument("--threshold", type=float, default=8.0,
                         help="PAM threshold used for margin/crossing calculations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_prefix = f"{args.dataset}_personalized" if args.personalized else args.dataset

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU instead")
        args.device = "cpu"

    artifact = load_artifact(args.dataset)
    feature_names = artifact["metadata"]["feature_names"]
    direction_map = load_direction_map(args.dataset)
    model = load_model(artifact, args.checkpoint, args.personalized, device=args.device)
    idx = participant_index(artifact) if args.personalized else None

    counts = participant_counts(artifact, split_name="test")
    print(f"[{args.dataset}] test set: {counts['n_windows']} windows, "
          f"{counts['n_participants']} distinct participants")

    device = torch.device(args.device)
    train_targets = artifact["train"]["y"].float().to(device)
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0).clamp_min(1e-6)

    # Univariate slope/curvature/margin per direction
    rows, continuous_rows = profile_split(
        model=model,
        artifact=artifact,
        feature_names=feature_names,
        direction_map=direction_map,
        threshold=args.threshold,
        max_windows=args.max_windows,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=args.batch_size,
        personalized=args.personalized,
        participant_index=idx,
    )
    aggregates = aggregate_profiles(rows)
    export_profiles(rows, aggregates, PROCESSED_DIR / "sensitivity", prefix=output_prefix)
    export_continuous_profiles(
        continuous_rows,
        PROCESSED_DIR / "sensitivity",
        prefix=output_prefix,
    )
    # Pairwise interaction sensitivity
    interaction_rows = profile_direction_pairs(
        model=model,
        artifact=artifact,
        feature_names=feature_names,
        direction_map=direction_map,
        max_windows=args.max_windows,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        personalized=args.personalized,
        participant_index=idx,
    )
    interaction_aggregates = aggregate_interaction_profiles(interaction_rows)
    export_interaction_profiles(
        interaction_rows,
        interaction_aggregates,
        PROCESSED_DIR / "sensitivity",
        prefix=output_prefix,
    )

    print(f"[{args.dataset}] wrote univariate + interaction profiles to "
          f"{PROCESSED_DIR / 'sensitivity'}")


if __name__ == "__main__":
    main()