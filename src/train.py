"""Train and evaluate the population PAM GRU baseline.

This script is GPU-ready for Colab/Kaggle and remains CPU-safe for small
smoke tests. It trains one dataset at a time and never mixes CES with
StudentLife in the baseline experiment.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CES_SEQUENCES_CACHE,
    MODEL_CHECKPOINT_DIR,
    MODEL_LOG_DIR,
    STUDENTLIFE_SEQUENCES_CACHE,
)
from src.model import PopulationGRU


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_sequence_artifact(dataset: str) -> dict:
    cache_path = (
        STUDENTLIFE_SEQUENCES_CACHE
        if dataset == "studentlife"
        else CES_SEQUENCES_CACHE
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run: "
            f"python -m src.build_sequences {dataset} --force"
        )
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def make_loader(split: dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(split["X"].float(), split["y"].float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: PopulationGRU,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        predictions = model(features)
        loss = loss_fn(predictions, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        rows = features.shape[0]
        total_loss += loss.item() * rows
        total_rows += rows

    return total_loss / max(total_rows, 1)


def evaluate(
    model: PopulationGRU,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            predictions.append(model(features.to(device)).cpu())
            targets.append(target)

    if not predictions:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    predicted = torch.cat(predictions)
    observed = torch.cat(targets)
    error = predicted - observed
    mse = torch.mean(error.square()).item()
    return {
        "mse": mse,
        "mae": torch.mean(error.abs()).item(),
        "rmse": float(np.sqrt(mse)),
    }


def train(
    dataset: str,
    epochs: int = 30,
    batch_size: int = 128,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 7,
    device: str = "auto",
    seed: int = 42,
    max_train_windows: int | None = None,
) -> dict[str, float]:
    set_seed(seed)
    device = resolve_device(device)
    artifact = load_sequence_artifact(dataset)
    metadata = artifact.get("metadata", {})
    feature_count = artifact["train"]["X"].shape[-1]
    target_count = artifact["train"]["y"].shape[-1]

    train_split = artifact["train"]
    if max_train_windows is not None:
        limit = min(max_train_windows, len(train_split["X"]))
        train_split = {
            key: value[:limit]
            for key, value in train_split.items()
            if torch.is_tensor(value) and value.ndim > 0
        }

    train_loader = make_loader(train_split, batch_size, shuffle=True)
    val_loader = make_loader(artifact["val"], batch_size, shuffle=False)
    test_loader = make_loader(artifact["test"], batch_size, shuffle=False)

    model = PopulationGRU(
        input_size=feature_count,
        hidden_size=hidden_size,
        output_size=target_count,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_path = MODEL_LOG_DIR / f"{dataset}_population_gru.csv"
    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{dataset}_population_gru.pt"

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss", "val_loss"],
        )
        writer.writeheader()

        print(f"dataset={dataset} device={device} features={feature_count}")
        print(f"targets={metadata.get('target_names', target_count)}")
        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer,
            )
            val_loss = run_epoch(model, val_loader, loss_fn, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            log_file.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"early stopping at epoch {epoch}")
                    break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "feature_count": feature_count,
            "target_names": metadata.get("target_names", []),
            "hidden_size": hidden_size,
            "best_val_loss": best_val,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"test_metrics={test_metrics}")
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows", type=int, default=None)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
