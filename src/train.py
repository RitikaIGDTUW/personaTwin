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
    MODEL_CHECKPOINT_DIR,
    MODEL_LOG_DIR,
    sequence_cache_path,
)
from src.model import (
    PersonalizedGRU,
    PopulationGRU,
    UncertaintyPersonalizedGRU,
    UncertaintyPopulationGRU,
)


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


def load_sequence_artifact(dataset: str, predict_delta: bool = False) -> dict:
    cache_path = sequence_cache_path(dataset, predict_delta=predict_delta)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run: "
            f"python -m src.build_sequences {dataset}"
            f" {'--predict-delta' if predict_delta else ''} --force"
        )
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def target_statistics(split: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target scaling statistics from the training split only."""
    targets = split["y"].float()
    mean = targets.mean(dim=0)
    std = targets.std(dim=0).clamp_min(1e-6)
    return mean, std


def make_loader(
    split: dict,
    batch_size: int,
    shuffle: bool,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> DataLoader:
    targets = split["y"].float()
    if target_mean is not None and target_std is not None:
        targets = (targets - target_mean) / target_std
    dataset = TensorDataset(split["X"].float(), targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: PopulationGRU,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        rows = features.shape[0]
        total_loss += loss.item() * rows
        total_rows += rows

    return total_loss / max(total_rows, 1)


def _safe_corr(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    """Pearson correlation between predicted and observed values.

    In predict_delta mode this IS the "first-difference correlation" the
    original issue flagged as very low — the direct check for whether the
    model tracks day-to-day change rather than just predicting a constant.
    Unlike mae/rmse, this is NOT translation-invariant, so it isn't
    redundant with the delta-scale error metrics above.
    """
    pred_flat = predicted.flatten().double()
    obs_flat = observed.flatten().double()
    if pred_flat.numel() < 2:
        return float("nan")
    pred_centered = pred_flat - pred_flat.mean()
    obs_centered = obs_flat - obs_flat.mean()
    denom = pred_centered.norm() * obs_centered.norm()
    if denom.item() == 0:
        return float("nan")
    return (pred_centered @ obs_centered / denom).item()


def evaluate(
    model: PopulationGRU,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute error metrics.

    If ``baseline`` is given (the artifact's per-window previous-day target,
    present whenever the sequence cache was built with predict_delta=True),
    also reports abs_mae/abs_rmse on RECONSTRUCTED absolute PAM
    (baseline + prediction), so delta-trained models stay comparable to
    absolute-trained ones. Relies on the loader having shuffle=False so row
    order matches ``baseline`` order exactly.
    """
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            predictions.append(model(features.to(device)).cpu())
            targets.append(target)

    if not predictions:
        metrics = {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
        if baseline is not None:
            metrics.update({"abs_mae": float("nan"), "abs_rmse": float("nan")})
        return metrics

    predicted = torch.cat(predictions)
    observed = torch.cat(targets)
    if target_mean is not None and target_std is not None:
        predicted = predicted * target_std + target_mean
        observed = observed * target_std + target_mean
    error = predicted - observed
    mse = torch.mean(error.square()).item()
    metrics = {
        "mse": mse,
        "mae": torch.mean(error.abs()).item(),
        "rmse": float(np.sqrt(mse)),
        "corr": _safe_corr(predicted, observed),
    }
    if baseline is not None:
        baseline = baseline.float()
        abs_predicted = baseline + predicted
        abs_observed = baseline + observed
        abs_error = abs_predicted - abs_observed
        abs_mse = torch.mean(abs_error.square()).item()
        metrics["abs_mae"] = torch.mean(abs_error.abs()).item()
        metrics["abs_rmse"] = float(np.sqrt(abs_mse))
    return metrics
    """Create a stable participant-to-embedding index across all splits."""
    identifiers = []
    for split_name in ("train", "val", "test"):
        values = artifact[split_name]["uid"]
        values = values.tolist() if torch.is_tensor(values) else values
        identifiers.extend(str(value) for value in values)
    return {identifier: index for index, identifier in enumerate(sorted(set(identifiers)))}


def participant_index(artifact: dict) -> dict[str, int]:
    """Create a stable participant-to-embedding index across all splits."""
    identifiers = []
    for split_name in ("train", "val", "test"):
        values = artifact[split_name]["uid"]
        values = values.tolist() if torch.is_tensor(values) else values
        identifiers.extend(str(value) for value in values)
    return {identifier: index for index, identifier in enumerate(sorted(set(identifiers)))}


def make_personalized_loader(
    split: dict,
    index: dict[str, int],
    batch_size: int,
    shuffle: bool,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> DataLoader:
    """Build batches containing features, targets, and participant indices."""
    values = split["uid"]
    values = values.tolist() if torch.is_tensor(values) else values
    participant_ids = torch.tensor(
        [index[str(value)] for value in values],
        dtype=torch.long,
    )
    targets = split["y"].float()
    if target_mean is not None and target_std is not None:
        targets = (targets - target_mean) / target_std
    dataset = TensorDataset(
        split["X"].float(),
        targets,
        participant_ids,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_personalized_epoch(
    model: PersonalizedGRU,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    for features, targets, participants in loader:
        features = features.to(device)
        targets = targets.to(device)
        participants = participants.to(device)
        predictions = model(features, participants)
        loss = loss_fn(predictions, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        rows = features.shape[0]
        total_loss += loss.item() * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def evaluate_personalized(
    model: PersonalizedGRU,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
) -> dict[str, float]:
    """See evaluate() for what `baseline` does (reconstructs absolute PAM
    from a delta prediction for reporting, when the artifact has one)."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target, participants in loader:
            predictions.append(
                model(
                    features.to(device),
                    participants.to(device),
                ).cpu()
            )
            targets.append(target)
    if not predictions:
        metrics = {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
        if baseline is not None:
            metrics.update({"abs_mae": float("nan"), "abs_rmse": float("nan")})
        return metrics
    predicted = torch.cat(predictions)
    observed = torch.cat(targets)
    if target_mean is not None and target_std is not None:
        predicted = predicted * target_std + target_mean
        observed = observed * target_std + target_mean
    error = predicted - observed
    mse = torch.mean(error.square()).item()
    metrics = {
        "mse": mse,
        "mae": torch.mean(error.abs()).item(),
        "rmse": float(np.sqrt(mse)),
        "corr": _safe_corr(predicted, observed),
    }
    if baseline is not None:
        baseline = baseline.float()
        abs_predicted = baseline + predicted
        abs_observed = baseline + observed
        abs_error = abs_predicted - abs_observed
        abs_mse = torch.mean(abs_error.square()).item()
        metrics["abs_mae"] = torch.mean(abs_error.abs()).item()
        metrics["abs_rmse"] = float(np.sqrt(abs_mse))
    return metrics


def gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL in standardized target space."""
    return 0.5 * (
        torch.exp(-logvar) * (targets - mean).square() + logvar
    ).mean()


def uncertainty_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    targets: torch.Tensor,
    mean_loss_weight: float,
) -> torch.Tensor:
    """Combine Gaussian NLL with a direct mean-prediction loss."""
    nll = gaussian_nll(mean, logvar, targets)
    mean_loss = torch.mean((targets - mean).square())
    return nll + mean_loss_weight * mean_loss


def run_uncertainty_epoch(
    model: UncertaintyPopulationGRU | UncertaintyPersonalizedGRU,
    loader: DataLoader,
    device: torch.device,
    personalized: bool,
    optimizer: torch.optim.Optimizer | None = None,
    mean_loss_weight: float = 0.25,
    mse_only: bool = False,
) -> float:
    """mse_only=True runs a pure-MSE warm-start epoch: the mean head is trained
    to actually fit the target before the logvar head is allowed to compete with
    it via the NLL term. This avoids the classic heteroscedastic-loss collapse
    where the model lowers its loss by inflating predicted variance instead of
    improving the mean prediction."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        features = batch[0].to(device)
        targets = batch[1].to(device)
        if personalized:
            mean, logvar = model(features, batch[2].to(device))
        else:
            mean, logvar = model(features)
        if mse_only:
            loss = torch.mean((targets - mean).square())
        else:
            loss = uncertainty_loss(mean, logvar, targets, mean_loss_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        rows = features.shape[0]
        total_loss += loss.item() * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def evaluate_uncertainty(
    model: UncertaintyPopulationGRU | UncertaintyPersonalizedGRU,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    personalized: bool,
) -> dict[str, float]:
    model.eval()
    means = []
    logvars = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            if personalized:
                mean, logvar = model(features, batch[2].to(device))
            else:
                mean, logvar = model(features)
            means.append(mean.cpu())
            logvars.append(logvar.cpu())
            targets.append(batch[1])
    if not means:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "nll": float("nan"), "mean_std": float("nan")}
    mean = torch.cat(means)
    logvar = torch.cat(logvars)
    standardized_targets = torch.cat(targets)
    nll = gaussian_nll(mean, logvar, standardized_targets).item()
    mean = mean * target_std + target_mean
    observed = standardized_targets * target_std + target_mean
    predictive_std = torch.exp(0.5 * logvar) * target_std
    error = mean - observed
    mse = torch.mean(error.square()).item()
    return {
        "mse": mse,
        "mae": torch.mean(error.abs()).item(),
        "rmse": float(np.sqrt(mse)),
        "nll": nll,
        "mean_std": predictive_std.mean().item(),
    }


def train(
    dataset: str,
    predict_delta: bool = False,
    epochs: int = 30,
    batch_size: int = 128,
    hidden_size: int = 64,
    projection_size: int | None = None,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    patience: int = 7,
    device: str = "auto",
    seed: int = 42,
    max_train_windows: int | None = None,
    result_tag: str | None = None,
) -> dict[str, float]:
    set_seed(seed)
    device = resolve_device(device)
    artifact = load_sequence_artifact(dataset, predict_delta=predict_delta)
    metadata = artifact.get("metadata", {})
    feature_count = artifact["train"]["X"].shape[-1]
    target_count = artifact["train"]["y"].shape[-1]
    target_mean, target_std = target_statistics(artifact["train"])

    train_split = artifact["train"]
    if max_train_windows is not None:
        limit = min(max_train_windows, len(train_split["X"]))
        train_split = {
            key: value[:limit]
            for key, value in train_split.items()
            if torch.is_tensor(value) and value.ndim > 0
        }

    train_loader = make_loader(
        train_split,
        batch_size,
        shuffle=True,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_loader = make_loader(
        artifact["val"],
        batch_size,
        shuffle=False,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_loader = make_loader(
        artifact["test"],
        batch_size,
        shuffle=False,
        target_mean=target_mean,
        target_std=target_std,
    )

    model = PopulationGRU(
        input_size=feature_count,
        hidden_size=hidden_size,
        projection_size=projection_size,
        dropout=dropout,
        output_size=target_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    suffix = f"_{result_tag}" if result_tag else ""
    log_path = MODEL_LOG_DIR / f"{dataset}_population_gru{suffix}.csv"
    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{dataset}_population_gru{suffix}.pt"

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
            scheduler.step(val_loss)
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
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        target_mean=target_mean,
        target_std=target_std,
        baseline=artifact["test"].get("baseline") if metadata.get("predict_delta") else None,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "feature_count": feature_count,
            "target_names": metadata.get("target_names", []),
            "hidden_size": hidden_size,
            "projection_size": projection_size,
            "dropout": dropout,
            "target_mean": target_mean,
            "target_std": target_std,
            "best_val_loss": best_val,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"test_metrics={test_metrics}")
    return test_metrics


def train_personalized(
    dataset: str,
    predict_delta: bool = False,
    epochs: int = 30,
    batch_size: int = 128,
    hidden_size: int = 64,
    embedding_size: int = 16,
    projection_size: int | None = None,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    patience: int = 7,
    device: str = "auto",
    seed: int = 42,
    max_train_windows: int | None = None,
    result_tag: str | None = None,
) -> dict[str, float]:
    """Train the participant-embedding GRU using the same split as baseline."""
    set_seed(seed)
    device = resolve_device(device)
    artifact = load_sequence_artifact(dataset, predict_delta=predict_delta)
    metadata = artifact.get("metadata", {})
    feature_count = artifact["train"]["X"].shape[-1]
    target_count = artifact["train"]["y"].shape[-1]
    target_mean, target_std = target_statistics(artifact["train"])
    index = participant_index(artifact)

    train_split = artifact["train"]
    if max_train_windows is not None:
        limit = min(max_train_windows, len(train_split["X"]))
        train_split = {
            key: value[:limit]
            for key, value in train_split.items()
            if key == "uid"
            or (torch.is_tensor(value) and value.ndim > 0)
        }
    train_loader = make_personalized_loader(
        train_split,
        index,
        batch_size,
        shuffle=True,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_loader = make_personalized_loader(
        artifact["val"],
        index,
        batch_size,
        shuffle=False,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_loader = make_personalized_loader(
        artifact["test"],
        index,
        batch_size,
        shuffle=False,
        target_mean=target_mean,
        target_std=target_std,
    )

    model = PersonalizedGRU(
        input_size=feature_count,
        num_participants=len(index),
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        projection_size=projection_size,
        dropout=dropout,
        output_size=target_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    suffix = f"_{result_tag}" if result_tag else ""
    log_path = MODEL_LOG_DIR / f"{dataset}_personalized_gru{suffix}.csv"
    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{dataset}_personalized_gru{suffix}.pt"

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss", "val_loss"],
        )
        writer.writeheader()
        print(
            f"dataset={dataset} model=personalized_gru "
            f"device={device} features={feature_count} "
            f"participants={len(index)}"
        )
        print(f"targets={metadata.get('target_names', target_count)}")
        for epoch in range(1, epochs + 1):
            train_loss = run_personalized_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer,
            )
            val_loss = run_personalized_epoch(
                model,
                val_loader,
                loss_fn,
                device,
            )
            scheduler.step(val_loss)
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
        raise RuntimeError("Personalized training produced no checkpoint state")
    model.load_state_dict(best_state)
    test_metrics = evaluate_personalized(
        model,
        test_loader,
        device,
        target_mean=target_mean,
        target_std=target_std,
        baseline=artifact["test"].get("baseline") if metadata.get("predict_delta") else None,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "feature_count": feature_count,
            "target_names": metadata.get("target_names", []),
            "hidden_size": hidden_size,
            "embedding_size": embedding_size,
            "projection_size": projection_size,
            "dropout": dropout,
            "target_mean": target_mean,
            "target_std": target_std,
            "participant_index": index,
            "best_val_loss": best_val,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"test_metrics={test_metrics}")
    return test_metrics


def train_uncertainty(
    dataset: str,
    personalized: bool = False,
    predict_delta: bool = False,
    epochs: int = 30,
    batch_size: int = 128,
    hidden_size: int = 64,
    embedding_size: int = 16,
    projection_size: int | None = None,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    patience: int = 7,
    device: str = "auto",
    seed: int = 42,
    max_train_windows: int | None = None,
    result_tag: str | None = None,
    mean_loss_weight: float = 1.0,
    warmup_epochs: int = 5,
) -> dict[str, float]:
    """Train an uncertainty-aware population or personalized GRU.

    warmup_epochs: number of initial epochs trained with plain MSE only (the
    logvar head still runs forward but doesn't receive gradient), so the mean
    head is well-fit before the NLL term is introduced. mean_loss_weight was
    raised from 0.25 to 1.0 by default so the direct mean-accuracy term isn't
    dominated by the variance-inflation shortcut in the NLL term after warmup.
    """
    set_seed(seed)
    device = resolve_device(device)
    artifact = load_sequence_artifact(dataset, predict_delta=predict_delta)
    metadata = artifact.get("metadata", {})
    feature_count = artifact["train"]["X"].shape[-1]
    target_count = artifact["train"]["y"].shape[-1]
    target_mean, target_std = target_statistics(artifact["train"])
    index = participant_index(artifact) if personalized else None

    train_split = artifact["train"]
    if max_train_windows is not None:
        limit = min(max_train_windows, len(train_split["X"]))
        train_split = {
            key: value[:limit]
            for key, value in train_split.items()
            if key == "uid"
            or (torch.is_tensor(value) and value.ndim > 0)
        }

    if personalized:
        make_args = lambda split, shuffle: make_personalized_loader(
            split, index, batch_size, shuffle, target_mean, target_std
        )
        model = UncertaintyPersonalizedGRU(
            input_size=feature_count,
            num_participants=len(index),
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            projection_size=projection_size,
            dropout=dropout,
            output_size=target_count,
        ).to(device)
        suffix = "uncertainty_personalized_gru"
    else:
        make_args = lambda split, shuffle: make_loader(
            split, batch_size, shuffle, target_mean, target_std
        )
        model = UncertaintyPopulationGRU(
            input_size=feature_count,
            hidden_size=hidden_size,
            projection_size=projection_size,
            dropout=dropout,
            output_size=target_count,
        ).to(device)
        suffix = "uncertainty_population_gru"

    train_loader = make_args(train_split, True)
    val_loader = make_args(artifact["val"], False)
    test_loader = make_args(artifact["test"], False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    result_suffix = f"_{result_tag}" if result_tag else ""
    log_path = MODEL_LOG_DIR / f"{dataset}_{suffix}{result_suffix}.csv"
    checkpoint_path = MODEL_CHECKPOINT_DIR / f"{dataset}_{suffix}{result_suffix}.pt"

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file, fieldnames=["epoch", "train_loss", "val_nll", "val_mae", "warmup"]
        )
        writer.writeheader()
        print(f"dataset={dataset} model={suffix} device={device} features={feature_count} "
              f"warmup_epochs={warmup_epochs} mean_loss_weight={mean_loss_weight}")
        for epoch in range(1, epochs + 1):
            in_warmup = epoch <= warmup_epochs
            train_loss = run_uncertainty_epoch(
                model, train_loader, device, personalized, optimizer,
                mean_loss_weight, mse_only=in_warmup,
            )
            # Always evaluate with the full metric set (mae/rmse/nll), even
            # during warmup, so checkpoint selection is on real accuracy from
            # epoch 1 rather than only once warmup ends.
            val_metrics = evaluate_uncertainty(
                model, val_loader, device, target_mean, target_std, personalized
            )
            val_nll = val_metrics["nll"]
            val_mae = val_metrics["mae"]
            scheduler.step(val_mae)
            writer.writerow({
                "epoch": epoch, "train_loss": train_loss,
                "val_nll": val_nll, "val_mae": val_mae, "warmup": in_warmup,
            })
            log_file.flush()
            print(f"epoch={epoch:03d} warmup={in_warmup} train_loss={train_loss:.6f} "
                  f"val_nll={val_nll:.6f} val_mae={val_mae:.6f}")
            # Select the checkpoint by validation MAE (real prediction accuracy
            # in original PAM units), not validation NLL. NLL can be lowered by
            # the model simply inflating predicted variance without getting any
            # better at the actual mean prediction — that's the collapse this
            # is fixing. We also skip warmup epochs for selection so a
            # partially-warmed-up model can't "win" on a fluke.
            if not in_warmup and val_mae < best_val:
                best_val = val_mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            elif not in_warmup:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"early stopping at epoch {epoch}")
                    break

    if best_state is None:
        raise RuntimeError("Uncertainty training produced no checkpoint state")
    model.load_state_dict(best_state)
    test_metrics = evaluate_uncertainty(
        model, test_loader, device, target_mean, target_std, personalized
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "dataset": dataset,
            "target_names": metadata.get("target_names", []),
            "feature_count": feature_count,
            "hidden_size": hidden_size,
            "embedding_size": embedding_size,
            "projection_size": projection_size,
            "dropout": dropout,
            "target_mean": target_mean,
            "target_std": target_std,
            "participant_index": index,
            "best_val_mae": best_val,
            "mean_loss_weight": mean_loss_weight,
            "warmup_epochs": warmup_epochs,
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
    parser.add_argument(
        "--predict-delta",
        action="store_true",
        help="load/train the separate next-day PAM-delta sequence cache",
    )
    parser.add_argument(
        "--model",
        choices=["population", "personalized", "uncertainty", "uncertainty_personalized"],
        default="population",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--projection-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5,
                         help="Epochs trained with plain MSE before the NLL term is introduced "
                              "(uncertainty/uncertainty_personalized models only)")
    args = parser.parse_args()
    arguments = vars(args)
    model_type = arguments.pop("model")
    # mean_loss_weight / warmup_epochs only apply to the uncertainty models;
    # the deterministic population/personalized trainers don't accept them.
    uncertainty_only_keys = ("mean_loss_weight", "warmup_epochs")
    if model_type == "personalized":
        for key in uncertainty_only_keys:
            arguments.pop(key, None)
        train_personalized(**arguments)
    elif model_type == "uncertainty":
        train_uncertainty(**arguments)
    elif model_type == "uncertainty_personalized":
        train_uncertainty(personalized=True, **arguments)
    else:
        for key in uncertainty_only_keys:
            arguments.pop(key, None)
        train(**arguments)


if __name__ == "__main__":
    main()