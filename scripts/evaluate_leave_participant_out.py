"""Leave-participant-out evaluation for the frozen Lasso modeling protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.config import sequence_cache_path
from scripts.frozen_feature_models import SCENARIO_LEVERS

DEFAULT_OUTPUT = Path("data/processed/ces_lopo_results.json")
DEFAULT_TABLE = Path("data/processed/ces_lopo_per_participant.csv")


def _values(split: dict[str, object], key: str) -> np.ndarray:
    value = split[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = observed.reshape(-1)
    predicted = predicted.reshape(-1)
    error = predicted - observed
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "corr": float(np.corrcoef(predicted, observed)[0, 1])
        if len(observed) > 1 and np.std(predicted) > 0 and np.std(observed) > 0
        else float("nan"),
    }


def _select_top_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    top_k: int,
    feature_names: list[str],
    lookback: int,
    scenario_aware: bool,
) -> np.ndarray:
    finite = np.isfinite(train_x).all(axis=0)
    variable = np.nanstd(train_x, axis=0) > 1e-8
    usable = finite & variable
    scores, _ = f_regression(train_x[:, usable], train_y)
    usable_indices = np.flatnonzero(usable)
    ranking = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    pinned = []
    if scenario_aware:
        for day_offset in range(lookback):
            for lever in SCENARIO_LEVERS:
                pinned.append(day_offset * len(feature_names) + feature_names.index(lever))
    ranked_indices = [int(index) for index in usable_indices[ranking]]
    return np.asarray(list(dict.fromkeys(pinned + ranked_indices))[:top_k], dtype=np.int64)


def _fit_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    top_k: int,
    feature_names: list[str],
    lookback: int,
    scenario_aware: bool,
) -> np.ndarray:
    selected = _select_top_features(
        train_x, train_y, top_k, feature_names, lookback, scenario_aware
    )
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x[:, selected])
    scaled_heldout = scaler.transform(heldout_x[:, selected])
    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(scaled_train, train_y)
    return model.predict(scaled_heldout)


def evaluate(
    dataset: str,
    predict_delta: bool,
    top_k: int,
    output_path: Path,
    table_path: Path,
    scenario_aware: bool,
) -> dict[str, object]:
    cache_path = sequence_cache_path(dataset, predict_delta=predict_delta)
    artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
    split = artifact["train"]
    features = _values(split, "X").reshape(len(split["X"]), -1).astype(np.float64)
    targets = _values(split, "y").reshape(-1).astype(np.float64)
    groups = _values(split, "uid").reshape(-1)
    feature_names = list(artifact.get("metadata", {}).get("feature_names", []))
    lookback = int(artifact.get("metadata", {}).get("lookback_days", split["X"].shape[1]))
    logo = LeaveOneGroupOut()
    participant_rows: list[dict[str, object]] = []

    splits = list(logo.split(features, targets, groups))
    for fold_index, (train_index, heldout_index) in enumerate(splits, start=1):
        heldout_uid = str(groups[heldout_index[0]])
        print(
            f"[lopo] fold {fold_index}/{len(splits)} participant={heldout_uid} "
            f"train_windows={len(train_index)} test_windows={len(heldout_index)}",
            flush=True,
        )
        predictions = _fit_fold(
            features[train_index],
            targets[train_index],
            features[heldout_index],
            top_k,
            feature_names,
            lookback,
            scenario_aware,
        )
        metrics = _metrics(targets[heldout_index], predictions)
        participant_rows.append(
            {
                "fold": fold_index,
                "participant": heldout_uid,
                "windows": len(heldout_index),
                **metrics,
            }
        )

    aggregate = {}
    for metric in ("mae", "rmse", "corr"):
        values = np.asarray(
            [row[metric] for row in participant_rows if np.isfinite(row[metric])],
            dtype=float,
        )
        aggregate[metric] = {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=0)) if len(values) else float("nan"),
            "count": int(len(values)),
        }

    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["fold", "participant", "windows", "mae", "rmse", "corr"],
        )
        writer.writeheader()
        writer.writerows(participant_rows)

    result = {
        "dataset": dataset,
        "predict_delta": predict_delta,
        "cache": str(cache_path),
        "protocol": {
            "split": "LeaveOneGroupOut over training windows",
            "groups": "participant uid",
            "feature_selection": "top absolute train-fold f_regression score",
            "top_k": top_k,
            "scenario_aware": scenario_aware,
            "scenario_levers": SCENARIO_LEVERS if scenario_aware else [],
            "standardization": "fit on each fold training participants",
            "model": "StandardScaler + Lasso(alpha=0.01)",
            "participants": len(participant_rows),
        },
        "aggregate": aggregate,
        "participant_table": str(table_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--predict-delta", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--scenario-aware", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.dataset,
        args.predict_delta,
        args.top_k,
        args.output,
        args.table,
        args.scenario_aware,
    )
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
