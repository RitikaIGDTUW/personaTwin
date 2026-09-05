"""Evaluate simple next-day PAM-delta baselines before deep models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import sequence_cache_path


def _values(split: dict[str, object], key: str) -> np.ndarray:
    value = split[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted.reshape(-1) - observed.reshape(-1)
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "corr": float(np.corrcoef(predicted.reshape(-1), observed.reshape(-1))[0, 1])
        if len(observed) > 1 and np.std(predicted) > 0 and np.std(observed) > 0
        else float("nan"),
    }


def _participant_mean_predictions(
    train: dict[str, object], split: dict[str, object], global_mean: float
) -> np.ndarray:
    train_uids = _values(train, "uid").reshape(-1)
    train_y = _values(train, "y").reshape(-1)
    split_uids = _values(split, "uid").reshape(-1)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for uid, target in zip(train_uids, train_y):
        key = str(uid)
        sums[key] = sums.get(key, 0.0) + float(target)
        counts[key] = counts.get(key, 0) + 1
    return np.asarray(
        [sums.get(str(uid), global_mean) / counts.get(str(uid), 1) for uid in split_uids],
        dtype=np.float32,
    )


def _feature_schema_report(feature_names: list[str]) -> dict[str, int]:
    suffixes = (
        "_delta1",
        "_delta3",
        "_roll_mean7",
        "_roll_std7",
        "_trend7",
        "_dev_from_own_mean",
    )
    return {
        "total_features": len(feature_names),
        "base_features": sum(
            not any(name.endswith(suffix) for suffix in suffixes)
            for name in feature_names
        ),
        **{
            suffix.removeprefix("_"): sum(name.endswith(suffix) for name in feature_names)
            for suffix in suffixes
        },
    }


def evaluate(dataset: str, predict_delta: bool, top_k: int) -> dict[str, object]:
    path = sequence_cache_path(dataset, predict_delta=predict_delta)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    train, val, test = artifact["train"], artifact["val"], artifact["test"]
    train_x = _values(train, "X").reshape(len(train["X"]), -1).astype(np.float64)
    train_y = _values(train, "y").reshape(-1).astype(np.float64)
    val_x = _values(val, "X").reshape(len(val["X"]), -1).astype(np.float64)
    val_y = _values(val, "y").reshape(-1).astype(np.float64)
    test_x = _values(test, "X").reshape(len(test["X"]), -1).astype(np.float64)
    test_y = _values(test, "y").reshape(-1).astype(np.float64)

    finite_columns = np.isfinite(train_x).all(axis=0)
    variable_columns = np.nanstd(train_x, axis=0) > 1e-8
    usable_columns = finite_columns & variable_columns
    train_x = train_x[:, usable_columns]
    val_x = val_x[:, usable_columns]
    test_x = test_x[:, usable_columns]

    global_mean = float(train_y.mean())
    predictions = {
        "zero_change": (np.zeros_like(val_y), np.zeros_like(test_y)),
        "global_mean": (np.full_like(val_y, global_mean), np.full_like(test_y, global_mean)),
        "participant_mean": (
            _participant_mean_predictions(train, val, global_mean),
            _participant_mean_predictions(train, test, global_mean),
        ),
    }

    k = min(top_k, train_x.shape[1], max(1, train_x.shape[0] - 1))
    models = {
        "ridge": make_pipeline(
            SelectKBest(f_regression, k=k),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "lasso": make_pipeline(
            SelectKBest(f_regression, k=k),
            StandardScaler(),
            Lasso(alpha=0.01, max_iter=5000),
        ),
    }
    for name, model in models.items():
        model.fit(train_x, train_y)
        predictions[name] = (model.predict(val_x), model.predict(test_x))

    result: dict[str, object] = {
        "dataset": dataset,
        "predict_delta": predict_delta,
        "cache": str(path),
        "feature_schema": _feature_schema_report(
            artifact.get("metadata", {}).get("feature_names", [])
        ),
        "window_shapes": {
            split_name: list(artifact[split_name]["X"].shape)
            for split_name in ("train", "val", "test")
        },
        "split": {
            "method": "per-participant chronological 70/15/15",
            "train_participants": len(set(map(str, _values(train, "uid").reshape(-1)))),
            "val_participants": len(set(map(str, _values(val, "uid").reshape(-1)))),
            "test_participants": len(set(map(str, _values(test, "uid").reshape(-1)))),
        },
        "top_k": k,
        "usable_flattened_features": int(usable_columns.sum()),
        "dropped_flattened_features": int((~usable_columns).sum()),
        "validation": {},
        "test": {},
    }
    for name, (val_pred, test_pred) in predictions.items():
        result["validation"][name] = _metrics(val_y, val_pred)
        result["test"][name] = _metrics(test_y, test_pred)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--predict-delta", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.predict_delta, args.top_k)
    print(json.dumps(result, indent=2, allow_nan=True))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
