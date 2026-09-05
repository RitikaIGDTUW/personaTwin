"""Freeze train-only features and compare bounded linear/tree delta models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import sequence_cache_path

DEFAULT_FEATURE_FILE = Path("selected_features.json")
SCENARIO_LEVERS = ["sleep_duration", "sleep_start", "sleep_end"]


def _values(split: dict[str, object], key: str) -> np.ndarray:
    value = split[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten(split: dict[str, object]) -> np.ndarray:
    return _values(split, "X").reshape(len(split["X"]), -1).astype(np.float64)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted.reshape(-1) - observed.reshape(-1)
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "corr": float(np.corrcoef(predicted, observed)[0, 1])
        if np.std(predicted) > 0 and np.std(observed) > 0
        else float("nan"),
    }


def _usable_columns(train_x: np.ndarray) -> np.ndarray:
    return np.isfinite(train_x).all(axis=0) & (np.nanstd(train_x, axis=0) > 1e-8)


def freeze_features(
    artifact: dict,
    output_path: Path,
    top_k: int,
    dataset: str,
    predict_delta: bool,
    scenario_aware: bool = False,
) -> dict[str, object]:
    train_x = _flatten(artifact["train"])
    train_y = _values(artifact["train"], "y").reshape(-1).astype(np.float64)
    usable = _usable_columns(train_x)
    scores, _ = f_regression(train_x[:, usable], train_y)
    usable_indices = np.flatnonzero(usable)
    ranking = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    metadata = artifact.get("metadata", {})
    feature_names = list(metadata.get("feature_names", []))
    lookback = int(metadata.get("lookback_days", artifact["train"]["X"].shape[1]))
    pinned_indices = []
    if scenario_aware:
        missing = [lever for lever in SCENARIO_LEVERS if lever not in feature_names]
        if missing:
            raise KeyError(f"Scenario levers missing from feature schema: {missing}")
        for day_offset in range(lookback):
            for lever in SCENARIO_LEVERS:
                pinned_indices.append(day_offset * len(feature_names) + feature_names.index(lever))
    ranked_indices = [int(index) for index in usable_indices[ranking]]
    selected_indices = list(dict.fromkeys(pinned_indices + ranked_indices))[:top_k]
    selected = []
    for flat_index in selected_indices:
        day_offset, feature_index = divmod(int(flat_index), len(feature_names))
        selected.append(
            {
                "flattened_index": int(flat_index),
                "day_offset": int(day_offset),
                "feature_index": int(feature_index),
                "feature_name": feature_names[feature_index],
            }
        )

    result = {
        "dataset": dataset,
        "predict_delta": predict_delta,
        "cache_feature_count": len(feature_names),
        "lookback_days": lookback,
        "selection": (
            "scenario levers pinned, remaining slots by train-only f_regression score"
            if scenario_aware
            else "top absolute train-only f_regression score"
        ),
        "scenario_levers": SCENARIO_LEVERS if scenario_aware else [],
        "top_k": len(selected),
        "selected_features": selected,
    }
    output_path.write_text(json.dumps(result, indent=2))
    return result


def _load_selected(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.asarray(
        [item["flattened_index"] for item in data["selected_features"]],
        dtype=np.int64,
    )


def _fit_linear_models(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000)),
    }


def _select_gbm_params(
    train_x: np.ndarray,
    train_y: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    candidates = [
        {"max_depth": depth, "n_estimators": estimators}
        for depth in (2, 3)
        for estimators in (100, 200)
    ]
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    cv_rows = []
    for params in candidates:
        fold_mae = []
        for train_index, val_index in splitter.split(train_x, train_y, groups):
            model = GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                random_state=42,
            )
            model.fit(train_x[train_index], train_y[train_index])
            fold_mae.append(
                _metrics(train_y[val_index], model.predict(train_x[val_index]))["mae"]
            )
        cv_rows.append({**params, "cv_mae": float(np.mean(fold_mae))})
    best = min(cv_rows, key=lambda row: row["cv_mae"])
    return {key: int(best[key]) for key in ("max_depth", "n_estimators")}, cv_rows


def evaluate(artifact: dict, selected_path: Path) -> dict[str, object]:
    selected_indices = _load_selected(selected_path)
    train_x = _flatten(artifact["train"])[:, selected_indices]
    val_x = _flatten(artifact["val"])[:, selected_indices]
    test_x = _flatten(artifact["test"])[:, selected_indices]
    train_y = _values(artifact["train"], "y").reshape(-1).astype(np.float64)
    val_y = _values(artifact["val"], "y").reshape(-1).astype(np.float64)
    test_y = _values(artifact["test"], "y").reshape(-1).astype(np.float64)
    groups = _values(artifact["train"], "uid").reshape(-1)

    models = _fit_linear_models(train_x, train_y)
    gbm_params, cv_rows = _select_gbm_params(train_x, train_y, groups)
    models["gradient_boosting"] = GradientBoostingRegressor(
        learning_rate=0.05,
        random_state=42,
        **gbm_params,
    )

    result: dict[str, object] = {
        "selected_features": str(selected_path),
        "selected_feature_count": len(selected_indices),
        "gbm_cv": cv_rows,
        "validation": {},
        "test": {},
    }
    for name, model in models.items():
        model.fit(train_x, train_y)
        result["validation"][name] = _metrics(val_y, model.predict(val_x))
        result["test"][name] = _metrics(test_y, model.predict(test_x))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--predict-delta", action="store_true")
    parser.add_argument("--top-k", type=int, choices=[50, 75, 100], default=50)
    parser.add_argument("--selected-features", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scenario-aware", action="store_true")
    args = parser.parse_args()

    path = sequence_cache_path(args.dataset, predict_delta=args.predict_delta)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    frozen = freeze_features(
        artifact,
        args.selected_features,
        args.top_k,
        args.dataset,
        args.predict_delta,
        args.scenario_aware,
    )
    print(json.dumps(frozen, indent=2))
    if args.freeze_only:
        return
    result = evaluate(artifact, args.selected_features)
    print(json.dumps(result, indent=2, allow_nan=True))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
