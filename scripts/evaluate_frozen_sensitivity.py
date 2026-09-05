"""Validate frozen Lasso/GBM counterfactual sensitivity on CES test windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from src.config import sequence_cache_path
from src.counterfactuals import perturb_sleep_schedule
from src.sensitivity import direction_feature_indices, perturb_window_for_direction


def _values(split: dict[str, object], key: str) -> np.ndarray:
    value = split[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_selected(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    return np.asarray(
        [item["flattened_index"] for item in data["selected_features"]],
        dtype=np.int64,
    )


def _selected_feature_names(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return [item["feature_name"] for item in data["selected_features"]]


def _flatten_windows(windows: torch.Tensor | np.ndarray) -> np.ndarray:
    values = windows.detach().cpu().numpy() if torch.is_tensor(windows) else np.asarray(windows)
    return values.reshape(len(values), -1).astype(np.float64)


def _fit_models(artifact: dict, selected: np.ndarray) -> tuple[object, object, StandardScaler]:
    train_x = _flatten_windows(artifact["train"]["X"])[:, selected]
    train_y = _values(artifact["train"], "y").reshape(-1).astype(np.float64)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    lasso = Lasso(alpha=0.01, max_iter=5000).fit(scaled_train, train_y)
    gbm = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=2,
        n_estimators=200,
        random_state=42,
    ).fit(train_x, train_y)
    return lasso, gbm, scaler


def _predict(model, scaler: StandardScaler | None, windows: torch.Tensor, selected: np.ndarray) -> np.ndarray:
    features = _flatten_windows(windows)[:, selected]
    if scaler is not None:
        features = scaler.transform(features)
    return np.asarray(model.predict(features), dtype=float)


def _scenario_windows(window: torch.Tensor, artifact: dict) -> dict[str, torch.Tensor]:
    return {
        "current": window.unsqueeze(0),
        "sleep_duration_plus_2h": perturb_sleep_schedule(
            window, artifact, duration_shift_hours=2.0
        ).unsqueeze(0),
        "bedtime_one_hour_earlier": perturb_sleep_schedule(
            window, artifact, bedtime_shift_hours=-1.0
        ).unsqueeze(0),
        "duration_plus_2h_and_bedtime_earlier": perturb_sleep_schedule(
            window,
            artifact,
            duration_shift_hours=2.0,
            bedtime_shift_hours=-1.0,
        ).unsqueeze(0),
    }


def evaluate(dataset: str, selected_path: Path, max_windows: int) -> dict[str, object]:
    artifact = torch.load(
        sequence_cache_path(dataset, predict_delta=True),
        map_location="cpu",
        weights_only=False,
    )
    selected = _load_selected(selected_path)
    selected_names = _selected_feature_names(selected_path)
    direction_map = json.loads(
        Path("data/interim")
        .joinpath(f"{dataset}_behavioral_direction_map.json")
        .read_text()
    )
    lasso, gbm, scaler = _fit_models(artifact, selected)
    windows = artifact["test"]["X"][:max_windows].float()
    scenario_names = list(_scenario_windows(windows[0], artifact))
    scenario_predictions = {"lasso": {}, "gradient_boosting": {}}
    for name in scenario_names:
        scenario_batch = torch.cat(
            [_scenario_windows(window, artifact)[name] for window in windows], dim=0
        )
        scenario_predictions["lasso"][name] = _predict(
            lasso, scaler, scenario_batch, selected
        )
        scenario_predictions["gradient_boosting"][name] = _predict(
            gbm, None, scenario_batch, selected
        )

    baseline_lasso = scenario_predictions["lasso"]["current"]
    baseline_gbm = scenario_predictions["gradient_boosting"]["current"]
    uids = _values(artifact["test"], "uid")[: len(windows)].reshape(-1)
    duration_changes = {
        "lasso": scenario_predictions["lasso"]["sleep_duration_plus_2h"] - baseline_lasso,
        "gradient_boosting": scenario_predictions["gradient_boosting"]["sleep_duration_plus_2h"] - baseline_gbm,
    }
    summary = {}
    for model_name, predictions in scenario_predictions.items():
        model_summary = {}
        for name in scenario_names[1:]:
            changes = predictions[name] - (
                baseline_lasso if model_name == "lasso" else baseline_gbm
            )
            model_summary[name] = {
                "mean_change": float(changes.mean()),
                "median_change": float(np.median(changes)),
                "positive_fraction": float(np.mean(changes > 0)),
                "negative_fraction": float(np.mean(changes < 0)),
            }
        summary[model_name] = model_summary

    heterogeneity = {}
    per_window_rows = []
    for model_name, changes in duration_changes.items():
        by_participant: dict[str, list[float]] = {}
        for uid, change in zip(uids, changes):
            by_participant.setdefault(str(uid), []).append(float(change))
            per_window_rows.append({
                "uid": str(uid),
                "model": model_name,
                "sleep_duration_plus_2h_change": float(change),
            })
        participant_means = np.asarray(
            [np.mean(values) for values in by_participant.values()], dtype=float
        )
        heterogeneity[model_name] = {
            "participants": int(len(participant_means)),
            "negative_fraction": float(np.mean(participant_means < 0)),
            "near_zero_fraction": float(np.mean(np.isclose(participant_means, 0.0, atol=0.05))),
            "positive_fraction": float(np.mean(participant_means > 0)),
            "mean": float(participant_means.mean()),
            "std": float(participant_means.std(ddof=0)),
            "quartiles": {
                "q25": float(np.percentile(participant_means, 25)),
                "median": float(np.percentile(participant_means, 50)),
                "q75": float(np.percentile(participant_means, 75)),
            },
        }

    directional_summary = {}
    direction_names = ("sleep", "activity", "social", "mobility", "screen")
    for direction in direction_names:
        direction_indices = direction_feature_indices(
            artifact["metadata"]["feature_names"], direction_map, direction
        )
        selected_direction = [
            name
            for name in selected_names
            if name in direction_map.get(direction, [])
            or any(
                name == f"{base}{suffix}"
                for base in direction_map.get(direction, [])
                for suffix in (
                    "_delta1",
                    "_delta3",
                    "_roll_mean7",
                    "_roll_std7",
                    "_trend7",
                    "_dev_from_own_mean",
                )
            )
        ]
        if not direction_indices:
            directional_summary[direction] = {
                "status": "unsupported_no_direction_features",
                "selected_features": selected_direction,
            }
            continue
        perturbed = torch.cat(
            [
                perturb_window_for_direction(
                    window,
                    artifact,
                    artifact["metadata"]["feature_names"],
                    direction_map,
                    direction,
                    alpha=1.0,
                ).unsqueeze(0)
                for window in windows
            ],
            dim=0,
        )
        directional_summary[direction] = {
            "status": "evaluated_plus_one_standardized_sd",
            "selected_features": selected_direction,
            "direction_feature_count": len(direction_indices),
            "lasso_mean_change": float(
                (_predict(lasso, scaler, perturbed, selected) - baseline_lasso).mean()
            ),
            "gradient_boosting_mean_change": float(
                (_predict(gbm, None, perturbed, selected) - baseline_gbm).mean()
            ),
        }

    lasso_selected_coefficients = scaler.scale_ * 0.0
    # A one-standard-deviation shift in one selected standardized input changes
    # Lasso output by exactly its fitted coefficient.
    lasso_selected_coefficients = lasso.coef_.copy()
    controlled = windows[0].unsqueeze(0).clone()
    flat = _flatten_windows(controlled)[:, selected]
    scaled = scaler.transform(flat)
    perturbed_scaled = scaled.copy()
    perturbed_scaled[0, 0] += 1.0
    coefficient_check = {
        "predicted_change": float(lasso.predict(perturbed_scaled)[0] - lasso.predict(scaled)[0]),
        "coefficient": float(lasso_selected_coefficients[0]),
        "absolute_error": float(
            abs(
                (lasso.predict(perturbed_scaled)[0] - lasso.predict(scaled)[0])
                - lasso_selected_coefficients[0]
            )
        ),
    }

    return {
        "dataset": dataset,
        "selected_features": str(selected_path),
        "selected_sleep_features": [
            name for name in selected_names if "sleep" in name
        ],
        "test_windows_evaluated": len(windows),
        "models": {
            "lasso": "StandardScaler + Lasso(alpha=0.01)",
            "gradient_boosting": "GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.05)",
        },
        "lasso_closed_form_check": coefficient_check,
        "sleep_scenario_summary": summary,
        "sleep_duration_heterogeneity": heterogeneity,
        "sleep_duration_per_window": per_window_rows,
        "directional_plus_one_sd_summary": directional_summary,
        "interpretation": (
            "Scenario outputs are model responses, not causal effects. "
            "Agreement is summarized by the sign of each model's change."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ces"])
    parser.add_argument("--selected-features", type=Path, default=Path("selected_features.json"))
    parser.add_argument("--max-windows", type=int, default=212)
    parser.add_argument("--output", type=Path, default=Path("data/processed/ces_frozen_sensitivity.json"))
    args = parser.parse_args()
    result = evaluate(args.dataset, args.selected_features, args.max_windows)
    print(json.dumps(result, indent=2, allow_nan=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
