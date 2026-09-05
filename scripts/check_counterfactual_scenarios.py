from pathlib import Path
import json
import sys

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import sequence_cache_path
from src.counterfactuals import perturb_sleep_schedule, raw_value_for_window
DIRECTION_MAP = ROOT / "data" / "interim" / "ces_behavioral_direction_map.json"
SELECTED_FEATURES = ROOT / "selected_features_scenario.json"


def _fit_model(artifact, model_name):
    selected = json.loads(SELECTED_FEATURES.read_text())
    indices = np.asarray(
        [item["flattened_index"] for item in selected["selected_features"]],
        dtype=np.int64,
    )
    train_x = artifact["train"]["X"].numpy().reshape(len(artifact["train"]["X"]), -1)[:, indices]
    train_y = artifact["train"]["y"].numpy().reshape(-1)
    scaler = StandardScaler() if model_name == "lasso" else None
    fit_x = scaler.fit_transform(train_x) if scaler is not None else train_x
    if model_name == "lasso":
        model = Lasso(alpha=0.01, max_iter=5000).fit(fit_x, train_y)
    else:
        model = GradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=2,
            n_estimators=200,
            random_state=42,
        ).fit(fit_x, train_y)
    return model, scaler, indices


def predict(model, scaler, indices, windows):
    features = windows.numpy().reshape(len(windows), -1)[:, indices]
    if scaler is not None:
        features = scaler.transform(features)
    return model.predict(features)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lasso", "gradient_boosting"], default="lasso")
    args = parser.parse_args()
    artifact = torch.load(
        sequence_cache_path("ces", predict_delta=True),
        map_location="cpu",
        weights_only=False,
    )
    model, scaler, indices = _fit_model(artifact, args.model)
    direction_map = json.loads(DIRECTION_MAP.read_text())
    feature_names = artifact["metadata"]["feature_names"]

    window = artifact["test"]["X"][0].float()
    uid = str(artifact["test"]["uid"][0])

    scenarios = {
        "current": (0.0, 0.0),
        "sleep_duration_plus_2h": (2.0, 0.0),
        "bedtime_one_hour_earlier": (0.0, -1.0),
        "duration_plus_2h_and_bedtime_earlier": (2.0, -1.0),
    }
    windows = [
        window if shifts == (0.0, 0.0) else perturb_sleep_schedule(
            window,
            artifact,
            duration_shift_hours=shifts[0],
            bedtime_shift_hours=shifts[1],
        )
        for shifts in scenarios.values()
    ]
    predictions = predict(model, scaler, indices, torch.stack(windows))
    baseline = float(predictions[0])
    base_values = raw_value_for_window(window, artifact, feature_names, direction_map, "sleep")

    print(f"model={args.model}")
    print(f"uid={uid}")
    print(f"current_sleep_duration={base_values['sleep_duration']:.3f}")
    print(f"current_sleep_start={base_values['sleep_start']:.3f}")
    print(f"current_sleep_end={base_values['sleep_end']:.3f}")
    print("--- counterfactual predictions ---")
    for index, (label, shifts) in enumerate(scenarios.items()):
        values = raw_value_for_window(windows[index], artifact, feature_names, direction_map, "sleep")
        print(
            f"scenario={label} duration_shift={shifts[0]:+.1f}h "
            f"bedtime_shift={shifts[1]:+.1f}h "
            f"duration={values['sleep_duration']:.3f} "
            f"start={values['sleep_start']:.3f} "
            f"end={values['sleep_end']:.3f} "
            f"predicted_delta={float(predictions[index]):.4f} "
            f"delta_from_current={float(predictions[index]) - baseline:+.4f} "
        )


if __name__ == "__main__":
    main()
