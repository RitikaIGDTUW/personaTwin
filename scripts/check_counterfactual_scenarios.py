from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_pre_sensitivity import load_artifact, participant_index
from src.counterfactuals import perturb_sleep_schedule, raw_value_for_window
from src.run_sensitivity import load_model


CHECKPOINT = ROOT / "data" / "processed" / "checkpoints" / "ces_uncertainty_personalized_gru.pt"
DIRECTION_MAP = ROOT / "data" / "interim" / "ces_behavioral_direction_map.json"


def predict(model, windows, participant_id, artifact):
    target_mean = artifact["train"]["y"].float().mean(dim=0)
    target_std = artifact["train"]["y"].float().std(dim=0).clamp_min(1e-6)
    participant = torch.tensor([participant_id] * len(windows), dtype=torch.long)
    with torch.no_grad():
        mean, logvar = model(windows, participant)
    mean = mean * target_std + target_mean
    std = torch.exp(0.5 * logvar) * target_std
    return mean[:, 0], std[:, 0]


def main():
    artifact = load_artifact("ces")
    model = load_model(artifact, CHECKPOINT, personalized=True, device="cpu")
    participant_map = participant_index(artifact)
    direction_map = json.loads(DIRECTION_MAP.read_text())
    feature_names = artifact["metadata"]["feature_names"]

    window = artifact["test"]["X"][0].float()
    uid = str(artifact["test"]["uid"][0])
    participant_id = participant_map[uid]

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
    predictions, stds = predict(model, torch.stack(windows), participant_id, artifact)
    baseline = float(predictions[0])
    base_values = raw_value_for_window(window, artifact, feature_names, direction_map, "sleep")

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
            f"predicted_pam={float(predictions[index]):.4f} "
            f"delta_from_current={float(predictions[index]) - baseline:+.4f} "
            f"predicted_std={float(stds[index]):.4f}"
        )


if __name__ == "__main__":
    main()
