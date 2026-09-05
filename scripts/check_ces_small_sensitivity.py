"""Run a small in-memory CES sensitivity check without overwriting full outputs."""

from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_pre_sensitivity import load_artifact, participant_index
from src.config import CES_DIRECTION_MAP_CACHE
from src.run_sensitivity import load_model
from src.sensitivity import aggregate_profiles, profile_split

CHECKPOINT = ROOT / "data" / "processed" / "checkpoints" / "ces_uncertainty_personalized_gru.pt"


def main():
    artifact = load_artifact("ces")
    feature_names = artifact["metadata"]["feature_names"]
    direction_map = json.loads(Path(CES_DIRECTION_MAP_CACHE).read_text())
    model = load_model(artifact, CHECKPOINT, personalized=True, device="cpu")
    participant_ids = participant_index(artifact)
    targets = artifact["train"]["y"].float()
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0).clamp_min(1e-6)

    rows, continuous = profile_split(
        model=model,
        artifact=artifact,
        feature_names=feature_names,
        direction_map=direction_map,
        max_windows=50,
        device="cpu",
        target_mean=target_mean,
        target_std=target_std,
        batch_size=16,
        personalized=True,
        participant_index=participant_ids,
    )
    summaries = aggregate_profiles(rows)
    print("windows=50")
    for direction, summary in sorted(summaries.items()):
        print(
            f"{direction}: count={summary['count']:.0f} "
            f"slope={summary.get('slope_mean')} "
            f"curvature={summary.get('curvature_mean')} "
            f"margin_count={summary.get('margin_count', 0):.0f}"
        )
    print(f"continuous_rows={len(continuous)}")
    print("full_output_files_untouched=True")


if __name__ == "__main__":
    main()
