from pathlib import Path
import math
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.counterfactuals import (
    alpha_for_real_shift,
    perturb_window_for_real_shift,
    perturb_sleep_schedule,
    raw_value_for_window,
)
from src.sensitivity import plausible_alpha_bounds

SENS_DIR = ROOT / "data" / "processed" / "sensitivity"
DIRECTIONS = ["sleep", "activity", "mobility", "screen"]
SEQUENCE_PATH = ROOT / "data" / "processed" / "ces_sequences.pt"
CES_DIRECTION_MAP_PATH = ROOT / "data" / "interim" / "ces_behavioral_direction_map.json"


def _finite_float(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def load_available_csvs():
    files = []
    for name in [
        "studentlife_personalized_continuous_sensitivity_profiles.csv",
        "ces_personalized_continuous_sensitivity_profiles.csv",
    ]:
        p = SENS_DIR / name
        if p.exists():
            files.append(p)
    return files


def build_profile(df, person_id, direction):
    sub = df[(df["uid"].astype(str) == person_id) & (df["direction"] == direction)].copy()
    if sub.empty:
        return None

    sub = sub[["uid", "direction", "alpha", "predicted_mean", "predicted_std"]].copy()
    sub["alpha"] = pd.to_numeric(sub["alpha"], errors="coerce")
    sub["predicted_mean"] = pd.to_numeric(sub["predicted_mean"], errors="coerce")
    sub["predicted_std"] = pd.to_numeric(sub["predicted_std"], errors="coerce")
    sub = sub[np.isfinite(sub[["alpha", "predicted_mean", "predicted_std"]].to_numpy()).all(axis=1)]
    if sub.empty:
        return None

    sub = sub.sort_values("alpha").copy()
    sub = sub.groupby("alpha", as_index=False).mean(numeric_only=True)
    if sub.empty:
        return None

    w = max(3, min(11, len(sub) // 4))
    sub["smooth"] = sub["predicted_mean"].rolling(window=w, center=True, min_periods=1).mean()

    xs = sub["alpha"].to_numpy(dtype=float)
    ys = sub["smooth"].to_numpy(dtype=float)
    if xs.size > 1 and np.unique(xs).size > 1:
        slopes = np.gradient(ys, xs)
    else:
        slopes = np.zeros_like(xs, dtype=float)
    sub["slope"] = slopes

    return sub


def radar_plot(scores, title):
    labels = list(scores.keys())
    vals = np.array([_finite_float(v, 0.0) for v in scores.values()], dtype=float)
    vals = np.abs(vals)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    if len(labels) == 0:
        raise ValueError("No labels for radar plot")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals2 = np.concatenate([vals, [vals[0]]])
    angles2 = angles + [angles[0]]

    max_val = float(np.max(vals)) if vals.size else 0.0
    if max_val <= 0:
        max_val = 1.0

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max_val * 1.4)
    ax.plot(angles2, vals2, color="#1f77b4", linewidth=2)
    ax.fill(angles2, vals2, color="#1f77b4", alpha=0.25)
    ax.set_title(title, pad=20)
    return fig


def main():
    csvs = load_available_csvs()
    if not csvs:
        raise FileNotFoundError("No sensitivity CSV found in data/processed/sensitivity")

    all_profiles = {}
    for csv_path in csvs:
        dataset = csv_path.name.replace("_continuous_sensitivity_profiles.csv", "")
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df["uid"] = df["uid"].astype(str)
        df["direction"] = df["direction"].astype(str)
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
        df["predicted_mean"] = pd.to_numeric(df["predicted_mean"], errors="coerce")
        df["predicted_std"] = pd.to_numeric(df["predicted_std"], errors="coerce")
        df = df.dropna(subset=["uid", "direction", "alpha", "predicted_mean", "predicted_std"])
        df = df[np.isfinite(df[["alpha", "predicted_mean", "predicted_std"]].to_numpy()).all(axis=1)]

        if df.empty:
            continue

        person = df["uid"].unique()[0]
        profiles = {}
        for d in DIRECTIONS:
            profiles[d] = build_profile(df, person, d)

        all_profiles[dataset] = {"person": person, "profiles": profiles}

    if not all_profiles:
        raise ValueError("No valid sensitivity data loaded")

    dataset_name = "ces_personalized" if "ces_personalized" in all_profiles else next(iter(all_profiles))
    meta = all_profiles[dataset_name]
    person = meta["person"]
    profiles = meta["profiles"]

    if dataset_name == "ces_personalized":
        artifact = torch.load(SEQUENCE_PATH, map_location="cpu", weights_only=False)
        direction_map = json.loads(CES_DIRECTION_MAP_PATH.read_text())
        sleep_feature = direction_map["sleep"][0]
        actual_window = artifact["test"]["X"][0]
        counterfactual_window = perturb_window_for_real_shift(
            actual_window,
            artifact,
            sleep_feature,
            real_shift=2.0,
        )
        current_value = raw_value_for_window(
            actual_window,
            artifact,
            artifact["metadata"]["feature_names"],
            direction_map,
            "sleep",
        )
        counterfactual_value = raw_value_for_window(
            counterfactual_window,
            artifact,
            artifact["metadata"]["feature_names"],
            direction_map,
            "sleep",
        )
        lower_alpha, upper_alpha = plausible_alpha_bounds(
            artifact,
            artifact["metadata"]["feature_names"],
            direction_map,
            "sleep",
        )
        requested_alpha = alpha_for_real_shift(2.0, sleep_feature, artifact)
        demo_alpha = float(np.clip(requested_alpha, lower_alpha, upper_alpha))
        print(f"sleep_feature={sleep_feature}")
        print("requested_sleep_shift_hours=2.0")
        print(f"requested_alpha={requested_alpha:.6f}")
        print(f"bounded_alpha={demo_alpha:.6f}")
        print(f"current_{sleep_feature}={current_value[sleep_feature]:.6f}")
        print(f"counterfactual_{sleep_feature}={counterfactual_value[sleep_feature]:.6f}")
        print(f"unchanged_sleep_start={current_value['sleep_start'] == counterfactual_value['sleep_start']}")
        print(f"unchanged_sleep_end={current_value['sleep_end'] == counterfactual_value['sleep_end']}")

        scenarios = {
            "duration_only": (2.0, 0.0),
            "bedtime_only": (0.0, -1.0),
            "duration_and_bedtime": (2.0, -1.0),
        }
        for label, (duration_shift, bedtime_shift) in scenarios.items():
            scenario_window = perturb_sleep_schedule(
                actual_window,
                artifact,
                duration_shift_hours=duration_shift,
                bedtime_shift_hours=bedtime_shift,
            )
            scenario_values = raw_value_for_window(
                scenario_window,
                artifact,
                artifact["metadata"]["feature_names"],
                direction_map,
                "sleep",
            )
            print(
                f"scenario={label} duration={scenario_values['sleep_duration']:.3f} "
                f"start={scenario_values['sleep_start']:.3f} "
                f"end={scenario_values['sleep_end']:.3f}"
            )

    radar_scores = {}
    for d in DIRECTIONS:
        p = profiles.get(d)
        if p is None or p.empty:
            radar_scores[d] = 0.0
            continue
        idx = int(np.argmin(np.abs(p["alpha"].to_numpy())))
        radar_scores[d] = _finite_float(p["slope"].iloc[idx], 0.0)

    radar_fig = radar_plot(radar_scores, f"{dataset_name.upper()} sensitivity profile")
    radar_fig.savefig(SENS_DIR / f"{dataset_name}_radar_profile.png", dpi=200)

    fig, axes = plt.subplots(len(DIRECTIONS), 2, figsize=(14, 16), sharex="col")
    fig.suptitle(f"{dataset_name.upper()} personalized sensitivity profile for {person[:8]}", fontsize=18, y=0.99)

    for i, d in enumerate(DIRECTIONS):
        p = profiles.get(d)
        ax_top = axes[i, 0]
        ax_bottom = axes[i, 1]

        if p is None or p.empty:
            ax_top.text(0.5, 0.5, f"No {d} data", ha="center", va="center")
            ax_bottom.axis("off")
            continue

        xs = p["alpha"].to_numpy(dtype=float)
        ys = p["smooth"].to_numpy(dtype=float)
        mean_vals = p["predicted_mean"].to_numpy(dtype=float)
        std_vals = p["predicted_std"].to_numpy(dtype=float)
        slope_vals = p["slope"].to_numpy(dtype=float)

        finite_mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(mean_vals) & np.isfinite(std_vals) & np.isfinite(slope_vals)
        if not np.any(finite_mask):
            ax_top.text(0.5, 0.5, f"Invalid {d} data", ha="center", va="center")
            ax_bottom.axis("off")
            continue

        xs = xs[finite_mask]
        ys = ys[finite_mask]
        mean_vals = mean_vals[finite_mask]
        std_vals = std_vals[finite_mask]
        slope_vals = slope_vals[finite_mask]

        idx0 = int(np.argmin(np.abs(xs)))
        baseline = _finite_float(ys[idx0], 0.0)
        slope0 = _finite_float(slope_vals[idx0], 0.0)

        ax_top.plot(xs, ys, color="#1f77b4", linewidth=2.5)
        ax_top.fill_between(xs, mean_vals - 1.96 * std_vals, mean_vals + 1.96 * std_vals, color="#1f77b4", alpha=0.15)
        ax_top.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax_top.axhline(baseline, color="gray", linestyle=":", linewidth=1.1)
        ax_top.set_title(f"{d.title()} sensitivity")
        ax_top.set_ylabel("Predicted outcome")
        ax_top.grid(alpha=0.25)

        ax_bottom.plot(xs, slope_vals, color="#2ca02c", linewidth=2)
        ax_bottom.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax_bottom.axhline(0, color="gray", linestyle=":", linewidth=1.0)
        ax_bottom.set_ylabel("Local slope")
        ax_bottom.grid(alpha=0.25)
        ax_bottom.set_xlabel("Perturbation alpha")

        ax_top.text(0.02, 0.96, f"slope@0={slope0:.3f}", transform=ax_top.transAxes,
                    va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(SENS_DIR / f"{dataset_name}_small_multiple_profile.png", dpi=200)

    print("saved:", SENS_DIR / f"{dataset_name}_radar_profile.png")
    print("saved:", SENS_DIR / f"{dataset_name}_small_multiple_profile.png")


if __name__ == "__main__":
    main()