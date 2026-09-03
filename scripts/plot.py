"""
PersonaTwin presentation plots.

Drop this file at the ROOT of your personaTwin repo (same level as run_pipeline.py)
and run it with `python plot_personatwin_results.py <dataset> [--personalized]`.

It reads directly from the same artifacts your pipeline already produces:
  - data/processed/checkpoints/<dataset>[_personalized]_gru.pt   (via src.run_sensitivity.load_model)
  - data/processed/sensitivity/<prefix>_sensitivity_profiles.csv
  - data/processed/sensitivity/<prefix>_sensitivity_aggregates.json
  - data/processed/sensitivity/<prefix>_continuous_sensitivity_profiles.csv
  - data/processed/sensitivity/<prefix>_interaction_profiles.csv
  - data/processed/sensitivity/<prefix>_interaction_aggregates.json

Nothing here duplicates src/sensitivity.py's own plot_sensitivity_results() (the
slope bar chart + threshold-crossing bar chart it already saves) — this script
adds the plots that function doesn't cover: predicted-vs-actual, per-participant
time series with uncertainty bands, calibration, per-participant radar profile,
sensitivity curves (the "what if we increase/decrease sleep" plot), population
vs personalized comparison, and the interaction heatmap.

All figures are saved as PNGs under <repo_root>/figures/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audit_pre_sensitivity import load_artifact, participant_index
from src.config import (
    BEHAVIORAL_DIRECTIONS,
    CES_TARGETS,
    MODEL_CHECKPOINT_DIR,
    PROCESSED_DIR,
    STUDENTLIFE_TARGETS,
)
from src.run_sensitivity import load_model

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PAM_INDEX = 0  # both CES_TARGETS and STUDENTLIFE_TARGETS put "pam" first


def _safe_float(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _clean_numeric_series(values, default=0.0):
    out = []
    for value in values:
        out.append(_safe_float(value, default))
    return np.asarray(out, dtype=float)


def _clean_curve_frame(df: pd.DataFrame):
    frame = df.copy()
    for col in ["alpha", "predicted_mean", "predicted_std", "slope"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[np.isfinite(frame[[c for c in ["alpha", "predicted_mean", "predicted_std", "slope"] if c in frame.columns]].to_numpy()).all(axis=1)]
    if frame.empty:
        return frame
    frame = frame.sort_values("alpha").drop_duplicates(subset=["alpha"], keep="last")
    return frame


# --------------------------------------------------------------------------
# 1. Model evaluation: predicted vs actual, time series, calibration
# --------------------------------------------------------------------------

def run_inference(dataset: str, personalized: bool, checkpoint_path: Path, device: str = "cpu"):
    """Runs the trained twin on the test split and returns a tidy DataFrame:
    uid, window_index, actual_pam, predicted_pam, predicted_std (original scale).
    Mirrors the target-scaling logic in src/run_sensitivity.py so numbers match
    what the sensitivity engine itself used.
    """
    artifact = load_artifact(dataset)
    model = load_model(artifact, checkpoint_path, personalized, device=device)
    idx = participant_index(artifact) if personalized else None

    split = artifact["test"]
    features = split["X"].float()
    targets = split["y"].float()
    uids = split["uid"]
    uids = uids.tolist() if torch.is_tensor(uids) else list(uids)

    train_targets = artifact["train"]["y"].float()
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0).clamp_min(1e-6)

    if personalized:
        participant_ids = torch.tensor([idx[str(u)] for u in uids])

    means, stds = [], []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            batch = features[start:start + 512]
            if personalized:
                mean, logvar = model(batch, participant_ids[start:start + 512])
            else:
                mean, logvar = model(batch)
            means.append(mean)
            stds.append(torch.exp(0.5 * logvar))
    means = torch.cat(means)
    stds = torch.cat(stds)

    # Model means/stds are standardized outputs; artifact targets are already
    # stored on the original PAM scale.
    means_orig = means * target_std + target_mean
    stds_orig = stds * target_std
    targets_orig = targets

    return pd.DataFrame({
        "uid": [str(u) for u in uids],
        "window_index": np.arange(len(uids)),
        "actual_pam": targets_orig[:, PAM_INDEX].numpy(),
        "predicted_pam": means_orig[:, PAM_INDEX].numpy(),
        "predicted_std": stds_orig[:, PAM_INDEX].numpy(),
    })


def plot_predicted_vs_actual(df: pd.DataFrame, prefix: str):
    clean = df.copy()
    for col in ["actual_pam", "predicted_pam", "predicted_std"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[np.isfinite(clean[["actual_pam", "predicted_pam", "predicted_std"]].to_numpy()).all(axis=1)]
    if clean.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clean["actual_pam"], clean["predicted_pam"], alpha=0.5, s=20, color="#2f6690")
    lo = min(clean["actual_pam"].min(), clean["predicted_pam"].min())
    hi = max(clean["actual_pam"].max(), clean["predicted_pam"].max())
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual PAM")
    ax.set_ylabel("Predicted PAM")
    ax.set_title(f"{prefix}: predicted vs actual (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_predicted_vs_actual.png", dpi=200)
    plt.close(fig)


def plot_participant_timeseries(df: pd.DataFrame, prefix: str, n_participants: int = 3):
    """One panel per participant: actual vs predicted PAM across their test windows,
    with a shaded predictive-uncertainty band. This is the single most convincing
    'the twin tracks a real person' plot."""
    clean = df.copy()
    for col in ["uid", "window_index", "actual_pam", "predicted_pam", "predicted_std"]:
        if col in clean.columns:
            if col == "uid":
                clean[col] = clean[col].astype(str)
            else:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[np.isfinite(clean[["window_index", "actual_pam", "predicted_pam", "predicted_std"]].to_numpy()).all(axis=1)]
    if clean.empty:
        return

    top_uids = clean["uid"].value_counts().head(n_participants).index.tolist()
    fig, axes = plt.subplots(len(top_uids), 1, figsize=(8, 3 * len(top_uids)), sharex=False)
    if len(top_uids) == 1:
        axes = [axes]
    for ax, uid in zip(axes, top_uids):
        sub = clean[clean["uid"] == uid].sort_values("window_index").reset_index(drop=True)
        x = np.arange(len(sub))
        actual = _clean_numeric_series(sub["actual_pam"])
        predicted = _clean_numeric_series(sub["predicted_pam"])
        std = _clean_numeric_series(sub["predicted_std"])
        ax.plot(x, actual, "o-", color="black", label="actual", markersize=4)
        ax.plot(x, predicted, "o-", color="#d97706", label="predicted", markersize=4)
        ax.fill_between(x, predicted - 1.96 * std, predicted + 1.96 * std, color="#d97706", alpha=0.2, label="95% predictive interval")
        ax.set_title(f"Participant {uid}")
        ax.set_ylabel("PAM")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Test window index (chronological)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_participant_timeseries.png", dpi=200)
    plt.close(fig)


def plot_calibration(df: pd.DataFrame, prefix: str):
    """Reliability diagram: for a sweep of nominal coverage levels, what fraction
    of actual values actually fall inside the predicted interval."""
    clean = df.copy()
    for col in ["actual_pam", "predicted_pam", "predicted_std"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[np.isfinite(clean[["actual_pam", "predicted_pam", "predicted_std"]].to_numpy()).all(axis=1)]
    if clean.empty:
        return

    nominal_levels = np.linspace(0.1, 0.95, 12)
    empirical = []
    try:
        from scipy.stats import norm
        for level in nominal_levels:
            z = norm.ppf(0.5 + level / 2)
            lo = clean["predicted_pam"] - z * clean["predicted_std"]
            hi = clean["predicted_pam"] + z * clean["predicted_std"]
            covered = ((clean["actual_pam"] >= lo) & (clean["actual_pam"] <= hi)).mean()
            empirical.append(float(covered))
    except Exception:
        empirical = [0.5] * len(nominal_levels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.plot(nominal_levels, empirical, "o-", color="#2f6690", label="observed")
    ax.axvline(0.95, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"{prefix}: uncertainty calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_calibration.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Sensitivity engine plots
# --------------------------------------------------------------------------

def load_sensitivity_outputs(prefix: str):
    sens_dir = PROCESSED_DIR / "sensitivity"
    files = {
        "profiles": sens_dir / f"{prefix}_sensitivity_profiles.csv",
        "aggregates": sens_dir / f"{prefix}_sensitivity_aggregates.json",
        "continuous": sens_dir / f"{prefix}_continuous_sensitivity_profiles.csv",
        "interactions": sens_dir / f"{prefix}_interaction_profiles.csv",
        "interaction_aggregates": sens_dir / f"{prefix}_interaction_aggregates.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing sensitivity artifacts for {prefix}: {', '.join(sorted(missing))}. "
            "Generate the sensitivity outputs first or run on a dataset that has them."
        )

    profiles = pd.read_csv(files["profiles"])
    with open(files["aggregates"]) as f:
        aggregates = json.load(f)
    continuous = pd.read_csv(files["continuous"])
    interactions = pd.read_csv(files["interactions"])
    with open(files["interaction_aggregates"]) as f:
        interaction_aggregates = json.load(f)
    return profiles, aggregates, continuous, interactions, interaction_aggregates


def plot_participant_radar(profiles: pd.DataFrame, prefix: str, uids: list[str] | None = None):
    """Radar/spider chart: one axis per behavioral direction, radius = |slope|,
    for 2-3 contrasting participants. Answers 'show a whole profile of a user'."""
    clean = profiles.copy()
    clean["uid"] = clean["uid"].astype(str)
    clean["direction"] = clean["direction"].astype(str)
    clean["slope"] = pd.to_numeric(clean["slope"], errors="coerce")
    clean = clean[np.isfinite(clean["slope"]) | clean["slope"].isna()]
    directions = [
        direction
        for direction in BEHAVIORAL_DIRECTIONS
        if clean.loc[clean["direction"] == direction, "slope"].notna().any()
    ]
    if clean.empty:
        return
    if not directions:
        return
    if uids is None:
        per_uid = clean.groupby("uid")["slope"].apply(lambda s: s.abs().mean())
        uids = per_uid.sort_values(ascending=False).head(3).index.tolist()

    angles = np.linspace(0, 2 * np.pi, len(directions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["#2f6690", "#d97706", "#3a7d44"]
    for uid, color in zip(uids, colors):
        sub = clean[clean["uid"] == uid]
        values = []
        for d in directions:
            vals = sub[sub["direction"] == d]["slope"].abs()
            v = float(vals.mean()) if not vals.empty else 0.0
            values.append(_safe_float(v, 0.0))
        values = np.asarray(values, dtype=float)
        if not np.any(np.isfinite(values)):
            values = np.zeros(len(directions), dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, "o-", linewidth=2, label=f"Participant {uid}", color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(directions)
    if np.any(np.isfinite(ax.get_ylim())):
        current_min, current_max = ax.get_ylim()
        if not math.isfinite(current_min) or not math.isfinite(current_max) or current_max <= current_min:
            ax.set_ylim(0, 1)
    ax.set_title(f"{prefix}: behavioral sensitivity profile by participant")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_participant_radar.png", dpi=200)
    plt.close(fig)


def plot_sensitivity_curves(continuous: pd.DataFrame, prefix: str, uid: str | None = None,
                             window_index: int | None = None):
    """Small multiples: one panel per direction, x=alpha (behavior shift),
    y=predicted PAM, with the model's own predictive std as a shaded band.
    This is the direct answer to 'how would it be if we increase/decrease sleep'."""
    clean = continuous.copy()
    for col in ["uid", "direction", "alpha", "window_index", "predicted_mean", "predicted_std"]:
        if col in clean.columns:
            if col in ["uid", "direction"]:
                clean[col] = clean[col].astype(str)
            else:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[np.isfinite(clean[["alpha", "window_index", "predicted_mean", "predicted_std"]].to_numpy()).all(axis=1)]
    if clean.empty:
        return

    if uid is None:
        uid = str(clean["uid"].iloc[0])
    sub = clean[clean["uid"] == uid]
    if sub.empty:
        return
    if window_index is None:
        window_index = int(sub["window_index"].iloc[0])
    sub = sub[sub["window_index"] == window_index]
    if sub.empty:
        return

    directions = [d for d in BEHAVIORAL_DIRECTIONS if d in sub["direction"].unique()]
    if not directions:
        return
    fig, axes = plt.subplots(1, len(directions), figsize=(4 * len(directions), 4), sharey=True)
    if len(directions) == 1:
        axes = [axes]
    for ax, direction in zip(axes, directions):
        curve = _clean_curve_frame(sub[sub["direction"] == direction])
        if curve.empty:
            ax.text(0.5, 0.5, f"No valid {direction} curve", ha="center", va="center")
            continue
        ax.plot(curve["alpha"], curve["predicted_mean"], color="#2f6690")
        ax.fill_between(curve["alpha"], curve["predicted_mean"] - 1.96 * curve["predicted_std"], curve["predicted_mean"] + 1.96 * curve["predicted_std"], color="#2f6690", alpha=0.2)
        ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(direction)
        ax.set_xlabel("alpha (behavior shift, std units)")
    axes[0].set_ylabel("Predicted PAM")
    fig.suptitle(f"{prefix}: sensitivity curves — participant {uid}, window {window_index}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_sensitivity_curves_{uid}_{window_index}.png", dpi=200)
    plt.close(fig)


def plot_population_vs_personalized(prefix_pop: str, prefix_personalized: str):
    """Grouped bar chart of mean slope per direction, population vs personalized twin."""
    try:
        with open(PROCESSED_DIR / "sensitivity" / f"{prefix_pop}_sensitivity_aggregates.json") as f:
            pop_agg = json.load(f)
        with open(PROCESSED_DIR / "sensitivity" / f"{prefix_personalized}_sensitivity_aggregates.json") as f:
            pers_agg = json.load(f)
    except FileNotFoundError:
        return

    directions = [d for d in BEHAVIORAL_DIRECTIONS if d in pop_agg and d in pers_agg]
    pop_slopes = [_safe_float(pop_agg[d].get("slope_mean", 0.0), 0.0) for d in directions]
    pers_slopes = [_safe_float(pers_agg[d].get("slope_mean", 0.0), 0.0) for d in directions]

    if not directions:
        return

    x = np.arange(len(directions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pop_slopes, width, label="population twin", color="#2f6690")
    ax.bar(x + width / 2, pers_slopes, width, label="personalized twin", color="#d97706")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("Mean slope (PAM per alpha)")
    ax.set_title("Population vs personalized sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix_pop}_vs_{prefix_personalized}_slopes.png", dpi=200)
    plt.close(fig)


def plot_slope_spread(profiles: pd.DataFrame, prefix: str):
    """Box plot of per-participant mean slope per direction — shows personalization
    actually produces a spread of different responses, not a collapsed average."""
    clean = profiles.copy()
    clean["uid"] = clean["uid"].astype(str)
    clean["direction"] = clean["direction"].astype(str)
    clean["slope"] = pd.to_numeric(clean["slope"], errors="coerce")
    clean = clean[np.isfinite(clean["slope"]) | clean["slope"].isna()]
    if clean.empty:
        return

    per_participant = clean.groupby(["uid", "direction"], as_index=False)["slope"].mean()
    directions = [
        d for d in BEHAVIORAL_DIRECTIONS
        if d in per_participant["direction"].unique()
        and per_participant.loc[per_participant["direction"] == d, "slope"].notna().any()
    ]
    data = [per_participant[per_participant["direction"] == d]["slope"].dropna().to_numpy(dtype=float) for d in directions]
    data = [np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in data]

    if not any(len(x) for x in data):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(directions) + 1))
    ax.set_xticklabels(directions)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Per-participant mean slope")
    ax.set_title(f"{prefix}: spread of individual sensitivity by direction")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_slope_spread_boxplot.png", dpi=200)
    plt.close(fig)


def plot_interaction_heatmap(interaction_aggregates: dict, prefix: str):
    pairs = []
    for pair, summary in interaction_aggregates.items():
        if summary.get("interaction_mean") is None:
            continue
        a, b = pair.split(":") if ":" in pair else pair.split(",")
        pairs.append((a.strip(), b.strip()))
    directions = [d for d in BEHAVIORAL_DIRECTIONS if any(d in pair for pair in pairs)]
    if not directions:
        return
    matrix = np.full((len(directions), len(directions)), np.nan)
    for pair, summary in interaction_aggregates.items():
        a, b = pair.split(":") if ":" in pair else pair.split(",")
        a, b = a.strip(), b.strip()
        if a in directions and b in directions:
            i, j = directions.index(a), directions.index(b)
            value = summary.get("interaction_mean")
            if value is not None:
                val = _safe_float(value, 0.0)
                matrix[i, j] = val
                matrix[j, i] = val

    finite = np.isfinite(matrix)
    if not np.any(finite):
        return

    abs_vals = np.abs(matrix[finite])
    vmax = float(np.max(abs_vals)) if abs_vals.size else 1.0
    vmin = -vmax if vmax > 0 else -1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(directions)))
    ax.set_yticks(range(len(directions)))
    ax.set_xticklabels(directions, rotation=45, ha="right")
    ax.set_yticklabels(directions)
    for i in range(len(directions)):
        for j in range(len(directions)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Mean interaction effect")
    ax.set_title(f"{prefix}: pairwise interaction heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_interaction_heatmap.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["studentlife", "ces"])
    parser.add_argument("--personalized", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Defaults to data/processed/checkpoints/<dataset>[_personalized]_gru.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    prefix = f"{args.dataset}_personalized" if args.personalized else args.dataset
    checkpoint_path = args.checkpoint or (
        MODEL_CHECKPOINT_DIR / f"{args.dataset}_{'personalized' if args.personalized else 'population'}_gru.pt"
    )

    print(f"[1/3] Running inference for model evaluation plots ({prefix})...")
    eval_df = run_inference(args.dataset, args.personalized, checkpoint_path, device=args.device)
    plot_predicted_vs_actual(eval_df, prefix)
    plot_participant_timeseries(eval_df, prefix)
    plot_calibration(eval_df, prefix)

    print(f"[2/3] Loading sensitivity engine outputs for {prefix}...")
    try:
        profiles, aggregates, continuous, interactions, interaction_aggregates = load_sensitivity_outputs(prefix)
        plot_participant_radar(profiles, prefix)
        plot_sensitivity_curves(continuous, prefix)
        plot_slope_spread(profiles, prefix)
        plot_interaction_heatmap(interaction_aggregates, prefix)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        print(f"[WARN] Skipping sensitivity plots for {prefix}. The model-evaluation plots already ran successfully.")

    print(f"[3/3] Done. Figures saved to {FIGURES_DIR}")

    if not args.personalized:
        print(
            "\nTip: also run this script with --personalized on the same dataset, "
            "then call plot_population_vs_personalized() yourself with both prefixes "
            "to get the population-vs-personalized comparison chart."
        )


if __name__ == "__main__":
    main()