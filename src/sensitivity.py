"""Input-space sensitivity analysis for behavioral directions.

The Sensitivity Engine should perturb the actual feature window, not the hidden
state z. This keeps the perturbation in a unit-bearing, interpretable space and
makes the resulting slope/curvature/margin diagnostics defensible.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def direction_feature_indices(
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
) -> list[int]:
    """Return the feature indices associated with one behavioral direction."""
    if direction not in direction_map:
        raise KeyError(f"Unknown direction: {direction}")
    names = list(feature_names)
    return [
        index
        for index, feature_name in enumerate(names)
        if feature_name in direction_map[direction]
    ]


def default_direction_alphas(
    lower: float,
    upper: float,
    steps: int = 21,
) -> list[float]:
    """Create a regular alpha sweep over a plausible direction range."""
    if steps <= 1:
        return [float(lower)]
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        lower, upper = upper, lower
    return [float(value) for value in np.linspace(lower, upper, steps)]


def plausible_alpha_bounds(
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
    lower_cap: float | None = None,
    upper_cap: float | None = None,
) -> tuple[float, float]:
    """Bound alpha to the observed training-window range for a direction.

    The perturbation is applied in standardized feature space, so a direction's
    alpha range is derived from the training split's empirical spread for that
    direction's feature columns and then clipped to a sensible paper-facing
    range.
    """
    if direction not in direction_map:
        raise KeyError(f"Unknown direction: {direction}")
    indices = direction_feature_indices(feature_names, direction_map, direction)
    if not indices:
        return (float("nan"), float("nan"))

    train_x = artifact.get("train", {}).get("X")
    if train_x is None:
        raise KeyError("artifact['train']['X'] is required to estimate alpha bounds")
    train_x = train_x.float()
    direction_values = train_x[:, :, indices]
    feature_sd = direction_values.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    composite = direction_values.mean(dim=2)
    composite_mean = composite.mean()
    # Alpha shifts every selected feature by alpha * its own SD. Convert the
    # observed composite range into that same shared alpha unit.
    step_scale = feature_sd.mean().clamp_min(1e-6)
    lower = float(((composite.min() - composite_mean) / step_scale).item())
    upper = float(((composite.max() - composite_mean) / step_scale).item())
    if lower_cap is not None:
        lower = max(lower, float(lower_cap))
    if upper_cap is not None:
        upper = min(upper, float(upper_cap))
    if lower > upper:
        lower, upper = upper, lower
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    return lower, upper


def direction_map_feature_counts(
    direction_map: dict[str, list[str]],
    directions: Sequence[str] = ("sleep", "activity", "social", "mobility", "screen"),
) -> dict[str, int]:
    """Return feature counts used to audit cross-dataset direction coverage."""
    return {direction: len(direction_map.get(direction, [])) for direction in directions}


def compare_direction_maps(
    maps: dict[str, dict[str, list[str]]],
    directions: Sequence[str] = ("sleep", "activity", "social", "mobility", "screen"),
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Compare feature lists without assuming equivalent measurement semantics."""
    result: dict[str, dict[str, dict[str, list[str]]]] = {}
    dataset_names = list(maps)
    for direction in directions:
        result[direction] = {}
        for dataset_name in dataset_names:
            result[direction][dataset_name] = {
                "features": list(maps[dataset_name].get(direction, [])),
            }
    return result


def mobility_target_correlations(
    frame,
    direction_map: dict[str, list[str]],
    target: str = "pam",
) -> dict[str, float]:
    """Compute raw mobility-feature/target correlations for leakage auditing."""
    if target not in frame.columns:
        raise KeyError(f"Target column {target!r} is missing from the dataframe")
    correlations = {}
    for feature in direction_map.get("mobility", []):
        if feature not in frame.columns:
            continue
        correlations[feature] = float(frame[feature].corr(frame[target]))
    return correlations


def participant_counts(artifact: dict, split_name: str = "test") -> dict[str, int]:
    """Report windows and distinct participants for a cached sequence split."""
    split = artifact[split_name]
    uids = split.get("uid", [])
    values = uids.tolist() if torch.is_tensor(uids) else list(uids)
    return {
        "n_windows": len(split["X"]),
        "n_participants": len({str(value) for value in values}),
    }


def perturb_window_for_direction(
    window: torch.Tensor | np.ndarray,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
    alpha: float,
) -> torch.Tensor:
    """Add a direction-specific alpha perturbation to a real input window."""
    tensor = torch.as_tensor(window, dtype=torch.float32)
    if tensor.dim() not in (2, 3):
        raise ValueError("window must have shape (T, F) or (B, T, F)")

    indices = direction_feature_indices(feature_names, direction_map, direction)
    if not indices:
        return tensor.clone()

    train_x = artifact.get("train", {}).get("X")
    if train_x is None:
        raise KeyError("artifact['train']['X'] is required for directional perturbations")
    train_x = train_x.float()
    feature_sd = train_x[:, :, indices].std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

    output = tensor.clone()
    if output.dim() == 2:
        perturbation = alpha * feature_sd.view(1, -1)
        output[:, indices] = output[:, indices] + perturbation
    else:
        perturbation = alpha * feature_sd.view(1, 1, -1)
        output[:, :, indices] = output[:, :, indices] + perturbation
    return output


def _coerce_prediction(
    prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(prediction, tuple):
        mean, logvar = prediction
    else:
        mean, logvar = prediction, None

    mean = torch.as_tensor(mean).float()
    if mean.dim() == 0:
        mean = mean.unsqueeze(0)
    if target_mean is not None and target_std is not None:
        mean = mean * target_std + target_mean

    if logvar is not None:
        logvar = torch.as_tensor(logvar).float()
        if logvar.dim() == 0:
            logvar = logvar.unsqueeze(0)
        if target_mean is not None and target_std is not None:
            logvar = logvar.clone()
    return mean, logvar


def probe_direction(
    model: torch.nn.Module,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
    window: torch.Tensor | np.ndarray,
    alphas: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    calibrated_std: torch.Tensor | float | None = None,
    personalized: bool = False,
    participant_id: int | str | None = None,
) -> list[dict[str, float | None]]:
    """Probe a behavioral direction by perturbing the real input window.

    A direction is varied over an empirically plausible alpha range, the full
    trained model is re-run on each perturbed window, and the predicted mean/std
    are stored for later slope/curvature/margin analysis.
    """
    if direction not in direction_map:
        raise KeyError(f"Unknown direction: {direction}")
    indices = direction_feature_indices(feature_names, direction_map, direction)
    if not indices:
        raise ValueError(
            f"Direction {direction!r} has no features in the supplied feature schema"
        )

    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else "cpu"
    device = torch.device(device)

    if alphas is None:
        lower, upper = plausible_alpha_bounds(artifact, feature_names, direction_map, direction)
        alphas = default_direction_alphas(lower, upper, steps=21)

    window_tensor = torch.as_tensor(window, dtype=torch.float32).to(device)
    if window_tensor.dim() == 2:
        window_tensor = window_tensor.unsqueeze(0)

    results: list[dict[str, float | None]] = []
    model.eval()
    with torch.no_grad():
        alpha_values = [float(alpha) for alpha in alphas]
        perturbed_windows = torch.cat(
            [
                perturb_window_for_direction(
                    window_tensor.cpu(),
                    artifact,
                    feature_names,
                    direction_map,
                    direction,
                    alpha,
                )
                for alpha in alpha_values
            ],
            dim=0,
        ).to(device)
        if personalized:
            if participant_id is None:
                raise ValueError("participant_id is required for personalized probes")
            participant = torch.tensor(
                [participant_id] * len(alpha_values),
                device=device,
                dtype=torch.long,
            )
            prediction = model(perturbed_windows, participant)
        else:
            prediction = model(perturbed_windows)

        mean, logvar = _coerce_prediction(prediction, target_mean, target_std)
        means = mean.detach().cpu().reshape(len(alpha_values), -1)[:, 0]
        stds = None
        if logvar is not None:
            predictive_std = torch.exp(0.5 * logvar)
            if target_std is not None:
                predictive_std = predictive_std * target_std
            stds = predictive_std.detach().cpu().reshape(len(alpha_values), -1)[:, 0]
        elif calibrated_std is not None:
            calibrated = torch.as_tensor(calibrated_std, dtype=torch.float32)
            stds = calibrated.expand(len(alpha_values)).cpu()
        for index, alpha in enumerate(alpha_values):
            mean_value = float(means[index])
            std_value = None if stds is None else float(stds[index])
            results.append({
                "alpha": alpha,
                "predicted_mean": mean_value,
                "predicted_std": std_value,
            })
    return results


def summarize_direction_response(
    response: Sequence[dict[str, float | None]],
    threshold: float = 0.5,
    bootstrap_samples: int = 200,
    random_seed: int = 42,
) -> dict[str, float]:
    """Reduce alpha/response pairs into uncertainty-weighted summaries."""
    if not response:
        return {"slope": 0.0, "curvature": 0.0, "margin": float("inf")}

    alphas = np.asarray([float(item["alpha"]) for item in response], dtype=float)
    means = np.asarray([float(item["predicted_mean"]) for item in response], dtype=float)
    std_values = [item.get("predicted_std") for item in response]
    stds = None
    if all(value is not None and np.isfinite(float(value)) for value in std_values):
        stds = np.asarray([float(value) for value in std_values], dtype=float)

    _, slope, curvature = fit_weighted_curve(alphas, means, stds)
    intervals = bootstrap_curve_intervals(
        alphas,
        means,
        stds,
        n_boot=bootstrap_samples,
        random_seed=random_seed,
    )

    crossing = [
        float(abs(alpha))
        for alpha, mean in zip(alphas, means)
        if mean >= threshold
    ]
    margin = min(crossing) if crossing else float("inf")

    return {
        "slope": slope,
        "curvature": curvature,
        "margin": float(margin),
        **intervals,
    }


def fit_weighted_curve(
    alphas: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float] | None = None,
    degree: int = 2,
) -> tuple[np.poly1d, float, float]:
    """Fit a curve weighted by inverse predictive variance."""
    alpha_array = np.asarray(alphas, dtype=float)
    mean_array = np.asarray(means, dtype=float)
    fit_degree = min(degree, max(len(alpha_array) - 1, 1))
    weights = None
    if stds is not None:
        std_array = np.maximum(np.asarray(stds, dtype=float), 1e-6)
        weights = 1.0 / (std_array**2 + 1e-6)
    coefficients = np.polyfit(alpha_array, mean_array, deg=fit_degree, w=weights)
    polynomial = np.poly1d(coefficients)
    first_derivative = polynomial.deriv()
    second_derivative = first_derivative.deriv()
    return (
        polynomial,
        float(first_derivative(0.0)),
        float(second_derivative(0.0)) if fit_degree >= 2 else 0.0,
    )


def bootstrap_curve_intervals(
    alphas: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float] | None,
    n_boot: int = 200,
    random_seed: int = 42,
) -> dict[str, float]:
    """Estimate 95% slope/curvature intervals using predictive uncertainty."""
    if stds is None or len(alphas) < 2:
        return {
            "slope_ci_low": float("nan"),
            "slope_ci_high": float("nan"),
            "curvature_ci_low": float("nan"),
            "curvature_ci_high": float("nan"),
        }
    rng = np.random.default_rng(random_seed)
    alpha_array = np.asarray(alphas, dtype=float)
    mean_array = np.asarray(means, dtype=float)
    std_array = np.asarray(stds, dtype=float)
    slopes = []
    curvatures = []
    for _ in range(n_boot):
        sampled_means = mean_array + rng.normal(0.0, std_array)
        _, slope, curvature = fit_weighted_curve(alpha_array, sampled_means, std_array)
        slopes.append(slope)
        curvatures.append(curvature)
    return {
        "slope_ci_low": float(np.percentile(slopes, 2.5)),
        "slope_ci_high": float(np.percentile(slopes, 97.5)),
        "curvature_ci_low": float(np.percentile(curvatures, 2.5)),
        "curvature_ci_high": float(np.percentile(curvatures, 97.5)),
    }


def profile_all_directions(
    model: torch.nn.Module,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    window: torch.Tensor | np.ndarray,
    threshold: float,
    directions: Sequence[str] = ("sleep", "activity", "social", "mobility", "screen"),
    alphas: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    calibrated_std: torch.Tensor | float | None = None,
    personalized: bool = False,
    participant_id: int | str | None = None,
) -> dict[str, dict[str, float] | None]:
    """Run every available direction for one window.

    A direction with no columns in the artifact is reported as ``None`` rather
    than being treated as a no-op or causing the whole profile to fail.
    """
    profile = {}
    for direction in directions:
        if not direction_feature_indices(feature_names, direction_map, direction):
            profile[direction] = None
            continue
        response = probe_direction(
            model=model,
            artifact=artifact,
            feature_names=feature_names,
            direction_map=direction_map,
            direction=direction,
            window=window,
            alphas=alphas,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            calibrated_std=calibrated_std,
            personalized=personalized,
            participant_id=participant_id,
        )
        profile[direction] = summarize_direction_response(response, threshold)
    return profile


def profile_split(
    model: torch.nn.Module,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    split_name: str = "test",
    threshold: float = 8.0,
    directions: Sequence[str] = ("sleep", "activity", "social", "mobility", "screen"),
    max_windows: int | None = None,
    alphas: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    calibrated_std: torch.Tensor | float | None = None,
) -> list[dict[str, object]]:
    """Build one sensitivity row per direction and real window in a split."""
    if split_name not in artifact:
        raise KeyError(f"Unknown artifact split: {split_name}")
    split = artifact[split_name]
    windows = split["X"]
    limit = len(windows) if max_windows is None else min(max_windows, len(windows))
    uids = split.get("uid")
    rows: list[dict[str, object]] = []

    for window_index in range(limit):
        profile = profile_all_directions(
            model=model,
            artifact=artifact,
            feature_names=feature_names,
            direction_map=direction_map,
            window=windows[window_index],
            threshold=threshold,
            directions=directions,
            alphas=alphas,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            calibrated_std=calibrated_std,
        )
        uid = uids[window_index].item() if torch.is_tensor(uids) else uids[window_index]
        for direction, summary in profile.items():
            rows.append(
                {
                    "window_index": window_index,
                    "uid": str(uid),
                    "direction": direction,
                    "slope": None if summary is None else summary["slope"],
                    "curvature": None if summary is None else summary["curvature"],
                    "margin": None if summary is None else summary["margin"],
                    "slope_ci_low": None if summary is None else summary["slope_ci_low"],
                    "slope_ci_high": None if summary is None else summary["slope_ci_high"],
                    "curvature_ci_low": None if summary is None else summary["curvature_ci_low"],
                    "curvature_ci_high": None if summary is None else summary["curvature_ci_high"],
                }
            )
    return rows


def _cluster_bootstrap_interval(
    rows: Sequence[dict[str, object]],
    metric: str,
    n_boot: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a metric mean over participant-level clusters."""
    by_participant: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(float(value)):
            continue
        by_participant.setdefault(str(row["uid"]), []).append(float(value))
    participant_means = np.asarray(
        [np.mean(values) for values in by_participant.values()],
        dtype=float,
    )
    if not len(participant_means):
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(
        participant_means,
        size=(n_boot, len(participant_means)),
        replace=True,
    ).mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def aggregate_profiles(
    rows: Sequence[dict[str, object]],
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Aggregate profiles by direction with participant-clustered CIs."""
    directions = sorted({str(row["direction"]) for row in rows})
    aggregates: dict[str, dict[str, float]] = {}
    for direction in directions:
        direction_rows = [row for row in rows if row["direction"] == direction]
        summary: dict[str, float] = {
            "count": float(len(direction_rows)),
            "participant_count": float(
                len({str(row["uid"]) for row in direction_rows})
            ),
        }
        for metric in ("slope", "curvature", "margin"):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in direction_rows
                    if row[metric] is not None and np.isfinite(float(row[metric]))
                ],
                dtype=float,
            )
            summary[f"{metric}_count"] = float(len(values))
            if len(values):
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_std"] = float(values.std(ddof=0))
            else:
                summary[f"{metric}_mean"] = float("nan")
                summary[f"{metric}_median"] = float("nan")
                summary[f"{metric}_std"] = float("nan")
            low, high = _cluster_bootstrap_interval(
                direction_rows,
                metric,
                n_boot=bootstrap_samples,
                random_seed=random_seed,
            )
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        aggregates[direction] = summary
    return aggregates


def export_profiles(
    rows: Sequence[dict[str, object]],
    aggregates: dict[str, dict[str, float]],
    output_dir: str | Path,
    prefix: str = "ces",
) -> tuple[Path, Path]:
    """Save per-window profiles and aggregate summaries as CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows_path = output_path / f"{prefix}_sensitivity_profiles.csv"
    aggregates_path = output_path / f"{prefix}_sensitivity_aggregates.json"

    with rows_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "window_index",
                "uid",
                "direction",
                "slope",
                "curvature",
                "margin",
                "slope_ci_low",
                "slope_ci_high",
                "curvature_ci_low",
                "curvature_ci_high",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    serializable = {
        direction: {
            key: (None if not np.isfinite(value) else value)
            for key, value in summary.items()
        }
        for direction, summary in aggregates.items()
    }
    with aggregates_path.open("w") as file:
        json.dump(serializable, file, indent=2, allow_nan=False)

    return rows_path, aggregates_path


def plot_sensitivity_results(
    aggregates: dict[str, dict[str, float]],
    output_dir: str | Path,
    prefix: str = "ces",
) -> tuple[Path, Path]:
    """Save slope and threshold-crossing summary plots."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    available = [
        direction
        for direction, summary in aggregates.items()
        if summary.get("slope_count", 0.0) > 0
    ]
    if not available:
        raise ValueError("No finite sensitivity summaries are available to plot")

    slopes = [aggregates[direction]["slope_mean"] for direction in available]
    slope_sd = [aggregates[direction]["slope_std"] for direction in available]
    crossing_rates = [
        aggregates[direction].get("margin_count", 0.0)
        / max(aggregates[direction].get("count", 0.0), 1.0)
        for direction in available
    ]

    slope_path = output_path / f"{prefix}_sensitivity_slopes.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(available, slopes, yerr=slope_sd, capsize=4, color="#2f6690")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean slope (PAM units per alpha)")
    axis.set_title("CES sensitivity by behavioral direction")
    figure.tight_layout()
    figure.savefig(slope_path, dpi=200)
    plt.close(figure)

    crossing_path = output_path / f"{prefix}_threshold_crossing_rates.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(available, crossing_rates, color="#3a7d44")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction crossing PAM threshold")
    axis.set_title("Threshold crossing frequency by direction")
    figure.tight_layout()
    figure.savefig(crossing_path, dpi=200)
    plt.close(figure)

    return slope_path, crossing_path
