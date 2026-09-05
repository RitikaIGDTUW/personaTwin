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
    """Return feature indices associated with one behavioral direction."""
    if direction not in direction_map:
        raise KeyError(f"Unknown direction: {direction}")
    return [index for index, name in enumerate(feature_names) if name in direction_map[direction]]


def default_direction_alphas(lower: float, upper: float, steps: int = 21) -> list[float]:
    """Create a regular alpha sweep over a plausible direction range."""
    if steps <= 1:
        return [float(lower)]
    lower, upper = sorted((float(lower), float(upper)))
    return [float(value) for value in np.linspace(lower, upper, steps)]


def plausible_alpha_bounds(
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
    lower_cap: float | None = None,
    upper_cap: float | None = None,
) -> tuple[float, float]:
    """Estimate an empirical standardized alpha range for a direction."""
    indices = direction_feature_indices(feature_names, direction_map, direction)
    if not indices:
        return (float("nan"), float("nan"))
    train_x = artifact.get("train", {}).get("X")
    if train_x is None:
        raise KeyError("artifact['train']['X'] is required to estimate alpha bounds")
    values = train_x.float()[:, :, indices]
    feature_sd = values.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    composite = values.mean(dim=2)
    scale = feature_sd.mean().clamp_min(1e-6)
    center = composite.mean()
    lower = float(((composite.min() - center) / scale).item())
    upper = float(((composite.max() - center) / scale).item())
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
    feature_sd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a direction-specific alpha perturbation to a real input window.

    A uniform shift across the window also shifts the rolling mean and
    deviation-from-own-mean derived features. Difference and standard
    deviation features remain unchanged by that uniform shift.
    """
    tensor = torch.as_tensor(window, dtype=torch.float32)
    if tensor.dim() not in (2, 3):
        raise ValueError("window must have shape (T, F) or (B, T, F)")

    indices = direction_feature_indices(feature_names, direction_map, direction)
    if not indices:
        return tensor.clone()

    if feature_sd is None:
        train_x = artifact.get("train", {}).get("X")
        if train_x is None:
            raise KeyError("artifact['train']['X'] is required for directional perturbations")
        train_x = train_x.float()
        feature_sd = train_x[:, :, indices].std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

    name_to_index = {name: index for index, name in enumerate(feature_names)}
    derived_indices: list[int] = []
    derived_sd: list[float] = []
    for local_index, base_index in enumerate(indices):
        base_name = feature_names[base_index]
        for suffix in ("_roll_mean7", "_dev_from_own_mean"):
            derived_index = name_to_index.get(f"{base_name}{suffix}")
            if derived_index is not None:
                derived_indices.append(derived_index)
                derived_sd.append(float(feature_sd[local_index]))

    output = tensor.clone()
    if output.dim() == 2:
        perturbation = alpha * feature_sd.view(1, -1)
        output[:, indices] = output[:, indices] + perturbation
        if derived_indices:
            derived_shift = alpha * torch.tensor(derived_sd, dtype=output.dtype)
            output[:, derived_indices] = output[:, derived_indices] + derived_shift.view(1, -1)
    else:
        perturbation = alpha * feature_sd.view(1, 1, -1)
        output[:, :, indices] = output[:, :, indices] + perturbation
        if derived_indices:
            derived_shift = alpha * torch.tensor(derived_sd, dtype=output.dtype)
            output[:, :, derived_indices] = output[:, :, derived_indices] + derived_shift.view(1, 1, -1)
    return output


def _direction_score(
    window: torch.Tensor | np.ndarray,
    feature_indices: Sequence[int],
) -> float:
    """Reduce a directional feature block to a single scalar summary score."""
    tensor = torch.as_tensor(window, dtype=torch.float32)
    if tensor.dim() == 2:
        return float(tensor[:, list(feature_indices)].mean())
    if tensor.dim() == 3:
        return float(tensor[:, :, list(feature_indices)].mean())
    raise ValueError("window must have shape (T, F) or (B, T, F)")


def _pair_plausibility_stats(
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction_a: str,
    direction_b: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the train-set joint mean, covariance, and 97.5% Mahalanobis threshold."""
    train_x = artifact.get("train", {}).get("X")
    if train_x is None:
        return np.asarray([0.0, 0.0], dtype=float), np.eye(2), 0.0
    train_x = train_x.float()
    indices_a = direction_feature_indices(feature_names, direction_map, direction_a)
    indices_b = direction_feature_indices(feature_names, direction_map, direction_b)
    if not indices_a or not indices_b:
        return np.asarray([0.0, 0.0], dtype=float), np.eye(2), 0.0

    a_train = train_x[:, :, indices_a].mean(dim=2).reshape(-1).numpy()
    b_train = train_x[:, :, indices_b].mean(dim=2).reshape(-1).numpy()
    joint_train = np.column_stack([a_train, b_train])
    mean_vec = joint_train.mean(axis=0)
    covariance = np.cov(joint_train.T)
    if covariance.shape == ():
        covariance = np.asarray([[float(covariance)]])
    covariance = np.atleast_2d(covariance)
    if np.allclose(covariance, 0.0):
        covariance = np.eye(2) * 1e-6
    if np.linalg.matrix_rank(covariance) < 2:
        covariance = covariance + np.eye(2) * 1e-6
    distances = np.einsum(
        "ij,jk,ik->i",
        joint_train - mean_vec,
        np.linalg.pinv(covariance),
        joint_train - mean_vec,
    )
    threshold = float(np.quantile(distances, 0.975))
    return mean_vec, covariance, threshold


def _joint_plausibility_check(
    perturbation: torch.Tensor | np.ndarray,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction_a: str,
    direction_b: str,
    stats: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> tuple[bool, float, float, float]:
    """Return whether a joint perturbation is inside the pair's empirical training range."""
    if stats is None:
        stats = _pair_plausibility_stats(
            artifact, feature_names, direction_map, direction_a, direction_b
        )
    mean_vec, covariance, threshold = stats
    inverse_covariance = np.linalg.pinv(covariance)
    if threshold <= 0.0:
        return True, 0.0, float(mean_vec[0]), float(mean_vec[1])
    current_vec = np.asarray([
        _direction_score(perturbation, direction_feature_indices(feature_names, direction_map, direction_a)),
        _direction_score(perturbation, direction_feature_indices(feature_names, direction_map, direction_b)),
    ], dtype=float)
    delta = current_vec - mean_vec
    mahalanobis = float(delta.T @ inverse_covariance @ delta)
    return mahalanobis <= threshold, mahalanobis, threshold, float(mean_vec[0])


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
        print(
        f"[ALPHA] direction={direction}, "
        f"min={lower:.4f}, max={upper:.4f}",
        flush=True,
        )
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

    order = np.argsort(alphas)
    sorted_alphas = alphas[order]
    sorted_means = means[order]

    margin = float("inf")
    for i in range(len(sorted_alphas) - 1):
        a0, a1 = sorted_alphas[i], sorted_alphas[i + 1]
        m0, m1 = sorted_means[i], sorted_means[i + 1]
        if (m0 - threshold) * (m1 - threshold) <= 0 and m1 != m0:
            crossing_alpha = a0 + (threshold - m0) * (a1 - a0) / (m1 - m0)
            margin = min(margin, abs(float(crossing_alpha)))

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


def profile_direction_pairs(
    model: torch.nn.Module,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    split_name: str = "test",
    directions: Sequence[str] = ("sleep", "activity", "social", "mobility", "screen"),
    max_windows: int | None = None,
    alphas_a: Sequence[float] | None = None,
    alphas_b: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    personalized: bool = False,
    participant_index: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Profile every available direction pair on real windows in a split."""
    if split_name not in artifact:
        raise KeyError(f"Unknown artifact split: {split_name}")
    if personalized and participant_index is None:
        raise ValueError("participant_index is required when personalized=True")
    available = [
        direction
        for direction in directions
        if direction_feature_indices(feature_names, direction_map, direction)
    ]
    pair_count = sum(len(available) - idx - 1 for idx in range(len(available)))
    windows = artifact[split_name]["X"]
    limit = len(windows) if max_windows is None else min(max_windows, len(windows))
    print(
        f"[interaction] starting {limit} windows across {len(available)} directions "
        f"and {pair_count} direction pairs",
        flush=True,
    )
    uids = artifact[split_name].get("uid")
    rows = []
    train_x = artifact["train"]["X"].float()
    feature_sds = {
        direction: train_x[:, :, direction_feature_indices(
            feature_names, direction_map, direction
        )].std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
        for direction in available
    }
    alpha_grids = {}
    for direction in available:
        lower, upper = plausible_alpha_bounds(
            artifact, feature_names, direction_map, direction
        )
        alpha_grids[direction] = default_direction_alphas(lower, upper, steps=7)
    pair_stats = {
        (direction_a, direction_b): _pair_plausibility_stats(
            artifact, feature_names, direction_map, direction_a, direction_b
        )
        for pair_index, direction_a in enumerate(available)
        for direction_b in available[pair_index + 1:]
    }
    for index in range(limit):
        raw_uid = None if uids is None else uids[index]
        raw_uid = raw_uid.item() if torch.is_tensor(raw_uid) else raw_uid
        embedding_id = None
        if personalized and raw_uid is not None:
            embedding_id = participant_index[str(raw_uid)]
        pair_index = 0
        for pair_index, direction_a in enumerate(available):
            for direction_b in available[pair_index + 1:]:
                response = probe_interaction(
                    model=model,
                    artifact=artifact,
                    feature_names=feature_names,
                    direction_map=direction_map,
                    direction_a=direction_a,
                    direction_b=direction_b,
                    window=windows[index],
                    alphas_a=alphas_a or alpha_grids[direction_a],
                    alphas_b=alphas_b or alpha_grids[direction_b],
                    device=device,
                    target_mean=target_mean,
                    target_std=target_std,
                    personalized=personalized,
                    participant_id=embedding_id,
                    feature_sds=feature_sds,
                    plausibility_stats=pair_stats[(direction_a, direction_b)],
                )
                rows.append({
                    "window_index": index,
                    "uid": raw_uid,
                    "direction_a": direction_a,
                    "direction_b": direction_b,
                    **summarize_interaction_response(response),
                })
        progress_interval = max(1, limit // 20)
        if (index + 1) % progress_interval == 0 or index == limit - 1:
            print(
                f"[interaction] processed {index + 1}/{limit} windows "
                f"({len(rows)} rows so far)",
                flush=True,
            )
    return rows


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
    batch_size: int = 32,
    personalized: bool = False,
    participant_index: dict[str, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build one sensitivity row per direction and real window in a split.

    Windows and alpha perturbations are batched together so GPU execution is
    used efficiently instead of launching one small forward pass per window.
    """
    if split_name not in artifact:
        raise KeyError(f"Unknown artifact split: {split_name}")
    if personalized and participant_index is None:
        raise ValueError("participant_index is required when personalized=True")
    split = artifact[split_name]
    windows = split["X"]
    limit = len(windows) if max_windows is None else min(max_windows, len(windows))
    uids = split.get("uid")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else "cpu"
    device = torch.device(device)
    rows: list[dict[str, object]] = []
    continuous_rows: list[dict[str, object]] = []
    model.eval()

    for direction in directions:
        indices = direction_feature_indices(feature_names, direction_map, direction)
        if not indices:
            summaries = [None] * limit
        else:
            if alphas is None:
                lower, upper = plausible_alpha_bounds(
                    artifact, feature_names, direction_map, direction
                )
                print(
                f"[ALPHA] direction={direction}, "
                f"min={lower:.4f}, max={upper:.4f}",
                flush=True,
            )

                direction_alphas = default_direction_alphas(lower, upper, steps=101)
            else:
                direction_alphas = [float(alpha) for alpha in alphas]

            train_x = artifact["train"]["X"].float()
            feature_sd = train_x[:, :, indices].std(
                dim=(0, 1), unbiased=False
            ).clamp_min(1e-6).to(device)
            summaries = []
            for start in range(0, limit, batch_size):
                stop = min(start + batch_size, limit)
                base = windows[start:stop].float().to(device)
                alpha_tensor = torch.tensor(direction_alphas, device=device)
                perturbed = base[:, None, :, :].expand(
                    -1, len(direction_alphas), -1, -1
                ).clone()
                perturbed[:, :, :, indices] += (
                    alpha_tensor[None, :, None, None] * feature_sd[None, None, None, :]
                )
                flat = perturbed.reshape(
                    -1, perturbed.shape[2], perturbed.shape[3]
                )
                with torch.no_grad():
                    if personalized:
                        batch_uids = uids[start:stop]
                        batch_uids = batch_uids.tolist() if torch.is_tensor(batch_uids) else list(batch_uids)
                        embedding_ids = [participant_index[str(uid)] for uid in batch_uids]
                        participant_tensor = torch.tensor(
                            embedding_ids, device=device, dtype=torch.long
                        ).repeat_interleave(len(direction_alphas))
                        prediction = model(flat, participant_tensor)
                    else:
                        prediction = model(flat)
                mean, logvar = _coerce_prediction(
                    prediction, target_mean, target_std
                )
                means = mean.reshape(stop - start, len(direction_alphas), -1)[:, :, 0]
                std_values = None
                if logvar is not None:
                    predictive_std = torch.exp(0.5 * logvar)
                    if target_std is not None:
                        predictive_std = predictive_std * target_std
                    std_values = predictive_std.reshape(
                        stop - start, len(direction_alphas), -1
                    )[:, :, 0]
                elif calibrated_std is not None:
                    std_values = torch.full_like(means, float(torch.as_tensor(calibrated_std)))
                for row_index in range(stop - start):
                    response = []

                    for alpha_index, alpha in enumerate(direction_alphas):
                        predicted_mean = float(
                            means[row_index, alpha_index].cpu()
                        )

                        predicted_std = (
                            None
                            if std_values is None
                            else float(std_values[row_index, alpha_index].cpu())
                        )

                        response.append({
                            "alpha": float(alpha),
                            "predicted_mean": predicted_mean,
                            "predicted_std": predicted_std,
                        })

                        # Preserve the continuous sensitivity profile.
                        window_index = start + row_index
                        uid = (
                            uids[window_index].item()
                            if torch.is_tensor(uids)
                            else uids[window_index]
                        )

                        continuous_rows.append({
                            "window_index": window_index,
                            "uid": str(uid),
                            "direction": direction,
                            "alpha": float(alpha),
                            "predicted_mean": predicted_mean,
                            "predicted_std": predicted_std,
                        })

                    summaries.append(
                        summarize_direction_response(response, threshold)
                    )

        for window_index, summary in enumerate(summaries):
            uid = uids[window_index].item() if torch.is_tensor(uids) else uids[window_index]
            rows.append({
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
            })
    return rows, continuous_rows


def _cluster_bootstrap_interval(
    rows: Sequence[dict[str, object]],
    metric: str,
    n_boot: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap the window-weighted metric mean by participant clusters."""
    by_participant: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None and metric == "interaction":
            value = row.get("interaction_mean")
        if value is None or not np.isfinite(float(value)):
            continue
        by_participant.setdefault(str(row["uid"]), []).append(float(value))
    participant_values = list(by_participant.values())
    if not participant_values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    sampled_means = []
    for _ in range(n_boot):
        sampled_participants = rng.integers(
            0,
            len(participant_values),
            size=len(participant_values),
        )
        sampled_values = np.concatenate(
            [participant_values[index] for index in sampled_participants]
        )
        sampled_means.append(float(sampled_values.mean()))
    sampled = np.asarray(sampled_means, dtype=float)
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


def aggregate_interaction_profiles(
    rows: Sequence[dict[str, object]],
    reporting_threshold: float = 0.0,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Aggregate interaction rows by pair with participant-clustered CIs."""
    pairs = sorted({
        (str(row["direction_a"]), str(row["direction_b"]))
        for row in rows
    })
    aggregates: dict[str, dict[str, float]] = {}
    for direction_a, direction_b in pairs:
        pair_rows = [
            row for row in rows
            if str(row.get("direction_a")) == direction_a
            and str(row.get("direction_b")) == direction_b
        ]
        interaction_values = np.asarray(
            [
                float(row.get("interaction", row.get("interaction_mean")))
                for row in pair_rows
                if row.get("interaction", row.get("interaction_mean")) is not None
                and np.isfinite(float(row.get("interaction", row.get("interaction_mean"))))
            ],
            dtype=float,
        )
        summary: dict[str, float] = {
            "count": float(len(pair_rows)),
            "participant_count": float(len({str(row.get("uid", "unknown")) for row in pair_rows})),
        }
        if interaction_values.size:
            summary["interaction_mean"] = float(interaction_values.mean())
            summary["interaction_median"] = float(np.median(interaction_values))
            summary["interaction_std"] = float(interaction_values.std(ddof=0))
            summary["interaction_abs_mean"] = float(np.abs(interaction_values).mean())
            summary["max_synergy"] = float(interaction_values.max())
            summary["max_antagonism"] = float(interaction_values.min())
            summary["interaction_large_fraction"] = float(
                (np.abs(interaction_values) > abs(reporting_threshold)).mean()
            )
        else:
            summary["interaction_mean"] = float("nan")
            summary["interaction_median"] = float("nan")
            summary["interaction_std"] = float("nan")
            summary["interaction_abs_mean"] = float("nan")
            summary["max_synergy"] = float("nan")
            summary["max_antagonism"] = float("nan")
            summary["interaction_large_fraction"] = float("nan")
        low, high = _interaction_cluster_interval(
            pair_rows,
            metric="interaction",
            n_boot=bootstrap_samples,
            random_seed=random_seed,
        )
        summary["interaction_ci_low"] = low
        summary["interaction_ci_high"] = high
        aggregates[f"{direction_a}:{direction_b}"] = summary
    return aggregates


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman rank correlation of two aligned score vectors."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return 1.0
    ranks_x = np.argsort(np.argsort(np.asarray(x, dtype=float))) + 1
    ranks_y = np.argsort(np.argsort(np.asarray(y, dtype=float))) + 1
    corr = np.corrcoef(ranks_x, ranks_y)[0, 1]
    return float(np.nan_to_num(corr, nan=1.0))


def interaction_seed_stability(
    interaction_summaries: Sequence[dict[str, float]],
    pairs: Sequence[str],
) -> float:
    """Return mean pairwise Spearman stability across all seed rankings."""
    if not interaction_summaries:
        return 1.0
    scores = []
    for summary in interaction_summaries:
        pair_scores = []
        for pair in pairs:
            pair_scores.append(abs(summary.get(pair, 0.0)))
        scores.append(pair_scores)
    if len(scores) < 2 or len(pairs) < 2:
        return 1.0
    correlations = [
        spearman_rank_correlation(scores[first], scores[second])
        for first in range(len(scores))
        for second in range(first + 1, len(scores))
    ]
    return float(np.mean(correlations))


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

def export_continuous_profiles(
    rows: Sequence[dict[str, object]],
    output_dir: str | Path,
    prefix: str = "ces",
) -> Path:
    """Save continuous alpha-response sensitivity profiles as CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows_path = output_path / f"{prefix}_continuous_sensitivity_profiles.csv"

    with rows_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "window_index",
                "uid",
                "direction",
                "alpha",
                "predicted_mean",
                "predicted_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows_path

def export_interaction_profiles(
    rows: Sequence[dict[str, object]],
    aggregates: dict[str, dict[str, float]],
    output_dir: str | Path,
    prefix: str = "ces",
) -> tuple[Path, Path]:
    """Save per-window interaction profiles and aggregate summaries as CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows_path = output_path / f"{prefix}_interaction_profiles.csv"
    aggregates_path = output_path / f"{prefix}_interaction_aggregates.json"

    with rows_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "window_index",
                "uid",
                "direction_a",
                "direction_b",
                "interaction_mean",
                "interaction_std",
                "max_synergy",
                "max_antagonism",
                "interaction_ci_low",
                "interaction_ci_high",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    serializable = {
        pair: {
            key: (None if not np.isfinite(value) else value)
            for key, value in summary.items()
        }
        for pair, summary in aggregates.items()
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


def probe_interaction(
    model: torch.nn.Module,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction_a: str,
    direction_b: str,
    window: torch.Tensor | np.ndarray,
    alphas_a: Sequence[float] | None = None,
    alphas_b: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    personalized: bool = False,
    participant_id: int | str | None = None,
    feature_sds: dict[str, torch.Tensor] | None = None,
    plausibility_stats: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> list[dict[str, float | None]]:
    """Measure joint input-space sensitivity for a pair of directions.

    The interaction value is the joint prediction minus both marginal
    predictions plus the unperturbed prediction. Positive values indicate
    model-predicted synergy; negative values indicate antagonism. This is not
    a causal interaction effect.
    """
    if direction_a == direction_b:
        raise ValueError("Interaction directions must be different")
    indices_a = direction_feature_indices(feature_names, direction_map, direction_a)
    indices_b = direction_feature_indices(feature_names, direction_map, direction_b)
    if not indices_a or not indices_b:
        raise ValueError("Both interaction directions must have matched features")
    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else "cpu"
    device = torch.device(device)

    if alphas_a is None:
        lower, upper = plausible_alpha_bounds(
            artifact, feature_names, direction_map, direction_a
        )
        alphas_a = default_direction_alphas(lower, upper, steps=7)
    if alphas_b is None:
        lower, upper = plausible_alpha_bounds(
            artifact, feature_names, direction_map, direction_b
        )
        alphas_b = default_direction_alphas(lower, upper, steps=7)
    alphas_a = list(dict.fromkeys([0.0, *[float(value) for value in alphas_a]]))
    alphas_b = list(dict.fromkeys([0.0, *[float(value) for value in alphas_b]]))

    base = torch.as_tensor(window, dtype=torch.float32)
    if base.dim() == 2:
        base = base.unsqueeze(0)
    if base.dim() != 3:
        raise ValueError("window must have shape (T, F) or (B, T, F)")

    combinations = [(alpha_a, alpha_b) for alpha_a in alphas_a for alpha_b in alphas_b]
    perturbed = []
    for alpha_a, alpha_b in combinations:
        changed = perturb_window_for_direction(
            base,
            artifact,
            feature_names,
            direction_map,
            direction_a,
            alpha_a,
            None if feature_sds is None else feature_sds[direction_a],
        )
        changed = perturb_window_for_direction(
            changed,
            artifact,
            feature_names,
            direction_map,
            direction_b,
            alpha_b,
            None if feature_sds is None else feature_sds[direction_b],
        )
        perturbed.append(changed)
    inputs = torch.cat(perturbed, dim=0).to(device)
    if personalized:
        if participant_id is None:
            raise ValueError("participant_id is required for personalized probes")
        participant = torch.tensor(
            [participant_id] * len(combinations),
            device=device,
            dtype=torch.long,
        )
        prediction = model(inputs, participant)
    else:
        prediction = model(inputs)
    means, logvars = _coerce_prediction(prediction, target_mean, target_std)
    means = means.detach().cpu().reshape(len(combinations), -1)[:, 0]
    stds = None
    if logvars is not None:
        predictive_std = torch.exp(0.5 * logvars)
        if target_std is not None:
            predictive_std = predictive_std * target_std
        stds = predictive_std.detach().cpu().reshape(len(combinations), -1)[:, 0]

    values = {
        combination: float(means[index])
        for index, combination in enumerate(combinations)
    }
    baseline = values[(0.0, 0.0)]
    results = []
    for index, (alpha_a, alpha_b) in enumerate(combinations):
        joint = values[(alpha_a, alpha_b)]
        marginal_a = values[(alpha_a, 0.0)]
        marginal_b = values[(0.0, alpha_b)]
        perturbed_window = perturb_window_for_direction(
            perturb_window_for_direction(
                base,
                artifact,
                feature_names,
                direction_map,
                direction_a,
                alpha_a,
                None if feature_sds is None else feature_sds[direction_a],
            ),
            artifact,
            feature_names,
            direction_map,
            direction_b,
            alpha_b,
            None if feature_sds is None else feature_sds[direction_b],
        )
        plausible, mahalanobis, threshold, _ = _joint_plausibility_check(
            perturbed_window,
            artifact,
            feature_names,
            direction_map,
            direction_a,
            direction_b,
            stats=plausibility_stats,
        )
        results.append({
            "alpha_a": alpha_a,
            "alpha_b": alpha_b,
            "predicted_mean": joint,
            "predicted_std": None if stds is None else float(stds[index]),
            "interaction": joint - marginal_a - marginal_b + baseline,
            "plausible": plausible,
            "mahalanobis_distance": mahalanobis,
            "mahalanobis_threshold": threshold,
            "uid": participant_id,
        })
    return results


def _interaction_cluster_interval(
    rows: Sequence[dict[str, object]],
    metric: str = "interaction",
    n_boot: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap the interaction mean by participant clusters, mirroring the marginal CI logic."""
    by_participant: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None and metric == "interaction":
            value = row.get("interaction_mean")
        if value is None or not np.isfinite(float(value)):
            continue
        uid = str(row.get("uid", "unknown"))
        by_participant.setdefault(uid, []).append(float(value))
    participant_values = list(by_participant.values())
    if not participant_values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_seed)
    sampled_means = []
    for _ in range(n_boot):
        sampled_participants = rng.integers(0, len(participant_values), size=len(participant_values))
        sampled_values = np.concatenate([participant_values[int(index)] for index in sampled_participants])
        sampled_means.append(float(sampled_values.mean()))
    sampled = np.asarray(sampled_means, dtype=float)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def summarize_interaction_response(
    response: Sequence[dict[str, float | None]],
    n_boot: int = 200,
    random_seed: int = 42,
) -> dict[str, float]:
    """Summarize joint-minus-marginal model-predicted interaction values.

    The CI is computed by participant-cluster resampling when UIDs are available,
    matching the marginal sensitivity pipeline. Otherwise, a standard row-level
    bootstrap is used as the fallback.
    """
    if not response:
        return {
            "interaction_mean": 0.0,
            "interaction_std": 0.0,
            "max_synergy": 0.0,
            "max_antagonism": 0.0,
            "interaction_ci_low": 0.0,
            "interaction_ci_high": 0.0,
        }

    plausible_mask = np.asarray([
        bool(item.get("plausible", True)) for item in response
    ], dtype=bool)
    filtered = list(response) if plausible_mask.all() else [item for item, keep in zip(response, plausible_mask) if keep]
    if not filtered:
        filtered = list(response)

    interactions = np.asarray([
        float(item["interaction"]) for item in filtered
    ], dtype=float)
    if interactions.size == 0:
        return {
            "interaction_mean": 0.0,
            "interaction_std": 0.0,
            "max_synergy": 0.0,
            "max_antagonism": 0.0,
            "interaction_ci_low": 0.0,
            "interaction_ci_high": 0.0,
        }

    ci_low, ci_high = float("nan"), float("nan")
    has_uid = any("uid" in item for item in filtered)
    if has_uid:
        ci_low, ci_high = _interaction_cluster_interval(filtered, metric="interaction", n_boot=n_boot, random_seed=random_seed)
    else:
        std_values = np.asarray([
            float(item.get("predicted_std", 0.0) or 0.0)
            for item in filtered
        ], dtype=float)
        std_values = np.where(np.isfinite(std_values), std_values, 0.0)
        if std_values.size > 0 and np.any(std_values > 0.0):
            rng = np.random.default_rng(random_seed)
            bootstrap_means = []
            for _ in range(n_boot):
                sampled = interactions + rng.normal(0.0, std_values)
                bootstrap_means.append(float(np.mean(sampled)))
            ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        else:
            ci_low = float(interactions.mean())
            ci_high = float(interactions.mean())

    return {
        "interaction_mean": float(interactions.mean()),
        "interaction_std": float(interactions.std(ddof=0)),
        "max_synergy": float(interactions.max()),
        "max_antagonism": float(interactions.min()),
        "interaction_ci_low": float(ci_low),
        "interaction_ci_high": float(ci_high),
    }
