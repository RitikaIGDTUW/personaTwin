"""Input-space sensitivity analysis for behavioral directions.

The Sensitivity Engine should perturb the actual feature window, not the hidden
state z. This keeps the perturbation in a unit-bearing, interpretable space and
makes the resulting slope/curvature/margin diagnostics defensible.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    lower_cap: float = -2.0,
    upper_cap: float = 2.0,
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
        return (float(lower_cap), float(upper_cap))

    train_x = artifact.get("train", {}).get("X")
    if train_x is None:
        raise KeyError("artifact['train']['X'] is required to estimate alpha bounds")
    train_x = train_x.float()
    direction_values = train_x[:, :, indices]
    feature_mean = direction_values.mean(dim=(0, 1), keepdim=True)
    feature_sd = direction_values.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    z_scaled = (direction_values - feature_mean) / feature_sd.view(1, 1, -1)
    lower = float(z_scaled.min().item())
    upper = float(z_scaled.max().item())
    lower = float(np.clip(lower, lower_cap, upper_cap))
    upper = float(np.clip(upper, lower_cap, upper_cap))
    if lower > upper:
        lower, upper = upper, lower
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    return lower, upper


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
) -> dict[str, float]:
    """Reduce alpha/response pairs into interpretable sensitivity summaries."""
    if not response:
        return {"slope": 0.0, "curvature": 0.0, "margin": float("inf")}

    alphas = np.asarray([float(item["alpha"]) for item in response], dtype=float)
    means = np.asarray([float(item["predicted_mean"]) for item in response], dtype=float)

    if len(alphas) >= 2:
        slope = float(np.polyfit(alphas, means, 1)[0])
    else:
        slope = 0.0

    if len(alphas) >= 3:
        curvature_coefficients = np.polyfit(alphas, means, 2)
        curvature = float(curvature_coefficients[0] * 2.0)
    else:
        curvature = 0.0

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
                }
            )
    return rows


def aggregate_profiles(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate per-window sensitivity rows by behavioral direction."""
    directions = sorted({str(row["direction"]) for row in rows})
    aggregates: dict[str, dict[str, float]] = {}
    for direction in directions:
        direction_rows = [row for row in rows if row["direction"] == direction]
        summary: dict[str, float] = {"count": 0.0}
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
        aggregates[direction] = summary
    return aggregates
