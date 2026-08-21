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

    if device is None:
        device = next(model.parameters()).device
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
        for alpha in alphas:
            perturbed = perturb_window_for_direction(
                window_tensor.cpu(),
                artifact,
                feature_names,
                direction_map,
                direction,
                float(alpha),
            ).to(device)
            if personalized:
                if participant_id is None:
                    raise ValueError("participant_id is required for personalized probes")
                prediction = model(perturbed, torch.tensor([participant_id], device=device, dtype=torch.long))
            else:
                prediction = model(perturbed)

            mean, logvar = _coerce_prediction(prediction, target_mean, target_std)
            mean_value = float(mean.detach().cpu().reshape(-1)[0])
            std_value = None
            if logvar is not None:
                std_value = float(torch.exp(0.5 * logvar).detach().cpu().reshape(-1)[0].item())
            results.append({
                "alpha": float(alpha),
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
