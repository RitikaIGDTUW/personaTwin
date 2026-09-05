"""User-facing, raw-unit counterfactual transformations.

The Sensitivity Engine measures domain-level model response. This module
constructs interpretable scenarios for the frontend, where a change should be
applied to a named feature or a coupled group of features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from src.sensitivity import direction_feature_indices


def alpha_for_real_shift(real_shift: float, feature_name: str, artifact: dict) -> float:
    """Convert a requested raw-unit feature change to standardized alpha."""
    raw_std = artifact.get("metadata", {}).get("feature_raw_std", {}).get(feature_name)
    if raw_std is None or not np.isfinite(float(raw_std)) or float(raw_std) <= 0:
        raise KeyError(f"No positive raw standard deviation for feature {feature_name!r}")
    return float(real_shift) / float(raw_std)


def raw_range_for_feature(feature_name: str, artifact: dict) -> tuple[float, float]:
    """Return the observed train-split raw range for one feature."""
    metadata = artifact.get("metadata", {})
    raw_min = metadata.get("feature_raw_min", {}).get(feature_name)
    raw_max = metadata.get("feature_raw_max", {}).get(feature_name)
    if raw_min is None or raw_max is None:
        raise KeyError("Raw feature range metadata is missing")
    return float(raw_min), float(raw_max)


def raw_value_for_window(
    window: torch.Tensor,
    artifact: dict,
    feature_names: Sequence[str],
    direction_map: dict[str, list[str]],
    direction: str,
    day_index: int = -1,
) -> dict[str, float]:
    """Return raw-unit values for one direction on one window day."""
    metadata = artifact.get("metadata", {})
    raw_means = metadata.get("feature_raw_mean")
    raw_stds = metadata.get("feature_raw_std")
    if raw_means is None or raw_stds is None:
        raise KeyError("Raw feature statistics are missing from artifact metadata")
    indices = direction_feature_indices(feature_names, direction_map, direction)
    names = list(feature_names)
    day = torch.as_tensor(window, dtype=torch.float32)[day_index]
    return {
        names[index]: float(day[index]) * float(raw_stds[names[index]]) + float(raw_means[names[index]])
        for index in indices
    }


def perturb_window_for_real_shift(
    window: torch.Tensor | np.ndarray,
    artifact: dict,
    feature_name: str,
    real_shift: float,
) -> torch.Tensor:
    """Apply a real-unit shift to one feature and leave all others unchanged."""
    names = list(artifact.get("metadata", {}).get("feature_names", []))
    if feature_name not in names:
        raise KeyError(f"Feature {feature_name!r} is not in the artifact schema")
    raw_std = artifact.get("metadata", {}).get("feature_raw_std", {}).get(feature_name)
    if raw_std is None or not np.isfinite(float(raw_std)) or float(raw_std) <= 0:
        raise KeyError(f"No positive raw standard deviation for feature {feature_name!r}")
    tensor = torch.as_tensor(window, dtype=torch.float32).clone()
    if tensor.dim() not in (2, 3):
        raise ValueError("window must have shape (T, F) or (B, T, F)")
    index = names.index(feature_name)
    standardized_shift = float(real_shift) / float(raw_std)
    if tensor.dim() == 2:
        tensor[:, index] += standardized_shift
    else:
        tensor[:, :, index] += standardized_shift
    return tensor


def perturb_window_for_real_shifts(
    window: torch.Tensor | np.ndarray,
    artifact: dict,
    shifts: dict[str, float],
) -> torch.Tensor:
    """Apply explicit raw-unit shifts to several named features.

    This supports scenarios such as ``screen_time +30 minutes`` together
    with ``unlock_count -5`` without shifting every feature in the screen
    direction. Each feature keeps its own unit and scaling.
    """
    output = torch.as_tensor(window, dtype=torch.float32).clone()
    for feature_name, real_shift in shifts.items():
        output = perturb_window_for_real_shift(output, artifact, feature_name, real_shift)
    return output


def perturb_sleep_schedule(
    window: torch.Tensor | np.ndarray,
    artifact: dict,
    duration_shift_hours: float = 0.0,
    bedtime_shift_hours: float = 0.0,
    sleep_duration_feature: str = "sleep_duration",
    sleep_start_feature: str = "sleep_start",
    sleep_end_feature: str = "sleep_end",
    timing_units_per_hour: float = 8.0,
) -> torch.Tensor:
    """Create a coupled sleep counterfactual across a full input window.

    Duration changes derive the end time. Bedtime changes move start and end
    together. CES timing features use eighths of an hour and a 192-unit clock.
    Other sleep features remain unchanged.
    """
    names = list(artifact.get("metadata", {}).get("feature_names", []))
    required = [sleep_duration_feature, sleep_start_feature, sleep_end_feature]
    missing = [name for name in required if name not in names]
    if missing:
        raise KeyError(f"Sleep schedule features missing from artifact: {missing}")
    metadata = artifact.get("metadata", {})
    raw_means = metadata.get("feature_raw_mean", {})
    raw_stds = metadata.get("feature_raw_std", {})
    if any(name not in raw_means or name not in raw_stds for name in required):
        raise KeyError("Raw sleep feature statistics are required")

    tensor = torch.as_tensor(window, dtype=torch.float32).clone()
    if tensor.dim() not in (2, 3):
        raise ValueError("window must have shape (T, F) or (B, T, F)")

    duration_index = names.index(sleep_duration_feature)
    start_index = names.index(sleep_start_feature)
    end_index = names.index(sleep_end_feature)
    duration_mean, duration_std = float(raw_means[sleep_duration_feature]), float(raw_stds[sleep_duration_feature])
    start_mean, start_std = float(raw_means[sleep_start_feature]), float(raw_stds[sleep_start_feature])
    end_mean, end_std = float(raw_means[sleep_end_feature]), float(raw_stds[sleep_end_feature])
    if min(duration_std, start_std, end_std) <= 0:
        raise ValueError("Sleep feature standard deviations must be positive")

    if tensor.dim() == 2:
        duration = tensor[:, duration_index] * duration_std + duration_mean + float(duration_shift_hours)
        start = tensor[:, start_index] * start_std + start_mean
        start = torch.remainder(start + float(bedtime_shift_hours) * timing_units_per_hour, 24.0 * timing_units_per_hour)
        end = torch.remainder(start + duration * timing_units_per_hour, 24.0 * timing_units_per_hour)
        tensor[:, duration_index] = (duration - duration_mean) / duration_std
        tensor[:, start_index] = (start - start_mean) / start_std
        tensor[:, end_index] = (end - end_mean) / end_std
    else:
        duration = tensor[:, :, duration_index] * duration_std + duration_mean + float(duration_shift_hours)
        start = tensor[:, :, start_index] * start_std + start_mean
        start = torch.remainder(start + float(bedtime_shift_hours) * timing_units_per_hour, 24.0 * timing_units_per_hour)
        end = torch.remainder(start + duration * timing_units_per_hour, 24.0 * timing_units_per_hour)
        tensor[:, :, duration_index] = (duration - duration_mean) / duration_std
        tensor[:, :, start_index] = (start - start_mean) / start_std
        tensor[:, :, end_index] = (end - end_mean) / end_std
    return tensor
