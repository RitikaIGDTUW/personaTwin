"""Rule-based mapping from engineered features to behavioral directions."""

from __future__ import annotations

from pathlib import Path

from src.caching import cache_json
from src.config import (
    BEHAVIORAL_DIRECTION_MAP_CACHE,
    BEHAVIORAL_DIRECTIONS,
)

_DIRECTION_KEYWORDS = {
    "sleep": ("sleep", "dark", "night"),
    "activity": ("activity", "steps", "still", "walking", "running"),
    "social": ("conversation", "social", "call", "sms"),
    "mobility": ("gps", "location", "distance", "cluster"),
    "screen": ("phonelock", "screen", "unlock", "app"),
}


def _build_uncached(feature_names: list[str]) -> dict[str, list[str]]:
    direction_map = {direction: [] for direction in BEHAVIORAL_DIRECTIONS}
    direction_map["other"] = []

    for feature_name in feature_names:
        name = feature_name.lower()
        assigned = False
        for direction in BEHAVIORAL_DIRECTIONS:
            if any(keyword in name for keyword in _DIRECTION_KEYWORDS[direction]):
                direction_map[direction].append(feature_name)
                assigned = True
                break
        if not assigned:
            direction_map["other"].append(feature_name)

    return direction_map


def build_direction_map(
    feature_names: list[str],
    cache_path: Path | None = None,
    force: bool = False,
) -> dict[str, list[str]]:
    """Build and optionally cache a defensible feature-to-direction mapping."""
    feature_names = list(dict.fromkeys(feature_names))
    path = cache_path or BEHAVIORAL_DIRECTION_MAP_CACHE
    direction_map = cache_json(
        path,
        build_fn=lambda: _build_uncached(feature_names),
        force=force,
    )

    expected = set(feature_names)
    mapped = {
        feature
        for features in direction_map.values()
        for feature in features
    }
    if mapped != expected:
        raise ValueError("Cached behavioral direction map does not match features")

    return direction_map


if __name__ == "__main__":
    example = [
        "sleep_duration",
        "activity_mean",
        "gps_distance_km",
        "conversation_events",
        "app_usage_events",
    ]
    print(build_direction_map(example, force=True))
