import pytest
import torch

from src.behavioral_directions import build_direction_map
from src.sensitivity import (
    default_direction_alphas,
    direction_feature_indices,
    perturb_window_for_direction,
    plausible_alpha_bounds,
    summarize_direction_response,
)


def test_direction_feature_indices_and_alpha_bounds():
    feature_names = ["sleep_mean", "activity_steps", "social_calls", "gps_distance", "screen_unlocks"]
    direction_map = {
        "sleep": ["sleep_mean"],
        "activity": ["activity_steps"],
        "social": ["social_calls"],
        "mobility": ["gps_distance"],
        "screen": ["screen_unlocks"],
    }

    artifact = {
        "train": {
            "X": torch.tensor(
                [
                    [[0.0, 10.0, 3.0, 1.0, 20.0]],
                    [[1.0, 12.0, 4.0, 2.0, 22.0]],
                    [[-1.0, 8.0, 2.0, 0.5, 18.0]],
                ],
                dtype=torch.float32,
            )
        }
    }

    indices = direction_feature_indices(feature_names, direction_map, "activity")
    assert indices == [1]

    lower, upper = plausible_alpha_bounds(artifact, feature_names, direction_map, "activity")
    assert lower < 0.0
    assert upper > 0.0
    assert default_direction_alphas(lower, upper, 5)[0] <= lower
    assert default_direction_alphas(lower, upper, 5)[-1] >= upper

    window = torch.tensor([[0.0, 10.0, 3.0, 1.0, 20.0]], dtype=torch.float32)
    expected_shift = artifact["train"]["X"][:, :, 1].std(unbiased=False).item()
    perturbed = perturb_window_for_direction(window, artifact, feature_names, direction_map, "activity", alpha=1.0)
    assert torch.allclose(perturbed[:, 1], window[:, 1] + torch.tensor([expected_shift], dtype=torch.float32))
    assert torch.allclose(perturbed[:, [0, 2, 3, 4]], window[:, [0, 2, 3, 4]])


def test_summarize_direction_response_returns_guarded_metrics():
    response = [
        {"alpha": -1.0, "predicted_mean": 2.0},
        {"alpha": 0.0, "predicted_mean": 2.5},
        {"alpha": 1.0, "predicted_mean": 3.0},
        {"alpha": 2.0, "predicted_mean": 3.5},
    ]
    summary = summarize_direction_response(response, threshold=0.4)

    assert "slope" in summary
    assert "curvature" in summary
    assert "margin" in summary
    assert summary["slope"] == summary["slope"]
    assert summary["margin"] >= 0.0


def test_probe_rejects_direction_with_no_matching_features():
    from src.sensitivity import probe_direction

    with pytest.raises(ValueError, match="no features"):
        probe_direction(
            model=torch.nn.Linear(2, 1),
            artifact={"train": {"X": torch.zeros(1, 1, 2)}},
            feature_names=["sleep_mean", "quality_activity"],
            direction_map={"activity": ["old_activity_feature"]},
            direction="activity",
            window=torch.zeros(1, 2),
        )


def test_ces_location_features_are_not_activity_features():
    direction_map = build_direction_map(
        [
            "act_still_ep_0",
            "quality_activity",
            "loc_home_still",
            "loc_food_still",
        ],
        force=True,
    )

    assert direction_map["activity"] == ["act_still_ep_0", "quality_activity"]
    assert direction_map["mobility"] == ["loc_home_still", "loc_food_still"]


def test_profile_all_directions_runs_each_direction():
    from src.sensitivity import profile_all_directions

    class SumModel(torch.nn.Module):
        def forward(self, features):
            return features.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)

    feature_names = ["sleep", "activity", "social", "mobility", "screen"]
    direction_map = {name: [name] for name in feature_names}
    artifact = {"train": {"X": torch.ones(4, 2, 5)}}
    profile = profile_all_directions(
        model=SumModel(),
        artifact=artifact,
        feature_names=feature_names,
        direction_map=direction_map,
        window=torch.zeros(2, 5),
        threshold=1.0,
        alphas=[0.0, 1.0],
    )

    assert set(profile) == set(feature_names)
    assert all("slope" in summary for summary in profile.values())
