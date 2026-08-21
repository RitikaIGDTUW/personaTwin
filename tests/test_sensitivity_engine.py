import torch

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
