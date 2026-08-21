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


def test_profile_split_and_aggregate_profiles():
    from src.sensitivity import aggregate_profiles, profile_split

    class SumModel(torch.nn.Module):
        def forward(self, features):
            return features.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)

    feature_names = ["activity"]
    direction_map = {"activity": ["activity"]}
    artifact = {
        "train": {"X": torch.ones(4, 2, 1)},
        "test": {"X": torch.zeros(2, 2, 1), "uid": torch.tensor([10, 11])},
    }
    rows = profile_split(
        model=SumModel(),
        artifact=artifact,
        feature_names=feature_names,
        direction_map=direction_map,
        directions=["activity"],
        max_windows=2,
        threshold=1.0,
        alphas=[0.0, 1.0],
    )
    aggregate = aggregate_profiles(rows)
    assert len(rows) == 2
    assert aggregate["activity"]["slope_count"] == 2.0


def test_probe_returns_one_result_per_alpha_with_batched_forward():
    from src.sensitivity import probe_direction

    class SumModel(torch.nn.Module):
        def forward(self, features):
            return features.sum(dim=(1, 2)).unsqueeze(-1)

    artifact = {"train": {"X": torch.ones(4, 2, 1)}}
    response = probe_direction(
        model=SumModel(),
        artifact=artifact,
        feature_names=["activity"],
        direction_map={"activity": ["activity"]},
        direction="activity",
        window=torch.zeros(2, 1),
        alphas=[-1.0, 0.0, 1.0],
    )

    assert [item["alpha"] for item in response] == [-1.0, 0.0, 1.0]
    assert response[0]["predicted_mean"] < response[-1]["predicted_mean"]


def test_export_profiles_writes_csv_and_json(tmp_path):
    import json

    from src.sensitivity import export_profiles

    rows = [{
        "window_index": 0,
        "uid": "p1",
        "direction": "activity",
        "slope": 1.0,
        "curvature": 0.1,
        "margin": None,
    }]
    aggregates = {
        "activity": {
            "count": 1.0,
            "slope_mean": 1.0,
            "margin_mean": float("nan"),
        }
    }
    rows_path, aggregates_path = export_profiles(rows, aggregates, tmp_path)

    assert rows_path.exists()
    assert aggregates_path.exists()
    assert json.loads(aggregates_path.read_text())["activity"]["margin_mean"] is None


def test_plot_sensitivity_results_writes_figures(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from src.sensitivity import plot_sensitivity_results

    aggregates = {
        "activity": {
            "count": 2.0,
            "slope_count": 2.0,
            "slope_mean": 0.5,
            "slope_std": 0.1,
            "margin_count": 1.0,
        },
        "social": {"count": 2.0, "slope_count": 0.0},
    }
    slope_path, crossing_path = plot_sensitivity_results(aggregates, tmp_path)

    assert slope_path.exists()
    assert crossing_path.exists()


def test_stage4_audit_helpers():
    from src.sensitivity import (
        compare_direction_maps,
        direction_map_feature_counts,
        mobility_target_correlations,
        participant_counts,
    )

    maps = {
        "ces": {"mobility": ["loc_distance"], "screen": ["unlock_time"]},
        "studentlife": {"mobility": ["gps_distance"], "screen": ["app_usage"]},
    }
    assert direction_map_feature_counts(maps["ces"])["mobility"] == 1
    comparison = compare_direction_maps(maps)
    assert comparison["mobility"]["ces"]["features"] == ["loc_distance"]

    frame = __import__("pandas").DataFrame({
        "pam": [1.0, 2.0, 3.0],
        "loc_distance": [1.0, 2.0, 4.0],
    })
    correlations = mobility_target_correlations(
        frame,
        {"mobility": ["loc_distance"]},
    )
    assert correlations["loc_distance"] > 0.0

    artifact = {"test": {"X": torch.zeros(3, 2, 1), "uid": torch.tensor([1, 1, 2])}}
    assert participant_counts(artifact) == {"n_windows": 3, "n_participants": 2}


def test_uncertainty_weighted_summary_reports_bootstrap_intervals():
    response = [
        {"alpha": -1.0, "predicted_mean": 1.0, "predicted_std": 0.1},
        {"alpha": 0.0, "predicted_mean": 2.0, "predicted_std": 0.1},
        {"alpha": 1.0, "predicted_mean": 3.0, "predicted_std": 0.1},
    ]
    summary = summarize_direction_response(
        response,
        threshold=2.5,
        bootstrap_samples=50,
    )

    assert summary["slope_ci_low"] < summary["slope"]
    assert summary["slope"] < summary["slope_ci_high"]


def test_probe_accepts_calibrated_std_for_deterministic_models():
    from src.sensitivity import probe_direction

    class SumModel(torch.nn.Module):
        def forward(self, features):
            return features.sum(dim=(1, 2)).unsqueeze(-1)

    response = probe_direction(
        model=SumModel(),
        artifact={"train": {"X": torch.ones(4, 2, 1)}},
        feature_names=["activity"],
        direction_map={"activity": ["activity"]},
        direction="activity",
        window=torch.zeros(2, 1),
        alphas=[-1.0, 0.0, 1.0],
        calibrated_std=0.5,
    )

    assert all(item["predicted_std"] == 0.5 for item in response)
