"""
Run the full preprocessing pipeline end to end (uses cache where available).
Usage:
    python run_pipeline.py            # uses cache, only computes what's missing
    python run_pipeline.py --force    # ignores cache, rebuilds everything
"""
import argparse

from src.preprocess_studentlife import build_studentlife_targets
from src.preprocess_ces import build_ces_model_df, build_ces_features_final


def main(force: bool):
    print("=" * 80)
    print("STUDENTLIFE")
    print("=" * 80)
    build_studentlife_targets(force=force)

    print("\n" + "=" * 80)
    print("CES")
    print("=" * 80)
    build_ces_model_df(force=force)
    build_ces_features_final(force=force)

    print("\nPipeline complete. Check data/interim/ for cached outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="ignore cache, rebuild everything")
    args = parser.parse_args()
    main(force=args.force)
