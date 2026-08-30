"""
CES preprocessing — ports the essential logic from cells 38, 54, 58 of
the original notebook. Cells 39-53, 55-57 were audit/inspection only
(target distributions, coverage prints, correlation checks) and are
intentionally not reproduced here — re-run those in
notebooks/exploration.ipynb against the cached model_df below if you
want to re-inspect something.
"""
import pandas as pd

from src.config import (
    CES_TABLES_CACHE,
    CES_MODEL_DF_CACHE,
    CES_FEATURES_FINAL_CACHE,
    CES_TARGETS,
)
from src.caching import cache_pickle, cache_parquet, cache_json
from src.io_loaders import load_ces_tables


def get_ces_tables(force: bool = False) -> dict:
    """Cell 38 equivalent: raw CES CSVs loaded, cached."""
    return cache_pickle(CES_TABLES_CACHE, build_fn=load_ces_tables, force=force)


def build_ces_model_df(force: bool = False) -> pd.DataFrame:
    """
    Cell 54 equivalent: merges EMA + sensing + steps into one
    participant-day modeling table.
    """
    def _build():
        ces = get_ces_tables()
        print("Available keys in ces:", ces.keys())
        ema = ces["general_ema"].copy()

        target_df = ema[["uid", "day"] + CES_TARGETS].copy()

        sensing = ces["sensing"].copy()
        steps = ces["steps"].copy()

        model_df = target_df.merge(sensing, on=["uid", "day"], how="left")
        model_df = model_df.merge(steps, on=["uid", "day"], how="left", suffixes=("", "_steps"))

        for target in CES_TARGETS:
            n = model_df[target].notna().sum()
            print(f"{target:15s}: {n:7d} observations ({n / len(model_df) * 100:6.2f}%)")

        print("model_df shape:", model_df.shape)
        return model_df

    return cache_parquet(CES_MODEL_DF_CACHE, build_fn=_build, force=force)


def build_ces_features_final(force: bool = False, max_missing: float = 0.75) -> list:
    """
    Cell 58 equivalent: the real filtering step — drop features with
    >max_missing missingness on labeled rows, then drop zero-variance
    features. Returns the final feature-name list.
    """
    def _build():
        model_df = build_ces_model_df()
        labeled = model_df[model_df[CES_TARGETS].notna().any(axis=1)].copy()

        id_cols = ["uid", "day"]
        feature_cols = [c for c in labeled.columns if c not in id_cols + CES_TARGETS]

        missing_rate = labeled[feature_cols].isna().mean()
        features_missing_ok = missing_rate[missing_rate <= max_missing].index.tolist()

        variance = labeled[features_missing_ok].var(skipna=True, numeric_only=True)
        zero_variance = variance[variance == 0].index.tolist()

        features_final = [c for c in features_missing_ok if c not in zero_variance]

        print(f"candidate={len(feature_cols)} after_missing_filter={len(features_missing_ok)} "
              f"zero_var_removed={len(zero_variance)} final={len(features_final)}")
        return features_final

    return cache_json(CES_FEATURES_FINAL_CACHE, build_fn=_build, force=force)


if __name__ == "__main__":
    features = build_ces_features_final()
    print("\nCES pipeline ready. Final feature count:", len(features))
