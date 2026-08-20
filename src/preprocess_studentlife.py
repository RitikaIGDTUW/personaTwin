"""
StudentLife preprocessing — ports the essential logic from cells
4, 5, 8, 24 of the original notebook. Everything else in that section
(cells 9-23, 25-30, 32) was audit/inspection only and is intentionally
not reproduced here; re-run those ad hoc in notebooks/exploration.ipynb
if you want to re-inspect something, using the cached tables below as
input so you never reload raw RDS files again.
"""
import pandas as pd

from src.config import (
    STUDENTLIFE_TABLES_CACHE,
    STUDENTLIFE_TARGETS_CACHE,
    STUDENTLIFE_TARGETS,
)
from src.caching import cache_pickle
from src.io_loaders import load_all_studentlife_tables


def get_studentlife_tables(force: bool = False) -> dict:
    """Cell 8 equivalent: all raw StudentLife tables, cached."""
    return cache_pickle(
        STUDENTLIFE_TABLES_CACHE,
        build_fn=load_all_studentlife_tables,
        force=force,
    )


def build_studentlife_targets(force: bool = False) -> dict:
    """
    Cell 24 equivalent: the actual fix — StudentLife target timestamps
    are Unix seconds and must be normalized before any temporal work.
    """
    def _build():
        tables = get_studentlife_tables()
        targets = {}
        for name in STUDENTLIFE_TARGETS:
            df = tables[name].copy()
            df["timestamp_raw"] = df["timestamp"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            targets[name] = df
            n_invalid = df["timestamp"].isna().sum()
            print(f"{name:7s} rows={len(df):5d} participants={df['uid'].nunique():3d} invalid={n_invalid}")
        return targets

    return cache_pickle(STUDENTLIFE_TARGETS_CACHE, build_fn=_build, force=force)


if __name__ == "__main__":
    # Running this file directly does the full StudentLife stage end to end,
    # using cache where available.
    targets = build_studentlife_targets()
    print("\nStudentLife targets ready:", list(targets.keys()))
