"""
Generic disk-cache helpers so preprocessing never has to be re-run
after a restart/disconnect unless you explicitly ask for it (force=True).
"""
import pickle
import json
from pathlib import Path
import pandas as pd


def cache_torch(path: Path, build_fn, force: bool = False):
    """Cache a PyTorch object using torch.save and torch.load."""
    import torch

    if path.exists() and not force:
        print(f"[cache hit] {path.name}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"[cache miss] building {path.name} ...")
    obj = build_fn()
    torch.save(obj, path)
    print(f"[cache saved] {path.name}")
    return obj


def cache_pickle(path: Path, build_fn, force: bool = False):
    """Cache any Python object (dict of dataframes, list, etc.) as pickle."""
    if path.exists() and not force:
        print(f"[cache hit] {path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"[cache miss] building {path.name} ...")
    obj = build_fn()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[cache saved] {path.name}")
    return obj


def cache_parquet(path: Path, build_fn, force: bool = False) -> pd.DataFrame:
    """Cache a single dataframe as parquet (faster + smaller than pickle for tabular data)."""
    if path.exists() and not force:
        print(f"[cache hit] {path.name}")
        return pd.read_parquet(path)

    print(f"[cache miss] building {path.name} ...")
    df = build_fn()
    df.to_parquet(path)
    print(f"[cache saved] {path.name}  shape={df.shape}")
    return df


def cache_json(path: Path, build_fn, force: bool = False):
    """Cache small objects (e.g. a filtered feature-name list) as json."""
    if path.exists() and not force:
        print(f"[cache hit] {path.name}")
        with open(path, "r") as f:
            return json.load(f)

    print(f"[cache miss] building {path.name} ...")
    obj = build_fn()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[cache saved] {path.name}")
    return obj
