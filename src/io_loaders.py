"""
Raw-data loaders. No google.colab imports here — this runs the same
locally or in any notebook. Point config.py's RAW_DIR at wherever you've
copied the raw StudentLife/CES folders locally (or a synced Drive folder
if you use Google Drive Desktop / rclone).
"""
import pandas as pd
from pathlib import Path
import pyreadr  # pip install pyreadr

from src.config import STUDENTLIFE_RAW, CES_RAW


def load_rds(path: Path) -> pd.DataFrame:
    """Load the first object from an RDS file."""
    result = pyreadr.read_r(str(path))
    if len(result) == 0:
        raise ValueError(f"No object found in {path}")
    return list(result.values())[0]


def list_rds_files():
    if not STUDENTLIFE_RAW.exists():
        raise FileNotFoundError(f"StudentLife raw dir not found: {STUDENTLIFE_RAW}")
    return list(STUDENTLIFE_RAW.rglob("*.Rds"))


def load_all_studentlife_tables() -> dict:
    """Loads every StudentLife RDS file into {table_name: dataframe}."""
    rds_files = list_rds_files()
    tables = {}
    for path in sorted(rds_files):
        try:
            df = load_rds(path)
            key = path.stem.lower()
            tables[key] = df
            print(f"loaded {key:25s} shape={df.shape}")
        except Exception as e:
            print(f"FAILED {path}: {e!r}")
    return tables


def load_ces_tables() -> dict:
    """Loads the five core CES CSVs into {table_name: dataframe}."""
    if not CES_RAW.exists():
        raise FileNotFoundError(f"CES raw dir not found: {CES_RAW}")

    files_to_load = {
        "demographics": CES_RAW / "Demographics" / "demographics.csv",
        "general_ema": CES_RAW / "EMA" / "general_ema.csv",
        "covid_ema": CES_RAW / "EMA" / "covid_ema.csv",
        "sensing": CES_RAW / "Sensing" / "sensing.csv",
        "steps": CES_RAW / "Sensing" / "steps.csv",
    }

    ces = {}
    for name, path in files_to_load.items():
        if not path.exists():
            print(f"MISSING: {name} at {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        ces[name] = df
        print(f"loaded {name:15s} shape={df.shape}")
    return ces
