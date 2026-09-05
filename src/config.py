"""
Central config — every path used anywhere in the project lives here.
Never hardcode a path in another file; import it from here instead.
"""
from pathlib import Path

# Project root = the personatwin/ folder this file's parent's parent lives in
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

STUDENTLIFE_RAW = RAW_DIR / "studentlife" / "dataset_rds"
CES_RAW = RAW_DIR / "CES"

# Make sure interim/processed dirs exist so cache writes never fail
for d in [INTERIM_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cache file paths — one constant per cached artifact, referenced everywhere
STUDENTLIFE_TABLES_CACHE = INTERIM_DIR / "studentlife_tables.pkl"
STUDENTLIFE_TARGETS_CACHE = INTERIM_DIR / "studentlife_targets.pkl"
STUDENTLIFE_MODEL_DF_CACHE = INTERIM_DIR / "studentlife_model_df.parquet"
CES_TABLES_CACHE = INTERIM_DIR / "ces_tables.pkl"
CES_MODEL_DF_CACHE = INTERIM_DIR / "ces_model_df.parquet"
CES_FEATURES_FINAL_CACHE = INTERIM_DIR / "ces_features_final.json"

CES_TARGETS = ["pam", "stress", "phq4_score"]
STUDENTLIFE_TARGETS = ["pam", "stress", "mood"]

# Stage 2: shared sequence/model configuration
SEQUENCE_LOOKBACK_DAYS = 7
MIN_SEQUENCE_DAYS = 14
BEHAVIORAL_DIRECTIONS = [
    "sleep",
    "activity",
    "social",
    "mobility",
    "screen",
]

CES_SEQUENCES_CACHE = PROCESSED_DIR / "ces_sequences.pt"
STUDENTLIFE_SEQUENCES_CACHE = PROCESSED_DIR / "studentlife_sequences.pt"
CES_DELTA_SEQUENCES_CACHE = PROCESSED_DIR / "ces_sequences_delta.pt"
STUDENTLIFE_DELTA_SEQUENCES_CACHE = PROCESSED_DIR / "studentlife_sequences_delta.pt"
BEHAVIORAL_DIRECTION_MAP_CACHE = INTERIM_DIR / "behavioral_direction_map.json"
CES_DIRECTION_MAP_CACHE = INTERIM_DIR / "ces_behavioral_direction_map.json"
STUDENTLIFE_DIRECTION_MAP_CACHE = INTERIM_DIR / "studentlife_behavioral_direction_map.json"
MODEL_CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"
MODEL_LOG_DIR = PROCESSED_DIR / "logs"
MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)


def sequence_cache_path(dataset: str, predict_delta: bool = False) -> Path:
    """Return the dataset cache path for absolute or delta PAM targets."""
    caches = {
        ("studentlife", False): STUDENTLIFE_SEQUENCES_CACHE,
        ("studentlife", True): STUDENTLIFE_DELTA_SEQUENCES_CACHE,
        ("ces", False): CES_SEQUENCES_CACHE,
        ("ces", True): CES_DELTA_SEQUENCES_CACHE,
    }
    try:
        return caches[(dataset, predict_delta)]
    except KeyError as error:
        raise ValueError("dataset must be 'studentlife' or 'ces'") from error
