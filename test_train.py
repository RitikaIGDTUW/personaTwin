import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

from src.datasets import build_sequences
from src.train import train_personalized, train

np.random.seed(1)

rows = []

for uid in range(1, 9):
    base = np.random.uniform(6, 8)

    for day in range(60):
        rows.append({
            "uid": uid,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
            "sleep": base + np.random.randn() * 0.5,
            "screen": 100 + np.random.randn() * 10,
            "pam": 8 + np.random.randn() * 2,
        })

df = pd.DataFrame(rows)

df["pam_history"] = df["pam"]

feature_names = [
    "sleep",
    "screen",
    "pam_history"
]

direction_map = {
    "sleep": ["sleep"],
    "screen": ["screen"]
}

tmpdir = Path(tempfile.mkdtemp())

cache_path = tmpdir / "ces_sequences.pt"

build_sequences(
    df,
    feature_names,
    ["pam"],
    direction_map,
    cache_path=cache_path,
    predict_delta=True,
    force=True
)

import src.train as train_mod

train_mod.CES_SEQUENCES_CACHE = cache_path

metrics = train_personalized(
    "ces",
    epochs=2,
    batch_size=16,
    hidden_size=8,
    embedding_size=4,
    patience=5
)

print("PERSONALIZED TEST METRICS:", metrics)

assert "corr" in metrics and "abs_mae" in metrics

metrics2 = train(
    "ces",
    epochs=2,
    batch_size=16,
    hidden_size=8,
    patience=5
)

print("POPULATION TEST METRICS:", metrics2)

assert "corr" in metrics2 and "abs_mae" in metrics2

print("OK: both train() and train_personalized() work with predict_delta artifacts")