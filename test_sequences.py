import sys, types
import numpy as np

torch_stub = types.ModuleType("torch")

class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    @property
    def shape(self):
        return self.arr.shape

    def flatten(self):
        return self.arr.flatten()

    def __getitem__(self, idx):
        return FakeTensor(self.arr[idx])

torch_stub.from_numpy = lambda a: FakeTensor(a)
torch_stub.tensor = lambda a: FakeTensor(a)

sys.modules["torch"] = torch_stub

import pandas as pd
from src.datasets import build_sequences

np.random.seed(0)

rows = []

for uid in [1, 2]:
    for day in range(30):
        rows.append({
            "uid": uid,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
            "sleep": 6 + np.random.randn() * 0.5,
            "screen": 100 + np.random.randn() * 10,
            "pam": 8 + np.random.randn() * 2,
        })

df = pd.DataFrame(rows)

df["pam_history"] = df["pam"]

feature_names = ["sleep", "screen", "pam_history"]

direction_map = {
    "sleep": ["sleep"],
    "screen": ["screen"]
}

seq = build_sequences(
    df,
    feature_names,
    ["pam"],
    direction_map,
    cache_path=None
)

for split in ["train", "val", "test"]:
    s = seq[split]
    print(
        split,
        s["X"].shape,
        s["y"].shape,
        s["baseline"].shape
    )

print(
    "num features after augmentation:",
    len(seq["metadata"]["feature_names"])
)

print(seq["metadata"]["feature_names"])

seq_delta = build_sequences(
    df,
    feature_names,
    ["pam"],
    direction_map,
    cache_path=None,
    predict_delta=True
)

s = seq_delta["train"]

recon = s["baseline"].flatten() + s["y"].flatten()

print(
    "reconstruction matches absolute y:",
    np.allclose(
        seq["train"]["y"].flatten(),
        recon
    )
)

print("OK")