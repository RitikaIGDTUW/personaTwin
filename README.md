# PersonaTwin

PersonaTwin is a research pipeline for personalized mental-state dynamics
using multimodal behavioral sensing. It processes StudentLife and CES
separately, converts each dataset into participant-day features, builds
leakage-safe temporal sequences, and will later train personalized GRU models
with a model-predicted behavioral sensitivity engine.

## Datasets

Raw datasets are intentionally excluded from GitHub. Place them locally at:

```text
data/raw/studentlife/dataset_rds/
data/raw/CES/Demographics/
data/raw/CES/EMA/
data/raw/CES/Sensing/
```

StudentLife is the development and personalization dataset. CES is the
larger replication and generalization dataset. They are preprocessed
independently and then use the shared sequence framework.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `.venv` already exists, activate it and skip creation.

## Cache Preprocessing

Expensive outputs are cached under `data/interim/` and `data/processed/`.
Those folders are local and ignored by Git.

### StudentLife

Build or load cached StudentLife RDS tables and normalized targets:

```powershell
python -m src.preprocess_studentlife
```

Build the StudentLife participant-day model dataframe:

```powershell
python -m src.preprocess_slife_daily
```

Use the direct Python API with `force=True` after changing preprocessing:

```powershell
python -c "from src.preprocess_slife_daily import build_studentlife_model_df; build_studentlife_model_df(force=True)"
```

Main cache:

```text
data/interim/studentlife_model_df.parquet
```

### CES

Build or load the cached CES participant-day dataframe and final feature list:

```powershell
python -c "from src.preprocess_ces import build_ces_model_df, build_ces_features_final; build_ces_model_df(); build_ces_features_final()"
```

Force a complete CES rebuild:

```powershell
python -c "from src.preprocess_ces import build_ces_model_df, build_ces_features_final; build_ces_model_df(force=True); build_ces_features_final(force=True)"
```

Main caches:

```text
data/interim/ces_model_df.parquet
data/interim/ces_features_final.json
```

### Temporal Sequence Caches

The current primary target is PAM for both datasets. Sequence construction
uses seven days of history, predicts the next calendar day, splits each
participant chronologically, and computes imputation/normalization statistics
from training data only.

```powershell
python -m src.build_sequences studentlife --force
python -m src.build_sequences ces --force
```

Sequence caches:

```text
data/processed/studentlife_sequences.pt
data/processed/ces_sequences.pt
```

Audit either artifact:

```powershell
python -m src.audit_sequences studentlife
python -m src.audit_sequences ces
```

## Current Validated Sequence Sizes

```text
StudentLife PAM: train 1458, validation 88, test 59, 17 features
CES PAM:        train 25576, validation 4112, test 3599, 570 features
```

## Full Preprocessing Entry Point

```powershell
python run_pipeline.py
python run_pipeline.py --force
```

## Project Status

Completed:

- Cached StudentLife RDS and CES CSV loading
- Dataset-specific daily feature engineering
- StudentLife target extraction for PAM, stress, and mood
- CES feature filtering
- Behavioral direction mapping
- Leakage-safe StudentLife and CES sequence artifacts
- Sequence artifact audits

Next:

- Population PAM GRU baseline
- Separate StudentLife and CES evaluation
- Participant personalization
- Predictive uncertainty
- Continuous model-predicted sensitivity profiles
- Margin, slope, and curvature analysis

The Sensitivity Engine must describe model-predicted sensitivity, not causal
effects.

## Train the Baseline GRU

Training is designed for Google Colab or Kaggle with a CUDA GPU. Upload the
repository and the generated sequence caches, or regenerate the caches there.
Train one dataset at a time:

```powershell
python -m src.train studentlife --device cuda --epochs 30
python -m src.train ces --device cuda --epochs 30
```

For a small local CPU smoke test only:

```powershell
python -m src.train studentlife --device cpu --epochs 1 --max-train-windows 64
```

The trainer automatically uses CUDA with `--device auto` when available. It
writes training history to `data/processed/logs/` and the best checkpoint to
`data/processed/checkpoints/`. These generated files are ignored by Git.

## Train the Personalized GRU

After the population baseline has been trained, compare the participant-
embedding model on the same sequence caches:

```powershell
python -m src.train studentlife --model personalized --device cuda --epochs 30 --batch-size 128
python -m src.train ces --model personalized --device cuda --epochs 30 --batch-size 128
```

The personalized model uses a shared GRU plus a learned embedding for each
participant. It is a predictive personalization experiment, not a causal
intervention model. Results are saved separately:

```text
data/processed/checkpoints/studentlife_personalized_gru.pt
data/processed/checkpoints/ces_personalized_gru.pt
data/processed/logs/studentlife_personalized_gru.csv
data/processed/logs/ces_personalized_gru.csv
```

For local smoke tests:

```powershell
python -m src.train studentlife --model personalized --device cpu --epochs 1 --max-train-windows 64
python -m src.train ces --model personalized --device cpu --epochs 1 --max-train-windows 8
```

## Repository Contents

- `src/`: preprocessing, caching, direction mapping, sequence construction,
  and audits
- `CONTRIBUTOR_GUIDE.md`: detailed architecture and onboarding guide
- `PersonaTwin_Major_Project_Framework.md`: research architecture and novelty
- `PersonaTwin_Complete_Implementation_Handoff(2).md`: implementation history
- `requirements.txt`: Python dependencies

Raw data, local caches, virtual environments, and generated model artifacts
are excluded through `.gitignore`.
