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
- Population and personalized PAM GRU baselines (both datasets)
- Predictive uncertainty heads (population and personalized, both datasets)
- Pre-sensitivity audit (model vs. mean baseline, interval coverage)
- Step 0 data-validation audit: direction-map consistency check and
  mobility/PAM leakage check on real training data (see
  `src/step0_audit_checks.py`)
- Sensitivity Engine core: continuous model-predicted slope, curvature,
  and margin profiles per behavioral direction, with participant-clustered
  bootstrap confidence intervals (`src/sensitivity.py`, `src/run_sensitivity.py`)
- Pairwise/interaction sensitivity across behavioral direction pairs
  (synergy/antagonism detection)
- Personalized-model participant-embedding support in the Sensitivity Engine
- Interpolated margin computation (exact threshold-crossing alpha via linear
  interpolation between grid points, rather than snapping to the nearest
  alpha grid value)
- Structural verification protocol for Sensitivity Engine outputs
  (`src/verify_sensitivity_engine.py`)
- StudentLife Sensitivity Engine run: fully executed and verified
  (population + personalized, univariate + interaction). Sleep's margin is
  correctly `inf` for all windows — sleep alone does not push predicted PAM
  across threshold within its plausible bounds; this is a validated finding,
  not a defect.

Next:

- CES Sensitivity Engine run (population + personalized, univariate +
  interaction) — code is ready and verified on StudentLife, but CES itself
  has not yet been executed (pending GPU/Colab run)
- Verify CES output with `src/verify_sensitivity_engine.py` once available
- Note: CES has no `social` direction (0 features — this dataset never
  captured conversation/call/SMS sensing). Cross-dataset comparisons are
  limited to the four directions CES and StudentLife share: sleep, activity,
  mobility, screen.
- Final sensitivity table combining both datasets, with per-dataset
  participant/window counts and small-N caveats for StudentLife (N=23
  participants, 59 test windows)
- Integrate Sensitivity Engine outputs with the twin/dashboard layer
  (not yet started)

The Sensitivity Engine must describe model-predicted sensitivity, not causal
effects.

## Running the Sensitivity Engine

Once uncertainty checkpoints exist for a dataset, run the Step 0 audit first,
then the engine itself, then verify the output:

```powershell
python -m src.step0_audit_checks

python -m src.run_sensitivity studentlife --checkpoint data/processed/checkpoints/studentlife_uncertainty_population_gru.pt
python -m src.run_sensitivity studentlife --checkpoint data/processed/checkpoints/studentlife_uncertainty_personalized_gru.pt --personalized

python -m src.verify_sensitivity_engine studentlife
```

For CES, cap the interaction sweep since it has 570 features and 3,599 test
windows:

```powershell
python -m src.run_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_population_gru.pt --max-windows 200
python -m src.run_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_personalized_gru.pt --personalized --max-windows 200

python -m src.verify_sensitivity_engine ces
```

Output files land in `data/processed/sensitivity/`:

```text
{dataset}_sensitivity_profiles.csv
{dataset}_sensitivity_aggregates.json
{dataset}_interaction_profiles.csv
{dataset}_interaction_aggregates.json
```

`verify_sensitivity_engine.py` checks file existence, absence of NaN values,
correct direction sets per dataset, CI ordering, and real (non-degenerate)
variance in margin and interaction values. A clean pass means the output is
structurally sound — it does not mean the substantive results are correct;
slope/curvature signs and magnitudes should still be reviewed against
domain expectations before being reported.

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

Training uses target mean/std computed from the training split only. Loss is
optimized on the standardized target, while reported MSE, MAE, and RMSE are
converted back to the original PAM scale. AdamW, validation-based learning
rate reduction, and gradient clipping are enabled for stability.

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

For CES, the high-dimensional input can be compressed and regularized before
the GRU. This is the recommended comparison configuration; it does not change
the cached data or StudentLife defaults:

```powershell
python -m src.train ces --device cuda --epochs 30 --batch-size 128 --hidden-size 128 --projection-size 128 --dropout 0.25
python -m src.train ces --model personalized --device cuda --epochs 30 --batch-size 128 --hidden-size 128 --projection-size 128 --dropout 0.25
```

Compare these CES runs against the existing 570-input configuration using the
same seeds. Keep the lower validation-loss checkpoint and report test metrics
only after selecting the configuration on validation data.

## Train the Uncertainty-Aware Models

The uncertainty stage predicts both a PAM mean and a Gaussian predictive
variance. The reported `mse`, `mae`, and `rmse` remain on the original PAM
scale; `nll` is the standardized Gaussian negative log-likelihood and
`mean_std` is the average predictive standard deviation in PAM units.

StudentLife:

```powershell
python -m src.train studentlife --model uncertainty --device cuda --epochs 30 --batch-size 128
python -m src.train studentlife --model uncertainty_personalized --device cuda --epochs 30 --batch-size 128
```

CES with the improved projection/dropout configuration:

```powershell
python -m src.train ces --model uncertainty --device cuda --epochs 30 --batch-size 128 --hidden-size 128 --projection-size 128 --dropout 0.25
python -m src.train ces --model uncertainty_personalized --device cuda --epochs 30 --batch-size 128 --hidden-size 128 --projection-size 128 --dropout 0.25
```

Uncertainty checkpoints are saved under:

```text
data/processed/checkpoints/*uncertainty*.pt
```

## Pre-Sensitivity Audit

Before using the Sensitivity Engine, compare the uncertainty model against a
test-set prediction of the training PAM mean and inspect empirical interval
coverage. Run this only with full-training checkpoints, not one-epoch smoke
checkpoints:

```powershell
python -m src.audit_pre_sensitivity studentlife --checkpoint data/processed/checkpoints/studentlife_uncertainty_population_gru.pt
python -m src.audit_pre_sensitivity studentlife --checkpoint data/processed/checkpoints/studentlife_uncertainty_personalized_gru.pt --personalized
python -m src.audit_pre_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_population_gru.pt
python -m src.audit_pre_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_personalized_gru.pt --personalized
```

The audit reports model MSE/MAE/RMSE, mean-baseline metrics, average
predictive standard deviation, and empirical 68%/95% interval coverage. A
model should beat the mean baseline, and nominal coverage should be checked
before interpreting sensitivity profiles.

For StudentLife, use the hybrid uncertainty loss to prioritize mean prediction
while retaining the variance head:

```powershell
python -m src.train studentlife --model uncertainty --device cuda --epochs 30 --batch-size 128 --mean-loss-weight 0.5 --result-tag hybrid_seed42
```

Then rerun `audit_pre_sensitivity` on the new checkpoint. The required result
is model RMSE below the mean-baseline RMSE, followed by a separate check that
the 68% and 95% interval coverage is reasonable.

## Repository Contents

- `src/`: preprocessing, caching, direction mapping, sequence construction,
  and audits
- `CONTRIBUTOR_GUIDE.md`: detailed architecture and onboarding guide
- `PersonaTwin_Major_Project_Framework.md`: research architecture and novelty
- `PersonaTwin_Complete_Implementation_Handoff(2).md`: implementation history
- `requirements.txt`: Python dependencies

Raw data, local caches, virtual environments, and generated model artifacts
are excluded through `.gitignore`.