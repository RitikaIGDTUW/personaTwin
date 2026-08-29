# PersonaTwin Contributor Guide

## 1. Project In One Sentence

PersonaTwin is a personalized temporal modeling system for mental-state dynamics. It combines behavioral sensing data with affective targets, learns a temporal representation for each participant, and eventually estimates a continuous model-predicted sensitivity profile across behavioral directions such as activity, mobility, social interaction, sleep-related behavior, and screen use.

The project is built around two datasets:

- **StudentLife**: development and personalization dataset.
- **CES (College Experience Study)**: larger-scale replication and generalization dataset.

The two datasets are **not concatenated blindly**. Each dataset has its own raw-data loader and preprocessing logic. They are converted separately into compatible participant-day tables and then passed through the same sequence and modeling framework.

---

## 2. Architecture

```text
StudentLife raw data                    CES raw data
        |                                      |
        v                                      v
Dataset-specific loading             Dataset-specific loading
and cleaning                         and daily aggregation
        |                                      |
        v                                      v
StudentLife model dataframe          CES model dataframe
        |                                      |
        +------------------+-------------------+
                           |
                           v
             Shared feature/direction logic
                           |
                           v
                Leakage-safe sequences
                           |
                           v
                 Population GRU baseline
                           |
                           v
                 Personalized temporal model
                           |
                           v
                    Latent state z_t
                           |
                           v
                 Sensitivity Engine
                           |
                           v
        margin / slope / curvature per person
        and behavioral direction
```

The intended research contribution is the **Sensitivity Engine**. It should describe how the trained model's predicted state changes as a latent behavioral direction is varied continuously. These are model-predicted sensitivities, not causal effects.

---

## 3. Dataset Roles

### 3.1 StudentLife

StudentLife contains intensive mobile sensing from university participants. It is useful for developing the personalized temporal modeling methodology because it contains participant-specific longitudinal sensing and established affective targets.

Important raw tables include:

| Family | Tables used or available |
|---|---|
| Activity | `activity` |
| Mobility | `gps`, `wifi`, `wifi_location` |
| Social interaction | `conversation`, `bluetooth` |
| Phone behavior | `phonelock`, `phonecharge`, `app_usage`, `call_log`, `sms` |
| Screen behavior | `dark` |
| Affect/EMA | `pam`, `stress`, `mood` |
| Sleep/surveys | `sleep`, Big Five, PHQ-9, PSQI, PANAS and other survey tables |

Verified raw inventory:

```text
RDS files found                 : 50
Successfully loaded tables     : 49
Core sensing participants      : 49
```

Important verified raw table sizes:

```text
activity       : 22,842,191 rows
conversation   :     79,023 rows
gps            :    202,877 rows
app_usage      :  1,990,510 rows
call_log       :     71,801 rows
sms            :     92,584 rows
phonelock      :      9,275 rows
pam            :      9,040 rows
stress         :      2,017 rows
mood           :        277 rows
sleep          :      1,644 rows
```

StudentLife timestamps are Unix seconds. They are converted with:

```python
pd.to_datetime(timestamp, unit="s", errors="coerce")
```

The current primary StudentLife target is **PAM** because it is sufficiently dense for longitudinal modeling.

Current raw target counts:

```text
PAM    : 9,040 observations / 49 participants
Stress : 2,017 observations / 46 participants
Mood   :   277 observations / 38 participants
```

After daily aggregation into the current model dataframe:

```text
PAM    : 2,296 observed participant-days
Stress :    46 observed participant-days
Mood   :   212 observed participant-days
```

Stress and mood have not been discarded. They are retained as scientifically meaningful secondary targets, but they are too sparse for the first strict seven-day next-day forecasting experiment.

### 3.2 CES

CES is a larger College Experience Study dataset containing daily EMA, sensing, steps, demographics, and many engineered sensing variables. It is intended for replication, scale, and cross-dataset generalization.

The local raw path is configured in `src/config.py` as:

```text
data/raw/CES/
```

The loader expects:

```text
data/raw/CES/Demographics/demographics.csv
data/raw/CES/EMA/general_ema.csv
data/raw/CES/EMA/covid_ema.csv
data/raw/CES/Sensing/sensing.csv
data/raw/CES/Sensing/steps.csv
```

Verified loaded CES tables:

```text
demographics : (216, 3)
general_ema  : (217155, 19)
covid_ema    : (16511, 12)
sensing      : (216065, 651)
steps        : (176458, 30)
```

CES participant and time facts:

```text
EMA participants              : 220
Sensing participants          : 220
Steps participants            : 198
EMA ∩ Sensing                 : 220
EMA ∩ Steps                   : 198
All three                    : 198
```

The 22 participants without steps should not automatically be removed. Steps are an additional modality and can be missing for some participants.

CES day values are integers in `YYYYMMDD` format, for example `20170907`. The sequence code explicitly converts them to calendar dates before checking consecutive days.

Verified temporal ranges include:

```text
GENERAL_EMA : 20170907 to 20220704
SENSING     : 20170907 to 20220615
STEPS       : 20170907 to 20220614
COVID_EMA   : 20200317 to 20220426
```

The current cached CES model dataframe is:

```text
217,155 rows
682 total columns
570 final numeric feature columns
```

For the first common experiment, CES also uses **PAM** as the target. Stress and PHQ-4-related targets remain available in the CES preprocessing layer for later experiments, but the canonical sequence entry point currently builds PAM-only sequences to keep the first experiment scientifically and computationally focused.

---

## 4. Existing Code Ownership

### Configuration and caching

- `src/config.py`
  - Project paths
  - Raw/interim/processed cache locations
  - Stage 2 constants
  - Sequence cache paths
  - Model checkpoint path

- `src/caching.py`
  - `cache_pickle`
  - `cache_parquet`
  - `cache_json`
  - `cache_torch`

The rule is: expensive or reusable work should be cached, and every builder should support `force=True` when a cache must be rebuilt.

### Raw data loading

- `src/io_loaders.py`
  - StudentLife RDS loading
  - CES CSV loading

- `src/preprocess_studentlife.py`
  - Cached StudentLife table dictionary
  - Timestamp-normalized StudentLife target tables

### Participant-day model tables

- `src/preprocess_slife_daily.py`
  - StudentLife sensing aggregation
  - GPS distance and location features
  - Conversation, phone-lock, dark, app, call, and SMS daily features
  - PAM, stress, and mood daily target extraction
  - Final StudentLife participant-day dataframe

- `src/preprocess_ces.py`
  - CES EMA/sensing/steps daily table construction
  - CES final feature filtering

### Direction mapping and sequences

- `src/behavioral_directions.py`
  - Rule-based feature-to-direction mapping

- `src/datasets.py`
  - Shared sequence construction for both datasets
  - Date normalization
  - Participant-wise chronological split
  - Train-only imputation and normalization
  - Seven-day windows and next-day target alignment
  - Direction vectors

- `src/build_sequences.py`
  - Canonical command-line entry point for either dataset

- `src/audit_sequences.py`
  - Validates sequence cache shapes, finite values, target dimensions, and participant coverage

---

## 5. Current Cached Artifacts

### Interim artifacts

```text
data/interim/studentlife_tables.pkl
data/interim/studentlife_targets.pkl
data/interim/studentlife_model_df.parquet
data/interim/ces_tables.pkl
data/interim/ces_model_df.parquet
data/interim/ces_features_final.json
data/interim/studentlife_behavioral_direction_map.json
data/interim/ces_behavioral_direction_map.json
```

### Processed sequence artifacts

```text
data/processed/studentlife_sequences.pt
data/processed/ces_sequences.pt
```

These are PyTorch dictionaries with this structure:

```python
{
    "train": {
        "X": ...,                  # windows x 7 x features
        "y": ...,                  # windows x targets
        "uid": ...,                # participant ID per window
        "direction_vectors": ..., # windows x directions x features
    },
    "val": {...},
    "test": {...},
    "metadata": {
        "feature_names": [...],
        "target_names": [...],
        "direction_names": [...],
        "lookback_days": 7,
    },
}
```

---

## 6. Current Sequence Results

The sequence builder uses:

- seven days of history
- the next calendar day as the prediction target
- participant-wise chronological splitting
- first 70% of each participant's rows for training
- next 15% for validation
- final 15% for testing
- forward-fill within participant
- remaining missing feature values filled with training medians
- z-score normalization using training statistics only
- no random temporal shuffling

### StudentLife PAM

```text
features : 17
train    : X=(1458, 7, 17), y=(1458, 1)
val      : X=(88, 7, 17),   y=(88, 1)
test     : X=(59, 7, 17),   y=(59, 1)
```

### CES PAM

```text
features : 570
train    : X=(25576, 7, 570), y=(25576, 1)
val      : X=(4112, 7, 570),  y=(4112, 1)
test     : X=(3599, 7, 570),  y=(3599, 1)
```

The CES build initially appeared slow because it processes 570 features and creates direction-vector tensors. It completed successfully after using PAM-only targets.

Run the sequence audit with:

```powershell
.\.venv\Scripts\python.exe -m src.audit_sequences studentlife
.\.venv\Scripts\python.exe -m src.audit_sequences ces
```

---

## 7. Behavioral Directions

The current rule-based mapping uses these semantic groups:

```text
sleep
activity
social
mobility
screen
other
```

Examples:

| Direction | Example feature names |
|---|---|
| Sleep | names containing `sleep`, `dark`, or `night` |
| Activity | `activity`, `steps`, `still`, `walking`, `running` |
| Social | `conversation`, `social`, `call`, `sms` |
| Mobility | `gps`, `location`, `distance`, `cluster` |
| Screen | `phonelock`, `screen`, `unlock`, `app` |
| Other | features matching no rule |

This is intentionally manual and interpretable. It is not a learned causal graph. The `other` category is important because forcing every feature into a semantic direction would overstate what the feature means.

For StudentLife, the current mapping covers the 17 daily features across the five primary directions. The sequence artifact stores a direction vector for every direction and window. These vectors are normalized feature summaries and are intended as inputs to the future Sensitivity Engine.

---

## 8. What Is Complete vs. What Remains

### Complete

```text
[done] Raw StudentLife inventory and cached loading
[done] Raw CES loading and cached loading
[done] StudentLife target normalization
[done] StudentLife daily behavioral aggregation
[done] StudentLife participant-day model dataframe
[done] CES participant-day model dataframe
[done] CES feature filtering
[done] Target feasibility audit
[done] Behavioral direction mapping
[done] Shared leakage-safe sequence builder
[done] StudentLife PAM sequence artifact
[done] CES PAM sequence artifact
[done] Sequence artifact audits
[done] Population-level GRU baseline (StudentLife and CES)
[done] Participant personalization layer (embedding-based, both datasets)
[done] Predictive uncertainty head (population and personalized, both datasets)
[done] Pre-sensitivity audit (mean-baseline comparison, interval coverage)
[done] Step 0 data-validation audit (direction-map consistency, mobility/PAM
       leakage check on real training data)
[done] Sensitivity Engine: slope/curvature/margin with participant-clustered
       bootstrap CIs, interpolated (non-quantized) margin, pairwise
       interaction/synergy-antagonism detection, personalized-model
       participant-embedding support
[done] Sensitivity Engine structural verification protocol
[done] StudentLife Sensitivity Engine run, fully verified (population +
       personalized, univariate + interaction)
```

### Not yet implemented

```text
[pending] CES Sensitivity Engine run (code ready, not yet executed - GPU/Colab)
[pending] Cross-dataset transfer/generalization write-up (limited to the four
          directions CES and StudentLife share - CES has no social direction)
[pending] Final combined sensitivity table (both datasets)
[pending] Evaluation notebook and result plots
[pending] Twin/dashboard integration
```

The original handoff markdown says some completed preprocessing items are still pending. That status section is now stale and should be interpreted together with the actual cached files and commands in this repository. This section (Section 8) itself required a similar correction as of the Sensitivity Engine build-out - it previously listed the population baseline, personalization, uncertainty, and Sensitivity Engine as not yet implemented, which was no longer accurate.

---

## 9. Recommended Implementation Order From Here

### Stage 2.4A: population PAM baseline

Implement a simple GRU that receives:

```text
X shape = windows x 7 x features
```

and predicts:

```text
y shape = windows x 1
```

Start with a population-level model without participant embeddings. This answers whether the sequence features contain predictive signal at all.

Train separately on:

1. StudentLife PAM sequences.
2. CES PAM sequences.

Do not mix the datasets in the first baseline.

### Stage 2.4B: personalized model

After the population baseline is validated, add participant embeddings or adapters. Compare the personalized model against the population baseline using the same chronological test split.

### Stage 2.4C: uncertainty

Add a mean/log-variance output head and Gaussian negative log-likelihood only after the point-prediction baseline is working. This keeps model debugging separate from uncertainty debugging.

### Stage 2.4D: Sensitivity Engine

Only after a trained model has a validated test result should the Sensitivity Engine:

1. Encode a participant's recent seven-day history into a latent state.
2. Select one behavioral direction.
3. Perturb the latent state across a continuous alpha range.
4. Decode model-predicted PAM values.
5. Fit a smooth curve.
6. Report slope, curvature, and margin.

Every result must be described as **model-predicted sensitivity**, never as a causal effect.

### Stage 2.5: cross-dataset evaluation

Use the separate dataset models and shared ontology to compare:

- performance
- feature families
- direction behavior
- personalization gains
- sensitivity-profile stability

Do not claim direct causal transfer between CES and StudentLife without additional alignment and validation.

---

## 10. Commands for a New Contributor

From the repository root:

```powershell
# Activate the environment if needed
.\.venv\Scripts\Activate.ps1

# Rebuild StudentLife daily model data when raw inputs changed
.\.venv\Scripts\python.exe -m src.preprocess_slife_daily

# Rebuild StudentLife PAM sequences
.\.venv\Scripts\python.exe -m src.build_sequences studentlife --force

# Rebuild CES model data/features if needed
.\.venv\Scripts\python.exe -c "from src.preprocess_ces import build_ces_model_df, build_ces_features_final; build_ces_model_df(); build_ces_features_final()"

# Rebuild CES PAM sequences
.\.venv\Scripts\python.exe -m src.build_sequences ces --force

# Audit both sequence artifacts
.\.venv\Scripts\python.exe -m src.audit_sequences studentlife
.\.venv\Scripts\python.exe -m src.audit_sequences ces
```

If raw data or target logic changes, use `force=True` or `--force` for the affected cache. Do not delete unrelated caches manually unless you understand their dependencies.

---

## 11. Contribution Rules

1. Keep StudentLife and CES preprocessing separate.
2. Reuse the shared sequence API after dataset-specific preprocessing.
3. Use cache helpers for expensive work.
4. Never compute imputation or normalization statistics using validation or test data.
5. Never randomly shuffle temporal rows before the chronological split.
6. Do not use StudentLife stress or mood as primary targets without a new feasibility analysis.
7. Keep PAM as the first common target unless the project decision changes.
8. Do not describe model-predicted sensitivity as a causal intervention effect.
9. Add or update an audit whenever a cache schema changes.
10. Prefer small executable validations after each implementation change.

---

## 12. The Immediate Next Task

The population baseline, personalization, uncertainty heads, and the
Sensitivity Engine itself are all implemented and have been run and
structurally verified on StudentLife (`python -m src.verify_sensitivity_engine
studentlife` passes cleanly, including the interpolated-margin fix and the
personalized-model participant-embedding path).

The next task is narrower: run the same pipeline on CES.

```powershell
python -m src.run_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_population_gru.pt --max-windows 200
python -m src.run_sensitivity ces --checkpoint data/processed/checkpoints/ces_uncertainty_personalized_gru.pt --personalized --max-windows 200
python -m src.verify_sensitivity_engine ces
```

CES is 570 features and 3,599 test windows, so this is meaningfully more
compute than StudentLife's 59-window run - offload it to a GPU (Colab) rather
than running the full sweep on CPU. The trained checkpoints
(`ces_uncertainty_population_gru.pt`, `ces_uncertainty_personalized_gru.pt`)
and the current `src/sensitivity.py` (with the interpolated-margin fix) must
both be copied into whatever environment runs this - checkpoints are not in
git, and an older copy of `sensitivity.py` without the interpolation fix will
reproduce the margin-quantization issue already fixed on StudentLife.

Once CES passes verification, remember: CES has no `social` direction (the
raw data never captured conversation/call/SMS sensing), so it will correctly
report only 4 directions where StudentLife reports 5. Do not force a
cross-dataset comparison on `social` - report it as StudentLife-only, and
limit direct CES-vs-StudentLife comparisons to sleep, activity, mobility,
and screen.

After both datasets are verified, the remaining work is the final combined
sensitivity table, the cross-dataset write-up, and eventually integrating
these outputs with the twin/dashboard layer - none of which are started yet.