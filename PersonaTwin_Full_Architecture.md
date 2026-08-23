# PersonaTwin — Full Project Architecture (Current State)

This reflects what you've actually built so far, plus the corrected
Sensitivity Engine design (input-space perturbation, not latent-space).

---

## End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — RAW DATA → PARTICIPANT-DAY TABLES        [DONE]           │
│                                                                       │
│   StudentLife raw (RDS)        CES raw (CSV)                        │
│         │                            │                              │
│   load + cache                 load + cache                        │
│         │                            │                              │
│   timestamp fix                merge EMA+sensing+steps              │
│         │                            │                              │
│   daily aggregation            missingness + zero-var filter        │
│         │                            │                              │
│   studentlife_model_df         ces_model_df (677→570 features)      │
└─────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 2 — SEQUENCE CONSTRUCTION                     [DONE]          │
│                                                                       │
│   Per participant: chronological 70/15/15 split                     │
│   Forward-fill → train-median fill → train-only z-score normalize   │
│   7-day lookback window → next-day PAM target                       │
│                                                                       │
│   StudentLife: 17 features,  1458/88/59 windows                     │
│   CES:        570 features, 25576/4112/3599 windows                 │
└─────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 3 — MODEL PROGRESSION                         [DONE]          │
│                                                                       │
│   Population GRU  →  Personalized GRU (+ participant embedding)     │
│         │                       │                                   │
│   baseline                RMSE improvement:                         │
│                             StudentLife 5.2%, CES 1.7%               │
│                                       │                              │
│                             Uncertainty head (Gaussian NLL)          │
│                                       │                              │
│              ┌────────────────────────┴────────────────────────┐    │
│              ▼                                                  ▼    │
│   CES: uncertainty head WORKS                    StudentLife: uncertainty
│   RMSE 3.997, 68%→69.3%, 95%→96.6%               head made mean WORSE
│   → use as-is                                    → use deterministic
│                                                     personalized GRU +
│                                                     validation-residual
│                                                     calibrated std
│                                                     (RMSE 3.277, but
│                                                     95% coverage only
│                                                     89.8% — label
│                                                     preliminary)
└─────────────────────────────┬───────────────────────────────────────┘
                               │
                    Two finalized "twins":
              ┌────────────────┴────────────────┐
              ▼                                  ▼
      CES twin (primary,                StudentLife twin
      uncertainty-native)                (preliminary, residual-calibrated)
              │                                  │
              └────────────────┬─────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 4 — SENSITIVITY ENGINE (core contribution)   [DONE —          │
│                                      corrected; interaction hardened] │
│                                                                       │
│   For a real person's real 7-day window, measure marginal and       │
│   pairwise behavioral sensitivity (sleep / activity / social /      │
│   mobility / screen):                                                │
│                                                                       │
│   1. Take the REAL input window x  (not z — this was the fix)        │
│   2. For alpha in the empirical range for this direction:             │
│        x' = x + alpha * (per-feature training-set SD),               │
│             applied only to that direction's feature columns          │
│        use the composite direction's observed training range          │
│        as the plausibility guard                                     │
│   3. Batch windows × alpha values on the selected device              │
│      and run the full trained model forward → (mean, std)             │
│   4. Fit an inverse-variance-weighted curve → compute:                │
│        slope     = responsiveness near current state                  │
│        curvature = diminishing / increasing returns                   │
│        margin    = smallest alpha crossing a chosen PAM threshold     │
│   5. Bootstrap uncertainty and aggregate by participant clusters        │
│   6. Repeat for all 5 directions → per-person/population profiles       │
│   7. For each available pair, evaluate x_ab = x + alpha_a*sd_a         │
│      + alpha_b*sd_b and subtract the additive expectation:              │
│      interaction = joint - marginal_a - marginal_b + baseline           │
│      Only plausible joint perturbations are retained using the empirical  │
│      training joint distribution. Pair summaries use participant-clustered│
│      bootstrap CIs. This is still model sensitivity, not causal effect    │
│      estimation.                                                          │
│                                                                        │
│   CES: uncertainty-native GRU, 3,599 windows / 202 participants        │
│   StudentLife: residual-calibrated deterministic GRU,                 │
│                59 windows / 23 participants                           │
│   Outputs: CSV + JSON summaries + PNG plots                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 5 — VALIDATION                                [NEXT]          │
│                                                                       │
│   - Manual sanity check on 3-5 individuals (curves look physiologically│
│     sane, no wild discontinuities) before trusting scale results      │
│   - Stability check: same person, multiple training seeds → do        │
│     slope/margin rankings stay consistent?                            │
│   - Cross-dataset check (RQ5): does the same Sensitivity Engine code   │
│     run unmodified on both twins?                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ STAGE 6 — DASHBOARD + PAPER                          [LATER]        │
│                                                                       │
│   Per-person view: predicted PAM + uncertainty band, and a            │
│   sensitivity-profile panel (bar/curve per direction showing          │
│   margin/slope/curvature) — the actual research deliverable           │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Why the Stage 4 fix matters (one-line version for your records)

Original plan: shift the model's internal number `z` directly along a
direction. Problem: `z` has no real-world units, so "shifting" it isn't
interpretable or defensible in a paper. Fix: shift the *real input
behavior numbers*, then let the full trained model re-process them
normally — every step stays in units you can explain and defend.

---

## What's genuinely dataset-specific vs. shared

| Component | Shared code? |
|---|---|
| Sequence builder, GRU architecture, personalization layer | Same code, different data — shared |
| Uncertainty mechanism | **Different per dataset** — CES uses the learned Gaussian head; StudentLife uses deterministic + validation-residual calibration. Document this divergence explicitly in the paper's methods section, don't hide it. |
| Sensitivity Engine | Same code for both, once built — this is what RQ5 (generalization) actually tests |

---

## Stage 4 empirical results

**Status of the numbers below:** the original exploratory tables are retained
for provenance but are superseded. In particular, the previously displayed
CES confidence intervals were invalid because their bootstrap statistic did
not match the reported window-weighted mean. The corrected implementation now
uses direction-specific empirical alpha ranges, uncertainty-weighted curve
fits, bootstrap intervals, and participant-clustered population intervals.
The corrected full CES and StudentLife runs are complete; the tables below are
the current results for reporting. Earlier exploratory values are retained in
git history only.

The implementation supports real input-space perturbations, all five
behavioral directions, per-window profiles, population aggregation, CSV/JSON
export, and summary plots. Alpha is swept over 21 values in the plausibility
guarded empirical range for each direction. These are model-sensitivity
results, not causal effects.

### Interaction hardening status

The interaction extension was hardened before beginning HBN/ADHD work:

| Improvement | Current status |
|---|---|
| Participant-clustered interaction bootstrap | Complete. Interaction responses preserve participant IDs, and pair-level aggregate CIs resample participant clusters using the same participant resampling unit as marginal profiles. |
| Empirical joint plausibility | Complete. Each pair uses Mahalanobis distance against its training joint distribution, with the 97.5th percentile of training distances as the cutoff rather than a fixed heuristic. |
| Aggregate interaction summaries | Complete. Pair summaries include mean, median, standard deviation, absolute interaction magnitude, large-interaction fraction, and clustered CI. |
| Seed-stability ranking check | Complete in code. The helper now reports mean pairwise Spearman agreement across all supplied seed rankings; the actual multi-seed experiment remains Stage 5 validation work. |

The focused sensitivity regression suite has 19 passing tests. Therefore the
Stage 4 interaction implementation is finalized at the code and regression
test level. Empirical claims about ranking stability still require rerunning
selected profiles with at least three independently trained seeds.

### CES (primary, calibration-ready)

The corrected run used all 3,599 test windows from 202 participants, producing
17,995 profile rows (3,599 windows x 5 directions), with PAM threshold 8.0.
The intervals below are participant-clustered bootstrap intervals for the
same window-weighted means shown in the table.

| Direction | Mean slope | 95% slope CI | Mean curvature | 95% curvature CI | Threshold crossings |
|---|---:|---:|---:|---:|---:|
| Activity | +0.5656 | [0.5039, 0.6233] | -0.1840 | [-0.2017, -0.1687] | 2056/3599 |
| Mobility | +0.1632 | [0.1308, 0.1944] | -0.0120 | [-0.0138, -0.0101] | 3599/3599 |
| Screen | -0.4781 | [-0.5001, -0.4567] | +0.1049 | [+0.0900, +0.1191] | 716/3599 |
| Sleep | -0.0622 | [-0.0677, -0.0561] | +0.0028 | [+0.0018, +0.0039] | 667/3599 |
| Social | unavailable | unavailable | unavailable | unavailable | unavailable |

CES has no features assigned to the social direction. Activity and mobility
show positive modeled sensitivity, screen and sleep show negative modeled
sensitivity, and the intervals bracket every reported mean. Mobility crosses
the threshold in every CES window under the empirical direction range; this
is a model-threshold result, not a claim that mobility causally raises PAM.

### StudentLife (preliminary)

The corrected run used all 59 test windows from 23 participants, producing
295 profile rows (59 windows x 5 directions). StudentLife uses a deterministic
GRU with validation residual standard deviation 3.1469; its results remain
preliminary. The PAM threshold was 12.5 and no direction crossed it.

| Direction | Features | Mean slope | 95% slope CI | Mean curvature | 95% curvature CI | Threshold crossings |
|---|---:|---:|---:|---:|---:|---:|
| Screen | 4 | +0.3220 | [+0.3060, +0.3374] | -0.0123 | [-0.0226, +0.0033] | 0/59 |
| Sleep | 2 | +0.1016 | [+0.0941, +0.1106] | -0.0136 | [-0.0192, -0.0085] | 0/59 |
| Social | 5 | +0.0824 | [+0.0673, +0.1003] | -0.0120 | [-0.0133, -0.0111] | 0/59 |
| Activity | 3 | +0.0449 | [+0.0342, +0.0542] | +0.0109 | [+0.0061, +0.0162] | 0/59 |
| Mobility | 3 | -0.2786 | [-0.2933, -0.2677] | +0.0330 | [+0.0317, +0.0362] | 0/59 |

The StudentLife ranking by absolute mean slope is screen, mobility, sleep,
social, then activity. No margin is available because no window crossed 12.5
within the empirical alpha range. This threshold is dataset-specific and
should not be compared directly with the CES threshold of 8.0.

### Stage 4 validity audits

Direction-map consistency was checked before cross-dataset interpretation.
CES mobility is dominated by location-derived features such as `loc_dist_*`,
whereas StudentLife mobility contains `gps_n`, `gps_distance_km`, and
`gps_unique_locations`. CES screen uses unlock-duration/count signals;
StudentLife screen uses phone-lock and app-usage signals. The directions are
therefore dataset-specific operationalizations and their slopes should be
reported independently, not as a direct cross-dataset effect comparison.

The training-data mobility/PAM correlation audit found no suspicious feature
with absolute correlation above 0.6: the largest absolute correlation was
approximately 0.157 for CES and 0.057 for StudentLife. This is a screening
check, not proof that leakage is impossible.

The corrected engine fits uncertainty-weighted quadratic curves using inverse
predictive variance and estimates 95% intervals by bootstrap. Pairwise
interaction summaries use the joint-minus-additive term, filter joint probes
against the empirical training joint range, and aggregate rows with
participant-clustered bootstrap intervals. Population intervals resample
participant-level clusters, pool all sampled windows, and recompute the same
window-weighted mean reported in the table rather than treating overlapping
windows as independent. StudentLife
deterministic/calibrated uncertainty must be passed explicitly before its
uncertainty-weighted intervals can be called calibrated.

Methodologically, this extends Individual Conditional Expectation (ICE) and
derivative-ICE analysis (Goldstein et al., 2015) from static tabular models to
a personalized temporal GRU: behavioral feature groups replace individual
features, perturbations are bounded by real training data, and uncertainty is
used in curve fitting and interval estimation. The ICEbox package is the
reference implementation for the classical ICE formulation; this project
implements the temporal, grouped-direction extension directly in PyTorch.

## Current implementation architecture

```text
processed sequence artifact
        |
        +--> feature schema + generated direction map
        |       |
        |       +--> empirical alpha bounds per direction
        |
        +--> dataset-specific trained twin
        |       +--> CES: uncertainty mean/logvar heads
        |       +--> StudentLife: deterministic mean + residual std
        |
        +--> batched sensitivity engine
                +--> real 7-day windows
                +--> direction-specific SD perturbations
                +--> windows x alpha values on CPU/GPU
                +--> weighted slope/curvature/margin
                +--> participant-cluster bootstrap CIs
                +--> per-window rows and population summaries
                        +--> CSV / JSON / PNG outputs
```

The shared engine is dataset-agnostic at the model-forward and aggregation
layers. Dataset-specific configuration is limited to feature schemas,
direction maps, target thresholds, checkpoint paths, and uncertainty source.

## Next action: Stage 5 validation

- Manually inspect curves for 3-5 individuals for discontinuities or
        physiologically implausible responses.
- Repeat selected marginal and interaction profiles across at least three
        training seeds and compare slope and absolute-interaction rankings;
        use the seed-stability helper to report rank consistency.
- Confirm that the same Sensitivity Engine runs on CES and StudentLife
        without dataset-specific code changes.
