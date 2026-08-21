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
│ STAGE 4 — SENSITIVITY ENGINE (core contribution)   [CORRECTED —     │
│                                                      rerun required] │
│                                                                       │
│   For a real person's real 7-day window, and one behavioral          │
│   direction (sleep / activity / social / mobility / screen):         │
│                                                                       │
│   1. Take the REAL input window x  (not z — this was the fix)        │
│   2. For alpha in plausible range for this direction:                │
│        x' = x + alpha * (per-feature training-set SD),               │
│             applied only to that direction's feature columns          │
│        clip alpha range to the empirically observed range for        │
│        that direction (plausibility guard)                            │
│   3. Run the FULL trained model forward on x'  →  (mean, std)         │
│   4. Collect (alpha, mean, std) across the range                      │
│   5. Fit a smooth curve → compute:                                     │
│        slope     = responsiveness near current state                  │
│        curvature = diminishing / increasing returns                   │
│        margin    = smallest alpha crossing a chosen PAM threshold     │
│   6. Repeat for all 5 directions → per-person sensitivity profile      │
│                                                                        │
│   Run on CES first (calibration-ready). StudentLife results carry     │
│   the same "preliminary" label as its uncertainty numbers.             │
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
not match the reported window-weighted mean. The corrected implementation now uses
direction-specific empirical alpha ranges, uncertainty-weighted curve fits,
bootstrap intervals, and participant-clustered population intervals. CES and
StudentLife results must be regenerated before final reporting.

The implementation supports real input-space perturbations, all five
behavioral directions, per-window profiles, population aggregation, CSV/JSON
export, and summary plots. Alpha is swept over 21 values in the plausibility
guarded empirical range for each direction. These are model-sensitivity
results, not causal effects.

### CES (primary, calibration-ready)

The superseded exploratory results used 100 real test windows, producing 500
profile rows (100 windows x 5 directions), with PAM threshold 8.0. CES has
3,599 test windows from 202 distinct participants. Its mobility group has 118
location-derived features, while its screen group has 56 unlock-derived
features. These operationalizations should not be treated as equivalent to
the corresponding StudentLife groups.

| Direction | Mean slope | Slope SD | Mean curvature | Threshold crossings |
|---|---:|---:|---:|---:|
| Mobility | +0.6628 | 0.1395 | +0.3576 | 75/100 |
| Activity | +0.3685 | 0.1854 | +0.1965 | 26/100 |
| Screen | -0.3924 | 0.0850 | +0.1059 | 4/100 |
| Sleep | -0.0636 | 0.0232 | +0.0036 | 2/100 |
| Social | unavailable | unavailable | unavailable | 0/100 |

CES has no features assigned to the social direction. Mobility is the
strongest positive modeled sensitivity, activity is also positive, screen
is negative, and sleep is nearly flat. Margins are reported only for windows
that cross the threshold within the alpha range.

### StudentLife (preliminary)

The superseded exploratory results used all 59 available test windows,
producing 295 profile rows
(59 windows x 5 directions). StudentLife has features in all five
directions and 23 distinct participants. Its uncertainty and sensitivity
results remain preliminary.

| Direction | Features | Mean slope | Slope SD | Mean curvature | Threshold crossings |
|---|---:|---:|---:|---:|---:|
| Social | 5 | +0.4287 | 0.0391 | +0.0062 | not reached |
| Screen | 4 | +0.3203 | 0.0372 | -0.0131 | not reached |
| Mobility | 3 | -0.2636 | 0.0218 | +0.0053 | not reached |
| Activity | 3 | +0.0752 | 0.0263 | +0.0034 | not reached |
| Sleep | 2 | -0.0184 | 0.0287 | -0.0004 | not reached |

The StudentLife exploratory threshold was calibrated to 12.5, approximately the training
target's 90th percentile. No direction crossed 12.5 within [-2, 2], so all
StudentLife margins are unavailable rather than zero. This threshold should
not be compared directly with the CES threshold of 8.0.

The StudentLife ranking by absolute mean slope is social, screen, mobility,
activity, then sleep. The near-zero sleep slope and small curvature indicate
little modeled response to the sleep direction in this preliminary run.

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
predictive variance and estimates 95% intervals by bootstrap. Population
intervals resample participant-level clusters, pool all sampled windows, and
recompute the same window-weighted mean reported in the table rather than
treating overlapping windows as independent. StudentLife
deterministic/calibrated uncertainty must be passed explicitly before its
uncertainty-weighted intervals can be called calibrated.

Methodologically, this extends Individual Conditional Expectation (ICE) and
derivative-ICE analysis (Goldstein et al., 2015) from static tabular models to
a personalized temporal GRU: behavioral feature groups replace individual
features, perturbations are bounded by real training data, and uncertainty is
used in curve fitting and interval estimation. The ICEbox package is the
reference implementation for the classical ICE formulation; this project
implements the temporal, grouped-direction extension directly in PyTorch.

## Next action: Stage 5 validation

- Manually inspect curves for 3-5 individuals for discontinuities or
        physiologically implausible responses.
- Repeat selected profiles across training seeds and compare slope rankings.
- Confirm that the same Sensitivity Engine runs on CES and StudentLife
        without dataset-specific code changes.
