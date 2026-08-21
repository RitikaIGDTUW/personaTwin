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
│ STAGE 4 — SENSITIVITY ENGINE (core contribution)   [NEXT — corrected │
│                                                      design below]   │
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

## Immediate next action

Implement Stage 4 exactly as specified above (input-space perturbation,
full forward pass, plausibility-clipped alpha range) on the CES twin
only. Once that produces sane per-person profiles on a handful of test
individuals, extend to StudentLife with the preliminary label carried
through, then move to Stage 5.
