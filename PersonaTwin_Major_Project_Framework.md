# PersonaTwin: A Personalized Multimodal Digital Twin with Continuous Behavioral-Sensitivity Profiles for Mental-State Dynamics

**BTech Major Project Framework — CES + StudentLife**

---

## 1. Problem Statement

Given an individual's longitudinal multimodal behavioral data (sleep, activity, mobility, social interaction, phone use) and affective/mental-state labels, build a personalized digital twin that:
1. Learns that individual's own temporal dynamics (not a population average)
2. Estimates their current latent mental-state
3. **Maps how predicted future stress/mood changes continuously along each behavioral dimension** — not just at a handful of discrete "what-if" points — including the *margin* (how much change is needed to matter), *slope* (current responsiveness), and *curvature* (diminishing/increasing returns) of that dimension
4. Reports this as an individual-specific, uncertainty-aware **behavioral-sensitivity profile**, rather than a single deterministic recommendation

---

## 2. Honest Related Work and the Precise Gap

This space has real, direct prior work on this exact dataset family. State it plainly rather than hoping a reviewer misses it:

| Work | What it already does |
|---|---|
| Jaques et al. 2017 — personalized multitask learning + domain adaptation for mood/health/stress | Personalized, individual-specific temporal prediction on mobile-sensing data — the foundational result this project's personalization layer builds on |
| CALM-Net (2019) | Auto-encoder + multitask personalization for StudentLife stress prediction |
| Branched CALM-Net (*Scientific Reports*, 2024) | Dynamic clustering + continuous adaptation, solving cold-start personalization on StudentLife specifically |
| Continual learning for personalized well-being monitoring (2025) | Ongoing extension of the same personalization lineage |
| GlyTwin (2025) | Digital twin + counterfactual reasoning to find **minimal discrete behavioral modifications** that shift a predicted health outcome (glucose, not mental state) |
| 2025 mental-health digital twin frameworks/reviews | Establish "digital twin + intervention simulation" as a named pattern for mental health, generally at a conceptual/review level |

**What none of them do**: none model the *continuous* shape of how a predicted mental-state outcome responds to a behavioral change — only discrete scenario points (GlyTwin-style) or point predictions (personalization lineage). None compute margin/slope/curvature diagnostics per individual per behavioral dimension. None apply the counterfactual-digital-twin mechanism (GlyTwin's contribution) to personalized multimodal mental-state data specifically.

**PersonaTwin's stated contribution**: transplanting a continuous sensitivity-landscape mechanism (margin, slope, curvature along a direction) — validated in an adjacent context in the reliability/robustness literature — into personalized mental-state digital twins, replacing discrete what-if ranking with a richer, individually diagnostic profile. This is an honest, specific, checkable claim — not "we invented personalized mental-health prediction."

---

## 3. Research Questions

- **RQ1**: Does personalizing the shared temporal model (adapter/branching, following the CALM-Net lineage) measurably improve next-state prediction over a population-level baseline, on CES and StudentLife?
- **RQ2**: Does multimodality (behavior + context + affect jointly) outperform behavior-only models?
- **RQ3**: For each behavioral dimension (sleep, activity, social interaction, mobility), can a continuous sensitivity function `S(individual, v, α)` be estimated reliably, and do its margin/slope/curvature diagnostics differ meaningfully across individuals?
- **RQ4**: Do individuals with different sensitivity *shapes* (e.g. gradual vs. cliff-like response to sleep change) differ in ways clinically/behaviorally interpretable (e.g. against PHQ/PAM subscales), without claiming causal effect?
- **RQ5**: Does the twin generalize across datasets (train CES, test StudentLife) where feature overlap permits?

---

## 4. Architecture

```
        CES + StudentLife
               │
      Data alignment layer
               │
     Multimodal features (behavior/context/affect)
               │
       Temporal representation (GRU)
               │
     ┌─────────┴─────────┐
     │  Shared dynamics   │   (population-level, from Jaques/CALM-Net lineage)
     └─────────┬─────────┘
               │
     Personalization layer (adapter / branching)
               │
     Personal latent state  z_t
               │
     ┌─────────┴──────────────────┐
     │                            │
Future-state predictor      Sensitivity Engine (NEW — core contribution)
     │                            │
Point prediction         For each behavioral direction v:
+ uncertainty              S(z_t, v, α), α ∈ [-a, a]
                            → margin, slope, curvature
                            → per-individual sensitivity profile
               │
     Dashboard: predicted state + uncertainty +
     ranked behavioral-sensitivity profile per dimension
```

---

## 5. Core Algorithm — Sensitivity Engine

```
Inputs: personalized twin (temporal model + personal latent state z_t),
        behavioral directions V = {sleep, activity, social, mobility, ...},
        future-state predictor f(z) → (state, uncertainty)

For each direction v in V:
  1. For α in [-a, ..., a] (e.g. -2h to +2h sleep change):
       z' ← apply_shift(z_t, v, α)     # shift along behavioral dimension
       S(v, α), unc(v, α) ← f(z')
  2. Fit smooth curve to {S(v, α)} across α
  3. Compute:
       margin(v)    = min |α| such that S(v, α) crosses model-estimated
                        threshold τ (a pattern reference point, not a clinical cutoff)
       slope(v)     = dS/dα at α=0 (current responsiveness)
       curvature(v) = d²S/dα² (diminishing/increasing returns)
  4. Store (margin, slope, curvature, uncertainty band) per direction

Output: ranked behavioral-sensitivity profile — e.g.
  "sleep shows a low margin and steep slope (high leverage);
   social interaction shows high margin, flat slope (low leverage 
   for this individual)"
```

This directly extends GlyTwin's discrete "which behavioral change helps" into a continuous, diagnosable shape — the delta stated in Section 2.

---

## 6. Datasets

- **Primary**: CES (College Experience Study)
- **Secondary/generalization**: StudentLife (for RQ5 cross-dataset check, and because it's the dataset the personalization-lineage papers above were built on — direct comparability)
- **Optional later**: Brighten / GLOBEM for external validation (per your original document's phased plan — keep this, it's sound)

---

## 7. Evaluation Plan

| RQ | Metric |
|---|---|
| RQ1 | Prediction error (population vs. personalized), following Branched CALM-Net's evaluation protocol for direct comparability |
| RQ2 | Ablation: behavior-only vs. +context vs. +affect |
| RQ3 | Held-out consistency of estimated `S(v, α)` curve shape (e.g. bootstrap stability); comparison against naive discrete GlyTwin-style scenario baseline |
| RQ4 | Correlation of sensitivity-shape clusters against PHQ/PAM subscales (correlational only — explicitly non-causal, per your original document's Section 5 caveat, which is correct and should stay) |
| RQ5 | Cross-dataset generalization error |

---

## 8. Deliverable

Dashboard per individual: current predicted state + uncertainty, and a **sensitivity profile panel** — bar/curve view per behavioral dimension showing margin/slope/curvature, so a user can see not just "more sleep helps" but "how much, how reliably, and whether returns diminish."

**Dashboard language spec** (lock this in before building): all labels must use pattern/sensitivity framing, not clinical-alert framing. Concretely:
- ✅ "Model-flagged low-mood pattern" — ❌ not "CRITICAL ALERT – LOW MOOD"
- ✅ "crosses model-estimated threshold" — ❌ not "crosses safety limit"
- ✅ "high sensitivity to sleep reduction detected" — ❌ not "early-warning system flags depressive/anxious episode"

This is an **explanatory tool** that surfaces what the model predicts under behavioral scenarios — not a clinical monitoring or alerting system.

---

## 9. Suggested Timeline

- Weeks 1-2: Data alignment + multimodal feature pipeline (CES, then StudentLife)
- Weeks 3-4: Shared temporal model (GRU baseline) → RQ1/RQ2 groundwork
- Weeks 5-6: Personalization layer (adapter/branching, reusing CALM-Net-style design) → RQ1
- Weeks 7-9: Sensitivity Engine (core new contribution) → RQ3
- Week 10: RQ4 correlational analysis
- Week 11: RQ5 cross-dataset check
- Week 12: Dashboard + writing buffer

---

## 10. Key Risks

- **Sensitivity curves may be noisy per individual** with limited longitudinal data per person → mitigate with uncertainty bands and bootstrap confidence, report honestly rather than overclaiming precision
- **Reviewer asks "how is this different from GlyTwin/CALM-Net"** → Section 2's table answers this directly; keep it in the paper's related-work section verbatim in spirit
- **Causal misreading risk** → keep your original document's explicit non-causal language throughout ("the model predicts a lower state under this scenario," never "this intervention reduces stress") — this was already correctly handled in your draft, don't lose it in revision
- **Clinical-label drift risk** → dashboard and any written description must never use clinical-alert framing ("CRITICAL ALERT," "safety limit," "flags depressive/anxious episode"). Use pattern/sensitivity language throughout (see Section 8 language spec). This is an explanatory research tool, not a diagnostic or monitoring system — conflating the two creates ethical and regulatory problems and will be challenged in defense.
