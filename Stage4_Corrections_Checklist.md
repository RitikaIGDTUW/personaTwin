# Stage 4 — Technical Corrections Checklist

Work through in this order. Items 1-2 determine whether your current
results are trustworthy at all; do these before anything else, including
before showing the current numbers to anyone.

---

## 1. Direction-map consistency check (do this first — 5 minutes, decides everything else)

**Problem**: CES mobility slope +0.66, StudentLife mobility slope -0.26 —
opposite signs. Could be a real cohort difference, or could mean the two
"mobility" groups aren't measuring the same construct.

**Action**:
```python
import json
ces_map = json.load(open("data/interim/behavioral_direction_map_ces.json"))
sl_map = json.load(open("data/interim/behavioral_direction_map_studentlife.json"))

for direction in ["mobility", "screen"]:
    print(f"--- {direction} ---")
    print("CES:", ces_map.get(direction, []))
    print("StudentLife:", sl_map.get(direction, []))
```
**Check**: do the underlying raw signals actually describe the same
behavior (e.g. both dominated by distance-traveled, or both dominated by
location-cluster-count)? If the dominant feature type differs between the
two lists, **do not report this as a cross-dataset comparison** — report
each dataset's result independently and state explicitly in the paper that
"mobility" is operationalized differently per dataset, with the two
feature lists shown in an appendix table.

---

## 2. Verify plausibility clipping is per-direction, not a flat cap

**Problem**: doc says alpha swept over "guarded range [-2, 2]" — reads
like one fixed cap applied everywhere, not the per-direction empirical
range that was specified.

**Action**: in `SensitivityEngine.probe_direction`, confirm the alpha
range is computed per direction from the actual training data, not
hardcoded:
```python
def get_alpha_range(direction: str, train_df, direction_map) -> tuple[float, float]:
    cols = direction_map[direction]
    # observed range of the composite direction signal in training data,
    # in units of training-set SD (matching how the perturbation is applied)
    observed = train_df[cols].mean(axis=1)  # or however the direction composite is built
    z = (observed - observed.mean()) / observed.std()
    return (z.min(), z.max())  # NOT a hardcoded (-2, 2)
```
Re-run Stage 4 with this fix and check whether the CES mobility 75/100
threshold-crossing result survives. If the true observed range for
mobility features is narrower than ±2 SD, this number will drop — that's
expected and correct, not a bug.

---

## 3. Check for a leaky/near-duplicate feature in the mobility group

**Action**: for each feature in the mobility direction, compute raw
correlation with the PAM target on the training set:
```python
for col in direction_map["mobility"]:
    print(col, train_df[col].corr(train_df["pam"]))
```
Any single feature with |correlation| > ~0.6-0.7 is suspicious for a
sensing-derived proxy that mechanically tracks the target rather than
behaviorally influencing it. If found, remove it from the direction and
re-run before trusting the 75/100 crossing number.

---

## 4. Report distinct participant counts, not just window counts

**Action**: add to every results table:
```python
n_windows = len(test_windows)
n_participants = test_windows["uid"].nunique()
print(f"{n_windows} windows from {n_participants} distinct participants")
```
Do this for both datasets, but especially StudentLife's 59-window test
set — if that's fewer than ~15-20 distinct people, say so explicitly next
to the "0/100 threshold crossings" result, since a flat curve from a thin
sample is a data-limitation finding, not a behavioral one.

---

## 5. Uncertainty-weight the curve fit (upgrades this from a plain ICE/d-ICE
computation to something that uses what your twin actually offers)

**Problem**: margin/slope/curvature are currently point estimates fit to
the mean prediction curve only, ignoring the model's own predicted `std`
at each alpha. Since your CES model outputs calibrated uncertainty, use
it — this is also your strongest novelty lever (see below).

**Action**: in the curve-fitting step of `probe_direction`:
```python
import numpy as np

def fit_weighted_curve(alphas, means, stds, degree=2):
    weights = 1.0 / (np.array(stds) ** 2 + 1e-6)
    coeffs = np.polyfit(alphas, means, deg=degree, w=weights)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()
    d2poly = dpoly.deriv()
    slope_at_0 = dpoly(0)
    curvature_at_0 = d2poly(0)
    return poly, slope_at_0, curvature_at_0

def bootstrap_ci(alphas, means, stds, n_boot=200, degree=2):
    slopes, curvatures = [], []
    for _ in range(n_boot):
        sampled = np.array(means) + np.random.normal(0, stds)
        _, s, c = fit_weighted_curve(alphas, sampled, stds, degree)
        slopes.append(s); curvatures.append(c)
    return {
        "slope_ci": (np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)),
        "curvature_ci": (np.percentile(curvatures, 2.5), np.percentile(curvatures, 97.5)),
    }
```
Report slope/curvature with their bootstrap CI in every results table
from now on, not just the point value. A slope whose CI crosses zero
should not be reported as a directional finding.

---

## 6. Framing correction (not code, but affects what you write next)

State in your methods section: this Sensitivity Engine extends
Individual Conditional Expectation / derivative-ICE analysis (Goldstein
et al., 2015) — normally applied to static tabular models — to a
personalized temporal (GRU) digital twin, using (a) domain-grouped
behavioral directions instead of single raw features, (b)
plausibility-bounded perturbation ranges from real training data, and (c)
uncertainty-weighted curve fitting with bootstrap confidence intervals on
slope/margin/curvature, which the standard ICE/d-ICE formulation does not
provide. Cite Goldstein et al. and the `ICEbox` package explicitly.

---

## Order of operations

1 → 2 → 3 → 4 (validity — do all four before trusting any current number)
→ 5 (adds the uncertainty-weighted CIs — your actual novelty lever) → 6
(update the write-up framing) → **then** re-generate the CES and
StudentLife results tables from scratch and re-check whether the
mobility/screen sign-flip and the 75/100 crossing survive.
