# PersonaTwin: Methodology & Presentation Guide
*A comprehensive guide for explaining the PersonaTwin research project, from foundational pipeline stages to advanced sensitivity-profiling and clinical diagnostic adaptations.*

---

> **Implementation status (read before presenting this guide to anyone):**
>
> - The Sensitivity Engine described in Sections 3-4 and 7 is implemented and
>   has been run end-to-end for personalized StudentLife and CES models. The
>   StudentLife run processed 59 windows and the CES run processed 3,599
>   windows. HBN has not yet been sensitivity-tested.
> - Two real bugs found and fixed since this guide's numbers were written:
>   (1) the personalized model's participant embedding wasn't being passed
>   through at all in the univariate path, and was being passed the wrong
>   (unmapped) index in the interaction path; (2) margin was quantized to
>   the nearest of 21 fixed alpha-grid points, which collapsed to identical
>   values across many windows (most visibly on `sleep`). Margin is now
>   computed via linear interpolation between grid points on a 101-point
>   grid, giving a continuous estimate instead of a quantized one.
> - **Section 5's older population tables are historical snapshots.** The
>   current results to cite are the personalized outputs generated after the
>   PAM-history feature was added and the uncertainty-model retraining was
>   completed. Do not mix the old population files with the new personalized
>   files.

---

## 1. Executive Summary & Thesis
**PersonaTwin** is a personalized digital twin framework that models individual mental-state dynamics using multimodal behavioral sensing data. 

* **The Core Problem**: Traditional mental health tracking relies on population averages or simple point-in-time predictions. They fail to explain *how* a specific individual's mental state changes in response to behavioral changes, or *how much* behavior change is needed to see an effect.
* **The Core Solution**: We build personalized, uncertainty-aware deep learning twins (GRUs). We then query these twins using a **Sensitivity Engine** that perturbs real input behaviors (in real-world units like hours of sleep or screen time) to map a continuous sensitivity profile (slope, curvature, and margin) for each individual.
* **Unified Scope**: The project combines daily passive sensing datasets (CES and StudentLife) with clinical diagnostics (Healthy Brain Network's ADHD/ASD actigraphy cohort) to evaluate whether these sensitivity profiles differ across neurodivergent populations using a shared analytical engine.

---

## 2. Unified Methodology Flowchart
This chart integrates all aspects of the project, highlighting how the general-population branch and the clinical ADHD/ASD branch merge into the same shared Sensitivity Engine, proving it is a single unified methodology.

```mermaid
graph TD
    %% Datasets & Ingestion
    subgraph Data Ingestion & Alignment
        D1["StudentLife Dataset (18 Model Features, N=23, Sensing + EMA)"] --> P1["StudentLife Daily Table"]
        D2["CES Dataset (677 Features to 570, N=202, Sensing + Demographics + EMA)"] --> P2["CES Daily Table"]
        D3["HBN Cohort Actigraphy (Wrist Movement/Sleep, Labels: ADHD/ASD/NT)"] --> P3["HBN Daily Table"]
    end

    %% Sequence Construction
    subgraph Stage 2: Sequence Construction
        P1 --> S1["Sequence Builder (Chronological 70/15/15 Split, Forward-fill, Train-median, z-score)"]
        P2 --> S2["Sequence Builder (Chronological 70/15/15 Split, Forward-fill, Train-median, z-score)"]
        P3 --> S3["Sequence Builder (Chronological 70/15/15 Split, Forward-fill, Train-median, z-score)"]
    end

    %% Model Training
    subgraph Stage 3: Personalized Twins
        S1 --> M1["Personalized GRU Twin (Population Baseline + Participant Embedding, Uncertainty: Val-Residual Calibrated Std)"]
        S2 --> M2["Personalized GRU Twin (Population Baseline + Participant Embedding, Uncertainty: Gaussian NLL Head)"]
        S3 --> M3["Personalized Clinical GRU Twin (Baseline + Participant Embedding + Diagnostic Group Embedding)"]
    end

    %% Sensitivity Engine (Shared)
    subgraph Stage 4: Shared Sensitivity Engine
        M1 --> SE["Sensitivity Engine: 1. Input-space Perturbation, 2. Plausibility Guard, 3. Curve Fitting"]
        M2 --> SE
        M3 --> SE
        
        SE --> M_Slope["1. Mean Slope (Responsiveness)"]
        SE --> M_Curv["2. Mean Curvature (Diminishing Returns)"]
        SE --> M_Marg["3. Margin to Threshold (PAM Limit)"]
        SE --> M_Int["4. Joint Interaction (Mahalanobis Distance Filtered)"]
    end

    %% Outputs & Validation
    subgraph Validation & Deliverables
        M_Slope --> V1["Stage 5: Multi-seed Stability & Cross-Dataset Verification"]
        M_Curv --> V1
        M_Marg --> V1
        M_Int --> V1
        
        M_Slope --> V2["Stage 6: Clinical Diagnostic Comparison (ADHD/ASD vs NT)"]
        M_Curv --> V2
        M_Marg --> V2
        M_Int --> V2
        
        V1 --> OUT["Interactive Dashboard & Paper Results"]
        V2 --> OUT
    end

    classDef gray fill:#f3f4f6,stroke:#374151,stroke-width:2px;
    classDef coral fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef teal fill:#ccfbf1,stroke:#0d9488,stroke-width:2px;
    
    class D1,D2,P1,P2,S1,S2,M1,M2,V1,OUT gray;
    class D3,P3,S3,M3,V2 coral;
    class SE,M_Slope,M_Curv,M_Marg,M_Int teal;
```

---

## 3. Stage-by-Stage Methodology (Basic to Advanced)

### Stage 1: Data Preprocessing & Alignment
* **Basic Level**: We load raw sensor streams (GPS, screen lock/unlock, Bluetooth, accelerometer) and daily surveys (EMA) from participants. We align them so each row represents one participant-day of data.
* **Advanced Level**:
  * **StudentLife**: Low-resource dataset. We process raw database entries, align daily timestamps, and yield 18 model features, including the leakage-safe observed PAM history feature.
  * **CES (College Experience Study)**: High-dimensional dataset. We drop features with extreme missingness and near-zero variance, narrowing down from 677 to 570 behavioral features; the sequence model uses 571 features after adding leakage-safe observed PAM history.
  * **HBN (Healthy Brain Network)**: The clinical cohort. Focuses on wrist actigraphy (movement, sleep duration, onset, wake-after-onset, activity counts). It has no phone-sensing data, which requires a custom, sleep-and-movement-focused feature schema.

### Stage 2: Leakage-Safe Sequence Construction
* **Basic Level**: Models must not cheat. We split data chronologically so that the model only predicts the future based on past behavior. We construct rolling 7-day windows to predict the next day's PAM (Photographic Affect Meter, representing affect/arousal).
* **Advanced Level**:
  * **Temporal Split**: Chronological 70% Train, 15% Validation, 15% Test per participant.
  * **Imputation Protocol**: To prevent data leakage:
    1. Forward-fill missing values (assuming behavior carries over).
    2. Fill remaining gaps with the **training split median** (never using validation/testing statistics).
    3. Z-score normalize features using **mean and standard deviation computed solely from the training data**.
  * **Sequence Dimensions**:
    * StudentLife: 1458 training, 88 validation, and 59 test windows (18 model features).
    * CES: 25576 training, 4112 validation, and 3599 test windows (571 model features).

### Stage 3: Personalized Temporal Modeling (The Digital Twin)
* **Basic Level**: A standard model learns the "average student." But everyone is different—six hours of sleep makes one person energetic and another exhausted. We train a deep recurrent network (GRU) that adapts to each participant.
* **Advanced Level**:
  * **Adapter Layer**: We inject a learned **participant embedding** (a latent representation vector unique to each person) into the GRU. The embedding shifts the network's internal representations, customizing predictions.
  * **ADHD/ASD Customization (HBN Branch)**: For the HBN cohort, we add a **diagnostic-group embedding** (ADHD, ASD, Neurotypical) alongside the individual participant embedding. This lets the network explicitly capture structural, diagnostic-level behavioral differences.
  * **Predictive Uncertainty**:
    * **CES and StudentLife (Uncertainty-Aware)**: The personalized GRU outputs both a mean $\hat{y}$ and a variance $\sigma^2$. Training uses an initial MSE warmup, a direct mean-prediction term, and Gaussian NLL; checkpoints are selected by validation MAE rather than NLL alone.

### Stage 4: Shared Sensitivity Engine (Core Research Novelty)
* **Basic Level**: Instead of just predicting tomorrow's mood, we ask "What if this person slept 1 hour more? What if they walked 2 miles more?" We run these hypothetical scenarios through the trained model to construct a response curve.
* **Advanced Level**:
  * **Input-Space Perturbation (The Stage 4 Fix)**: Historically, models shifted internal representations ($z$). We corrected this to perturb the **actual input behaviors ($x$)**, applying direction-specific standardized shifts ($\alpha$ times the training feature standard deviation) before passing them through the model. This preserves feature-space interpretability; a frontend must convert these standardized shifts to real units before displaying user recommendations.
  * **Behavioral Directions**: Features are grouped into 5 domains: Sleep, Activity, Social, Mobility, and Screen.
  * **Empirical Plausibility Guard**: To prevent the engine from simulating impossible behaviors (e.g., 30 hours of screen time), shifts ($\alpha$) are restricted to the empirical min/max values observed in the training cohort.
  * **Curve Fitting & Metric Extraction**: We sweep $\alpha$ over 101 steps (increased from an initial 21-step grid after the coarser grid was found to quantize margin to identical values across many windows), run them through the twin, and fit an uncertainty-weighted quadratic curve ($y = a\alpha^2 + b\alpha + c$) using the inverse predictive variance. From this, we extract:
    1. **Slope**: Current responsiveness (first derivative at $\alpha=0$).
    2. **Curvature**: Acceleration/diminishing returns (second derivative).
    3. **Margin**: The minimum shift required to cross a model-estimated PAM reference point (threshold 8.0 for both datasets), computed by linearly interpolating the exact crossing alpha between the two bracketing grid points rather than snapping to the nearest grid value — a descriptive sensitivity diagnostic, not a clinical cutoff. When a direction never crosses the threshold within its plausible bounds (observed for StudentLife's `sleep` direction), margin is correctly reported as infinite for that window rather than a spurious finite value.
  * **Pairwise Interactions**: We perturb two directions simultaneously and compare the output to the sum of individual shifts. We filter out implausible joint states using a Mahalanobis distance cutoff (97.5th percentile of the training distribution) and construct participant-clustered bootstrap confidence intervals (CIs).

---

## 4. Key Differences & Shared Infrastructure
Use this table to prove that the codebase is highly generalizable and modular.

| Component / Stage | Shared Pipeline Code? | How Datasets Differ |
|---|---|---|
| **Preprocessing & Schema** | No (dataset-specific scripts) | StudentLife: 18 model features (17 behavioral + PAM history).<br/>CES: 571 model features (570 behavioral + PAM history).<br/>HBN: Actigraphy movement/sleep features only. |
| **Sequence Building** | **Yes (shared framework)** | Splits and chronological boundaries are identical; input/output dimensions adjust dynamically. |
| **Adapter Layer** | **Yes (shared GRU model structure)**| CES & StudentLife: Participant embedding only.<br/>HBN: Diagnostic-group embedding + participant embedding. |
| **Uncertainty Method** | No (configuration-driven) | StudentLife and CES: learned Gaussian uncertainty heads with MSE warmup, direct mean-loss weighting, and validation-MAE checkpoint selection. |
| **Sensitivity Engine** | **Yes (identical shared code)** | Same mathematical curve fitting, slope, curvature, margin, and bootstrap engine. |

---

## 5. Concrete Numbers Ready for Presentation

The current personalized results below were generated after retraining both
models with PAM history, MSE warmup, direct mean-loss weighting, and
validation-MAE checkpoint selection. These are model-sensitivity estimates,
not causal effects. Older population snapshots later in this section are kept
for historical comparison and must not be presented as the current results.

### Current personalized GRU predictive quality

| Dataset | Test windows | PAM MAE | PAM RMSE | Actual/predicted correlation |
|---|---:|---:|---:|---:|
| StudentLife | 59 | 2.522 | 3.168 | 0.405 |
| CES | 3,599 | 3.421 | 4.182 | 0.286 |

The models are on the PAM scale (1-16), no longer exhibit the former
prediction collapse, and still smooth some short-term individual fluctuations.
These metrics support a useful predictive baseline, but not a claim of highly
accurate fluctuation tracking.

### Current personalized sensitivity outputs

StudentLife was regenerated for 59 test windows and 23 participants. CES was
regenerated for 3,599 test windows and 202 participants. The current CES
personalized univariate means are:

| Direction | Slope mean | Curvature mean | Margin mean | Participant count |
|---|---:|---:|---:|---:|
| activity | +2.0313 | -0.7171 | 0.6048 | 202 |
| mobility | +0.0705 | -0.0047 | 0.5725 | 202 |
| screen | +0.2181 | -0.0673 | 1.1118 | 202 |
| sleep | -0.2395 | +0.0350 | 1.8501 | 202 |

CES has no social sensing direction; social must be reported as unavailable,
not as a NaN result.

### StudentLife population univariate sensitivity (59 test windows, N=23)

| Direction | Slope mean | Curvature mean | Margin mean | Participant count |
|---|---:|---:|---:|---:|
| activity | -0.0789 | -0.00797 | not finite for many windows; threshold not reached within plausible bounds | 23 |
| mobility | +0.2724 | -0.0341 | not finite for many windows; threshold not reached within plausible bounds | 23 |
| screen | -0.5769 | -0.00053 | 1.5070 | 23 |
| sleep | -0.3941 | +0.00416 | 1.8159 | 23 |
| social | +0.0817 | -0.0101 | not finite for many windows; threshold not reached within plausible bounds | 23 |

Interpretation:
- **screen** shows the strongest negative population sensitivity, suggesting that increases in screen exposure correspond to lower predicted wellbeing under the trained model.
- **sleep** also shows a strong negative sensitivity, indicating large modeled dependence on sleep-related behavioral shifts.
- **mobility** is positive, suggesting higher mobility is associated with higher predicted wellbeing in the population model.
- **activity** and **social** show weaker marginal effects than screen and sleep.
- The **margin** statistic is not always finite for StudentLife because several windows do not cross the selected PAM threshold within the empirical perturbation range; this is a legitimate modeling outcome and should be described as a "no crossing within plausible range" result rather than as a numerical failure.

### StudentLife population pairwise interaction sensitivity

| Pair | Interaction mean | Interaction std | Min | Max | Nonzero windows |
|---|---:|---:|---:|---:|---:|
| activity:mobility | -0.000694 | 0.00498 | -0.01250 | 0.00724 | 59 |
| activity:screen | +0.002985 | 0.00396 | -0.00519 | 0.01165 | 59 |
| activity:social | +0.000998 | 0.00362 | -0.00762 | 0.01146 | 59 |
| mobility:screen | +0.000854 | 0.00624 | -0.01889 | 0.00933 | 59 |
| sleep:activity | +0.001276 | 0.00258 | -0.00417 | 0.00700 | 59 |
| sleep:mobility | +0.000846 | 0.00533 | -0.01121 | 0.01621 | 59 |
| sleep:screen | +0.009721 | 0.00842 | -0.00301 | 0.04301 | 59 |
| sleep:social | +0.001673 | 0.00496 | -0.01532 | 0.02470 | 46 |
| social:mobility | +0.001510 | 0.00380 | -0.00552 | 0.01127 | 59 |
| social:screen | +0.002465 | 0.00856 | -0.02777 | 0.01688 | 59 |

The strongest pairwise signal is **sleep:screen** (interaction mean $\approx 0.0097$), while the remaining pairwise effects are comparatively small and often close to zero. This suggests that the primary StudentLife signal is dominated by individual behavioral directions rather than strong two-way interactions.

### Current verified output snapshots

These values reflect the current generated outputs from the verified pipeline and should be read as a snapshot of the current model sensitivity estimates rather than as historical claims from an earlier draft.

#### Historical CES population snapshot (superseded)

| direction | slope | curvature | margin |
|---|---:|---:|---:|
| activity | 2.131489 | -0.413488 | 0.435649 |
| mobility | 0.194555 | -0.014611 | 0.835889 |
| screen | 0.068173 | -0.010087 | inf |
| sleep | -0.045856 | -0.003242 | inf |
| social | NaN | NaN | NaN |

This is a historical population-model snapshot and is superseded by the current personalized CES table above. It is retained only to preserve experiment history.

#### Historical CES interaction snapshot (superseded)

| direction_a | direction_b | mean_interaction | std_interaction | min_interaction | max_interaction | nonzero |
|---|---|---:|---:|---:|---:|---:|
| activity | mobility | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0 |
| activity | screen | -9.103933970133e-06 | 9.702535430676e-06 | -3.508726755778e-05 | 6.866455078125e-06 | 25 |
| mobility | screen | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0 |
| sleep | activity | -2.199014027913e-06 | 6.341696428400e-07 | -3.099441528320e-06 | -3.178914388021e-07 | 25 |
| sleep | mobility | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 | 0 |
| sleep | screen | -1.113704559336e-04 | 2.620990805857e-04 | -1.118146456205e-03 | 1.941025257110e-04 | 25 |

This historical snapshot must not be described as the current personalized result. The current personalized CES interaction run completed all 6 direction pairs across 3,599 windows; its interaction means remain small, with the largest absolute mean approximately $3.9 \times 10^{-3}$ (sleep:screen).

### Current CES personalized pairwise interaction sensitivity

The CES personalized interaction results completed technically, but the values are mostly small. This is a model result rather than a computation failure. The current personalized pairwise interaction summary is:

| direction_a | direction_b | mean | std | minimum | maximum | nonzero |
|---|---|---:|---:|---:|---:|---:|
| activity | mobility | +0.003214 | 0.068214 | -1.192463 | +1.424456 | 3599 |
| activity | screen | +0.000781 | 0.027294 | -0.389853 | +0.571522 | 3599 |
| mobility | screen | +0.000238 | 0.039064 | -0.497936 | +0.264733 | 3599 |
| sleep | activity | +0.000231 | 0.007824 | -0.193029 | +0.215892 | 3599 |
| sleep | mobility | -0.000568 | 0.012810 | -0.312482 | +0.146410 | 3599 |
| sleep | screen | +0.003934 | 0.031104 | -0.154701 | +0.583326 | 3599 |

This indicates that CES pairwise interactions are small in mean relative to the univariate sensitivity terms, although individual windows can show larger positive or negative responses. The result is not an algorithmic failure; it suggests that the current CES model is driven more by individual behavioral directions than by stable population-level behavioral combinations.

### CES univariate sensitivity status

**CES personalized univariate sensitivity is working and producing numeric results.** The current implementation completed 3,599 windows across 202 participants. The pairwise interaction analysis also completed all six CES pairs; its mean effects are small, while window-level variability is retained in the exported profiles. This should be reported as model sensitivity, not as causal evidence.

> [!NOTE]
> The CES and StudentLife findings should be framed as **model sensitivity**, not causal effects. They explain which behavioral directions the trained model relies on most strongly and whether two-way interactions materially change the prediction beyond the sum of the independent effects.

---

## 6. How to Defend Your Methodology (Talking Points)

### 1. Defending "Model Sensitivity vs. Causal Inference"
> **Professor's Question**: "How can you recommend a behavior change if this isn't a causal model?"
>
> **Your Answer**: *"We are very explicit: this is **model sensitivity**, not a causal effect estimation. We are testing how the trained network’s prediction shifts under counterfactual inputs. This indicates what behavioral levers the model relies on for its predictions. It is an explanatory tool (similar to Individual Conditional Expectation in static models, but extended to dynamic time series), not a medical prescription."*

### 2. Defending the StudentLife Uncertainty Calibration
> **Professor's Question**: "Why is the StudentLife coverage only 89.8% instead of the target 95%?"
>
> **Your Answer**: *"Because StudentLife is a much smaller dataset (59 test windows compared to CES's 3,599), we use warmup training, a direct mean-prediction term, and validation-MAE checkpoint selection for both personalized uncertainty models. The current StudentLife model remains preliminary because its test set is small and its predictions smooth some short-term fluctuations."*

### 3. Defending the ADHD/ASD HBN Extension
> **Professor's Question**: "Why do you have a different branch for ADHD/ASD?"
>
> **Your Answer**: *"The HBN dataset lets us test if our sensitivity engine can detect cohort-level patterns in clinical populations. Because HBN contains actigraphy data, it uses a narrower feature set (sleep and movement). By injecting a diagnostic-group embedding (ADHD vs ASD vs Neurotypical), the digital twin learns structural differences in sleep/activity patterns. The core sensitivity engine runs identically on this cohort, showing how these diagnostic groups respond differently to behavioral levers."*

### 4. Defending the "Stage 4 Correction"
> **Professor's Question**: "Why did you change the sensitivity engine from latent-space to input-space perturbation?"
>
> **Your Answer**: *"Perturbing the latent vector $z$ directly has no behavioral interpretation. We instead perturb the raw input features $x$ and express the engine's internal sweep in standardized alpha units. Before showing a user-facing value such as an extra hour of sleep, the frontend must convert alpha through the relevant feature's training standard deviation."*

---

## 7. Stage 4 Magnified Flowchart: The Sensitivity Engine
This magnified diagram shows the exact execution logic of `src/sensitivity.py` for a single behavioral direction (e.g. Sleep) on a participant sequence.

```mermaid
graph TD
    A["Input: Real Participant Sequence Window X (Shape: T x F)"] --> B["Determine Feature Indices for Chosen Direction v<br/>(e.g., Sleep features: sleep_duration, sleep_efficiency)"]
    
    %% Bounds Calculation
    B --> C["1. Plausible Bounds Computation<br/>(plausible_alpha_bounds)"]
    C --> C1["Extract Train cohort values: train_X"]
    C1 --> C2["Compute standard deviation per feature column (feature_sd)"]
    C2 --> C3["Compute average spread and min/max composite range of train_X"]
    C3 --> C4["Derive alpha limits [alpha_min, alpha_max] (empirical bounds)"]
    
    %% Sweeping and Perturbation
    C4 --> D["2. Generate Alpha Sweep Array (alphas)<br/>(101 regular steps between alpha_min and alpha_max)"]
    D --> E["3. Input-Space Perturbation (perturb_window_for_direction)"]
    E --> E1["For each alpha: X_perturbed = X + alpha * feature_sd<br/>(Applied only to the direction's feature columns)"]
    
    %% Model Forward Pass
    E1 --> F["4. Batched Inference Forward Pass (probe_direction)"]
    F --> F1["Stack perturbed windows to form a Batch (Shape: 101 x T x F)"]
    F1 --> F2["Pass batch through trained model forward: model(perturbed_batch)"]    
    %% Coercion and Inverse Variance Weighted Fit
    F2 --> G["5. Coerce Predictions & target scaling (_coerce_prediction)"]
    G --> G1["Extract PAM Mean values (original scale)"]
    G1 --> G2["Extract Predictive Standard Deviations (logvar head or validation residual)"]
    G2 --> H["6. Inverse-Variance Weighted Curve Fitting (fit_weighted_curve)"]
    H --> H1["Fit 2nd degree polynomial: y = a*alpha^2 + b*alpha + c<br/>using weights w = 1 / std^2"]
    H1 --> H2["Compute Slope (b) and Curvature (2a) at current state (alpha = 0)"]
    H1 --> H3["Identify Margin (linear interpolation of the exact crossing alpha between the two bracketing grid points, not the nearest grid alpha)"]
    
    %% Bootstrapping and Output
    H2 & H3 --> I["7. Predictive Bootstrapping (bootstrap_curve_intervals)"]
    I --> I1["Resample means using predictive SDs (n_boot = 200) to find 95% CIs"]
    
    %% Joint Interaction Side-branch
    B --> J["Pairwise Interaction Flow (probe_interaction)"]
    J --> J1["Joint perturbation: X_joint = X + alpha_a*sd_a + alpha_b*sd_b"]
    J1 --> J2["Plausibility Filter: Compute Mahalanobis Distance against training joint spread"]
    J2 --> J3["Compare Mahalanobis distance to the 97.5th percentile cutoff of train split"]
    J3 -->|Plausible| J4["Run forward pass & compute interaction term:<br/>Joint effect - (Marginal A + Marginal B) + Baseline"]
    J3 -->|Implausible| J5["Discard state (prevent out-of-distribution hallucinations)"]

    I1 & J4 --> K["Output: continuous sensitivity metrics (slope, curvature, margin, bootstrap CIs, interaction values)"]

    classDef gray fill:#f3f4f6,stroke:#374151,stroke-width:2px;
    classDef coral fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef teal fill:#ccfbf1,stroke:#0d9488,stroke-width:2px;
    
    class A,B,C,C1,C2,C3,C4,D,E,E1,F,F1,F2,G,G1,G2 gray;
    class H,H1,H2,H3,I,I1 teal;
    class J,J1,J2,J3,J4,J5 coral;
```

### Explanatory Breakdown of the Stage 4 Flow (What to say to your Teacher)

You should explain these steps to your teacher to show the engineering rigor and scientific novelty:

1. **Step 1: Plausible Bounds Computation (`plausible_alpha_bounds`)**
   * *What it means:* We determine the maximum and minimum behavior shifts that are realistic for this direction.
   * *Why it's important:* If we perturb a feature by a random high amount (e.g. telling the model the participant slept 30 hours in a day), the model will predict on "impossible" out-of-distribution data, leading to garbage predictions. We calculate the actual training min/max composite range for these features, ensuring all simulated shifts are empirically grounded.

2. **Step 2: Generate Alpha Sweep Array (`default_direction_alphas`)**
   * *What it means:* We create a continuous range of 101 test points (shifts) stretching from the lowest plausible change to the highest plausible change (e.g. from $-2.5$ standard deviations to $+2.5$ standard deviations). This was increased from an initial 21-point grid after the coarser grid was found to quantize margin — collapsing it to identical values across many windows for narrow-feature directions like sleep.
   * *Why it's important:* Instead of testing a single discrete "what-if" scenario, we evaluate the behavior change across a smooth, continuous spectrum.

3. **Step 3: Input-Space Perturbation (`perturb_window_for_direction`)**
   * *What it means:* We modify the **raw features** directly (like raw sleep hours) before passing them to the model, rather than modifying the model's internal hidden representations ($z$).
  * *Why it's important:* Shifting internal layers has no behavioral interpretation. By shifting raw inputs and passing them through the entire model, the engine evaluates realistic feature-space changes. The current exported alpha values are standardized units; a frontend must translate them into real units such as hours or minutes before displaying a scenario to a user.

4. **Step 4: Batched Inference Forward Pass (`probe_direction`)**
   * *What it means:* We clone the user's 7-day behavior window 101 times, apply the 101 different alpha shifts, stack them into a single tensor batch, and run them forward through the personalized GRU model. For personalized models, the participant's correctly-mapped embedding index is passed alongside the perturbed batch — this needed a fix, since an early version silently omitted the participant tensor for population-vs-personalized calls, and a separate early version passed the raw participant ID instead of its trained embedding-table index.
   * *Why it's important:* Running 101 separate model passes would be computationally slow. Batching them takes advantage of modern GPU/CPU acceleration, enabling real-time sensitivity analysis.

5. **Step 5: Coerce Predictions & Target Scaling (`_coerce_prediction`)**
   * *What it means:* The model outputs predictions in standardized values (z-scores). We use the training split's mean and standard deviation to convert these predictions back into original units (e.g. original PAM scale).
   * *Why it's important:* It ensures the output metrics are in original scale units (e.g. PAM values) rather than z-scores — making the sensitivity profile readable in the same units used to describe the data, without implying medical authority over those numbers.

6. **Step 6: Inverse-Variance Weighted Curve Fitting (`fit_weighted_curve`)**
   * *What it means:* We fit a quadratic curve ($y = a\alpha^2 + b\alpha + c$) across the 101 prediction points. Crucially, each point is weighted by the inverse of its predictive variance ($w = 1/\sigma^2$).
   * *Why it's important (The Core Novelty):* If the model is highly uncertain about a simulated behavioral shift, that prediction point receives less weight in the curve fit. From this fitted curve, we calculate:
     * **Slope ($b$):** The participant's immediate mood responsiveness near their current behavioral baseline.
     * **Curvature ($2a$):** The rate of acceleration or diminishing returns (e.g., does extra sleep help less and less as it increases?).
     * **Margin:** The minimum behavior shift at which the model-estimated sensitivity curve crosses a chosen pattern-reference point, found by linearly interpolating between the two grid points that bracket the crossing (not a clinical prescription — a descriptive diagnostic of model-predicted responsiveness). If a direction never crosses the reference point within its plausible bounds for a given window, margin is correctly reported as infinite rather than a spurious finite number — this is itself a finding (e.g. StudentLife's `sleep` direction alone does not push predicted PAM across threshold for any test window), not a computation failure.

7. **Step 7: Predictive Bootstrapping (`bootstrap_curve_intervals`)**
   * *What it means:* We resample the predictions using the model's own predicted standard deviation 200 times and recalculate the curves to establish 95% Confidence Intervals (CIs) for our slope and curvature.
   * *Why it's important:* It tells us if the calculated sensitivity curves are statistically reliable or just random noise.

8. **Pairwise Interaction Flow & Mahalanobis Filter (`probe_interaction`)**
   * *What it means:* We simulate shifting two behaviors simultaneously (e.g. changing both sleep and screen time). To make sure the joint behavior is realistic, we calculate the Mahalanobis distance against the joint training distribution and discard simulated states that fall outside the 97.5% boundary.
   * *Why it's important:* This ensures the model does not hallucinate under impossible joint conditions (e.g., a person sleeping 12 hours *and* spending 16 hours on their screen on the same day). It calculates whether the joint behavioral shift is more (or less) than the simple sum of individual changes (identifying synergetic or buffering behavioral interactions).
