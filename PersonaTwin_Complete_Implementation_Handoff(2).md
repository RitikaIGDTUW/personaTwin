# PersonaTwin — Complete Implementation Handoff Plan
## CES + StudentLife → Personalized Multimodal Digital Twin

**Purpose of this file:** This is a handoff document for a new ChatGPT conversation.  
A new chat should read this file first and continue implementation from the current state **without restarting the data-understanding work or inventing results that have not been established**.

---

# 1. Project identity

## Project name

**PersonaTwin — Personalized Multimodal Digital Twin for Mental-State Dynamics**

## Core research question

> Can a personalized multimodal Digital Twin learn an individual's longitudinal behavioral and affective dynamics and predict how that individual's future mental state may change under different realistic, low-risk behavioral or contextual scenarios?

The intended pipeline is:

```text
OBSERVE
  ↓
Understand multimodal longitudinal behavior
  ↓
Align participant × day
  ↓
Learn temporal dynamics
  ↓
Personalize to individual history
  ↓
Estimate current latent state
  ↓
Predict future mental state
  ↓
Run realistic what-if scenarios
  ↓
Estimate uncertainty
  ↓
Rank scenarios
  ↓
Observe new data
  ↓
Update the individual's twin
```

The project should NOT be presented as merely:

> "mental-health prediction using smartphone sensing."

The stronger research position is:

> **Personalized mental-state dynamics + Digital Twin + what-if simulation + uncertainty-aware scenario ranking.**

---

# 2. Important scientific limitation

CES and StudentLife are observational datasets.

Therefore:

- Do NOT claim that a simulated intervention causes a mental-state change.
- Do NOT call counterfactual predictions causal effects.
- Do NOT say "increasing sleep will reduce stress."
- Say instead:
  > "Under the learned model, the increased-sleep scenario produces a lower predicted future stress state."

A randomized intervention dataset such as Brighten could later strengthen intervention-response validation.

---

# 3. High-level architecture

```text
                              PERSONA TWIN
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
          STUDENTLIFE                              CES
     development + primary                replication + generalization
                 │                                   │
       ┌─────────┴─────────┐               ┌─────────┴─────────┐
       │                   │               │                   │
   EMA / targets        sensing        EMA / targets        sensing
       │                   │               │                   │
   PAM / stress /       activity,       PAM / stress /     activity,
   mood / surveys       sleep, GPS,     PHQ4 / social      sleep, GPS,
                        phone, audio                       phone, audio
       │                   │               │                   │
       └─────────┬─────────┘               └─────────┬─────────┘
                 │                                   │
          DATASET-SPECIFIC                     DATASET-SPECIFIC
          PREPROCESSING                        PREPROCESSING
                 │                                   │
          participant × day                    participant × day
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ↓
                         COMMON FEATURE ONTOLOGY
                                   ↓
                      COMMON MODELING FRAMEWORK
                                   ↓
                         POPULATION-LEVEL MODEL
                                   ↓
                         TEMPORAL DYNAMICS (GRU)
                                   ↓
                            PERSONALIZATION
                                   ↓
                         PERSONAL LATENT STATE
                                   ↓
                          PERSONAL DIGITAL TWIN
                                   ↓
                         NEXT-STATE PREDICTION
                                   ↓
                            WHAT-IF ENGINE
                                   ↓
                         UNCERTAINTY ESTIMATION
                                   ↓
                           SENSITIVITY ENGINE
                                   ↓
                       CROSS-DATASET VALIDATION
```

---

# 4. What the Digital Twin means here

The Digital Twin is NOT a literal replica of a person.

It is a computational representation of an individual's learned behavioral and affective dynamics.

Conceptually:

```text
Real student
    │
    │ longitudinal observations
    ↓
Behavior + Context + Affect
    ↓
Temporal model
    ↓
Personal latent state
    ↓
Digital Twin
```

The latent state can conceptually contain:

```text
z_t =
[
    behavioral state,
    contextual state,
    affective state,
    personal baseline,
    temporal trend
]
```

The important property is that the twin changes as new observations arrive.

---

# 5. Roles of the two datasets

## 5.1 StudentLife

StudentLife is the primary/development dataset in the original research plan.

Expected/complementary feature families include:

- activity
- sleep
- mobility/location
- audio/conversation-related sensing
- phone-related signals
- Bluetooth/WiFi-related signals
- mood
- stress
- social information
- PAM
- survey measures

The original plan considered PAM/affect as a possible primary target and stress as a secondary target, but target selection must ultimately follow the actual observed label frequency and coverage.

StudentLife should be used to develop the methodology and demonstrate personalization/dynamic modeling.

## 5.2 CES

CES is the currently inspected and processed dataset and is the first dataset ready for the next modeling stage.

CES contains:

- general EMA
- sensing
- steps
- demographics
- PAM
- PHQ-4-related variables
- stress
- social variables
- activity
- sleep
- mobility/location-derived variables
- communication
- phone interaction
- hourly and episode-level sensing features

CES is useful for longitudinal participant-day sequence construction and replication/generalization.

## 5.3 The datasets are NOT blindly concatenated

The two datasets should NOT simply be row-concatenated or column-concatenated.

Instead:

```text
StudentLife
   ↓
dataset-specific preprocessing
   ↓
common feature ontology

CES
   ↓
dataset-specific preprocessing
   ↓
common feature ontology

          ↓
   common model framework
```

Only semantically meaningful overlapping feature families should be treated as shared features.

---

# 6. Current StudentLife data state — VERIFIED

StudentLife is NOT merely a future integration dataset. Its raw-data inspection and target-feasibility work has already been completed and is recorded below. It is the **development + primary dataset** in the research architecture.

## 6.1 Verified StudentLife inventory

The project record reports:

```text
RDS files found                 : 50
Successfully loaded tables     : 49
Core sensing participants      : 49
```

Important verified tables include:

| Table | Rows | Columns |
|---|---:|---:|
| activity | 22,842,191 | 3 |
| audio | 99,298,223 | 3 |
| bluetooth | 1,288,526 | 5 |
| conversation | 79,023 | 3 |
| gps | 202,877 | 11 |
| wifi | 19,244,309 | 5 |
| wifi_location | 1,893,838 | 3 |
| app_usage | 1,990,510 | 11 |
| call_log | 71,801 | 12 |
| sms | 92,584 | 4 |
| phonelock | 9,275 | 3 |
| phonecharge | 3,318 | 3 |
| pam | 9,040 | 3 |
| stress | 2,017 | 3 |
| mood | 277 | 7 |
| sleep | 1,644 | 6 |

Survey tables also include BigFive, FlourishingScale, LonelinessScale, PHQ-9, PerceivedStressScale, PANAS, PSQI and VR-12.
```

These values are recovered from the verified project record, not estimated. fileciteturn21file1L314-L369

## 6.2 Verified StudentLife targets

```text
PAM    : 9,040 observations / 49 participants
Stress : 2,017 observations / 46 participants
Mood   :   277 observations / 38 participants
```

Verified participant-level coverage:

| Target | Participants | Median observations/person | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| PAM | 49 | 195 | 184.49 | 8 | 437 |
| Stress | 46 | 38 | 43.85 | 4 | 112 |
| Mood | 38 | 3 | 7.29 | 1 | 83 |

Therefore, PAM is currently the strongest **dense longitudinal StudentLife target**. Stress is scientifically important but substantially sparser, and mood is too sparse to be the first dynamic target. This does NOT permanently exclude stress or mood; it determines implementation order. fileciteturn20file0L135-L177

## 6.3 Verified StudentLife timestamp processing

The target timestamps were Unix seconds and were normalized with:

```python
pd.to_datetime(timestamp, unit="s", errors="coerce")
```

No invalid target timestamps were found. Verified ranges:

```text
PAM    : 2013-03-24 08:40:30 → 2013-07-13 23:47:02
Stress : 2013-03-24 08:40:01 → 2013-08-16 00:56:08
Mood   : 2013-04-24 23:05:35 → 2013-08-10 03:44:09
```

The project record also contains a gap-hours calculation with values around e-09; those values are treated as a timestamp-unit artifact and MUST NOT be used as scientific gap estimates. The more reliable within-window continuity results are retained.

## 6.4 Verified StudentLife target continuity

```text
Target   within 6h   within 12h   within 24h   within 48h   within 72h
PAM        66.78%      83.26%      95.50%      98.06%      98.93%
Stress     32.47%      46.83%      71.23%      85.03%      90.41%
Mood       14.64%      26.78%      49.79%      71.13%      79.08%
```

These results support using daily modeling for PAM first, while requiring explicit target-alignment rules for sparse stress/mood. fileciteturn20file8L957-L983

## 6.5 StudentLife processing status

```text
[✓] RDS inventory
[✓] table loading / inspection
[✓] participant coverage inspection
[✓] target identification
[✓] target observation counts
[✓] target participant coverage
[✓] longitudinal duration analysis
[✓] target continuity analysis
[✓] Unix-second timestamp normalization
[✓] target date-range verification
[✓] feature-level missingness inspection

[ ] final participant-day modeling table
[ ] daily feature engineering
[ ] target alignment for forecasting
[ ] sequence construction
[ ] leakage-safe chronological split
[ ] train-only imputation / normalization
[ ] baseline models
[ ] GRU
[ ] personalization
[ ] Digital Twin latent state
[ ] what-if / uncertainty / sensitivity
[ ] CES replication
```

**Important:** StudentLife does not yet have a final modeling dataframe. The completed work is the data-understanding and target-feasibility stage.

---

# 6. Current CES data state — VERIFIED

The CES data are stored in Google Drive so the ZIP does not have to be uploaded every time.

Current Drive structure:

```text
/content/drive/MyDrive/PersonaTwin/data/
    └── CES/
         ├── CES.zip
         └── CES/
```

The runtime must still remount Drive/reload tables after a Colab runtime disconnects, but the original ZIP no longer needs repeated manual upload.

Current loaded tables:

```text
demographics : (216, 3)
general_ema  : (217155, 19)
covid_ema    : (16511, 12)
sensing      : (216065, 651)
steps        : (176458, 30)
```

The CES loader uses:

```python
base = Path("/content/drive/MyDrive/PersonaTwin/data/CES/CES")
```

and loads:

```python
files_to_load = {
    "demographics": base / "Demographics" / "demographics.csv",
    "general_ema": base / "EMA" / "general_ema.csv",
    "covid_ema": base / "EMA" / "covid_ema.csv",
    "sensing": base / "Sensing" / "sensing.csv",
    "steps": base / "Sensing" / "steps.csv",
}
```

---

# 7. CES temporal coverage

Verified outputs:

```text
GENERAL_EMA
day dtype: int64
minimum day: 20170907
maximum day: 20220704
unique days: 1762

SENSING
day dtype: int64
minimum day: 20170907
maximum day: 20220615
unique days: 1743

STEPS
day dtype: int64
minimum day: 20170907
maximum day: 20220614
unique days: 1742

COVID_EMA
day dtype: int64
minimum day: 20200317
maximum day: 20220426
unique days: 771
```

---

# 8. CES participant overlap

Verified:

```text
EMA participants: 220
Sensing participants: 220
Steps participants: 198

EMA ∩ Sensing: 220
EMA ∩ Steps: 198
EMA ∩ Sensing ∩ Steps: 198
```

Thus:

- General EMA and sensing cover the same 220 participant IDs.
- Steps cover 198 of those participants.
- The 22 participants without steps should not automatically be discarded; steps should be treated as an additional modality with missing participant-level coverage.

---

# 9. CES EMA ↔ sensing day alignment

Verified:

```text
both          216057
left_only       1098
right_only         0
```

Sensing availability for EMA rows:

```text
99.49 %
```

This is a strong alignment result.

Interpretation:

- Almost all EMA participant-days have corresponding sensing data.
- Only 1,098 EMA rows lack a matching sensing participant-day.
- No sensing-only rows occurred in this left-join comparison.

---

# 10. CES participant coverage

General EMA participant-day coverage:

```text
<=30 days       3 participants
31-90           3
91-180          7
181-365        12
366-730        29
731-1000       26
1001-1500     140
>1500           0
```

So most CES participants have long longitudinal histories.

---

# 11. CES target variables inspected

The general EMA targets inspected were:

```text
pam
stress
phq4_score
phq4_resp_mean
phq4_resp_median
social_level
sse3_resp_mean
sse3_resp_median
```

Important observation:

Most of the mental-state targets have approximately:

```text
35,348 non-missing observations
181,807 missing observations
83.72% missing
```

The PAM target has:

```text
35,186 non-missing
181,969 missing
83.80% missing
```

This means the mental-state labels are sparse relative to the full EMA table.

That is expected in EMA-style longitudinal data and is a major reason we must not randomly split rows or fabricate labels.

---

# 12. CES target statistics

## PAM

```text
Non-missing: 35186
Missing: 181969
Missing %: 83.8
Unique: 16

mean: 7.258938
std: 4.330170
min: 1
25%: 3
50%: 7
75%: 10
max: 16
```

## Stress

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 5

mean: 2.520312
std: 1.095161
min: 1
25%: 2
50%: 2
75%: 3
max: 5
```

## PHQ4 score

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 13

mean: 2.316397
std: 2.620882
min: 0
25%: 0
50%: 2
75%: 4
max: 12
```

## PHQ4 response mean

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 34398

mean: 6.662954
std: 121.566681
min: -3.640500
25%: 1.060533
50%: 1.625579
75%: 2.617375
max: 13860.450489
```

This variable is extremely skewed and contains extreme values. Do not use it blindly as the first modeling target.

## PHQ4 response median

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 33120

mean: 1.799358
std: 2.515378
min: 0.103109
25%: 0.845009
50%: 1.332035
75%: 2.016919
max: 149.429673
```

## Social level

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 5

mean: 3.138905
std: 1.256505
min: 1
25%: 2
50%: 3
75%: 4
max: 5
```

## SSE3 response mean

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 34405

mean: 7.988441
std: 394.652521
min: -3.640500
25%: 1.347932
50%: 1.821408
75%: 2.655755
max: 69661.490631
```

Also extremely skewed.

## SSE3 response median

```text
Non-missing: 35348
Missing: 181807
Missing %: 83.72
Unique: 33122

mean: 1.993368
std: 2.061981
min: 0.046799
25%: 1.228678
50%: 1.623779
75%: 2.200024
max: 75.798566
```

---

# 13. CES target coverage per participant

For PAM:

```text
participants with >= 1 observations:   218 / 220 = 99.1%
>= 5:                                   217 / 220 = 98.6%
>= 10:                                  215 / 220 = 97.7%
>= 20:                                  211 / 220 = 95.9%
>= 30:                                  205 / 220 = 93.2%
>= 50:                                  192 / 220 = 87.3%
>= 100:                                 161 / 220 = 73.2%
>= 150:                                 131 / 220 = 59.5%
>= 200:                                  58 / 220 = 26.4%
```

For stress:

```text
>= 1:   218 / 220 = 99.1%
>= 5:   217 / 220 = 98.6%
>= 10:  215 / 220 = 97.7%
>= 20:  210 / 220 = 95.5%
>= 30:  205 / 220 = 93.2%
>= 50:  191 / 220 = 86.8%
>= 100: 161 / 220 = 73.2%
>= 150: 133 / 220 = 60.5%
>= 200:  62 / 220 = 28.2%
```

For PHQ4 score:

```text
>= 1:   218 / 220 = 99.1%
>= 5:   217 / 220 = 98.6%
>= 10:  215 / 220 = 97.7%
>= 20:  210 / 220 = 95.5%
>= 30:  205 / 220 = 93.2%
>= 50:  191 / 220 = 86.8%
>= 100: 161 / 220 = 73.2%
>= 150: 133 / 220 = 60.5%
>= 200:  62 / 220 = 28.2%
```

Median observations per participant:

```text
n_days median: 1153
target observations median: ~168
```

Maximum target observations per participant:

```text
441
```

Two participants had zero target observations.

---

# 14. CES target correlations

Using rows where the target values are available:

```text
                 pam       stress    phq4_score
pam              1.000000 -0.116643 -0.090894
stress          -0.116643  1.000000  0.479496
phq4_score      -0.090894  0.479496  1.000000
```

Stress and PHQ4 score show the strongest association among these three:

```text
r ≈ 0.4795
```

PAM and stress are only weakly negatively correlated:

```text
r ≈ -0.1166
```

This supports treating stress as a sensible first modeling target rather than assuming all mental-state variables are interchangeable.

---

# 15. CES target co-occurrence

```text
pam    stress  phq4_score    count
False  False   False         181594
True   True    True           34973
False  True    True             375
True   False   False            213
```

This is useful because most labeled stress/PHQ4 rows have all three targets simultaneously.

---

# 16. CES sensing feature groups

The 649 sensing features were conceptually grouped into families.

Observed families include:

### Activity
Approximately 220 features.

Examples:

```text
act_in_vehicle_*
act_on_bike_*
act_on_foot_*
activity-state episode/hour features
```

### Mobility

Approximately 144 features.

Examples:

```text
light_mean_*
light_std_*
loc_dist_*
location-derived variables
```

### Social

Approximately 168 features.

Examples:

```text
audio_amp_mean_*
audio_amp_std_*
audio_voice_*
conversation-related features
```

### Phone use

Approximately 56 features.

Examples:

```text
unlock_duration_*
unlock_num_*
phone interaction features
```

Other sensing families are present and should be mapped to the common ontology rather than simply being treated as an undifferentiated 649-dimensional block.

---

# 17. CES modeling dataset already constructed

The first modeling table was built by joining on:

```text
uid + day
```

No duplicate EMA `uid-day` rows were found.

Starting data:

```text
EMA:     (217155, 19)
Sensing: (216065, 651)
Steps:   (176458, 30)
```

Feature counts:

```text
Sensing features: 649
Steps features:    28
```

After EMA + sensing:

```text
(217155, 654)
```

After adding steps:

```text
(217155, 682)
```

Therefore the current broad CES modeling dataset is:

```text
Rows:    217155
Columns: 682
```

Target coverage:

```text
pam         35186 observations = 16.20%
stress      35348 observations = 16.28%
phq4_score  35348 observations = 16.28%
```

This is a participant-day/EMA-row modeling table with the sensing and steps modalities attached.

---

# 18. What has NOT yet been done

Important: the following should NOT be claimed as completed unless a later notebook output proves it:

- train/test split
- normalization fitted on training data
- final imputation strategy
- 7-day sequence construction
- next-day target construction
- neural network training
- GRU training
- personalization
- Digital Twin latent state
- what-if engine
- uncertainty estimation
- causal/counterfactual validation

The current state is still **pre-modeling**.

---

# 19. CES feature quality analysis

On labeled target days:

```text
Labeled rows: 35348
Participants: 218
Candidate features: 677
```

Feature missingness distribution:

```text
<=10%       321
10-25%      140
25-50%       34
50-75%       75
75-90%      105
90-95%        1
95-100%       1
```

Thus:

```text
677 candidate features
→ 570 after missingness filtering
```

107 features were removed by the missingness threshold.

No zero-variance features were found after this step.

---

# 20. Current CES feature filter

Current filter result:

```text
Original candidate features:   677
After missingness filter:      570
Zero variance removed:            0
Final feature count:            570
```

The remaining features can still have substantial missingness.

Examples of high-missingness features among those remaining include:

```text
loc_food_convo_num
loc_food_unlock_num
loc_food_audio_voice
loc_food_still
loc_food_unlock_duration
loc_food_convo_duration
loc_study_still
loc_study_convo_num
loc_study_unlock_num
loc_study_unlock_duration
loc_study_convo_duration
loc_study_audio_voice
audio_amp_mean_hr_5
audio_amp_std_hr_5
...
```

Important: filtering by missingness is only the first quality-control step. It is NOT the final feature engineering step.

---

# 21. Important issue in CES feature engineering

The raw CES sensing table contains hundreds of highly granular hourly and episode-level features.

Do not immediately feed all 570 features into a large neural network.

The intended scientific representation is:

```text
RAW SENSING
     ↓
feature families
     ↓
daily behavioral/contextual summaries
     ↓
person-relative features
     ↓
temporal windows
     ↓
model
```

The common ontology should include:

```text
Behavior
├── activity
├── sedentary behavior
├── sleep
├── physical activity
└── phone activity

Mobility / Context
├── mobility distance
├── location diversity
├── routine variability
└── contextual location behavior

Communication
├── call frequency
├── call duration
├── SMS frequency
└── conversation duration

Affect / Mental state
├── PAM
├── stress
├── mood
├── PHQ-related score
└── social state
```

Only features that can be meaningfully mapped between CES and StudentLife should be considered shared cross-dataset features.

---

# 22. Intended feature engineering

## Activity

Possible derived daily features:

```text
daily_activity
sedentary_duration
activity_variability
walking_duration
running_duration
vehicle_duration
foot_activity
```

## Sleep

Possible:

```text
sleep_duration
sleep_timing
sleep_regularity
sleep_missing
```

## Mobility

Possible:

```text
mobility_distance
location_diversity
mobility_variability
location_visit_count
maximum_distance
```

## Communication

Possible:

```text
communication_frequency
communication_duration
call_count
call_duration
sms_count
conversation_duration
social_interaction_proxy
```

## Phone interaction

Possible:

```text
phone_use_frequency
phone_use_duration
unlock_count
unlock_duration
```

These are conceptual feature engineering targets. The exact columns must be selected from the actual dataset and documented.

---

# 23. Missing-data policy

Do NOT delete every row containing a missing sensing value.

Recommended pipeline:

```text
Raw data
   ↓
Missingness analysis
   ↓
Feature-level filtering
   ↓
Participant/day coverage filtering
   ↓
Appropriate imputation
   ↓
Missingness indicators
```

Example:

```text
sleep_duration
sleep_missing
```

The missingness itself may carry information about sensor availability.

Normalization must be fitted using training data only.

Never calculate normalization statistics over the complete dataset before splitting.

---

# 24. Target choice for the first model

The recommended first target is:

> **Next-day stress**

Conceptually:

```text
behavior/context history through day t
             ↓
       predict stress at t+1
```

Initial formulation:

```text
X(i,t-6:t) → stress(i,t+1)
```

or, for the first 7-day sequence:

```text
Day t-6
Day t-5
Day t-4
Day t-3
Day t-2
Day t-1
Day t
   ↓
predict Day t+1 stress
```

Do not assume that every calendar day has a stress label.

If the next day has no observed target, do NOT fabricate one.

Possible strategies:

1. only construct examples where the target day is observed;
2. later investigate next-available-label prediction;
3. consider weekly aggregation if daily target sparsity makes the experiment infeasible.

The exact choice must be reported and justified.

---

# 25. Critical leakage rule

Do NOT randomly split the 217,155 rows.

Because this is longitudinal data, random row splitting can put the same participant's future observations into training.

That can cause:

- identity memorization
- personal-pattern leakage
- overly optimistic performance

Use chronological splitting.

For a first CES experiment:

```text
For each participant:

earlier period → TRAIN
later period   → TEST
```

The exact proportion should be determined from the available longitudinal span and the target coverage.

Potential initial split:

```text
70% earliest labeled/eligible temporal period → TRAIN
30% later temporal period                     → TEST
```

But the split must be chronological and participant-aware.

A second, stronger experiment can evaluate held-out participants.

---

# 26. Recommended train/test framework

We need at least two complementary evaluation regimes.

## A. Within-person future forecasting

Each participant contributes:

```text
past → training
future → test
```

This answers:

> Can the model learn population dynamics and personalize to a person using their earlier history?

## B. Leave-person-out / participant-held-out

Some participants are never seen during training.

This answers:

> Does the learned general model transfer to unseen people?

Then personalization can be evaluated with:

```text
0 days personal history
3 days
7 days
14 days
...
```

This directly tests sample efficiency.

---

# 27. Baseline progression

Do not jump directly to the final Digital Twin.

Use:

```text
Baseline 1
Persistence
prediction(t+1) = state(t)

        ↓

Baseline 2
Population mean

        ↓

Baseline 3
Linear / Elastic Net

        ↓

Baseline 4
Random Forest / Gradient Boosting

        ↓

Baseline 5
GRU without personalization

        ↓

Baseline 6
GRU + personal embedding

        ↓

Baseline 7
GRU + personalized adapter

        ↓

Final
Personalized Digital Twin
```

Every architectural addition must be evaluated against the previous stage.

---

# 28. Multimodal model architecture

The intended architecture is:

```text
                   PARTICIPANT-DAY INPUT
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
       Behavior         Mobility        Communication
          │                │                │
      encoder           encoder           encoder
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                       fusion
                           ↓
                    temporal GRU
                           ↓
                 shared population state
                           ↓
                 personalized adapter
                           ↓
                 personal latent state
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
      next-day prediction          twin state
                                        ↓
                                  what-if engine
                                        ↓
                                  uncertainty
                                        ↓
                                  sensitivity
```

The first implementation can be simpler:

```text
570-ish filtered features
        ↓
MLP
        ↓
GRU
        ↓
stress(t+1)
```

Then the multimodal separation can be introduced.

---

# 29. Personalization mechanism

The project should NOT train one neural network from scratch for every person.

Instead:

```text
Shared population model
        +
Individual history
        ↓
Personalized adapter
        ↓
Personal latent dynamics
```

Possible implementation progression:

### Level 1

Participant embedding.

### Level 2

Small participant-specific adapter.

### Level 3

Few-shot adaptation of only the adapter while the shared model is frozen.

This is important for the research contribution.

---

# 30. Digital Twin latent state

After the temporal model is working, expose a latent representation:

```text
z_t = f(
    recent behavioral history,
    contextual history,
    affective history,
    personal baseline,
    temporal trends
)
```

This latent state becomes the computational "twin".

It should evolve:

```text
z_t
 ↓ new observation
z_t+1
 ↓ new observation
z_t+2
...
```

The twin is therefore dynamic, not a static participant profile.

---

# 31. Personal baselines

For suitable variables, calculate deviations from an individual's own baseline.

Example:

```text
today_sleep - personal_mean_sleep
```

This lets the model distinguish:

```text
6 hours for a person who normally sleeps 6 hours
```

from:

```text
6 hours for a person who normally sleeps 8 hours
```

Potential personal-baseline features:

```text
sleep deviation
activity deviation
mobility deviation
phone-use deviation
communication deviation
```

All baseline statistics must be calculated using information available up to the prediction point, not future test data.

---

# 32. Temporal window

Initial window:

```text
7 days of history → next-day target
```

Later ablation:

```text
3 days
7 days
14 days
21 days
```

Research question:

> How much temporal history does the Digital Twin need before personalization becomes useful?

---

# 33. What-if engine

This is a major research component.

Given current input:

```text
X_t
```

construct realistic scenario versions:

```text
X_t^(0) = current pattern
X_t^(1) = sleep increased
X_t^(2) = activity increased
X_t^(3) = social interaction increased
X_t^(4) = combined change
```

Run all scenarios through the SAME personalized Digital Twin.

```text
                 Personal Twin
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
   Scenario 0    Scenario 1    Scenario 2
        │             │             │
        ↓             ↓             ↓
     y0(t+1)       y1(t+1)       y2(t+1)
```

Calculate:

```text
Δscenario =
predicted_scenario - predicted_baseline
```

For stress, a negative Δ means lower predicted stress under the model, but it must NOT be described as a causal effect.

---

# 34. Scenario constraints

Only simulate realistic, low-risk, data-supported variables.

Examples:

```text
sleep duration
physical activity
social interaction
mobility
routine regularity
phone-use pattern
```

Changes must stay within ranges observed in the data.

Example:

```text
Current sleep = 5.5 hours
Scenario      = 6.5 hours
```

not an arbitrary unrealistic value.

Do not let the what-if engine perturb variables that have no meaningful observational support.

---

# 35. Uncertainty estimation

The system should eventually output:

```text
prediction + uncertainty
```

Possible approaches:

- Monte Carlo dropout
- deep ensembles
- quantile regression
- prediction intervals

Practical first implementation:

> Monte Carlo dropout OR a small ensemble.

Example:

```text
Scenario: More sleep
Predicted stress: 3.4
Uncertainty: ±0.4
```

---

# 36. Scenario ranking

Example output:

```text
Scenario             Prediction     Uncertainty
------------------------------------------------
No change             3.8             ±0.5
More sleep            3.4             ±0.4
More activity         3.6             ±0.5
More social           3.5             ±0.6
```

The system should identify a:

> **model-preferred simulated scenario**

not:

> a medical prescription.

The ranking should consider both predicted change and uncertainty.

---

# 37. Sensitivity engine

The original conceptual sensitivity module is:

```text
Predict Y under each scenario A
       ↓
Y_hat(A=0), Y_hat(A=1)
       ↓
Individual response estimate

Δ_i = Y_hat(A=1) - Y_hat(A=0)
       ↓
Reliability / sensitivity analysis
```

This enables analysis of response heterogeneity:

```text
Person A → sleep scenario predicted stress decrease
Person B → little change
Person C → predicted increase
```

Do not force all individuals to have the same response.

---

# 38. Research hypotheses

## H1 — Personalization

A personalized latent Digital Twin improves future mental-state forecasting compared with population-level models.

## H2 — Multimodality

Combining multiple behavioral modalities improves prediction compared with single-modality models.

## H3 — Response dynamics

Maintaining an evolving individual latent state provides better forecasting and more stable individual response estimates than static personalization.

## H4 — Sample efficiency

A lightweight personalization mechanism reaches useful performance with substantially less personal data than training an independent model from scratch for every participant.

---

# 39. Evaluation metrics

For continuous targets:

```text
MAE
RMSE
R²
Pearson/Spearman correlation where appropriate
```

For participant-level analysis:

```text
per-person MAE
per-person RMSE
percentage of participants improved over global baseline
median individual improvement
distribution of individual improvements
```

For the Digital Twin:

```text
forecasting accuracy
temporal stability
personalization gain
sample efficiency
counterfactual sensitivity
uncertainty/reliability
```

Do NOT report only one aggregate RMSE.

The important question is:

> Across how many individuals does personalization actually help?

---

# 40. Statistical analysis

Compare models using participant-level performance wherever possible.

Report:

```text
mean
median
standard deviation
confidence interval
```

Use an appropriate paired statistical test when assumptions are satisfied.

The key comparison is:

```text
personalized performance - global performance
```

and the distribution of this gain across people.

---

# 41. Ablation studies

At minimum:

### A — No personalization

Remove participant-specific adapter.

### B — No temporal memory

Replace GRU with a static model.

### C — Single modality

Run one modality at a time.

### D — No contextual features

Remove context.

### E — No what-if module

Use only forecasting.

### F — Personalization size

Compare:

```text
embedding only
small adapter
larger adapter
```

Optional:

```text
adapter vs MAML/few-shot personalization
```

---

# 42. StudentLife ↔ CES harmonization

The datasets should share a common conceptual ontology:

```text
                 COMMON ONTOLOGY
                       │
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Behavior         Context          Affect
       │               │                │
 activity           mobility          stress
 sleep              location           PAM
 phone              routine             mood
 communication     environment          social
```

Example mapping:

| Common family | StudentLife | CES |
|---|---|---|
| Activity | activity inference | activity sensing |
| Sleep | sleep | sleep-related sensing |
| Mobility | GPS/location | location-derived sensing |
| Phone | phone/app signals | unlock/phone signals |
| Communication | calls/SMS/conversation | calls/SMS/audio/conversation |
| Affect | PAM/mood/stress | PAM/stress/PHQ4/social |
| Context | contextual signals | location/contextual features |

Do not pretend the datasets are identical.

Document:

- feature overlap
- feature mismatch
- sensor differences
- population differences
- temporal differences

---

# 43. What "combining both datasets" actually means

There are three different levels of combination.

## Level 1 — methodological combination

Use the same:

- participant-day representation
- feature-family ontology
- target formulation
- temporal model
- personalization method
- evaluation framework

This is the most important form of combination.

## Level 2 — pooled model training

If feature definitions and scaling are sufficiently harmonized, shared models may be trained using data from both datasets.

This should only happen after dataset-specific validation.

## Level 3 — cross-dataset evaluation

Train on one dataset and evaluate on another.

Examples:

```text
StudentLife → CES
CES → StudentLife
```

This is particularly useful for generalization.

Do NOT force Level 2 if the feature definitions are not genuinely comparable.

---

# 44. Recommended experimental order for StudentLife + CES

The two datasets must now proceed through **parallel dataset-specific preprocessing**, followed by a shared modeling framework. CES must not be completed first and StudentLife added at the end.

```text
                         TWO DATASET TRACKS
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
        STUDENTLIFE                               CES
      primary/development                 replication/generalization
             │                                     │
      participant × day                     participant × day
             │                                     │
      daily features                        daily features
             │                                     │
      target alignment                      target alignment
             │                                     │
             └──────────────────┬──────────────────┘
                                ↓
                      COMMON FEATURE ONTOLOGY
                                ↓
                       COMMON MODEL FRAMEWORK
                                ↓
                  within-dataset model evaluation
                                ↓
                      personalization experiments
                                ↓
                     Digital Twin representation
                                ↓
                      cross-dataset validation
```

## PHASE 1 — Dataset-specific modeling tables

### StudentLife

Build a participant-day table from the verified raw tables. Start with feature families that are sufficiently supported:

- activity
- sleep
- mobility/location
- phone/app usage
- communication
- audio/conversation
- contextual signals

PAM should be the first dense longitudinal target candidate. Stress remains a secondary target because it is much sparser.

### CES

Use the already-built CES reservoir:

```text
217,155 EMA rows
+ sensing aligned by uid + day
+ steps
= 682-column modeling reservoir

677 candidate labeled-day features
→ 570 after missingness filter
→ zero-variance removal: 0
```

The 570 features are **not yet the final scientific feature table**. They still need semantic grouping and daily feature engineering.

## PHASE 2 — Common feature ontology

Map both datasets into the same conceptual families:

```text
Activity
Sleep
Mobility / Location
Phone / Screen
Communication / Social
Audio / Conversation
Context / Routine
Affect / Mental state
```

Do not force feature-name equality. Preserve dataset-specific raw features and create a mapping table describing:

- common concept
- StudentLife source
- CES source
- aggregation rule
- unit / scale
- missingness
- whether directly comparable

## PHASE 3 — Target strategy

Do not assume the target must be identical simply because the datasets are both mental-state datasets. First compare target feasibility.

The initial candidate hierarchy is:

```text
StudentLife: PAM → first dense longitudinal experiment
             stress → secondary sparse-label experiment
             mood → later feasibility experiment

CES:         PAM / stress / PHQ4 → choose after harmonization
```

For cross-dataset replication, prioritize targets that are genuinely available and sufficiently comparable in both datasets. PAM is a strong candidate because it is dense in StudentLife and available in CES; stress is also scientifically important but much sparser in StudentLife.

## PHASE 4 — Leakage-safe temporal modeling

For each dataset independently:

```text
participant × day
      ↓
features at t
      ↓
observed target at t+1
      ↓
7-day history
      ↓
chronological split
```

Do NOT randomly split rows. Normalization, imputation and feature selection parameters must be fitted using training data only.

## PHASE 5 — Within-dataset baselines

Run the same baseline ladder on StudentLife and CES wherever the target is comparable:

1. population/global model
2. static personalization baseline
3. temporal GRU baseline
4. modality ablations
5. multimodal model
6. personalized adapter

This establishes whether improvements are due to temporal memory, multimodality or personalization rather than model size alone.

## PHASE 6 — PersonaTwin

Use the shared architecture:

```text
modality encoders
      ↓
fusion
      ↓
shared GRU
      ↓
personal embedding / low-rank adapter
      ↓
personal latent state z(i,t)
      ↓
next-state prediction
```

Do not create a completely independent neural network per participant. The intended design is shared population dynamics plus lightweight individual adaptation.

## PHASE 7 — What-if and sensitivity

Only after forecasting is stable:

```text
current personal state
        ↓
realistic perturbation
        ↓
predicted future state
        ↓
uncertainty
        ↓
individual response estimate
        ↓
sensitivity / scenario ranking
```

Because both datasets are observational, this remains model-based scenario simulation, not causal intervention estimation.

## PHASE 8 — Cross-dataset validation

Then evaluate:

```text
StudentLife → CES
CES → StudentLife
```

Use only overlapping feature families and compatible target definitions. Report domain shift rather than hiding it.

## PHASE 9 — Optional external datasets

After StudentLife + CES are working:

```text
GLOBEM → external robustness / transfer
Brighten → optional intervention-response validation
```

---

# 45. Immediate next step — BUILD BOTH MODELING TABLES BEFORE TRAIN/TEST

We are currently **before the train/test split on both datasets**. The correct next stage is not to train a model yet.

### Step 1 — StudentLife participant-day table

Construct:

```text
uid
day
activity features
sleep features
mobility/location features
phone/app features
communication/social features
audio/conversation features
context features
PAM / stress / mood labels where observed
```

The implementation must use the verified Unix-second timestamp normalization and aggregate raw sensing to day-level features without leaking future information.

### Step 2 — CES participant-day table

Use the existing `model_df` / CES reservoir and transform the 570 basic-filtered features into semantically grouped daily feature families. Do not treat the 570 columns as the final feature engineering result.

### Step 3 — Validate both tables

For each dataset report:

```text
rows
participants
date range
features
missingness
target observations
target observations per participant
number of consecutive day pairs
number of eligible next-day targets
```

### Step 4 — Decide the first common target

Use the actual observed coverage, not an arbitrary choice. PAM is currently the strongest common dense-target candidate because StudentLife has 9,040 PAM observations across 49 participants. Stress remains a secondary target because StudentLife has only 2,017 stress observations across 46 participants.

### Step 5 — Construct next-day targets

For each dataset, only assign a next-day target when the target is actually observed according to the explicit alignment rule. Never use a simple global `.shift(-1)`.

### Step 6 — Sequence construction

For a 7-day model:

```text
t-6  t-5  t-4  t-3  t-2  t-1  t
 └────────────── X history ──────────────┘
                                      ↓
                                  Y(t+1)
```

The exact window may be changed after target-density analysis, but it must be fixed before test evaluation.

### Step 7 — Chronological split

Split by time within participant, not randomly by row. The split must be performed before fitting imputation/scaling parameters.

### Step 8 — Only then train models

```text
Global baseline
      ↓
Static-personalized baseline
      ↓
Temporal GRU
      ↓
Multimodal fusion
      ↓
Personalized adapter
      ↓
PersonaTwin
```

---

# 46. Exact next notebook cells to implement

The new chat should generate code in this order. The first cells must build and validate **both StudentLife and CES modeling tables**. Do not jump directly to CES-only training.

## Cell A — verify both modeling sources

StudentLife: use the verified loaded tables and normalized target timestamps.

CES: use the already-built CES modeling dataset.

Verify:

```python
model_df.shape
model_df.columns
model_df["uid"].nunique()
model_df["day"].min()
model_df["day"].max()
```

## Cell B — sort

```python
model_df = model_df.sort_values(["uid", "day"]).reset_index(drop=True)
```

## Cell C — construct next-day target

Do NOT use a simple `.shift(-1)` across the entire dataframe.

It must be participant-aware and day-aware.

Conceptually:

```python
next_day_stress = stress shifted by one calendar day within uid
```

The code must verify that:

```text
day(t+1) == day(t) + 1
```

before assigning the target.

## Cell D — report target coverage

Report:

```text
eligible next-day stress rows
participants with at least 1 eligible row
participants with >= 5
participants with >= 10
participants with >= 20
```

## Cell E — build 7-day sequences

For each participant:

```text
t-6 ... t
   ↓
stress(t+1)
```

Only construct a sequence when the required history and target rules are satisfied.

Decide explicitly whether missing sensing values inside the 7-day window are handled by imputation or by a mask.

## Cell F — temporal split

Split chronologically.

No random row split.

## Cell G — fit preprocessing only on training

Examples:

```text
median imputation
missingness indicators
standardization
```

Fit all parameters on train only.

Apply to validation/test.

## Cell H — baseline models

At minimum:

```text
persistence
population mean
Elastic Net
Random Forest / Gradient Boosting
```

## Cell I — GRU

Only after the baseline results exist.

---

# 47. Important correction regarding sparse labels

The full CES modeling dataset has:

```text
217155 rows
```

but only about:

```text
35348 stress observations
```

Therefore the effective supervised sample size is much smaller than 217,155.

Do NOT say:

> "We have 217k labeled samples."

Correct:

> "We have 217k longitudinal EMA/sensing-aligned rows, of which approximately 35k contain observed stress labels."

This distinction is important for the research paper.

---

# 48. Why the current processed dataset is useful

The current CES table is valuable because it has already solved the difficult raw-data integration problem:

```text
general EMA
     +
sensing
     +
steps
     ↓
participant-day aligned modeling table
```

It preserves:

- participant identity
- longitudinal order
- behavioral sensing
- contextual sensing
- phone use
- mobility
- communication
- target variables

The 682-column broad table is therefore a **feature reservoir**, not yet the final neural-network input.

The next stage converts this reservoir into a scientifically meaningful temporal representation.

---

# 49. Why StudentLife is still important

StudentLife is not redundant.

It serves as:

1. development/complementary longitudinal evidence;
2. a second population and sensing environment;
3. a test of whether the modeling framework is dataset-specific;
4. a source for cross-dataset validation;
5. potentially a route toward external generalization.

The final story is not:

```text
CES + StudentLife = one giant dataframe
```

It is:

```text
CES ───────────┐
               ├──→ shared modeling methodology
StudentLife ───┘
                     ↓
             personalized dynamics
                     ↓
              Digital Twin
```

---

# 50. Optional future GLOBEM / Brighten stage

If available later:

```text
CES + StudentLife
        ↓
Initial Digital Twin
        ↓
What-if framework
        ↓
GLOBEM / Brighten
        ↓
External validation
        ↓
Intervention-response analysis
```

GLOBEM can provide longitudinal multimodal external validation.

Brighten's randomized intervention structure could be particularly useful for stronger intervention-response evaluation.

---

# 51. Few-shot personalization experiment

A key research experiment:

```text
Personal history available:
0 days
3 days
7 days
14 days
21 days
...
```

Compare:

```text
global model
vs
personalized model
```

Measure:

```text
personalized performance - global performance
```

This tests:

> How much individual history does a Digital Twin need before personalization becomes useful?

This is a central sample-efficiency experiment.

---

# 52. Closed-loop Digital Twin

Final system:

```text
Real-world observations
          ↓
      Digital Twin
          ↓
   Current latent state
          ↓
   Future prediction
          ↓
   What-if simulation
          ↓
    Scenario ranking
          ↓
Low-risk model-based output
          ↓
 New observations
          ↓
 Update Digital Twin
          ↓
        Repeat
```

This is what makes the system dynamic rather than a one-time predictor.

---

# 53. Important things NOT to do

Do NOT:

- train a huge Transformer immediately;
- train one neural network per participant;
- randomly split longitudinal rows;
- fabricate unobserved mental-state labels;
- calculate normalization using future/test data;
- impute mental-state labels as if they were measurements;
- claim causal intervention effects from observational data;
- force counterfactual predictions to have a predetermined direction;
- report only population-level RMSE;
- call a model a Digital Twin merely because it uses an LSTM/GRU;
- use all raw features without checking temporal leakage;
- blindly concatenate CES and StudentLife;
- wait for GLOBEM before completing the core CES/StudentLife pipeline.

---

# 54. Recommended project directory

```text
PersonaTwin/
│
├── data/
│   ├── CES/
│   │   ├── CES.zip
│   │   └── CES/
│   ├── studentlife/
│   └── processed/
│       ├── ces_daily/
│       ├── studentlife_daily/
│       └── harmonized/
│
├── notebooks/
│   ├── 01_CES_loading_and_inventory.ipynb
│   ├── 02_CES_target_and_feature_quality.ipynb
│   ├── 03_CES_target_ready_dataset.ipynb
│   ├── 04_CES_temporal_split_and_baselines.ipynb
│   ├── 05_CES_GRU.ipynb
│   ├── 06_personalization.ipynb
│   ├── 07_digital_twin.ipynb
│   ├── 08_counterfactuals.ipynb
│   ├── 09_uncertainty.ipynb
│   ├── 10_studentlife_alignment.ipynb
│   └── 11_cross_dataset_validation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── target_construction.py
│   ├── splits.py
│   ├── evaluation.py
│   ├── personalization.py
│   ├── counterfactual.py
│   └── models/
│       ├── baseline.py
│       ├── gru.py
│       ├── multimodal.py
│       └── digital_twin.py
│
├── experiments/
│   ├── configs/
│   └── results/
│
├── figures/
└── paper/
```

---

# 55. Paper-level contribution structure

Potential contributions:

### Contribution 1

A multimodal temporal representation of evolving student behavioral and affective states.

### Contribution 2

A personalization mechanism adapting population-level dynamics to individual patterns.

### Contribution 3

A Digital Twin latent representation for future-state simulation.

### Contribution 4

A what-if engine for hypothetical low-risk behavioral/contextual changes.

### Contribution 5

Uncertainty-aware scenario ranking rather than deterministic recommendations.

The final novelty claim must be validated against the literature before publication.

---

# 56. Expected paper experiments

```text
Experiment 1
Population-level forecasting

Experiment 2
Temporal GRU vs static model

Experiment 3
Multimodal vs single modality

Experiment 4
Global vs personalized model

Experiment 5
Personalization sample efficiency

Experiment 6
Latent Digital Twin analysis

Experiment 7
Counterfactual scenario sensitivity

Experiment 8
Uncertainty/reliability

Experiment 9
CES ↔ StudentLife cross-dataset generalization

Experiment 10
Optional GLOBEM/Brighten validation
```

---

# 57. Planned paper figures

### Figure 1

Overall Digital Twin architecture.

### Figure 2

StudentLife/CES temporal data pipeline.

### Figure 3

Forecasting performance:

```text
Global
Static-personalized
GRU
Multimodal
Digital Twin
```

### Figure 4

Distribution of per-person personalization gains.

### Figure 5

Sample-efficiency curve:

```text
personal history → prediction performance
```

### Figure 6

Counterfactual response distributions.

### Figure 7

Example individual Digital Twin trajectory:

```text
actual mental state
predicted mental state
counterfactual state
```

### Figure 8

Cross-dataset transfer/generalization.

---

# 58. Final conceptual research flowchart

```text
┌─────────────────────────────────────────────────────────────┐
│                    RAW LONGITUDINAL DATA                    │
│                                                             │
│        StudentLife                    CES                    │
│             │                          │                     │
└─────────────┼──────────────────────────┼─────────────────────┘
              ↓                          ↓
       Dataset-specific            Dataset-specific
       preprocessing              preprocessing
              ↓                          ↓
       participant × day          participant × day
              │                          │
              └────────────┬─────────────┘
                           ↓
                 COMMON FEATURE ONTOLOGY
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    Behavior            Context             Affect
        │                  │                  │
 activity/sleep       mobility/location   stress/PAM/mood
 phone/communication  routine/context      social/PHQ
        └──────────────────┼──────────────────┘
                           ↓
                    DAILY FEATURES
                           ↓
                MISSINGNESS + MASKING
                           ↓
                 PERSONAL BASELINES
                           ↓
                    7-DAY WINDOWS
                           ↓
                 LEAKAGE-SAFE SPLIT
                           ↓
                 BASELINE MODELS
                           ↓
                  TEMPORAL GRU
                           ↓
                MULTIMODAL FUSION
                           ↓
                PERSONALIZED ADAPTER
                           ↓
              PERSONAL LATENT STATE z_t
                           ↓
                 DIGITAL TWIN STATE
                           ↓
               PREDICT STATE t + 1
                           ↓
                    WHAT-IF ENGINE
                           ↓
       ┌───────────────────┼───────────────────┐
       ↓                   ↓                   ↓
   No change           More sleep         More activity
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ↓
                  PREDICTED FUTURE STATES
                           ↓
                     UNCERTAINTY
                           ↓
                  SENSITIVITY ANALYSIS
                           ↓
                  SCENARIO RANKING
                           ↓
                 NEW OBSERVATIONS
                           ↓
                UPDATE PERSONAL TWIN
```

---

# 59. Immediate handoff instruction to a new ChatGPT

**Do not restart the project from scratch.**

Start from this exact state:

```text
CES:
    raw data loaded and stored in Drive
    EMA + sensing + steps aligned
    217,155 broad modeling rows
    682 columns
    570 features after basic labeled-day missingness filter
    stress = 35,348 observed labels
    target sparsity ≈ 83.72%
    EMA↔sensing day alignment = 99.49%
    220 EMA/sensing participants
    198 participants with steps

StudentLife:
    intended complementary/development dataset
    common feature ontology and methodology established
    exact raw-data numerical outputs MUST be verified from the StudentLife
    notebook/files before being stated as fact
```

**The next code should construct the leakage-safe next-day stress target and inspect the resulting supervised sequence counts.**

Do not perform train/test splitting until the target-ready participant-day table has been validated.

Do not train a GRU until baseline construction and leakage checks are complete.

---

# 60. Exact next question the new chat should answer

The new chat should proceed by explaining:

1. what the current CES modeling table represents;
2. what still needs to be transformed into daily/temporal modeling examples;
3. how next-day stress will be constructed without leakage;
4. how the chronological train/test split will work;
5. what the first baseline models will be;
6. then provide the next Colab code cell-by-cell.

The new chat should preserve the research architecture in this document and should not replace it with an unrelated modeling strategy unless a data-driven reason is demonstrated.

---

# 61. One-sentence professor explanation

> **We propose a personalized multimodal Digital Twin that learns an individual's longitudinal behavioral-to-affective dynamics from StudentLife and CES data, predicts their future mental state, and simulates how that prediction changes under realistic low-risk behavioral or contextual scenarios with explicit uncertainty.**

---

# 62. Final implementation roadmap

```text
1. CES + StudentLife data inventory
        ↓
2. Dataset-specific participant-day tables
        ↓
3. Common feature ontology
        ↓
4. Feature engineering
        ↓
5. Target construction
        ↓
6. Missingness / masking
        ↓
7. Leakage-safe chronological split
        ↓
8. Persistence + mean + linear/tree baselines
        ↓
9. GRU temporal baseline
        ↓
10. Multimodal fusion
        ↓
11. Personal embedding / adapter
        ↓
12. Digital Twin latent state
        ↓
13. Future-state forecasting
        ↓
14. Per-person evaluation
        ↓
15. Ablations
        ↓
16. Few-shot personalization
        ↓
17. What-if engine
        ↓
18. Sensitivity analysis
        ↓
19. Uncertainty estimation
        ↓
20. Scenario ranking
        ↓
21. StudentLife ↔ CES cross-dataset evaluation
        ↓
22. Optional GLOBEM / Brighten validation
        ↓
23. Statistical analysis
        ↓
24. Figures + paper
```


---

# 62A. StudentLife — VERIFIED PROCESSING OUTPUTS RECOVERED FROM THE PROJECT RECORD

This section has been added after reviewing the **"My strongest recommendation.docx"** project record. These are the StudentLife values that were actually established during the prior inspection/processing work. They replace the earlier instruction that exact StudentLife numerical outputs must still be re-run before being stated.

## 62A.1 Raw StudentLife inventory

The project record reports:

```text
Number of RDS files: 50
Number of successfully loaded StudentLife tables: 49
```

The StudentLife dataset contains three broad groups:

```text
EMA
├── Activity
├── Behavior
├── Mood / Mood 1 / Mood 2
├── PAM
├── Sleep
├── Social
├── Stress
├── Study Spaces
└── other contextual EMA forms

OTHER
├── app_usage
├── calendar
├── call_log
├── dining
└── sms

SENSING
├── activity
├── audio
├── bluetooth
├── conversation
├── dark
├── gps
├── phonecharge
├── phonelock
├── wifi
└── wifi_location

SURVEY
├── BigFive
├── FlourishingScale
├── LonelinessScale
├── PHQ-9
├── PerceivedStressScale
├── PANAS
├── PSQI
└── VR-12
```

Important verified table sizes include:

| StudentLife table | Rows | Columns | Participants where reported |
|---|---:|---:|---:|
| activity | 22,842,191 | 3 | 49 |
| audio | 99,298,223 | 3 | 49 |
| bluetooth | 1,288,526 | 5 | 49 |
| conversation | 79,023 | 3 | 49 |
| gps | 202,877 | 11 | 49 |
| wifi | 19,244,309 | 5 | 49 |
| wifi_location | 1,893,838 | 3 | 49 |
| app_usage | 1,990,510 | 11 | 49 |
| call_log | 71,801 | 12 | 49 |
| sms | 92,584 | 4 | 49 |
| phonelock | 9,275 | 3 | 49 |
| phonecharge | 3,318 | 3 | 49 |
| pam | 9,040 | 3 | 49 |
| stress | 2,017 | 3 | 46 |
| mood | 277 | 7 | 38 |
| sleep | 1,644 | 6 | 49 |
| BigFive | 85 | 46 | 47 |
| FlourishingScale | 83 | 10 | 46 |
| LonelinessScale | 83 | 22 | 46 |
| PHQ-9 | 84 | 12 | 46 |
| PerceivedStressScale | 85 | 12 | 46 |
| PANAS | 85 | 20 | 47 |
| PSQI | 84 | 21 | 46 |
| VR-12 | 83 | 16 | 46 |

The project record also established that the StudentLife sensing layer is genuinely multimodal: activity inference, audio inference, Bluetooth, conversation, GPS, WiFi, WiFi-derived location, phone charging/locking and phone/application information are all available.

## 62A.2 StudentLife target coverage — VERIFIED

The three longitudinal EMA targets inspected for the proposed dynamic modeling work were:

```text
PAM   : 9,040 observations, 49 participants
Stress: 2,017 observations, 46 participants
Mood  :   277 observations, 38 participants
```

All three target timestamp columns were verified as Unix timestamps in seconds and normalized with:

```python
pd.to_datetime(timestamp, unit="s", errors="coerce")
```

No invalid timestamps were found:

```text
pam    | rows=9040 | participants=49 | invalid timestamps=0
stress | rows=2017 | participants=46 | invalid timestamps=0
mood   | rows= 277 | participants=38 | invalid timestamps=0
```

Verified normalized target ranges:

```text
PAM
start: 2013-03-24 08:40:30
end:   2013-07-13 23:47:02

Stress
start: 2013-03-24 08:40:01
end:   2013-08-16 00:56:08

Mood
start: 2013-04-24 23:05:35
end:   2013-08-10 03:44:09
```

## 62A.3 StudentLife longitudinal coverage per target

The verified participant-level summary was:

| Target | Participants | Median observations/person | Mean observations/person | Min | Max | Median duration (days) |
|---|---:|---:|---:|---:|---:|---:|
| Mood | 38 | 3 | 7.289474 | 1 | 83 | 21.007465 |
| PAM | 49 | 195 | 184.489796 | 8 | 437 | 67.345150 |
| Stress | 46 | 38 | 43.847826 | 4 | 112 | 57.764832 |

This is a critical modeling result:

```text
PAM   → relatively dense target
Stress → substantially sparser target
Mood  → very sparse target
```

Therefore PAM is the strongest StudentLife candidate for a dense longitudinal affect experiment, while stress remains scientifically important but must be treated as a sparse-label forecasting problem. Mood should not be used as the first dynamic target without a separate feasibility decision.

## 62A.4 StudentLife target-observation continuity

The prior analysis calculated the percentage of target observations followed by another observation within the specified window:

```text
Target   within 6h   within 12h   within 24h   within 48h   within 72h
PAM        66.7779      83.2610      95.4955      98.0647      98.9323
Stress     32.4708      46.8290      71.2329      85.0330      90.4110
Mood       14.6444      26.7782      49.7908      71.1297      79.0795
```

These numbers reinforce the sparsity difference between PAM, stress and mood. They should be used when deciding whether the first target should be strictly next-calendar-day, next-observed-label, or a broader temporal window.

> **Audit note:** the earlier project record also printed a target-gap-hours table whose numerical values were on the order of `e-09`. Those values are inconsistent with the actual multi-hour/day observation spacing and appear to be a timestamp-unit conversion artifact. They must NOT be used as scientific gap estimates. The directly reported within-window percentages above are retained as the more useful continuity result.

## 62A.5 StudentLife missingness findings that matter for modeling

Selected verified missingness findings include:

```text
Behavior
- behavior.null:              43.742255% missing
- class.null:                 79.865206% missing
- comment.null:               59.535655% missing
- social.null:                80.750799% missing
- stress.null:                88.101140% missing

Sleep
- sleep.location:             15.571776% missing
- sleep.hour:                 15.450122% missing
- sleep.rate:                 15.450122% missing
- sleep.social:               15.450122% missing

Mood
- mood.happy:                  0.361011% missing
- mood.sad:                    0.361011% missing
- mood.sadornot:               0.361011% missing
- mood.happyornot:             0.000000% missing

Exercise
- exercise.walk:               0.131062% missing

Call log
- CALLS_numberlabel:          99.988858% missing
- CALLS_name:                 93.098982% missing
- CALLS_numbertype:           91.068370% missing
- CALLS_duration:             91.014053% missing
- CALLS_type:                 91.014053% missing
- CALLS_date:                 91.014053% missing
```

This does NOT mean all high-missingness columns should simply be discarded. It means the final StudentLife feature table must be constructed at the feature-family level, with coverage, semantic meaning and sensor availability considered explicitly.

## 62A.6 StudentLife preprocessing that is actually complete

The verified StudentLife processing completed so far is:

```text
[✓] RDS inventory
[✓] 49 StudentLife tables loaded
[✓] table shapes and columns inspected
[✓] participant coverage inspected
[✓] target tables identified
[✓] target observation counts inspected
[✓] target participant coverage inspected
[✓] target longitudinal duration inspected
[✓] target continuity / next-observation windows inspected
[✓] raw target timestamps audited
[✓] Unix-second timestamp normalization completed
[✓] normalized target date ranges verified
[✓] feature-level missingness inspection completed

[ ] final participant-day table
[ ] final daily feature engineering
[ ] final cross-dataset feature mapping
[ ] next-day target construction
[ ] sequence construction
[ ] chronological train/test split
[ ] train-only imputation/normalization
[ ] baselines
[ ] GRU
[ ] personalization
[ ] Digital Twin latent state
[ ] what-if engine
[ ] uncertainty
[ ] sensitivity analysis
[ ] cross-dataset validation
```

**Do not describe StudentLife as already having a final modeling dataframe.** The verified work is data understanding, auditing, timestamp normalization and target-feasibility analysis. The next StudentLife implementation stage is the same conceptual participant-day transformation that CES now needs.

## 62A.7 What StudentLife now contributes to the research design

The verified StudentLife outputs change the strategy in an important but controlled way:

```text
StudentLife
   │
   ├── rich multimodal sensing
   ├── 49-person core sensing coverage
   ├── PAM: 9,040 observations / 49 participants
   ├── stress: 2,017 / 46 participants
   └── mood: 277 / 38 participants
          ↓
   target-specific feasibility
          ↓
   participant × day representation
          ↓
   common feature ontology
          ↓
   shared temporal/personalization framework
```

StudentLife should therefore remain the **development/primary research dataset in the original architecture**, while CES provides a larger, longer and independently collected environment for replication/generalization. The datasets are still NOT to be blindly concatenated.

---

---

# 63. Important provenance note

This handoff document deliberately distinguishes **verified outputs** from **planned methodology**.

The CES numerical outputs in Sections 6–21 are based on the processing results established in the current project conversation.

For StudentLife, exact numerical processing outputs have now been recovered from and cross-checked against the project record **"My strongest recommendation.docx"**. Section 62A records the verified StudentLife inventory, target counts, participant coverage, longitudinal ranges, continuity analysis, selected missingness findings, and timestamp normalization. Values not explicitly established in that record remain unverified and must not be invented.

---

# 64. Current status

```text
[✓] CES data stored in Drive
[✓] CES data loading
[✓] CES inventory
[✓] CES temporal coverage
[✓] participant overlap
[✓] EMA-sensing alignment
[✓] target inspection
[✓] target coverage
[✓] target co-occurrence
[✓] sensing feature-family inspection
[✓] broad EMA+sensing+steps modeling dataset
[✓] labeled-day feature quality analysis
[✓] missingness filter: 677 → 570
[✓] zero-variance check
[✓] StudentLife RDS inventory and table inspection
[✓] StudentLife target coverage / longitudinal feasibility analysis
[✓] StudentLife timestamp normalization and range verification
[✓] StudentLife missingness inspection
[ ] final daily feature engineering
[ ] next-day target construction
[ ] sequence construction
[ ] chronological train/test split
[ ] normalization/imputation fitted on train
[ ] baselines
[ ] GRU
[ ] personalization
[ ] Digital Twin latent state
[ ] what-if engine
[ ] uncertainty
[ ] sensitivity analysis
[ ] cross-dataset validation
[ ] paper experiments
```

**NEXT ACTION: construct and validate the participant-day modeling tables for BOTH StudentLife and CES, decide the first harmonized target from verified coverage, then build leakage-safe next-day sequences before any train/test split.**
