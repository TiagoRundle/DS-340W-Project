# Eye-Tracking Biometrics with GazeBase

This project investigates whether eye-movement patterns can be used for biometric identification and whether feature engineering can improve robustness across recording sessions.

The work begins by reproducing a classical eye-movement biometric approach, then uses the GazeBase dataset to explore acquisition protocols and identify a cross-session generalization problem. The proposed experiment combines motion-based and saccade-specific features to test whether a representation designed around eye-movement dynamics can improve identification across recording sessions.

## Research Question

Can motion- and saccade-specific feature engineering improve cross-session eye-movement biometric identification compared with a classical feature representation adapted from Harezlak et al.?

## Project Progression

### 1. Replication

The first part of the project adapts the classical machine-learning methodology from Harezlak et al. (2021) to GazeBase.

The approach uses:

- 1-second eye-movement responses sampled at approximately 1000 Hz
- 10 non-overlapping 100-sample segments per response
- classical velocity, acceleration, frequency-domain, and nonlinear features
- Random Forest classification
- subject-level aggregation of multiple eye-movement responses

The three-point Random Forest configuration reproduced the parent-paper performance reasonably closely, particularly at the subject level.

### 2. Exploratory Protocol Selection

The next part of the project explores GazeBase acquisition tasks to identify a practical protocol for further experimentation.

Horizontal Saccades (HSS) and Random Saccades (RAN) produced the strongest biometric signal and were selected for continued study.

Using fixed Random Forest settings and equal HSS/RAN evidence fusion:

- development accuracy reached approximately 95.8%
- evaluation on a separate recording session dropped to approximately 76.2%

This decrease motivates the central problem studied in the proposed experiment: eye-movement biometric features may perform well within a recording condition but generalize poorly when the recording session changes.

### 3. Proposed Method

The proposed method focuses on features designed to represent eye-movement dynamics more directly.

It combines:

- 123 motion-information features inspired by Yin et al. (2022)
- 19 principal-saccade dynamic features
- a training-derived saccadic-vigor feature
- a missing-saccade indicator

The final proposed representation contains 144 features.

The motion features describe displacement magnitude, direction, temporal structure, turning behavior, path length, and speed. The saccade features describe properties such as amplitude, duration, peak speed, acceleration, deceleration, trajectory efficiency, and shape.

A Random Forest classifier is used so that improvements can be attributed primarily to the feature representation rather than to a substantially different modeling architecture.

## Experimental Design

The primary task is closed-set biometric identification among 50 participants.

Development uses Round 1 of GazeBase with recording sessions kept strictly separate:

- train on R1-S1 and test on R1-S2
- train on R1-S2 and test on R1-S1

HSS and RAN are modeled separately and combined using equal subject-level probability fusion.

After the feature representation and evaluation procedure are fixed, Round 2 is reserved for external temporal evaluation. The proposed method is compared with the parent-inspired 230-feature representation under the same participant cohort, classifier settings, and evidence-aggregation procedure.

No model or feature tuning is intended to be performed using Round-2 classification performance.

## Repository Structure

```text
.
├── replication/
│   ├── data/
│   └── src/
│
├── exploration/
│   ├── data/
│   ├── results/
│   └── src/
│
├── proposed_method/
│   ├── data/
│   ├── results/
│   └── src/
│
├── data/
│   └── gazebase_manifest.csv
│
└── README.md
```

### `replication/`

Contains the preprocessing and evaluation code used to reproduce the parent-paper methodology on GazeBase.

### `exploration/`

Contains the experiments used to select the HSS/RAN protocol and evaluate its robustness across recording sessions.

### `proposed_method/`

Contains the fixed-cohort dataset construction, final feature engineering, development baseline, and planned comparative evaluation.

### `data/`

Contains a manifest describing the local GazeBase recordings. The original GazeBase dataset is not included in this repository.

## Dataset

This project uses the GazeBase dataset introduced by Griffith et al. (2021).

GazeBase contains repeated eye-tracking recordings collected at approximately 1000 Hz across multiple rounds, sessions, participants, and visual tasks.

The raw dataset must be obtained separately. The repository assumes extracted recordings are placed under:

```text
data/raw/
```

Raw GazeBase files and generated feature datasets are excluded from version control.

## Primary References

**Harezlak, K., Blasiak, P., & Kasprowski, P. (2021).**  
*Biometric Identification Based on Eye Movement Dynamic Features.*  
Provides the parent classical feature-engineering and machine-learning methodology.

**Griffith, H. et al. (2021).**  
*GazeBase, a large-scale, multi-stimulus, longitudinal eye movement dataset.*  
Provides the dataset used for replication, exploration, and comparative experimentation.

**Yin, J., Sun, J., Li, J., & Liu, K. (2022).**  
*An Effective Gaze-Based Authentication Method with the Spatiotemporal Feature of Eye Movement.*  
Motivates the motion-information representation used in the proposed method.

Additional saccade-dynamics ideas are informed by prior eye-movement biometric research on saccadic vigor and acceleration.

## Notes

The current task is biometric **identification**, not verification/authentication. The system chooses one identity from the enrolled participant set.

The project is designed to emphasize reproducibility, session-separated evaluation, and feature engineering that remains practical on standard CPU hardware.