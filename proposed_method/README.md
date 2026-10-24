# Proposed Cross-Session Method

This directory contains the proposed comparative experiment for improving cross-session eye-movement biometric identification.

The central question is whether motion- and saccade-specific feature engineering can improve robustness across recording sessions compared with the classical parent-inspired representation.

## Proposed Representation

The proposed representation contains 144 features:

- 123 Yin-inspired motion-information features
- 19 principal-saccade dynamic features
- 1 saccadic-vigor feature
- 1 missing-saccade indicator

### Motion Information

The motion features summarize displacement magnitude and direction throughout a 1-second gaze response.

They include:

- temporal motion bins
- displacement magnitude statistics
- direction distributions
- directional consistency
- turning behavior
- path length
- net displacement
- speed statistics

### Saccade Dynamics

The largest valid principal saccade in each response is characterized using features such as:

- onset latency
- duration
- amplitude
- path length
- trajectory efficiency
- peak and mean speed
- acceleration and deceleration
- time to peak speed
- trajectory deviation
- overshoot

Saccadic vigor is calculated using a parameter estimated only from the training data. Missing valid saccades are handled using training-derived imputation and an explicit missing-event indicator.

## Experimental Design

Development uses Round 1 with sessions kept strictly separate:

- train R1-S1 and test R1-S2
- train R1-S2 and test R1-S1

The fixed cohort contains 50 participants with sufficient HSS and RAN data.

The comparison keeps the classifier and evidence aggregation fixed so that the primary experimental difference is the feature representation.

The parent-inspired comparison uses 230 classical features.

The proposed method uses the 144-feature motion and saccade representation.

HSS and RAN are modeled separately and combined using equal subject-level probability fusion.

Round 2 is reserved as an external temporal evaluation after the method is fixed. No feature or model tuning should be performed using Round-2 classification performance.

## Files

```text
proposed_method/
├── data/
│   └── cohort_ids.csv
├── results/
└── src/
    ├── build_datasets.py
    ├── evaluate_development_baseline.py
    ├── feature_engineering.py
    └── evaluate_comparison.py
```

`build_datasets.py` constructs the fixed cohort and parent-inspired datasets.

`feature_engineering.py` contains the final motion and saccade feature extraction logic.

`evaluate_development_baseline.py` evaluates the classical cross-session comparison on Round 1.

`evaluate_comparison.py` implements the fixed comparison between the parent-inspired and proposed representations.

Generated feature caches and final evaluation outputs are stored locally and excluded from version control during the midterm stage.