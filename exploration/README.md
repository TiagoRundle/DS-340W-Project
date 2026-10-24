# Exploratory Protocol Selection

This directory contains the exploratory experiments used to select the GazeBase acquisition protocol for the main study and to identify the cross-session generalization problem.

The exploration uses a fixed 50-participant closed-set identification problem and compares evidence from different recording tasks.

## Main Finding

Horizontal Saccades (HSS) and Random Saccades (RAN) provided the strongest biometric signal and were selected for continued study.

Using fixed Random Forest settings and equal HSS/RAN subject-level evidence fusion:

- development accuracy reached approximately 95.8%
- performance on a separate recording session dropped to approximately 76.2%

The large decrease suggested that the classical representation was sensitive to session changes even when the same identities and acquisition tasks were used.

This finding motivates the research question addressed by the proposed method.

## Evaluation Setup

The exploratory evaluation uses:

- 50 enrolled identities
- HSS and RAN recordings
- 230 classical features per response
- Random Forest with 500 trees
- `balanced_subsample` class weighting
- mean predicted class probabilities across responses
- equal HSS/RAN evidence fusion
- multiple random seeds for stability

The development procedure is separated from the later recording-session evaluation so that the generalization result remains an honest out-of-session test.

## Files

```text
exploration/
├── data/
│   ├── development.csv
│   ├── test.csv
│   └── validation.csv
├── results/
│   ├── development_results.csv
│   └── validation_results.csv
└── src/
    ├── evaluate_development.py
    └── evaluate_validation.py
```

The processed datasets are generated locally and excluded from version control. The small result tables may be retained for reproducibility and presentation support.