# Parent-Paper Replication

This directory contains the reproduction and adaptation of the classical eye-movement biometric methodology described by Harezlak et al. (2021).

The goal of this section is to establish that the parent-paper feature engineering and Random Forest approach can be reproduced on the GazeBase dataset before extending the project to a new research question.

## Method

Each target response contributes approximately 1 second of gaze data sampled at 1000 Hz.

The response is divided into ten non-overlapping 100-sample segments. Classical eye-movement features are extracted from each segment, including:

- horizontal and vertical velocity statistics
- resultant velocity statistics
- acceleration statistics
- frequency-domain features
- nonlinear dynamics features

The segment features are concatenated into a fixed-length representation and evaluated with classical machine-learning models, with Random Forest serving as the main comparison model.

The replication includes both single-response and three-response representations. The three-response Random Forest configuration reproduced the parent-paper performance reasonably closely, especially after subject-level aggregation.

## Files

```text
replication/
├── data/
│   └── replication_features_100point.csv
└── src/
    ├── preprocess_replication.py
    └── evaluate_replication.py
```

`preprocess_replication.py` builds the classical eye-movement feature representation from GazeBase recordings.

`evaluate_replication.py` evaluates the resulting feature vectors using the parent-inspired classification procedure.

The generated feature CSV is kept locally and excluded from version control because it can be reproduced from the source data.