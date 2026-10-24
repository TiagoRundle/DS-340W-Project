#!/usr/bin/env python3
"""
FROZEN Stage-1 DEVELOPMENT REPRODUCTION.

Uses only the frozen Stage-1 TRAIN and TEST CSVs.

Locked methodology:
- 1PT representation (230 features)
- Random Forest
- 500 trees
- class_weight="balanced_subsample"
- soft aggregation: mean predicted class probabilities across test windows
- 10 RF seeds
- all seven tasks evaluated independently
- locked 50/50 HSS+RAN subject-evidence fusion

This script exists only to reproduce the final Stage-1 development result.
Do not change its modeling choices if the goal is Stage-1 reproduction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


TASKS = ("BLG", "FXS", "HSS", "RAN", "TEX", "VD1", "VD2")

SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORATION_DIR = SCRIPT_DIR.parent
DATA_DIR = EXPLORATION_DIR / "data"
RESULTS_DIR = EXPLORATION_DIR / "results"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^seg\d{2}_", c)]
    if len(cols) != 230:
        raise ValueError(f"Expected 230 feature columns, found {len(cols)}")
    return cols


def aligned_subject_evidence(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    cohort: list[int],
) -> dict[int, np.ndarray]:
    class_to_col = {int(c): i for i, c in enumerate(model_classes)}
    out: dict[int, np.ndarray] = {}

    for pid in cohort:
        mask = y_true == pid
        if not np.any(mask):
            continue

        mean_prob = probabilities[mask].mean(axis=0)
        aligned = np.zeros(len(cohort), dtype=np.float64)

        for j, class_id in enumerate(cohort):
            k = class_to_col.get(class_id)
            if k is not None:
                aligned[j] = mean_prob[k]

        out[pid] = aligned

    return out


def accuracy_from_evidence(
    evidence: dict[int, np.ndarray],
    cohort: list[int],
) -> tuple[float, int]:
    correct = 0

    for pid in cohort:
        scores = evidence.get(pid)
        if scores is None:
            continue
        pred = cohort[int(np.argmax(scores))]
        correct += int(pred == pid)

    return correct / len(cohort), correct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train",
        default=str(DATA_DIR / "development.csv"),
    )
    p.add_argument(
        "--test",
        default=str(DATA_DIR / "test.csv"),
    )
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output",
        default=str(RESULTS_DIR / "development_results.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(Path(args.train).resolve())
    test = pd.read_csv(Path(args.test).resolve())

    if set(train["split"].unique()) != {"train"}:
        raise ValueError("Training CSV contains non-training rows.")
    if set(test["split"].unique()) != {"test"}:
        raise ValueError("Testing CSV contains non-testing rows.")

    feature_cols = get_feature_cols(train)
    if get_feature_cols(test) != feature_cols:
        raise ValueError("Train/test feature columns differ.")

    cohort = sorted(
        set(train["participant_id"].astype(int))
        | set(test["participant_id"].astype(int))
    )
    if len(cohort) != 50:
        raise ValueError(f"Expected 50-subject cohort, found {len(cohort)}")

    print("=== FINAL STAGE-1 DEVELOPMENT RUN ===")
    print("All 7 tasks")
    print("Representation: 1PT")
    print(f"Random Forest trees: {args.trees}")
    print('Class weighting: "balanced_subsample"')
    print("Aggregation: mean RF class probabilities")
    print(f"Seeds: 0..{args.seeds - 1}")
    print("Selected task fusion: HSS/RAN = 0.50/0.50")
    print("Historical Stage-1 development reproduction")
    print("\nNO FURTHER TUNING SHOULD BE DONE FROM THIS RUN.")

    task_data = {}
    for task in TASKS:
        tr = train[train["task"] == task].copy()
        te = test[test["task"] == task].copy()

        task_data[task] = (
            tr[feature_cols].to_numpy(dtype=np.float32),
            tr["participant_id"].to_numpy(dtype=int),
            te[feature_cols].to_numpy(dtype=np.float32),
            te["participant_id"].to_numpy(dtype=int),
        )

    rows = []

    for seed in range(args.seeds):
        evidence_by_task = {}

        for task in TASKS:
            Xtr, ytr, Xte, yte = task_data[task]

            model = RandomForestClassifier(
                n_estimators=args.trees,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            model.fit(Xtr, ytr)

            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)

            vector_acc = accuracy_score(yte, pred)
            evidence = aligned_subject_evidence(
                yte,
                proba,
                model.classes_,
                cohort,
            )
            subject_acc, subject_correct = accuracy_from_evidence(
                evidence,
                cohort,
            )

            evidence_by_task[task] = evidence

            rows.append(
                {
                    "system": task,
                    "seed": seed,
                    "test_vector_accuracy": vector_acc,
                    "subject_accuracy": subject_acc,
                    "correct_out_of_50": subject_correct,
                }
            )

        # Locked 50/50 fusion of the two leading acquisition protocols.
        fused = {}
        for pid in cohort:
            h = evidence_by_task["HSS"].get(pid)
            r = evidence_by_task["RAN"].get(pid)

            if h is not None and r is not None:
                fused[pid] = 0.5 * h + 0.5 * r
            elif h is not None:
                fused[pid] = h.copy()
            elif r is not None:
                fused[pid] = r.copy()

        fused_acc, fused_correct = accuracy_from_evidence(fused, cohort)

        rows.append(
            {
                "system": "HSS+RAN_50_50",
                "seed": seed,
                "test_vector_accuracy": np.nan,
                "subject_accuracy": fused_acc,
                "correct_out_of_50": fused_correct,
            }
        )

    out = pd.DataFrame(rows)

    print("\n=== FINAL DEVELOPMENT SUMMARY ===")
    summary = (
        out.groupby("system")["subject_accuracy"]
        .agg(["mean", "std", "min", "max"])
        * 100
    )

    preferred_order = list(TASKS) + ["HSS+RAN_50_50"]
    summary = summary.reindex(preferred_order)
    print(summary.round(2).to_string())

    print("\n=== VECTOR ACCURACY BY SINGLE TASK ===")
    vector_summary = (
        out[out["system"].isin(TASKS)]
        .groupby("system")["test_vector_accuracy"]
        .agg(["mean", "std"])
        * 100
    ).reindex(TASKS)
    print(vector_summary.round(2).to_string())

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("Frozen Stage-1 development result reproduced.")
    print(
        "\nStage-1 development methodology is now frozen. "
        "Do not change settings based on these final development results."
    )


if __name__ == "__main__":
    main()
