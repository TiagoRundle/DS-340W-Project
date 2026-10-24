#!/usr/bin/env python3
"""
FROZEN Stage-1 VALIDATION REPRODUCTION.

Reproduces the historical Round-1 Session-2 validation result from the frozen
Stage-1 experiment.

Locked methodology:
- final fitting data = official Stage-1 TRAIN + TEST rows from R1-S1
- validation data = official frozen Stage-1 VALIDATION rows from R1-S2
- 1PT representation (230 features)
- Random Forest
- 500 trees
- class_weight="balanced_subsample"
- soft aggregation: mean predicted class probabilities
- 10 RF seeds
- all seven tasks evaluated independently
- locked 50/50 HSS+RAN subject-evidence fusion

This script exists only to reproduce the historical Stage-1 validation result.
Do not use it to retroactively tune Exploratory.
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


def aligned_subject_evidence(y_true, probabilities, model_classes, cohort):
    class_to_col = {int(c): i for i, c in enumerate(model_classes)}
    out = {}

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


def accuracy_from_evidence(evidence, cohort):
    correct = 0

    for pid in cohort:
        scores = evidence.get(pid)
        if scores is None:
            continue

        pred = cohort[int(np.argmax(scores))]
        correct += int(pred == pid)

    return correct / len(cohort), correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=str(DATA_DIR / "development.csv"))
    p.add_argument("--test", default=str(DATA_DIR / "test.csv"))
    p.add_argument("--validation", default=str(DATA_DIR / "validation.csv"))
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output",
        default=str(RESULTS_DIR / "validation_results.csv"),
    )
    return p.parse_args()


def main():
    args = parse_args()

    train = pd.read_csv(Path(args.train).resolve())
    test = pd.read_csv(Path(args.test).resolve())
    validation = pd.read_csv(Path(args.validation).resolve())

    if set(train["split"].unique()) != {"train"}:
        raise ValueError("Training CSV contains non-training rows.")
    if set(test["split"].unique()) != {"test"}:
        raise ValueError("Testing CSV contains non-testing rows.")
    if set(validation["split"].unique()) != {"validation"}:
        raise ValueError("Validation CSV contains non-validation rows.")

    feature_cols = get_feature_cols(train)
    if get_feature_cols(test) != feature_cols:
        raise ValueError("Train/test feature columns differ.")
    if get_feature_cols(validation) != feature_cols:
        raise ValueError("Train/validation feature columns differ.")

    development = pd.concat([train, test], ignore_index=True)

    cohort = sorted(
        set(development["participant_id"].astype(int))
        | set(validation["participant_id"].astype(int))
    )
    if len(cohort) != 50:
        raise ValueError(f"Expected 50-subject cohort, found {len(cohort)}")

    dev_subjects = set(development["participant_id"].astype(int))
    val_subjects = set(validation["participant_id"].astype(int))
    if dev_subjects != val_subjects:
        raise ValueError("Development/validation participant sets do not match.")

    print("=== FINAL STAGE-1 VALIDATION ===")
    print("Training data: ALL S1 development rows (official train + test)")
    print("Validation data: frozen historical S2 holdout")
    print("Representation: 1PT")
    print(f"Random Forest trees: {args.trees}")
    print('Class weighting: "balanced_subsample"')
    print("Aggregation: mean RF class probabilities")
    print(f"Seeds: 0..{args.seeds - 1}")
    print("Locked task fusion: HSS/RAN = 0.50/0.50")
    print("\nNO TUNING IS PERFORMED IN THIS SCRIPT.")

    task_data = {}
    for task in TASKS:
        dev_task = development[development["task"] == task].copy()
        val_task = validation[validation["task"] == task].copy()

        task_data[task] = (
            dev_task[feature_cols].to_numpy(dtype=np.float32),
            dev_task["participant_id"].to_numpy(dtype=int),
            val_task[feature_cols].to_numpy(dtype=np.float32),
            val_task["participant_id"].to_numpy(dtype=int),
        )

    rows = []

    for seed in range(args.seeds):
        evidence_by_task = {}

        for task in TASKS:
            Xdev, ydev, Xval, yval = task_data[task]

            model = RandomForestClassifier(
                n_estimators=args.trees,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            model.fit(Xdev, ydev)

            pred = model.predict(Xval)
            proba = model.predict_proba(Xval)

            vector_acc = accuracy_score(yval, pred)

            evidence = aligned_subject_evidence(
                yval, proba, model.classes_, cohort
            )
            subject_acc, subject_correct = accuracy_from_evidence(
                evidence, cohort
            )

            evidence_by_task[task] = evidence

            rows.append(
                {
                    "system": task,
                    "seed": seed,
                    "validation_vector_accuracy": vector_acc,
                    "validation_subject_accuracy": subject_acc,
                    "correct_out_of_50": subject_correct,
                }
            )

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
                "validation_vector_accuracy": np.nan,
                "validation_subject_accuracy": fused_acc,
                "correct_out_of_50": fused_correct,
            }
        )

    out = pd.DataFrame(rows)
    preferred_order = list(TASKS) + ["HSS+RAN_50_50"]

    print("\n=== VALIDATION SUBJECT-LEVEL SUMMARY ===")
    summary = (
        out.groupby("system")["validation_subject_accuracy"]
        .agg(["mean", "std", "min", "max"])
        * 100
    ).reindex(preferred_order)
    print(summary.round(2).to_string())

    print("\n=== VALIDATION VECTOR ACCURACY BY SINGLE TASK ===")
    vector_summary = (
        out[out["system"].isin(TASKS)]
        .groupby("system")["validation_vector_accuracy"]
        .agg(["mean", "std"])
        * 100
    ).reindex(TASKS)
    print(vector_summary.round(2).to_string())

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("\nFrozen Stage-1 validation result reproduced.")


if __name__ == "__main__":
    main()
