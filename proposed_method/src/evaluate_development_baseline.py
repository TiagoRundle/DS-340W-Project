#!/usr/bin/env python3
"""
Development cross-session baseline.

Purpose:
- Establish the frozen Stage-1 HSS/RAN Random Forest system as the Stage-2
  baseline on the fixed 50-person Stage-2 cohort.
- Evaluate both Round-1 session-separated directions:
    A) R1-S1 -> R1-S2
    B) R1-S2 -> R1-S1

Important:
- Round 2 validation files are NOT loaded or inspected.
- No tuning is performed here.
- Modeling choices are inherited from the frozen Stage-1 system:
    * 1PT / 230 features
    * RandomForestClassifier
    * 500 trees
    * class_weight="balanced_subsample"
    * 10 RF seeds
    * subject evidence = mean predicted class probabilities across windows
    * HSS/RAN fusion = 50/50 at subject-evidence level
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TASKS = ("HSS", "RAN")

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
DATA_DIR = METHOD_DIR / "data"
RESULTS_DIR = METHOD_DIR / "results"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^seg\d{2}_", c)]
    if len(cols) != 230:
        raise ValueError(f"Expected 230 feature columns, found {len(cols)}")
    return cols


def validate_input(df: pd.DataFrame, expected_session: str) -> None:
    if set(df["session"].unique()) != {expected_session}:
        raise ValueError(
            f"Expected only session {expected_session}, "
            f"found {sorted(df['session'].unique())}"
        )

    if set(df["task"].unique()) != set(TASKS):
        raise ValueError(
            f"Expected tasks {TASKS}, found {sorted(df['task'].unique())}"
        )

    participants = sorted(df["participant_id"].astype(int).unique())
    if len(participants) != 50:
        raise ValueError(
            f"Expected 50 participants in {expected_session}, found {len(participants)}"
        )

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features detected in {expected_session}")


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
            raise ValueError(f"Missing subject evidence for participant {pid}")

        pred = cohort[int(np.argmax(scores))]
        correct += int(pred == pid)

    return correct / len(cohort), correct


def evaluate_direction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_session: str,
    test_session: str,
    trees: int,
    seeds: int,
    feature_cols: list[str],
    cohort: list[int],
) -> list[dict]:
    rows: list[dict] = []

    task_data = {}
    for task in TASKS:
        tr = train[train["task"] == task]
        te = test[test["task"] == task]

        task_data[task] = (
            tr[feature_cols].to_numpy(dtype=np.float32),
            tr["participant_id"].to_numpy(dtype=int),
            te[feature_cols].to_numpy(dtype=np.float32),
            te["participant_id"].to_numpy(dtype=int),
        )

    for seed in range(seeds):
        evidence_by_task: dict[str, dict[int, np.ndarray]] = {}

        for task in TASKS:
            Xtr, ytr, Xte, yte = task_data[task]

            model = RandomForestClassifier(
                n_estimators=trees,
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
                    "direction": f"{train_session}->{test_session}",
                    "train_session": train_session,
                    "test_session": test_session,
                    "system": task,
                    "seed": seed,
                    "vector_accuracy": vector_acc,
                    "subject_accuracy": subject_acc,
                    "correct_out_of_50": subject_correct,
                }
            )

        fused = {}
        for pid in cohort:
            h = evidence_by_task["HSS"][pid]
            r = evidence_by_task["RAN"][pid]
            fused[pid] = 0.5 * h + 0.5 * r

        fused_acc, fused_correct = accuracy_from_evidence(fused, cohort)

        rows.append(
            {
                "direction": f"{train_session}->{test_session}",
                "train_session": train_session,
                "test_session": test_session,
                "system": "HSS+RAN_50_50",
                "seed": seed,
                "vector_accuracy": np.nan,
                "subject_accuracy": fused_acc,
                "correct_out_of_50": fused_correct,
            }
        )

    return rows


def print_summary(out: pd.DataFrame) -> None:
    preferred = list(TASKS) + ["HSS+RAN_50_50"]

    print("\n=== SUBJECT-LEVEL CROSS-SESSION BASELINE ===")
    for direction in ("S1->S2", "S2->S1"):
        print(f"\n{direction}")
        sub = out[out["direction"] == direction]
        summary = (
            sub.groupby("system")["subject_accuracy"]
            .agg(["mean", "std", "min", "max"])
            .mul(100)
            .reindex(preferred)
        )
        print(summary.round(2).to_string())

    print("\n=== VECTOR ACCURACY BY TASK ===")
    for direction in ("S1->S2", "S2->S1"):
        print(f"\n{direction}")
        sub = out[
            (out["direction"] == direction)
            & (out["system"].isin(TASKS))
        ]
        summary = (
            sub.groupby("system")["vector_accuracy"]
            .agg(["mean", "std"])
            .mul(100)
            .reindex(TASKS)
        )
        print(summary.round(2).to_string())

    fused = out[out["system"] == "HSS+RAN_50_50"].copy()
    per_seed = (
        fused.pivot(index="seed", columns="direction", values="subject_accuracy")
        .reindex(columns=["S1->S2", "S2->S1"])
    )
    per_seed["bidirectional_mean"] = per_seed.mean(axis=1)

    print("\n=== FUSED BIDIRECTIONAL SUMMARY ===")
    print(
        (per_seed.agg(["mean", "std", "min", "max"]) * 100)
        .round(2)
        .to_string()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--s1",
        default=str(DATA_DIR / "r1_s1_classical_features.csv"),
    )
    p.add_argument(
        "--s2",
        default=str(DATA_DIR / "r1_s2_classical_features.csv"),
    )
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output",
        default=str(RESULTS_DIR / "development_baseline_results.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    s1 = pd.read_csv(Path(args.s1).resolve())
    s2 = pd.read_csv(Path(args.s2).resolve())

    validate_input(s1, "S1")
    validate_input(s2, "S2")

    feature_cols = get_feature_cols(s1)
    if get_feature_cols(s2) != feature_cols:
        raise ValueError("S1 and S2 feature columns differ.")

    cohort_s1 = sorted(s1["participant_id"].astype(int).unique())
    cohort_s2 = sorted(s2["participant_id"].astype(int).unique())
    if cohort_s1 != cohort_s2:
        raise ValueError("S1 and S2 participant cohorts differ.")

    cohort = cohort_s1

    print("=== DEVELOPMENT CROSS-SESSION BASELINE ===")
    print("Data: Round 1 only")
    print("Directions: S1->S2 and S2->S1")
    print("Participants: 50")
    print("Tasks: HSS + RAN")
    print("Representation: 1PT / 230 features")
    print(f"Random Forest trees: {args.trees}")
    print('Class weighting: "balanced_subsample"')
    print("Aggregation: mean RF class probabilities")
    print("Fusion: HSS/RAN = 0.50/0.50")
    print(f"Seeds: 0..{args.seeds - 1}")
    print("Round 2 validation is NOT loaded by this script.")

    rows = []
    rows.extend(
        evaluate_direction(
            train=s1,
            test=s2,
            train_session="S1",
            test_session="S2",
            trees=args.trees,
            seeds=args.seeds,
            feature_cols=feature_cols,
            cohort=cohort,
        )
    )
    rows.extend(
        evaluate_direction(
            train=s2,
            test=s1,
            train_session="S2",
            test_session="S1",
            trees=args.trees,
            seeds=args.seeds,
            feature_cols=feature_cols,
            cohort=cohort,
        )
    )

    out = pd.DataFrame(rows)
    print_summary(out)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("\nDevelopment cross-session baseline complete.")


if __name__ == "__main__":
    main()
