#!/usr/bin/env python3
"""
Frozen parent-inspired vs proposed-method comparison.

This is the clean final evaluator for the project. It performs NO tuning.

Parent-inspired baseline:
- 230 Harezlak-style features.

Proposed method:
- 123 Yin-inspired motion-information features,
- 19 principal-saccade dynamics features,
- 1 training-only saccadic-vigor feature,
- 1 missing-event indicator,
- 144 total features.

Evaluation:
- Random Forest, 500 trees,
- class_weight="balanced_subsample",
- seeds 0..9,
- HSS and RAN modeled separately,
- mean predicted class probabilities across windows,
- fixed 50/50 HSS/RAN evidence fusion.

External validation:
- train on all Round-1 S1+S2 data,
- evaluate untouched Round-2 S1 and S2 separately,
- combined Round-2 evidence is secondary only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import (
    TASKS,
    KEYS,
    build_round_feature_caches,
    estimate_training_bavg,
    add_vigor,
)


SESSIONS = ("S1", "S2")
SYSTEMS = (
    "BASELINE_FULL230",
    "YIN_ONLY123",
    "FINAL_YIN_SACCADE144",
)

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
PROJECT_DIR = METHOD_DIR.parent
DATA_DIR = METHOD_DIR / "data"
RESULTS_DIR = METHOD_DIR / "results"
RAW_DIR = PROJECT_DIR / "data" / "raw"


def load_cohort(path: Path) -> list[int]:
    df = pd.read_csv(path)

    if "participant_id" not in df.columns:
        raise ValueError("Cohort file must contain participant_id")

    cohort = sorted(df["participant_id"].astype(int).tolist())

    if len(cohort) != 50 or len(set(cohort)) != 50:
        raise ValueError(
            f"Expected exactly 50 unique cohort IDs; found {len(set(cohort))}"
        )

    return cohort


def get_yin_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("yin_mi_")]
    if len(cols) != 123:
        raise ValueError(f"Expected 123 Yin features; found {len(cols)}")
    return cols


def get_saccade_base_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("sd_")]
    if len(cols) != 19:
        raise ValueError(
            f"Expected 19 base saccade features; found {len(cols)}"
        )
    return cols


def get_full230_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^seg\d{2}_", c)]
    if len(cols) != 230:
        raise ValueError(
            f"Expected 230 parent-inspired features; found {len(cols)}"
        )
    return cols


def make_full_saccade_table(
    yin: pd.DataFrame,
    saccade: pd.DataFrame,
) -> pd.DataFrame:
    out = yin.merge(
        saccade,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    out["sd_missing_event"] = (
        out["_merge"] == "left_only"
    ).astype(np.int8)
    return out.drop(columns=["_merge"])


def prepare_final_task(
    train_task: pd.DataFrame,
    test_task: pd.DataFrame,
    saccade_base_cols: list[str],
):
    """
    Add training-only vigor and training-only median imputation.
    """
    valid_train = train_task[
        train_task["sd_missing_event"] == 0
    ].copy()

    if valid_train.empty:
        raise ValueError("No valid training saccades for task")

    bavg = estimate_training_bavg(valid_train)

    train = add_vigor(train_task, bavg)
    test = add_vigor(test_task, bavg)

    imputed_cols = saccade_base_cols + ["sd_vigor"]

    training_medians = (
        train.loc[
            train["sd_missing_event"] == 0,
            imputed_cols,
        ]
        .median()
    )

    if training_medians.isna().any():
        bad = training_medians[
            training_medians.isna()
        ].index.tolist()
        raise ValueError(f"Undefined training medians: {bad}")

    train.loc[:, imputed_cols] = (
        train[imputed_cols].fillna(training_medians)
    )
    test.loc[:, imputed_cols] = (
        test[imputed_cols].fillna(training_medians)
    )

    model_saccade_cols = imputed_cols + ["sd_missing_event"]

    return train, test, model_saccade_cols, bavg


def aligned_subject_evidence(
    y_true,
    probabilities,
    model_classes,
    cohort,
):
    class_to_col = {int(c): i for i, c in enumerate(model_classes)}
    evidence = {}

    for pid in cohort:
        mask = y_true == pid

        if not np.any(mask):
            raise ValueError(f"No test windows for participant {pid}")

        mean_probability = probabilities[mask].mean(axis=0)
        aligned = np.zeros(len(cohort), dtype=np.float64)

        for j, class_id in enumerate(cohort):
            col = class_to_col.get(class_id)
            if col is not None:
                aligned[j] = mean_probability[col]

        evidence[pid] = aligned

    return evidence


def accuracy_from_evidence(evidence, cohort):
    correct = 0

    for pid in cohort:
        prediction = cohort[int(np.argmax(evidence[pid]))]
        correct += int(prediction == pid)

    return correct / len(cohort), correct


def fit_rf(train, feature_cols, seed, trees):
    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(
        train[feature_cols].to_numpy(dtype=np.float32),
        train["participant_id"].to_numpy(dtype=int),
    )
    return model


def predict_subject_evidence(
    model,
    test,
    feature_cols,
    cohort,
):
    probabilities = model.predict_proba(
        test[feature_cols].to_numpy(dtype=np.float32)
    )

    return aligned_subject_evidence(
        test["participant_id"].to_numpy(dtype=int),
        probabilities,
        model.classes_,
        cohort,
    )


def validate_baseline(
    r1_base,
    r2_base,
    full_cols,
    cohort,
    seeds,
    trees,
):
    rows = []

    for seed in range(seeds):
        store = {}

        for task in TASKS:
            train_task = r1_base[r1_base["task"] == task]
            model = fit_rf(
                train_task,
                full_cols,
                seed,
                trees,
            )

            store[task] = {}

            for session in SESSIONS:
                test_task = r2_base[
                    (r2_base["task"] == task)
                    & (r2_base["session"] == session)
                ]

                evidence = predict_subject_evidence(
                    model,
                    test_task,
                    full_cols,
                    cohort,
                )
                store[task][session] = evidence

                accuracy, correct = accuracy_from_evidence(
                    evidence,
                    cohort,
                )

                rows.append(
                    {
                        "system": "BASELINE_FULL230",
                        "seed": seed,
                        "evaluation": f"R2-{session}",
                        "task_system": task,
                        "subject_accuracy": accuracy,
                        "correct_out_of_50": correct,
                    }
                )

        for session in SESSIONS:
            fused = {
                pid: (
                    0.5 * store["HSS"][session][pid]
                    + 0.5 * store["RAN"][session][pid]
                )
                for pid in cohort
            }

            accuracy, correct = accuracy_from_evidence(
                fused,
                cohort,
            )

            rows.append(
                {
                    "system": "BASELINE_FULL230",
                    "seed": seed,
                    "evaluation": f"R2-{session}",
                    "task_system": "HSS+RAN_50_50",
                    "subject_accuracy": accuracy,
                    "correct_out_of_50": correct,
                }
            )

        combined_task = {}

        for task in TASKS:
            combined_task[task] = {
                pid: (
                    0.5 * store[task]["S1"][pid]
                    + 0.5 * store[task]["S2"][pid]
                )
                for pid in cohort
            }

        combined_fused = {
            pid: (
                0.5 * combined_task["HSS"][pid]
                + 0.5 * combined_task["RAN"][pid]
            )
            for pid in cohort
        }

        accuracy, correct = accuracy_from_evidence(
            combined_fused,
            cohort,
        )

        rows.append(
            {
                "system": "BASELINE_FULL230",
                "seed": seed,
                "evaluation": "R2-S1+S2_SECONDARY",
                "task_system": "HSS+RAN_50_50",
                "subject_accuracy": accuracy,
                "correct_out_of_50": correct,
            }
        )

    return rows


def validate_proposed(
    r1_final,
    r2_final,
    yin_cols,
    saccade_base_cols,
    cohort,
    seeds,
    trees,
):
    rows = []
    diagnostics = []
    prepared = {}

    for task in TASKS:
        train_task = r1_final[
            r1_final["task"] == task
        ].copy()
        test_task = r2_final[
            r2_final["task"] == task
        ].copy()

        (
            train_task,
            test_task,
            model_saccade_cols,
            bavg,
        ) = prepare_final_task(
            train_task,
            test_task,
            saccade_base_cols,
        )

        final_cols = yin_cols + model_saccade_cols

        if len(final_cols) != 144:
            raise ValueError(
                f"Expected 144 final features for {task}; "
                f"found {len(final_cols)}"
            )

        prepared[task] = (
            train_task,
            test_task,
            final_cols,
        )

        diagnostics.append(
            {
                "task": task,
                "r1_training_bavg": bavg,
                "r1_missing_events": int(
                    train_task["sd_missing_event"].sum()
                ),
                "r2_missing_events": int(
                    test_task["sd_missing_event"].sum()
                ),
            }
        )

    for seed in range(seeds):
        store = {
            "YIN_ONLY123": {},
            "FINAL_YIN_SACCADE144": {},
        }

        for task in TASKS:
            train_task, test_task, final_cols = prepared[task]

            yin_model = fit_rf(
                train_task,
                yin_cols,
                seed,
                trees,
            )
            final_model = fit_rf(
                train_task,
                final_cols,
                seed,
                trees,
            )

            store["YIN_ONLY123"][task] = {}
            store["FINAL_YIN_SACCADE144"][task] = {}

            for session in SESSIONS:
                test_session = test_task[
                    test_task["session"] == session
                ]

                yin_evidence = predict_subject_evidence(
                    yin_model,
                    test_session,
                    yin_cols,
                    cohort,
                )
                final_evidence = predict_subject_evidence(
                    final_model,
                    test_session,
                    final_cols,
                    cohort,
                )

                store["YIN_ONLY123"][task][session] = yin_evidence
                store["FINAL_YIN_SACCADE144"][task][session] = (
                    final_evidence
                )

                for system, evidence in (
                    ("YIN_ONLY123", yin_evidence),
                    ("FINAL_YIN_SACCADE144", final_evidence),
                ):
                    accuracy, correct = accuracy_from_evidence(
                        evidence,
                        cohort,
                    )

                    rows.append(
                        {
                            "system": system,
                            "seed": seed,
                            "evaluation": f"R2-{session}",
                            "task_system": task,
                            "subject_accuracy": accuracy,
                            "correct_out_of_50": correct,
                        }
                    )

        for system in (
            "YIN_ONLY123",
            "FINAL_YIN_SACCADE144",
        ):
            for session in SESSIONS:
                fused = {
                    pid: (
                        0.5
                        * store[system]["HSS"][session][pid]
                        + 0.5
                        * store[system]["RAN"][session][pid]
                    )
                    for pid in cohort
                }

                accuracy, correct = accuracy_from_evidence(
                    fused,
                    cohort,
                )

                rows.append(
                    {
                        "system": system,
                        "seed": seed,
                        "evaluation": f"R2-{session}",
                        "task_system": "HSS+RAN_50_50",
                        "subject_accuracy": accuracy,
                        "correct_out_of_50": correct,
                    }
                )

            combined_task = {}

            for task in TASKS:
                combined_task[task] = {
                    pid: (
                        0.5 * store[system][task]["S1"][pid]
                        + 0.5 * store[system][task]["S2"][pid]
                    )
                    for pid in cohort
                }

            combined_fused = {
                pid: (
                    0.5 * combined_task["HSS"][pid]
                    + 0.5 * combined_task["RAN"][pid]
                )
                for pid in cohort
            }

            accuracy, correct = accuracy_from_evidence(
                combined_fused,
                cohort,
            )

            rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "evaluation": "R2-S1+S2_SECONDARY",
                    "task_system": "HSS+RAN_50_50",
                    "subject_accuracy": accuracy,
                    "correct_out_of_50": correct,
                }
            )

    return rows, diagnostics


def print_summary(results):
    fused = results[
        results["task_system"] == "HSS+RAN_50_50"
    ]

    print("\n=== EXTERNAL VALIDATION: FUSED SUBJECT ACCURACY ===")

    summary_rows = []

    for system in SYSTEMS:
        for evaluation in (
            "R2-S1",
            "R2-S2",
            "R2-S1+S2_SECONDARY",
        ):
            sub = fused[
                (fused["system"] == system)
                & (fused["evaluation"] == evaluation)
            ]

            summary_rows.append(
                {
                    "system": system,
                    "evaluation": evaluation,
                    "mean_accuracy_pct": (
                        100 * sub["subject_accuracy"].mean()
                    ),
                    "std_pct": (
                        100 * sub["subject_accuracy"].std()
                    ),
                    "min_pct": (
                        100 * sub["subject_accuracy"].min()
                    ),
                    "max_pct": (
                        100 * sub["subject_accuracy"].max()
                    ),
                }
            )

    print(
        pd.DataFrame(summary_rows)
        .round(2)
        .to_string(index=False)
    )

    print("\n=== PRIMARY R2 TWO-SESSION MEAN ===")

    primary_rows = []

    for system in SYSTEMS:
        sub = fused[
            (fused["system"] == system)
            & fused["evaluation"].isin(["R2-S1", "R2-S2"])
        ]

        pivot = sub.pivot(
            index="seed",
            columns="evaluation",
            values="subject_accuracy",
        ).reindex(columns=["R2-S1", "R2-S2"])

        per_seed_mean = pivot.mean(axis=1)

        primary_rows.append(
            {
                "system": system,
                "R2-S1_mean": 100 * pivot["R2-S1"].mean(),
                "R2-S2_mean": 100 * pivot["R2-S2"].mean(),
                "two_session_mean": 100 * per_seed_mean.mean(),
                "two_session_std": 100 * per_seed_mean.std(),
            }
        )

    print(
        pd.DataFrame(primary_rows)
        .round(2)
        .to_string(index=False)
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cohort",
        default=str(DATA_DIR / "cohort_ids.csv"),
    )

    p.add_argument(
        "--r1-s1-base",
        default=str(DATA_DIR / "r1_s1_classical_features.csv"),
    )
    p.add_argument(
        "--r1-s2-base",
        default=str(DATA_DIR / "r1_s2_classical_features.csv"),
    )
    p.add_argument(
        "--r2-s1-base",
        default=str(DATA_DIR / "r2_s1_classical_features.csv"),
    )
    p.add_argument(
        "--r2-s2-base",
        default=str(DATA_DIR / "r2_s2_classical_features.csv"),
    )

    p.add_argument(
        "--r1-yin",
        default=str(DATA_DIR / "r1_motion_features.csv"),
    )
    p.add_argument(
        "--r1-saccade",
        default=str(DATA_DIR / "r1_saccade_features.csv"),
    )
    p.add_argument(
        "--r2-yin",
        default=str(DATA_DIR / "r2_motion_features.csv"),
    )
    p.add_argument(
        "--r2-saccade",
        default=str(DATA_DIR / "r2_saccade_features.csv"),
    )

    p.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
    )
    p.add_argument(
        "--rebuild-r1-features",
        action="store_true",
    )
    p.add_argument(
        "--rebuild-r2-features",
        action="store_true",
    )

    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--seeds", type=int, default=10)

    p.add_argument(
        "--output",
        default=str(
            RESULTS_DIR / "external_validation_results.csv"
        ),
    )
    p.add_argument(
        "--diagnostics-output",
        default=str(
            RESULTS_DIR / "external_validation_diagnostics.csv"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()
    cohort = load_cohort(Path(args.cohort).resolve())

    print("=== FROZEN PARENT-INSPIRED VS PROPOSED COMPARISON ===")
    print("NO TUNING IS PERFORMED.")
    print("Training: all Round-1 S1 + S2.")
    print("External validation: Round-2 S1 and S2 separately.")
    print("Participants: 50.")
    print("RF: 500 trees, balanced_subsample, seeds 0..9.")
    print("HSS/RAN evidence fusion: 0.50 / 0.50.")

    # Parent-inspired baseline data.
    r1_base = pd.concat(
        [
            pd.read_csv(Path(args.r1_s1_base).resolve()),
            pd.read_csv(Path(args.r1_s2_base).resolve()),
        ],
        ignore_index=True,
    )
    r2_base = pd.concat(
        [
            pd.read_csv(Path(args.r2_s1_base).resolve()),
            pd.read_csv(Path(args.r2_s2_base).resolve()),
        ],
        ignore_index=True,
    )

    full_cols = get_full230_cols(r1_base)

    if get_full230_cols(r2_base) != full_cols:
        raise ValueError("R1/R2 baseline feature columns differ")

    for name, df in (
        ("R1 baseline", r1_base),
        ("R2 baseline", r2_base),
    ):
        ids = sorted(df["participant_id"].astype(int).unique())
        if ids != cohort:
            raise ValueError(
                f"{name} cohort differs from frozen cohort"
            )

    r1_yin_path = Path(args.r1_yin).resolve()
    r1_saccade_path = Path(args.r1_saccade).resolve()
    r2_yin_path = Path(args.r2_yin).resolve()
    r2_saccade_path = Path(args.r2_saccade).resolve()

    if (
        args.rebuild_r1_features
        or not r1_yin_path.exists()
        or not r1_saccade_path.exists()
    ):
        build_round_feature_caches(
            raw_dir=Path(args.raw_dir).resolve(),
            round_id=1,
            cohort=cohort,
            yin_output=r1_yin_path,
            saccade_output=r1_saccade_path,
        )

    if (
        args.rebuild_r2_features
        or not r2_yin_path.exists()
        or not r2_saccade_path.exists()
    ):
        build_round_feature_caches(
            raw_dir=Path(args.raw_dir).resolve(),
            round_id=2,
            cohort=cohort,
            yin_output=r2_yin_path,
            saccade_output=r2_saccade_path,
        )

    r1_yin = pd.read_csv(r1_yin_path)
    r1_saccade = pd.read_csv(r1_saccade_path)
    r2_yin = pd.read_csv(r2_yin_path)
    r2_saccade = pd.read_csv(r2_saccade_path)

    yin_cols = get_yin_cols(r1_yin)

    if get_yin_cols(r2_yin) != yin_cols:
        raise ValueError("R1/R2 Yin feature columns differ")

    saccade_base_cols = get_saccade_base_cols(r1_saccade)

    if get_saccade_base_cols(r2_saccade) != saccade_base_cols:
        raise ValueError("R1/R2 saccade feature columns differ")

    r1_final = make_full_saccade_table(
        r1_yin,
        r1_saccade,
    )
    r2_final = make_full_saccade_table(
        r2_yin,
        r2_saccade,
    )

    for name, df in (
        ("R1 proposed", r1_final),
        ("R2 proposed", r2_final),
    ):
        ids = sorted(df["participant_id"].astype(int).unique())
        if ids != cohort:
            raise ValueError(
                f"{name} cohort differs from frozen cohort"
            )

    rows = validate_baseline(
        r1_base,
        r2_base,
        full_cols,
        cohort,
        args.seeds,
        args.trees,
    )

    proposed_rows, diagnostics = validate_proposed(
        r1_final,
        r2_final,
        yin_cols,
        saccade_base_cols,
        cohort,
        args.seeds,
        args.trees,
    )
    rows.extend(proposed_rows)

    results = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostics)

    print_summary(results)

    print("\n=== FROZEN PROPOSED-METHOD DIAGNOSTICS ===")
    print(diagnostics.round(4).to_string(index=False))

    output = Path(args.output).resolve()
    diagnostics_output = Path(args.diagnostics_output).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output, index=False)
    diagnostics.to_csv(diagnostics_output, index=False)

    print(f"\nSaved results: {output}")
    print(f"Saved diagnostics: {diagnostics_output}")


if __name__ == "__main__":
    main()
