#!/usr/bin/env python3
"""
Frozen Round-1 development comparison for the final eye-tracking biometric method.

Purpose:
- Reproduce the session-separated Round-1 development comparison used before
  external Round-2 validation.
- Evaluate both directions:
    R1-S1 -> R1-S2
    R1-S2 -> R1-S1
- Compare:
    1) parent-inspired 230-feature baseline,
    2) 123 Yin-inspired motion features,
    3) final 144-feature Yin + principal-saccade representation.

Important:
- Round 2 is never loaded.
- No tuning is performed.
- Saccadic-vigor parameters and missing-event imputations are estimated from
  the training session only for each direction/task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_engineering import TASKS
from evaluate_comparison import (
    load_cohort,
    get_yin_cols,
    get_saccade_base_cols,
    get_full230_cols,
    make_full_saccade_table,
    prepare_final_task,
    fit_rf,
    predict_subject_evidence,
    accuracy_from_evidence,
)

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent
DATA_DIR = METHOD_DIR / "data"
RESULTS_DIR = METHOD_DIR / "results"

SYSTEMS = (
    "BASELINE_FULL230",
    "YIN_ONLY123",
    "FINAL_YIN_SACCADE144",
)

DIRECTIONS = (
    ("S1", "S2"),
    ("S2", "S1"),
)


def fuse_task_evidence(evidence_by_task, cohort):
    return {
        pid: (
            0.5 * evidence_by_task["HSS"][pid]
            + 0.5 * evidence_by_task["RAN"][pid]
        )
        for pid in cohort
    }


def validate_cohort(df, cohort, name):
    ids = sorted(df["participant_id"].astype(int).unique())
    if ids != cohort:
        raise ValueError(f"{name} cohort differs from frozen cohort")


def evaluate_baseline_direction(
    r1_base,
    train_session,
    test_session,
    full_cols,
    cohort,
    seeds,
    trees,
):
    rows = []

    for seed in range(seeds):
        evidence_by_task = {}

        for task in TASKS:
            train_task = r1_base[
                (r1_base["session"] == train_session)
                & (r1_base["task"] == task)
            ]
            test_task = r1_base[
                (r1_base["session"] == test_session)
                & (r1_base["task"] == task)
            ]

            model = fit_rf(train_task, full_cols, seed, trees)
            evidence = predict_subject_evidence(
                model,
                test_task,
                full_cols,
                cohort,
            )
            evidence_by_task[task] = evidence

            accuracy, correct = accuracy_from_evidence(evidence, cohort)
            rows.append(
                {
                    "system": "BASELINE_FULL230",
                    "direction": f"{train_session}->{test_session}",
                    "task_system": task,
                    "seed": seed,
                    "subject_accuracy": accuracy,
                    "correct_out_of_50": correct,
                }
            )

        fused = fuse_task_evidence(evidence_by_task, cohort)
        accuracy, correct = accuracy_from_evidence(fused, cohort)

        rows.append(
            {
                "system": "BASELINE_FULL230",
                "direction": f"{train_session}->{test_session}",
                "task_system": "HSS+RAN_50_50",
                "seed": seed,
                "subject_accuracy": accuracy,
                "correct_out_of_50": correct,
            }
        )

    return rows


def prepare_proposed_direction(
    r1_final,
    train_session,
    test_session,
    yin_cols,
    saccade_base_cols,
):
    prepared = {}
    diagnostics = []

    for task in TASKS:
        train_task = r1_final[
            (r1_final["session"] == train_session)
            & (r1_final["task"] == task)
        ].copy()

        test_task = r1_final[
            (r1_final["session"] == test_session)
            & (r1_final["task"] == task)
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
                "direction": f"{train_session}->{test_session}",
                "task": task,
                "training_bavg": bavg,
                "train_missing_events": int(
                    train_task["sd_missing_event"].sum()
                ),
                "test_missing_events": int(
                    test_task["sd_missing_event"].sum()
                ),
            }
        )

    return prepared, diagnostics


def evaluate_proposed_direction(
    r1_final,
    train_session,
    test_session,
    yin_cols,
    saccade_base_cols,
    cohort,
    seeds,
    trees,
):
    rows = []

    prepared, diagnostics = prepare_proposed_direction(
        r1_final,
        train_session,
        test_session,
        yin_cols,
        saccade_base_cols,
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

            yin_evidence = predict_subject_evidence(
                yin_model,
                test_task,
                yin_cols,
                cohort,
            )
            final_evidence = predict_subject_evidence(
                final_model,
                test_task,
                final_cols,
                cohort,
            )

            store["YIN_ONLY123"][task] = yin_evidence
            store["FINAL_YIN_SACCADE144"][task] = final_evidence

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
                        "direction": f"{train_session}->{test_session}",
                        "task_system": task,
                        "seed": seed,
                        "subject_accuracy": accuracy,
                        "correct_out_of_50": correct,
                    }
                )

        for system in (
            "YIN_ONLY123",
            "FINAL_YIN_SACCADE144",
        ):
            fused = fuse_task_evidence(store[system], cohort)
            accuracy, correct = accuracy_from_evidence(
                fused,
                cohort,
            )
            rows.append(
                {
                    "system": system,
                    "direction": f"{train_session}->{test_session}",
                    "task_system": "HSS+RAN_50_50",
                    "seed": seed,
                    "subject_accuracy": accuracy,
                    "correct_out_of_50": correct,
                }
            )

    return rows, diagnostics


def print_summary(results):
    fused = results[
        results["task_system"] == "HSS+RAN_50_50"
    ].copy()

    print("\n=== ROUND-1 DEVELOPMENT: FUSED SUBJECT ACCURACY ===")

    directional_rows = []
    for system in SYSTEMS:
        for direction in ("S1->S2", "S2->S1"):
            sub = fused[
                (fused["system"] == system)
                & (fused["direction"] == direction)
            ]["subject_accuracy"]

            directional_rows.append(
                {
                    "system": system,
                    "direction": direction,
                    "mean_accuracy_pct": 100 * sub.mean(),
                    "std_pct": 100 * sub.std(),
                    "min_pct": 100 * sub.min(),
                    "max_pct": 100 * sub.max(),
                }
            )

    print(
        pd.DataFrame(directional_rows)
        .round(2)
        .to_string(index=False)
    )

    print("\n=== ROUND-1 BIDIRECTIONAL MEAN ===")

    summary_rows = []
    for system in SYSTEMS:
        sub = fused[fused["system"] == system]

        pivot = sub.pivot(
            index="seed",
            columns="direction",
            values="subject_accuracy",
        ).reindex(columns=["S1->S2", "S2->S1"])

        per_seed_mean = pivot.mean(axis=1)

        summary_rows.append(
            {
                "system": system,
                "S1->S2_mean": 100 * pivot["S1->S2"].mean(),
                "S2->S1_mean": 100 * pivot["S2->S1"].mean(),
                "bidirectional_mean": 100 * per_seed_mean.mean(),
                "bidirectional_std": 100 * per_seed_mean.std(),
            }
        )

    print(
        pd.DataFrame(summary_rows)
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
        "--r1-yin",
        default=str(DATA_DIR / "r1_motion_features.csv"),
    )
    p.add_argument(
        "--r1-saccade",
        default=str(DATA_DIR / "r1_saccade_features.csv"),
    )
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output",
        default=str(
            RESULTS_DIR / "development_proposed_method_results.csv"
        ),
    )
    p.add_argument(
        "--diagnostics-output",
        default=str(
            RESULTS_DIR / "development_proposed_method_diagnostics.csv"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    cohort = load_cohort(Path(args.cohort).resolve())

    print("=== FROZEN ROUND-1 DEVELOPMENT COMPARISON ===")
    print("NO TUNING IS PERFORMED.")
    print("Directions: R1-S1 -> R1-S2 and R1-S2 -> R1-S1.")
    print("Round 2 is NOT loaded.")
    print("Participants: 50.")
    print("RF: 500 trees, balanced_subsample, seeds 0..9.")
    print("HSS/RAN evidence fusion: 0.50 / 0.50.")

    r1_base = pd.concat(
        [
            pd.read_csv(Path(args.r1_s1_base).resolve()),
            pd.read_csv(Path(args.r1_s2_base).resolve()),
        ],
        ignore_index=True,
    )

    validate_cohort(r1_base, cohort, "R1 baseline")
    full_cols = get_full230_cols(r1_base)

    r1_yin = pd.read_csv(Path(args.r1_yin).resolve())
    r1_saccade = pd.read_csv(Path(args.r1_saccade).resolve())

    yin_cols = get_yin_cols(r1_yin)
    saccade_base_cols = get_saccade_base_cols(r1_saccade)

    r1_final = make_full_saccade_table(
        r1_yin,
        r1_saccade,
    )
    validate_cohort(r1_final, cohort, "R1 proposed")

    rows = []
    diagnostics = []

    for train_session, test_session in DIRECTIONS:
        rows.extend(
            evaluate_baseline_direction(
                r1_base,
                train_session,
                test_session,
                full_cols,
                cohort,
                args.seeds,
                args.trees,
            )
        )

        proposed_rows, proposed_diagnostics = (
            evaluate_proposed_direction(
                r1_final,
                train_session,
                test_session,
                yin_cols,
                saccade_base_cols,
                cohort,
                args.seeds,
                args.trees,
            )
        )

        rows.extend(proposed_rows)
        diagnostics.extend(proposed_diagnostics)

    results = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)

    print_summary(results)

    print("\n=== ROUND-1 PROPOSED-METHOD DIAGNOSTICS ===")
    print(diagnostics_df.round(4).to_string(index=False))

    output = Path(args.output).resolve()
    diagnostics_output = Path(args.diagnostics_output).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output, index=False)
    diagnostics_df.to_csv(diagnostics_output, index=False)

    print(f"\nSaved results: {output}")
    print(f"Saved diagnostics: {diagnostics_output}")


if __name__ == "__main__":
    main()
