"""Harezlak-style three-consecutive-point experiment on GazeBase.

Input: the corrected one-point feature CSV produced by preprocess_harezlak_gazebase.py.

Parent-paper second scenario:
- concatenate features from 3 consecutive stimulus points
- 27 sliding vectors/session from 29 points (0-2, 1-3, ..., 26-28)
- 690 features/vector (3 × 230)
- mixed S1/S2
- 75/25 stratified split
- 10 random runs
- kNN(k=5), CART, Random Forest, Gaussian Naive Bayes
- vector-level accuracy and subject-level majority vote

Because some GazeBase point epochs were rejected for excessive invalid samples,
a triplet is created only when all three consecutive point vectors are available.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

PARENT_VECTOR = {
    "kNN (k=5)": 0.24,
    "Decision Tree": 0.46,
    "Random Forest": 0.79,
    "Naive Bayes": 0.28,
}

PARENT_SUBJECT = {
    "kNN (k=5)": 0.41,
    "Decision Tree": 0.83,
    "Random Forest": 0.99,
    "Naive Bayes": 0.49,
}

META_COLUMNS = {
    "round",
    "participant_id",
    "session",
    "point_index",
    "target_x",
    "target_y",
    "invalid_ratio",
    "label",
}


def make_models(seed: int):
    return {
        "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
    }


def majority_vote_subject_accuracy(
    true_subjects: np.ndarray,
    predicted_labels: np.ndarray,
    encoder: LabelEncoder,
) -> tuple[float, int, int]:
    pred_subjects = encoder.inverse_transform(predicted_labels)
    correct = 0
    unique_subjects = np.unique(true_subjects)

    for subject in unique_subjects:
        preds = pred_subjects[true_subjects == subject]
        counts = Counter(preds.tolist())
        winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        correct += int(winner == subject)

    total = len(unique_subjects)
    return correct / total, correct, total


def fmt_mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{100*np.mean(arr):6.2f}% ± {100*np.std(arr, ddof=1):5.2f}%"


def build_three_point_dataset(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    for (participant, session), group in df.groupby(["participant_id", "session"]):
        group = group.sort_values("point_index")
        by_point = {int(row["point_index"]): row for _, row in group.iterrows()}

        # Create every available overlapping three-point vector.
        # 29 points -> 27 triplets; 100 points -> 98 triplets.
        max_point = max(by_point)
        for start in range(max_point - 1):
            points = [start, start + 1, start + 2]
            if not all(p in by_point for p in points):
                continue

            row = {
                "participant_id": participant,
                "session": session,
                "start_point": start,
            }

            for offset, point in enumerate(points):
                src = by_point[point]
                for col in feature_cols:
                    row[f"p{offset}_{col}"] = float(src[col])

            rows.append(row)

    return pd.DataFrame(rows)


def evaluate(csv_path: str, runs: int = 10) -> None:
    one = pd.read_csv(csv_path)

    required = {"participant_id", "session", "point_index"}
    missing = required - set(one.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    feature_cols = [c for c in one.columns if c not in META_COLUMNS]
    if len(feature_cols) != 230:
        raise ValueError(
            f"Expected 230 one-point features, found {len(feature_cols)}. "
            "Use the corrected Harezlak preprocessing output."
        )

    three = build_three_point_dataset(one, feature_cols)
    three_feature_cols = [c for c in three.columns if c.startswith(("p0_", "p1_", "p2_"))]

    encoder = LabelEncoder()
    three["label"] = encoder.fit_transform(three["participant_id"])

    points_per_session = int(one.groupby(["participant_id", "session"])["point_index"].nunique().median())
    expected = len(encoder.classes_) * 2 * max(points_per_session - 2, 0)
    counts = three.groupby(["participant_id", "session"]).size()

    print("=== HAREZLAK THREE-POINT GAZEBASE EXPERIMENT ===")
    print(f"Input one-point CSV: {csv_path}")
    print(f"Participants: {len(encoder.classes_)}")
    print(f"Three-point vectors: {len(three)} / expected {expected}")
    print(f"Points/session: {points_per_session}")
    print(f"Features/vector: {len(three_feature_cols)} (expected 690)")
    print(
        f"Per participant/session triplets: "
        f"min={counts.min()}, median={counts.median()}, max={counts.max()}"
    )

    X = three[three_feature_cols].to_numpy(dtype=float)
    y = three["label"].to_numpy()
    subjects = three["participant_id"].to_numpy()
    indices = np.arange(len(three))

    vector_scores = {name: [] for name in make_models(0)}
    subject_scores = {name: [] for name in make_models(0)}

    print("\n--- MIXED SESSIONS, 75/25, 10 RUNS ---")

    for run in range(runs):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=run,
            stratify=y,
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_subjects = subjects[test_idx]

        for name, model in make_models(run).items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            vector_scores[name].append(accuracy_score(y_test, pred))
            sub_acc, _, _ = majority_vote_subject_accuracy(
                test_subjects, pred, encoder
            )
            subject_scores[name].append(sub_acc)

    print("\nVector-level accuracy (parent Table 1, three-point comparator):")
    for name in vector_scores:
        mean = float(np.mean(vector_scores[name]))
        parent = PARENT_VECTOR[name]
        print(
            f"  {name:15s} ours {fmt_mean_std(vector_scores[name])} | "
            f"parent {100*parent:5.1f}% | delta {100*(mean-parent):+6.2f} pp"
        )

    print("\nSubject-level majority vote (parent Table 4, three-point comparator):")
    for name in subject_scores:
        mean = float(np.mean(subject_scores[name]))
        parent = PARENT_SUBJECT[name]
        print(
            f"  {name:15s} ours {fmt_mean_std(subject_scores[name])} | "
            f"parent {100*parent:5.1f}% | delta {100*(mean-parent):+6.2f} pp"
        )

    print("\nParent expectation:")
    print("  kNN 24%, DT 46%, RF 79%, NB 28% at vector level.")
    print("  RF should remain clearly strongest; parent subject-level RF was ~99%.")
    print("  Note: missing GazeBase epochs reduce the number of valid 3-point vectors.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser.add_argument(
        "--csv",
        default=os.path.join(
            project_root,
            "data",
            "processed",
            "harezlak_gazebase_r1_24sub_29pt.csv",
        ),
    )
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.csv, runs=args.runs)
