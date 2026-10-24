from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "proposed_method" / "data"


COHORT = [
    7, 10, 11, 21, 30,
    80, 99, 107, 128, 129,
    136, 137, 138, 140, 142,
    146, 147, 149, 176, 179,
    180, 181, 182, 183, 184,
    190, 191, 192, 207, 216,
    217, 219, 230, 231, 234,
    241, 248, 249, 252, 255,
    259, 261, 284, 291, 295,
    298, 314, 315, 332, 335,
]

SOURCE_FILES = {
    ("R1", "HSS"): DATA_DIR / "round1_HSS_features.csv",
    ("R1", "RAN"): DATA_DIR / "round1_RAN_features.csv",
    ("R2", "HSS"): DATA_DIR / "round2_HSS_features.csv",
    ("R2", "RAN"): DATA_DIR / "round2_RAN_features.csv",
}

OUTPUT_FILES = {
    ("R1", "S1"): DATA_DIR / "r1_s1_classical_features.csv",
    ("R1", "S2"): DATA_DIR / "r1_s2_classical_features.csv",
    ("R2", "S1"): DATA_DIR / "r2_s1_classical_features.csv",
    ("R2", "S2"): DATA_DIR / "r2_s2_classical_features.csv",
}


def load_selected(path, session):
    pieces = []

    # Chunked reading keeps memory use modest on the MacBook.
    for chunk in pd.read_csv(path, chunksize=10000):
        keep = chunk[
            chunk["participant_id"].isin(COHORT)
            & (chunk["session"] == session)
        ]
        if not keep.empty:
            pieces.append(keep)

    return pd.concat(pieces, ignore_index=True)


def validate_dataset(df, round_name, session):
    participants = sorted(df["participant_id"].astype(int).unique())

    if participants != sorted(COHORT):
        raise ValueError(
            f"{round_name}-{session}: participant cohort does not match."
        )

    if set(df["session"].unique()) != {session}:
        raise ValueError(f"{round_name}-{session}: wrong session rows present.")

    if set(df["task"].unique()) != {"HSS", "RAN"}:
        raise ValueError(f"{round_name}-{session}: expected HSS and RAN.")

    feature_cols = [
        c for c in df.columns if re.match(r"^seg\d{2}_", c)
    ]

    if len(feature_cols) != 230:
        raise ValueError(
            f"{round_name}-{session}: expected 230 features, "
            f"found {len(feature_cols)}."
        )

    if not np.isfinite(
        df[feature_cols].to_numpy(dtype=np.float32)
    ).all():
        raise ValueError(
            f"{round_name}-{session}: non-finite features detected."
        )

    counts = (
        df.groupby(["participant_id", "task"])
        .size()
        .unstack()
    )

    if (counts < 90).any().any():
        raise ValueError(
            f"{round_name}-{session}: participant below "
            f"90-window eligibility threshold."
        )

    print(f"\n=== {round_name}-{session} ===")
    print(f"Rows: {len(df)}")
    print(f"Participants: {len(participants)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"All features finite: True")

    print("\nRows by task:")
    print(df.groupby("task").size().to_string())

    print("\nUsable windows per participant/task:")
    print(
        counts.agg(["min", "median", "max"])
        .round(1)
        .to_string()
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the frozen identity list separately for reproducibility.
    pd.DataFrame({"participant_id": COHORT}).to_csv(
        DATA_DIR / "cohort_ids.csv",
        index=False,
    )

    for round_name in ("R1", "R2"):
        for session in ("S1", "S2"):

            task_frames = []

            for task in ("HSS", "RAN"):
                path = SOURCE_FILES[(round_name, task)]

                if not path.exists():
                    raise FileNotFoundError(path)

                selected = load_selected(path, session)
                task_frames.append(selected)

            combined = pd.concat(task_frames, ignore_index=True)

            validate_dataset(
                combined,
                round_name,
                session,
            )

            output = OUTPUT_FILES[(round_name, session)]
            combined.to_csv(output, index=False)

            print(f"\nSaved: {output}")

    print("\n=== DATASET BUILD COMPLETE ===")
    print("Frozen cohort: 50 participants")
    print("Eligibility rule: >=90 usable windows")
    print("Tasks: HSS + RAN")
    print("Round 2 remains external validation.")


if __name__ == "__main__":
    main()