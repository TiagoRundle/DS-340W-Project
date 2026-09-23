"""Faithful Harezlak-style feature extraction for GazeBase RAN.

Parent paper:
Harezlak, Blasiak, Kasprowski (2021),
"Biometric Identification Based on Eye Movement Dynamic Features".

Baseline design implemented here:
- GazeBase Round 1 RAN only
- 24 participants, both S1 and S2, preferring complete/high-quality first-29 epochs
- first 29 target epochs per session (to match the parent-paper sample count)
- first 1000 samples after each target presentation
- ten non-overlapping 100-sample (100 ms) segments
- 23 features per segment => 230 features per target/point vector

Important dataset adaptation:
GazeBase marks missing samples with val != 0 / NaN. The parent paper does not
specify missing-data handling, so this script linearly interpolates within an epoch
and rejects epochs whose invalid-sample ratio exceeds --max-invalid-ratio.

The paper computed the largest Lyapunov exponent (LLE) with R's tseriesChaos
package. To keep the pipeline Python-only and runnable on a laptop, this script
uses a self-contained Rosenstein/Kantz-style approximation with data-driven delay
and embedding estimates. All other features follow the paper directly. V2 also corrects the DFT features to the six first REAL fft values stated in the paper.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

SAMPLE_RATE_HZ = 1000.0
DT = 1.0 / SAMPLE_RATE_HZ
SEGMENT_SAMPLES = 100
EPOCH_SAMPLES = 1000
SEGMENTS_PER_EPOCH = 10

# GazeBase file names encode round and participant, e.g. S_1001_S1_RAN.csv
FILE_RE = re.compile(r"S_(\d)(\d{3})_S([12])_RAN\.csv$")


@dataclass(frozen=True)
class Recording:
    round_id: int
    participant_id: int
    session: str
    path: str


def _stats4(arr: np.ndarray) -> list[float]:
    """Parent-paper statistics: max, min, average, max-min range."""
    return [
        float(np.max(arr)),
        float(np.min(arr)),
        float(np.mean(arr)),
        float(np.max(arr) - np.min(arr)),
    ]


def _interpolate_signal(series: pd.Series) -> np.ndarray:
    return (
        series.astype(float)
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy()
    )


def _average_mutual_information_delay(signal: np.ndarray, max_lag: int = 10) -> int:
    """Estimate delay using the first local minimum of average mutual information.

    This mirrors the parent paper's use of mutual information, while remaining
    self-contained in Python. Histogram binning follows Scott's-rule spirit.
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n < 8 or sd < 1e-12:
        return 1

    width = 3.5 * sd / np.cbrt(n)
    span = float(np.max(x) - np.min(x))
    bins = int(np.ceil(span / width)) if width > 0 and span > 0 else 8
    bins = max(4, min(16, bins))

    edges = np.histogram_bin_edges(x, bins=bins)
    discrete = np.digitize(x, edges[1:-1])

    values: list[float] = []
    max_lag = min(max_lag, max(1, n // 4))
    for lag in range(1, max_lag + 1):
        values.append(mutual_info_score(discrete[:-lag], discrete[lag:]))

    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] <= values[i + 1]:
            return i + 1

    return int(np.argmin(values) + 1) if values else 1


def _delay_embed(signal: np.ndarray, dim: int, tau: int) -> np.ndarray | None:
    n_vectors = len(signal) - (dim - 1) * tau
    if n_vectors <= 1:
        return None
    return np.column_stack(
        [signal[offset * tau : offset * tau + n_vectors] for offset in range(dim)]
    )


def _estimate_embedding_dimension(
    signal: np.ndarray,
    tau: int,
    max_dim: int = 5,
    ratio_threshold: float = 10.0,
    fnn_threshold: float = 0.10,
) -> int:
    """Small-sample false-nearest-neighbour approximation."""
    x = np.asarray(signal, dtype=float)
    scale = float(np.std(x))
    if scale < 1e-12:
        return 1

    theiler = max(2, tau)

    for dim in range(1, max_dim + 1):
        emb_m = _delay_embed(x, dim, tau)
        emb_next = _delay_embed(x, dim + 1, tau)
        if emb_m is None or emb_next is None or len(emb_next) < 6:
            return dim

        n = len(emb_next)
        base = emb_m[:n]
        distances = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=2)
        idx = np.arange(n)
        distances[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf

        nn = np.argmin(distances, axis=1)
        d_m = distances[np.arange(n), nn]
        valid = np.isfinite(d_m) & (d_m > 1e-10)
        if np.sum(valid) < 5:
            return dim

        extra = np.abs(emb_next[np.arange(n), -1] - emb_next[nn, -1])
        d_next = np.sqrt(d_m[valid] ** 2 + extra[valid] ** 2)
        ratio = extra[valid] / d_m[valid]

        false = (ratio > ratio_threshold) | (d_next / scale > 2.0)
        if float(np.mean(false)) < fnn_threshold:
            return dim

    return max_dim


def _largest_lyapunov_approx(signal: np.ndarray, dt: float = DT) -> float:
    """Approximate largest Lyapunov exponent from a 100-point time series.

    The parent paper uses tseriesChaos::lyap_k/lyap (Kantz algorithm). This is a
    compact Python approximation intended to preserve the same nonlinear-dynamics
    information without requiring R.
    """
    x = np.asarray(signal, dtype=float)
    if len(x) < 20 or float(np.std(x)) < 1e-10:
        return 0.0

    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    tau = _average_mutual_information_delay(x)
    dim = _estimate_embedding_dimension(x, tau)
    emb = _delay_embed(x, dim, tau)
    if emb is None or len(emb) < 12:
        return 0.0

    n = len(emb)
    max_follow = min(10, n // 4)
    usable = n - max_follow
    if usable < 6:
        return 0.0

    base = emb[:usable]
    distances = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=2)
    idx = np.arange(usable)
    theiler = max(2, dim * tau)
    distances[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf

    nn = np.argmin(distances, axis=1)
    d0 = distances[np.arange(usable), nn]
    valid = np.isfinite(d0) & (d0 > 1e-8) & ((nn + max_follow) < n)
    refs = np.arange(usable)[valid]
    nbrs = nn[valid]
    if len(refs) < 5:
        return 0.0

    mean_log_distance: list[float] = []
    for step in range(max_follow + 1):
        d = np.linalg.norm(emb[refs + step] - emb[nbrs + step], axis=1)
        d = d[d > 1e-10]
        mean_log_distance.append(float(np.mean(np.log(d))) if len(d) else np.nan)

    y = np.asarray(mean_log_distance)
    good = np.flatnonzero(np.isfinite(y))
    if len(good) < 3:
        return 0.0

    # Estimate the slope from the early divergence region.
    chosen = good[: min(6, len(good))]
    slope = np.polyfit(chosen * dt, y[chosen], 1)[0]
    return float(slope)


def segment_features(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[str]]:
    # First and second derivatives of horizontal/vertical gaze position.
    vx = np.gradient(x, DT)
    vy = np.gradient(y, DT)
    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)

    resultant_velocity = np.sqrt(vx**2 + vy**2)
    resultant_acceleration = np.sqrt(ax**2 + ay**2)

    values: list[float] = []
    names: list[str] = []

    for prefix, arr in (
        ("vx", vx),
        ("vy", vy),
        ("v", resultant_velocity),
        ("a", resultant_acceleration),
    ):
        values.extend(_stats4(arr))
        names.extend([f"{prefix}_max", f"{prefix}_min", f"{prefix}_mean", f"{prefix}_range"])

    # Parent paper explicitly states that the SIX FIRST REAL VALUES from the
    # DFT output were used (R stats::fft), not the magnitudes.
    fft_values = np.real(np.fft.fft(vx))[:6]
    values.extend(float(v) for v in fft_values)
    names.extend([f"fft_{i}" for i in range(6)])

    values.append(_largest_lyapunov_approx(vx))
    names.append("lle")

    assert len(values) == 23
    return values, names


def find_target_epochs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return [start, end) row ranges for each constant GazeBase target position."""
    changed = df["xT"].ne(df["xT"].shift()) | df["yT"].ne(df["yT"].shift())
    starts = np.flatnonzero(changed.to_numpy())
    ends = np.r_[starts[1:], len(df)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def point_vector(
    epoch: pd.DataFrame,
    max_invalid_ratio: float,
) -> tuple[dict[str, float] | None, float]:
    epoch = epoch.iloc[:EPOCH_SAMPLES].copy()
    if len(epoch) < EPOCH_SAMPLES:
        return None, 1.0

    invalid = (epoch["val"] != 0) | epoch["x"].isna() | epoch["y"].isna()
    invalid_ratio = float(invalid.mean())
    if invalid_ratio > max_invalid_ratio:
        return None, invalid_ratio

    x_series = epoch["x"].mask(epoch["val"] != 0)
    y_series = epoch["y"].mask(epoch["val"] != 0)
    x = _interpolate_signal(x_series)
    y = _interpolate_signal(y_series)

    if np.isnan(x).any() or np.isnan(y).any():
        return None, invalid_ratio

    out: dict[str, float] = {}
    for seg_idx in range(SEGMENTS_PER_EPOCH):
        lo = seg_idx * SEGMENT_SAMPLES
        hi = lo + SEGMENT_SAMPLES
        vals, names = segment_features(x[lo:hi], y[lo:hi])
        for name, value in zip(names, vals):
            out[f"seg{seg_idx:02d}_{name}"] = float(value)

    assert len(out) == 230
    return out, invalid_ratio


def discover_recordings(raw_dir: str, round_id: int) -> dict[int, dict[str, Recording]]:
    recordings: dict[int, dict[str, Recording]] = {}
    for path in glob.glob(os.path.join(raw_dir, "*_RAN.csv")):
        match = FILE_RE.search(os.path.basename(path))
        if not match:
            continue
        r = int(match.group(1))
        participant = int(match.group(2))
        session = f"S{match.group(3)}"
        if r != round_id:
            continue
        rec = Recording(r, participant, session, path)
        recordings.setdefault(participant, {})[session] = rec
    return recordings


def _recording_quality(path: str, points_per_session: int, max_invalid_ratio: float) -> tuple[int, float]:
    """Return number of usable first-N epochs and their mean invalid ratio."""
    df = pd.read_csv(path, usecols=["x", "y", "val", "xT", "yT"])
    epochs = find_target_epochs(df)[:points_per_session]
    usable = 0
    ratios: list[float] = []

    for start, end in epochs:
        epoch = df.iloc[start:end].iloc[:EPOCH_SAMPLES]
        if len(epoch) < EPOCH_SAMPLES:
            ratios.append(1.0)
            continue
        invalid = (epoch["val"] != 0) | epoch["x"].isna() | epoch["y"].isna()
        ratio = float(invalid.mean())
        ratios.append(ratio)
        if ratio <= max_invalid_ratio:
            usable += 1

    while len(ratios) < points_per_session:
        ratios.append(1.0)

    return usable, float(np.mean(ratios))


def _select_quality_cohort(
    recordings: dict[int, dict[str, Recording]],
    max_subjects: int,
    points_per_session: int,
    max_invalid_ratio: float,
) -> list[int]:
    """Choose a complete/high-quality 24-person cohort.

    The parent paper has all 29 vectors in both sessions for every participant.
    To make GazeBase structurally comparable, prefer participants with 29/29
    usable epochs in BOTH sessions. If fewer than 24 are fully complete, fill
    remaining slots by the best balanced data quality.
    """
    scored = []
    for participant in sorted(recordings):
        sessions = recordings[participant]
        if not {"S1", "S2"} <= set(sessions):
            continue

        s1_count, s1_mean = _recording_quality(
            sessions["S1"].path, points_per_session, max_invalid_ratio
        )
        s2_count, s2_mean = _recording_quality(
            sessions["S2"].path, points_per_session, max_invalid_ratio
        )
        scored.append(
            (
                participant,
                min(s1_count, s2_count),
                s1_count + s2_count,
                (s1_mean + s2_mean) / 2.0,
            )
        )

    # Maximise worst-session completeness, then total completeness, then minimise
    # missingness. Participant id is only a deterministic final tie-break.
    scored.sort(key=lambda t: (-t[1], -t[2], t[3], t[0]))
    return [p for p, *_ in scored[:max_subjects]]


def build_dataset(
    raw_dir: str,
    output_csv: str,
    round_id: int = 1,
    max_subjects: int = 24,
    points_per_session: int = 29,
    max_invalid_ratio: float = 0.20,
) -> pd.DataFrame:
    recordings = discover_recordings(raw_dir, round_id)
    selected = _select_quality_cohort(
        recordings,
        max_subjects=max_subjects,
        points_per_session=points_per_session,
        max_invalid_ratio=max_invalid_ratio,
    )

    if len(selected) < max_subjects:
        raise RuntimeError(
            f"Only {len(selected)} Round-{round_id} participants have both S1 and S2; "
            f"need {max_subjects}."
        )

    print(f"Round {round_id}: using {len(selected)} participants: {selected}")
    print(f"Target design: {points_per_session} points/session, {EPOCH_SAMPLES} samples/point")

    rows: list[dict[str, float | int | str]] = []
    skipped: list[tuple[int, str, int, str]] = []

    for participant in selected:
        for session in ("S1", "S2"):
            rec = recordings[participant][session]
            df = pd.read_csv(rec.path)
            required = {"x", "y", "val", "xT", "yT"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{rec.path} is missing columns: {sorted(missing)}")

            epochs = find_target_epochs(df)
            if len(epochs) < points_per_session:
                raise RuntimeError(
                    f"{os.path.basename(rec.path)} has only {len(epochs)} target epochs; "
                    f"need {points_per_session}."
                )

            for point_idx, (start, end) in enumerate(epochs[:points_per_session]):
                epoch = df.iloc[start:end]
                feats, invalid_ratio = point_vector(epoch, max_invalid_ratio=max_invalid_ratio)
                if feats is None:
                    skipped.append((participant, session, point_idx, f"invalid_ratio={invalid_ratio:.3f}"))
                    continue

                row: dict[str, float | int | str] = {
                    "round": round_id,
                    "participant_id": participant,
                    "session": session,
                    "point_index": point_idx,
                    "target_x": float(epoch["xT"].iloc[0]),
                    "target_y": float(epoch["yT"].iloc[0]),
                    "invalid_ratio": invalid_ratio,
                }
                row.update(feats)
                rows.append(row)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result.to_csv(output_csv, index=False)

    expected = max_subjects * 2 * points_per_session
    print(f"\nSaved: {output_csv}")
    print(f"Rows: {len(result)} / expected {expected}")
    print(f"Feature columns: {sum(c.startswith('seg') for c in result.columns)} (expected 230)")
    print(f"Skipped epochs: {len(skipped)}")
    if skipped:
        for item in skipped[:20]:
            print("  skipped", item)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    counts = result.groupby(["participant_id", "session"]).size()
    print(f"Per participant/session vectors: min={counts.min()}, median={counts.median()}, max={counts.max()}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser.add_argument(
        "--raw-dir",
        default=os.path.join(project_root, "data", "raw"),
        help="Directory containing extracted GazeBase *_RAN.csv files.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "data", "processed", "harezlak_gazebase_r1_24sub_29pt_v2.csv"),
    )
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--points", type=int, default=29)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        raw_dir=args.raw_dir,
        output_csv=args.output,
        round_id=args.round,
        max_subjects=args.subjects,
        points_per_session=args.points,
        max_invalid_ratio=args.max_invalid_ratio,
    )
