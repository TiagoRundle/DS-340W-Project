#!/usr/bin/env python3
"""
Frozen feature engineering for the final eye-tracking biometric method.

This module contains only the feature logic used by the proposed method:
- the frozen HSS/RAN response-window eligibility rule,
- 123 Yin-inspired motion-information features,
- 19 principal-saccade dynamics features,
- the training-only saccadic-vigor transform,
- helpers for rebuilding Round-1/Round-2 feature caches.

It intentionally omits discarded exploratory representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


SAMPLE_RATE_HZ = 1000.0
DT = 1.0 / SAMPLE_RATE_HZ
WINDOW_SAMPLES = 1000
TASKS = ("HSS", "RAN")
KEYS = ["participant_id", "session", "task", "window_index"]

# Frozen response-window QC / eligibility.
MAX_INVALID_RATIO = 0.20
MIN_TARGET_JUMP = 0.5

# Yin-inspired motion-information representation.
LAG_SAMPLES = 10
TIME_BIN_SAMPLES = 40
N_TIME_BINS = WINDOW_SAMPLES // TIME_BIN_SAMPLES
DIR_BINS = 8

# Principal-saccade extraction.
SACCADE_VELOCITY_THRESHOLD = 30.0
MIN_SACCADE_AMPLITUDE = 0.5
MAX_SACCADE_AMPLITUDE = 100.0
MAX_SACCADE_PEAK_SPEED = 1000.0

FILE_RE = re.compile(
    r"^S_(\d)(\d{3})_S([12])_(HSS|RAN)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Recording:
    round_id: int
    participant_id: int
    session: str
    task: str
    path: Path


def interpolate_signal(series: pd.Series) -> np.ndarray:
    return (
        series.astype(float)
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy()
    )


def moving_average(x: np.ndarray, width: int = 9) -> np.ndarray:
    pad = width // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(padded, kernel, mode="valid")


def find_target_epochs(df: pd.DataFrame) -> list[tuple[int, int]]:
    tx = df["xT"]
    ty = df["yT"]

    changed = (
        tx.ne(tx.shift())
        | ty.ne(ty.shift())
        | (tx.isna() != tx.shift().isna())
        | (ty.isna() != ty.shift().isna())
    )

    starts = np.flatnonzero(changed.to_numpy())
    ends = np.r_[starts[1:], len(df)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def eligible_response_window(
    window: pd.DataFrame,
    prev_target: tuple[float, float],
    curr_target: tuple[float, float],
    max_invalid_ratio: float = MAX_INVALID_RATIO,
) -> bool:
    """
    Frozen eligibility gate used when the final motion representation was built.

    The original development code implemented this gate inside the
    target-relative experiment. The target-relative FEATURES are not used by
    the final model; only these pre-existing eligibility conditions are kept.
    """
    if len(window) != WINDOW_SAMPLES:
        return False

    invalid = (
        (window["val"] != 0)
        | window["x"].isna()
        | window["y"].isna()
    )
    if float(invalid.mean()) > max_invalid_ratio:
        return False

    x = interpolate_signal(window["x"].mask(window["val"] != 0))
    y = interpolate_signal(window["y"].mask(window["val"] != 0))

    if np.isnan(x).any() or np.isnan(y).any():
        return False

    p0 = np.asarray(prev_target, dtype=float)
    p1 = np.asarray(curr_target, dtype=float)

    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return False

    amplitude = float(np.linalg.norm(p1 - p0))
    return bool(np.isfinite(amplitude) and amplitude >= MIN_TARGET_JUMP)


def circular_entropy(hist: np.ndarray) -> float:
    p = hist.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0

    p = p / total
    nz = p[p > 0]
    ent = -np.sum(nz * np.log(nz))
    return float(ent / np.log(len(hist)))


def yin_motion_features(window: pd.DataFrame) -> dict[str, float]:
    """
    Frozen 123-feature Yin-inspired motion-information representation.
    """
    invalid = (
        (window["val"] != 0)
        | window["x"].isna()
        | window["y"].isna()
    )

    x = interpolate_signal(window["x"].mask(invalid))
    y = interpolate_signal(window["y"].mask(invalid))

    if len(x) != WINDOW_SAMPLES or len(y) != WINDOW_SAMPLES:
        raise ValueError("Expected a 1000-sample window")

    xs = moving_average(x, 9)
    ys = moving_average(y, 9)

    dx = xs[LAG_SAMPLES:] - xs[:-LAG_SAMPLES]
    dy = ys[LAG_SAMPLES:] - ys[:-LAG_SAMPLES]
    magnitude = np.sqrt(dx**2 + dy**2)

    eps = 1e-12
    cos_dir = np.divide(
        dx, magnitude, out=np.zeros_like(dx), where=magnitude > eps
    )
    sin_dir = np.divide(
        dy, magnitude, out=np.zeros_like(dy), where=magnitude > eps
    )
    angle = np.arctan2(dy, dx)

    out: dict[str, float] = {}

    end_sample_indices = np.arange(LAG_SAMPLES, WINDOW_SAMPLES)

    for b in range(N_TIME_BINS):
        lo = b * TIME_BIN_SAMPLES
        hi = lo + TIME_BIN_SAMPLES
        mask = (end_sample_indices >= lo) & (end_sample_indices < hi)

        if not np.any(mask):
            out[f"yin_mi_bin{b:02d}_mag_mean"] = 0.0
            out[f"yin_mi_bin{b:02d}_mag_std"] = 0.0
            out[f"yin_mi_bin{b:02d}_cos_mean"] = 0.0
            out[f"yin_mi_bin{b:02d}_sin_mean"] = 0.0
            continue

        out[f"yin_mi_bin{b:02d}_mag_mean"] = float(np.mean(magnitude[mask]))
        out[f"yin_mi_bin{b:02d}_mag_std"] = float(np.std(magnitude[mask]))
        out[f"yin_mi_bin{b:02d}_cos_mean"] = float(np.mean(cos_dir[mask]))
        out[f"yin_mi_bin{b:02d}_sin_mean"] = float(np.mean(sin_dir[mask]))

    out["yin_mi_mag_mean"] = float(np.mean(magnitude))
    out["yin_mi_mag_std"] = float(np.std(magnitude))
    out["yin_mi_mag_median"] = float(np.median(magnitude))
    out["yin_mi_mag_p90"] = float(np.percentile(magnitude, 90))
    out["yin_mi_mag_max"] = float(np.max(magnitude))

    hist, _ = np.histogram(angle, bins=DIR_BINS, range=(-np.pi, np.pi))
    hist_prop = hist / max(1, hist.sum())

    for i, val in enumerate(hist_prop):
        out[f"yin_mi_dirbin_{i:02d}"] = float(val)

    out["yin_mi_direction_entropy"] = circular_entropy(hist)
    out["yin_mi_resultant_length"] = float(
        np.sqrt(np.mean(cos_dir) ** 2 + np.mean(sin_dir) ** 2)
    )

    if len(angle) > 1:
        turn = np.angle(np.exp(1j * np.diff(angle)))
        abs_turn = np.abs(turn)
        out["yin_mi_turn_abs_mean"] = float(np.mean(abs_turn))
        out["yin_mi_turn_abs_std"] = float(np.std(abs_turn))
        out["yin_mi_turn_abs_p90"] = float(np.percentile(abs_turn, 90))
    else:
        out["yin_mi_turn_abs_mean"] = 0.0
        out["yin_mi_turn_abs_std"] = 0.0
        out["yin_mi_turn_abs_p90"] = 0.0

    path_length = float(
        np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2))
    )
    net_displacement = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))

    vx = np.gradient(xs, DT)
    vy = np.gradient(ys, DT)
    speed = np.sqrt(vx**2 + vy**2)

    out["yin_mi_path_length"] = path_length
    out["yin_mi_net_displacement"] = net_displacement
    out["yin_mi_speed_mean"] = float(np.mean(speed))
    out["yin_mi_speed_p90"] = float(np.percentile(speed, 90))
    out["yin_mi_speed_max"] = float(np.max(speed))

    if len(out) != 123:
        raise ValueError(f"Expected 123 Yin MI features; produced {len(out)}")

    vals = np.asarray(list(out.values()), dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Non-finite Yin motion feature generated")

    return out


def _saccade_velocity(x: np.ndarray, y: np.ndarray):
    vx = np.zeros(len(x))
    vy = np.zeros(len(y))
    vx[3:-3] = (x[6:] - x[:-6]) / (6 * DT)
    vy[3:-3] = (y[6:] - y[:-6]) / (6 * DT)
    return vx, vy


def _saccade_acceleration(speed: np.ndarray):
    accel = np.zeros(len(speed))
    accel[4:-4] = (speed[8:] - speed[:-8]) / (8 * DT)
    return accel


def _true_runs(mask: np.ndarray):
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1) - 1,
        )
    )


def saccade_dynamics_features(
    window: pd.DataFrame,
) -> dict[str, float] | None:
    """
    Frozen 19-feature principal-saccade representation.
    """
    invalid = (
        (window["val"] != 0)
        | window["x"].isna()
        | window["y"].isna()
    )

    x = interpolate_signal(window["x"].mask(invalid))
    y = interpolate_signal(window["y"].mask(invalid))

    vx, vy = _saccade_velocity(x, y)
    speed = np.hypot(vx, vy)
    accel = _saccade_acceleration(speed)

    candidates = []

    for start, end in _true_runs(speed > SACCADE_VELOCITY_THRESHOLD):
        if start < 4 or end >= len(x) - 4 or end - start + 1 < 6:
            continue

        amplitude = float(
            np.hypot(x[end] - x[start], y[end] - y[start])
        )
        peak_speed = float(speed[start : end + 1].max())

        if (
            MIN_SACCADE_AMPLITUDE <= amplitude <= MAX_SACCADE_AMPLITUDE
            and peak_speed <= MAX_SACCADE_PEAK_SPEED
        ):
            candidates.append((start, end, amplitude, peak_speed))

    if not candidates:
        return None

    start, end, amplitude, peak_speed = max(
        candidates, key=lambda z: (z[2], z[3])
    )

    xx = x[start : end + 1]
    yy = y[start : end + 1]
    sp = speed[start : end + 1]
    ac = accel[start : end + 1]

    path_length = float(
        np.hypot(np.diff(xx), np.diff(yy)).sum()
    )
    efficiency = (
        amplitude / path_length if path_length > 1e-12 else 0.0
    )

    peak_index = int(np.argmax(sp))
    time_to_peak_frac = (
        peak_index / (len(sp) - 1) if len(sp) > 1 else 0.0
    )

    before_peak = ac[: peak_index + 1]
    after_peak = ac[peak_index:]

    positive = before_peak[before_peak > 0]
    negative = after_peak[after_peak < 0]

    peak_accel = float(positive.max()) if len(positive) else 0.0
    peak_decel = float(abs(negative.min())) if len(negative) else 0.0
    mean_accel = float(positive.mean()) if len(positive) else 0.0
    mean_decel = float(abs(negative.mean())) if len(negative) else 0.0

    dx = xx[-1] - xx[0]
    dy = yy[-1] - yy[0]
    denom_amp = max(amplitude, 1e-12)

    ux = dx / denom_amp
    uy = dy / denom_amp
    px = -uy
    py = ux

    rx = xx - xx[0]
    ry = yy - yy[0]

    along = rx * ux + ry * uy
    cross = rx * px + ry * py

    out = {
        "sd_onset_ms": float(start),
        "sd_duration_ms": float(end - start + 1),
        "sd_amplitude_dva": amplitude,
        "sd_path_length_dva": path_length,
        "sd_efficiency": float(efficiency),
        "sd_peak_speed_dva_s": peak_speed,
        "sd_mean_speed_dva_s": float(sp.mean()),
        "sd_speed_std_dva_s": float(sp.std()),
        "sd_speed_shape": (
            float(sp.mean() / peak_speed)
            if peak_speed > 1e-12
            else 0.0
        ),
        "sd_time_to_peak_frac": float(time_to_peak_frac),
        "sd_peak_accel_dva_s2": peak_accel,
        "sd_peak_decel_dva_s2": peak_decel,
        "sd_peak_accel_ratio": (
            peak_accel / peak_decel if peak_decel > 1e-12 else 0.0
        ),
        "sd_mean_accel_dva_s2": mean_accel,
        "sd_mean_decel_dva_s2": mean_decel,
        "sd_mean_accel_ratio": (
            mean_accel / mean_decel if mean_decel > 1e-12 else 0.0
        ),
        "sd_cross_track_rms_norm": float(
            np.sqrt(np.mean(cross**2)) / denom_amp
        ),
        "sd_cross_track_max_norm": float(
            np.max(np.abs(cross)) / denom_amp
        ),
        "sd_overshoot_norm": float(
            max(0.0, np.max(along) - amplitude) / denom_amp
        ),
    }

    if len(out) != 19:
        raise ValueError(
            f"Expected 19 saccade-dynamics features; produced {len(out)}"
        )

    return out


def fit_subject_vigor_b(amplitude, peak_speed):
    x = np.asarray(amplitude, dtype=float)
    y = np.asarray(peak_speed, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= MIN_SACCADE_AMPLITUDE)
        & (y > 0)
    )
    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        return np.nan

    best_b = np.nan
    best_sse = np.inf

    for b in np.geomspace(0.25, 50.0, 300):
        g = 1.0 - np.exp(-x / b)
        denom = np.dot(g, g)

        if denom <= 1e-12:
            continue

        scale = np.dot(g, y) / denom
        resid = y - scale * g
        sse = float(np.dot(resid, resid))

        if sse < best_sse:
            best_sse = sse
            best_b = float(b)

    return best_b


def estimate_training_bavg(valid_train_task: pd.DataFrame) -> float:
    values = []

    for _, group in valid_train_task.groupby("participant_id"):
        b = fit_subject_vigor_b(
            group["sd_amplitude_dva"].to_numpy(),
            group["sd_peak_speed_dva_s"].to_numpy(),
        )
        if np.isfinite(b):
            values.append(b)

    if not values:
        raise ValueError("Could not estimate training vigor parameter")

    return float(np.mean(values))


def add_vigor(df: pd.DataFrame, bavg: float) -> pd.DataFrame:
    out = df.copy()

    amplitude = out["sd_amplitude_dva"].to_numpy(dtype=float)
    peak = out["sd_peak_speed_dva_s"].to_numpy(dtype=float)

    denom = 1.0 - np.exp(-amplitude / bavg)

    out["sd_vigor"] = np.divide(
        peak,
        denom,
        out=np.full_like(peak, np.nan),
        where=np.abs(denom) > 1e-12,
    )

    return out


def discover_recordings(
    raw_dir: Path,
    round_id: int,
    cohort: set[int],
) -> list[Recording]:
    recordings = []

    for path in raw_dir.rglob("*.csv"):
        match = FILE_RE.match(path.name)
        if not match:
            continue

        found_round = int(match.group(1))
        participant_id = int(match.group(2))
        session = f"S{match.group(3)}"
        task = match.group(4).upper()

        if found_round != round_id or participant_id not in cohort:
            continue

        recordings.append(
            Recording(
                round_id=found_round,
                participant_id=participant_id,
                session=session,
                task=task,
                path=path,
            )
        )

    return sorted(
        recordings,
        key=lambda r: (r.participant_id, r.session, r.task),
    )


def build_round_feature_caches(
    raw_dir: Path,
    round_id: int,
    cohort: list[int],
    yin_output: Path,
    saccade_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the exact frozen feature population for one GazeBase round.

    The first target epoch is skipped because the frozen response-window
    eligibility rule requires a preceding target position.
    """
    recordings = discover_recordings(
        raw_dir=raw_dir,
        round_id=round_id,
        cohort=set(cohort),
    )

    expected = len(cohort) * 2 * len(TASKS)
    if len(recordings) != expected:
        raise ValueError(
            f"Expected {expected} Round-{round_id} HSS/RAN recordings; "
            f"found {len(recordings)}"
        )

    yin_rows = []
    saccade_rows = []

    candidates = 0
    rejected = 0
    missing_events = 0

    print(f"=== BUILDING ROUND-{round_id} PROPOSED-METHOD FEATURES ===")

    for index, rec in enumerate(recordings, 1):
        df = pd.read_csv(rec.path)

        required = {"x", "y", "val", "xT", "yT"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{rec.path.name} missing required columns: {sorted(missing)}"
            )

        epochs = find_target_epochs(df)

        eligible_here = 0
        saccade_here = 0
        missing_here = 0

        for window_index in range(1, len(epochs)):
            start, end = epochs[window_index]
            prev_start, _ = epochs[window_index - 1]

            if end - start < WINDOW_SAMPLES:
                continue

            candidates += 1

            prev_target = (
                float(df["xT"].iloc[prev_start]),
                float(df["yT"].iloc[prev_start]),
            )
            curr_target = (
                float(df["xT"].iloc[start]),
                float(df["yT"].iloc[start]),
            )

            window = df.iloc[start : start + WINDOW_SAMPLES].copy()

            if not eligible_response_window(
                window,
                prev_target,
                curr_target,
            ):
                rejected += 1
                continue

            key = {
                "participant_id": rec.participant_id,
                "session": rec.session,
                "task": rec.task,
                "window_index": window_index,
            }

            yin_row = dict(key)
            yin_row.update(yin_motion_features(window))
            yin_rows.append(yin_row)
            eligible_here += 1

            saccade = saccade_dynamics_features(window)

            if saccade is None:
                missing_events += 1
                missing_here += 1
            else:
                saccade_row = dict(key)
                saccade_row.update(saccade)
                saccade_rows.append(saccade_row)
                saccade_here += 1

        print(
            f"[{index:3d}/{len(recordings)}] "
            f"P{rec.participant_id:03d} {rec.session} {rec.task} | "
            f"eligible={eligible_here:3d} "
            f"saccade={saccade_here:3d} "
            f"missing_event={missing_here:3d}"
        )

    yin = pd.DataFrame(yin_rows)
    saccade = pd.DataFrame(saccade_rows)

    yin_cols = [c for c in yin.columns if c.startswith("yin_mi_")]
    saccade_cols = [c for c in saccade.columns if c.startswith("sd_")]

    if len(yin_cols) != 123:
        raise ValueError(f"Expected 123 Yin features; found {len(yin_cols)}")
    if len(saccade_cols) != 19:
        raise ValueError(
            f"Expected 19 saccade features; found {len(saccade_cols)}"
        )

    yin_output.parent.mkdir(parents=True, exist_ok=True)
    saccade_output.parent.mkdir(parents=True, exist_ok=True)

    yin.to_csv(yin_output, index=False)
    saccade.to_csv(saccade_output, index=False)

    print(f"\nCandidate response windows: {candidates}")
    print(f"Eligibility rejected: {rejected}")
    print(f"Eligible Yin windows: {len(yin)}")
    print(f"Valid principal-saccade windows: {len(saccade)}")
    print(f"Missing principal-saccade events: {missing_events}")
    print("Eligible Yin rows by task/session:")
    print(
        yin.groupby(["task", "session"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print(f"Saved Yin cache: {yin_output}")
    print(f"Saved saccade cache: {saccade_output}")

    return yin, saccade
