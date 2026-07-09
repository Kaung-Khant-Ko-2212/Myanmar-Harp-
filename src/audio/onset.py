from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OnsetStrengthCache:
    envelope: np.ndarray
    times_sec: np.ndarray
    hop_length: int
    sr: int
    debug: dict[str, Any]


@dataclass(frozen=True)
class OnsetDetectionResult:
    onset_time_sec: float | None
    onset_score: float | None
    status: str
    debug: dict[str, Any]


def prepare_onset_strength(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
) -> OnsetStrengthCache:
    try:
        import librosa  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"librosa_required_for_onset:{exc}") from exc

    y = np.asarray(audio, dtype=np.float32)
    env = librosa.onset.onset_strength(y=y, sr=int(sr), hop_length=int(hop_length))
    env = np.asarray(env, dtype=np.float32)
    frames = np.arange(env.shape[0], dtype=np.int32)
    times = librosa.frames_to_time(frames, sr=int(sr), hop_length=int(hop_length))
    return OnsetStrengthCache(
        envelope=env,
        times_sec=np.asarray(times, dtype=np.float64),
        hop_length=int(hop_length),
        sr=int(sr),
        debug={"num_frames": int(env.shape[0])},
    )


def _slice_by_time(times: np.ndarray, t0: float, t1: float) -> slice:
    i0 = int(np.searchsorted(times, float(t0), side="left"))
    i1 = int(np.searchsorted(times, float(t1), side="right"))
    i0 = max(0, min(i0, len(times)))
    i1 = max(i0, min(i1, len(times)))
    return slice(i0, i1)


def detect_onset_in_window(
    cache: OnsetStrengthCache,
    *,
    touch_time_sec: float,
    onset_window_sec: float,
    baseline_window_sec: float,
    onset_threshold: float,
) -> OnsetDetectionResult:
    times = cache.times_sec
    env = cache.envelope
    if env.size == 0 or times.size == 0:
        return OnsetDetectionResult(None, None, "no_onset", {"reason": "empty_envelope"})

    action_t0 = float(touch_time_sec)
    action_t1 = float(touch_time_sec + max(0.0, onset_window_sec))
    base_t0 = float(touch_time_sec - max(0.0, baseline_window_sec))
    base_t1 = float(touch_time_sec)

    action_sl = _slice_by_time(times, action_t0, action_t1)
    base_sl = _slice_by_time(times, base_t0, base_t1)
    action_env = env[action_sl]
    base_env = env[base_sl]
    debug: dict[str, Any] = {
        "action_window": {"t0": action_t0, "t1": action_t1},
        "baseline_window": {"t0": base_t0, "t1": base_t1},
        "action_len": int(action_env.size),
        "baseline_len": int(base_env.size),
    }
    if action_env.size == 0:
        return OnsetDetectionResult(None, None, "no_onset", {**debug, "reason": "empty_action_window"})

    peak_rel = int(np.argmax(action_env))
    peak_idx = (action_sl.start or 0) + peak_rel
    peak_value = float(env[peak_idx])

    if base_env.size >= 2:
        baseline_mean = float(np.mean(base_env))
        baseline_std = float(np.std(base_env))
    else:
        baseline_mean = float(np.median(action_env))
        baseline_std = float(np.std(action_env))
    baseline_std = max(baseline_std, 1e-6)
    onset_z = float((peak_value - baseline_mean) / baseline_std)
    normalized = float(1.0 / (1.0 + np.exp(-onset_z)))
    accepted = onset_z >= float(onset_threshold)

    debug.update(
        {
            "peak_idx": int(peak_idx),
            "peak_value": peak_value,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "onset_z": onset_z,
            "onset_score_sigmoid": normalized,
            "threshold": float(onset_threshold),
        }
    )
    if not accepted:
        return OnsetDetectionResult(None, onset_z, "no_onset", debug)

    onset_time = float(times[peak_idx]) if peak_idx < times.size else None
    return OnsetDetectionResult(onset_time, onset_z, "ok", debug)

