from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_BASELINE_STD_FOR_Z = 0.05


@dataclass
class CandidateDecision:
    candidate_string_id: int | str
    baseline_mean: float
    baseline_std: float
    baseline_center: float
    baseline_scale: float
    baseline_median: float
    baseline_mad: float
    baseline_max: float
    z_action: np.ndarray
    peak: float
    duration: int
    impulse: float
    action_mean: float
    action_max: float
    candidate_score: float
    vibrates: bool
    z_thr: float
    thr_peak: float
    thr_duration_frames: int
    thr_impulse: float
    baseline_count: int
    action_count: int
    normalization_mode: str
    absolute_motion_pass: bool
    relative_pass: bool
    baseline_pass: bool
    reject_reasons: list[str] = field(default_factory=list)
    selected_offset: int | None = None
    tested_offsets: list[int] = field(default_factory=list)
    selected_action_frames: list[int] = field(default_factory=list)
    peak_action_index: int = 0


def _to_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return arr


def robust_center_scale(
    values: np.ndarray,
    *,
    mode: str,
    min_scale: float,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
) -> tuple[float, float, dict[str, float]]:
    baseline = _to_float_array(values)
    if baseline.size == 0:
        return 0.0, max(float(min_scale), 1e-6), {
            "baseline_median": 0.0,
            "baseline_mad": 0.0,
            "q_low": 0.0,
            "q_high": 0.0,
        }

    baseline_median = float(np.median(baseline))
    baseline_mad = float(np.median(np.abs(baseline - baseline_median)))
    q_low = float(np.percentile(baseline, percentile_low))
    q_high = float(np.percentile(baseline, percentile_high))
    mode_text = str(mode).strip().lower()

    if mode_text == "robust_mad":
        center = baseline_median
        scale = max(float(baseline_mad) * float(mad_scale_factor), float(min_scale))
    elif mode_text == "percentile":
        center = baseline_median
        scale = max(float(q_high - q_low), float(min_scale))
    else:
        center = float(np.mean(baseline))
        scale = max(float(np.std(baseline)), float(min_scale))
        mode_text = "zscore"

    return float(center), float(scale), {
        "baseline_median": baseline_median,
        "baseline_mad": baseline_mad,
        "q_low": q_low,
        "q_high": q_high,
    }


def evaluate_candidate_scores(
    *,
    candidate_string_id: int | str,
    baseline_scores: Any,
    action_scores: Any,
    z_thr: float,
    thr_peak: float,
    thr_duration_frames: int,
    thr_impulse: float,
    normalization_mode: str = "zscore",
    min_scale: float = MIN_BASELINE_STD_FOR_Z,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    selected_offset: int | None = None,
    tested_offsets: list[int] | None = None,
    selected_action_frames: list[int] | None = None,
) -> CandidateDecision:
    baseline = _to_float_array(baseline_scores)
    action = _to_float_array(action_scores)

    if baseline.size == 0:
        baseline_mean = 0.0
        baseline_std = 0.0
        baseline_max = 0.0
    else:
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline))
        baseline_max = float(np.max(baseline))

    baseline_center, baseline_scale, robust_debug = robust_center_scale(
        baseline,
        mode=normalization_mode,
        min_scale=min_scale,
        mad_scale_factor=mad_scale_factor,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    z_action = (action - float(baseline_center)) / max(float(baseline_scale), 1e-6)
    if z_action.size == 0:
        peak = 0.0
        duration = 0
        impulse = 0.0
        peak_action_index = 0
    else:
        peak_action_index = int(np.argmax(z_action))
        peak = float(np.max(z_action))
        duration = int(np.sum(z_action > float(z_thr)))
        impulse = float(np.sum(np.maximum(z_action - float(z_thr), 0.0)))

    action_mean = float(np.mean(action)) if action.size else 0.0
    action_max = float(np.max(action)) if action.size else 0.0

    relative_pass = bool(
        peak >= float(thr_peak)
        and duration >= int(thr_duration_frames)
        and impulse >= float(thr_impulse)
    )
    absolute_motion_pass = True
    if require_absolute_motion:
        absolute_motion_pass = bool(
            action_mean >= float(min_action_mean)
            or action_max >= float(min_action_max)
        )

    baseline_pass = True
    if max_baseline_mean is not None:
        baseline_pass = bool(baseline_mean <= float(max_baseline_mean))

    vibrates = bool(relative_pass and absolute_motion_pass and baseline_pass)
    reject_reasons: list[str] = []
    if not relative_pass:
        reject_reasons.append("relative_threshold_failed")
    if not absolute_motion_pass:
        reject_reasons.append("low_absolute_motion")
    if not baseline_pass:
        reject_reasons.append("high_baseline_motion")

    return CandidateDecision(
        candidate_string_id=candidate_string_id,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_center=float(baseline_center),
        baseline_scale=float(baseline_scale),
        baseline_median=float(robust_debug["baseline_median"]),
        baseline_mad=float(robust_debug["baseline_mad"]),
        baseline_max=baseline_max,
        z_action=z_action.astype(np.float32),
        peak=peak,
        duration=duration,
        impulse=impulse,
        action_mean=action_mean,
        action_max=action_max,
        candidate_score=float(impulse),
        vibrates=vibrates,
        z_thr=float(z_thr),
        thr_peak=float(thr_peak),
        thr_duration_frames=int(thr_duration_frames),
        thr_impulse=float(thr_impulse),
        baseline_count=int(baseline.size),
        action_count=int(action.size),
        normalization_mode=str(normalization_mode).strip().lower(),
        absolute_motion_pass=bool(absolute_motion_pass),
        relative_pass=bool(relative_pass),
        baseline_pass=bool(baseline_pass),
        reject_reasons=reject_reasons,
        selected_offset=selected_offset,
        tested_offsets=list(tested_offsets or ([] if selected_offset is None else [selected_offset])),
        selected_action_frames=list(selected_action_frames or []),
        peak_action_index=int(peak_action_index),
    )


def _series_value(candidate_series: Any, key: str, default: Any) -> Any:
    if isinstance(candidate_series, dict):
        return candidate_series.get(key, default)
    return getattr(candidate_series, key, default)


def _decision_metric_value(decision: CandidateDecision, metric: str) -> float:
    metric_text = str(metric).strip().lower()
    if metric_text == "max_peak":
        return float(decision.peak)
    if metric_text == "max_action_mean":
        return float(decision.action_mean)
    return float(decision.impulse)


def evaluate_candidate_series(
    candidate_series: Any,
    *,
    z_thr: float,
    thr_peak: float,
    thr_duration_frames: int,
    thr_impulse: float,
    normalization_mode: str = "zscore",
    min_scale: float = MIN_BASELINE_STD_FOR_Z,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    dynamic_select_metric: str = "max_impulse",
) -> CandidateDecision:
    candidate_id = _series_value(candidate_series, "candidate_string_id", "unknown")
    baseline_scores = _series_value(candidate_series, "baseline_seq", [])
    action_scores = _series_value(candidate_series, "action_seq", [])
    score_by_frame = _series_value(candidate_series, "score_by_frame", {})
    action_windows = _series_value(candidate_series, "action_windows", {})

    if isinstance(action_windows, dict) and action_windows:
        tested_offsets = sorted(int(k) for k in action_windows.keys())
        best: CandidateDecision | None = None
        for offset in tested_offsets:
            frames = [int(f) for f in action_windows.get(offset, [])]
            scores = [float(score_by_frame.get(int(f), 0.0)) for f in frames]
            decision = evaluate_candidate_scores(
                candidate_string_id=candidate_id,
                baseline_scores=baseline_scores,
                action_scores=scores,
                z_thr=z_thr,
                thr_peak=thr_peak,
                thr_duration_frames=thr_duration_frames,
                thr_impulse=thr_impulse,
                normalization_mode=normalization_mode,
                min_scale=min_scale,
                mad_scale_factor=mad_scale_factor,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                require_absolute_motion=require_absolute_motion,
                min_action_mean=min_action_mean,
                min_action_max=min_action_max,
                max_baseline_mean=max_baseline_mean,
                selected_offset=offset,
                tested_offsets=tested_offsets,
                selected_action_frames=frames,
            )
            if best is None:
                best = decision
                continue
            score_now = _decision_metric_value(decision, dynamic_select_metric)
            score_best = _decision_metric_value(best, dynamic_select_metric)
            tiebreak_now = (float(decision.peak), float(decision.action_max), -int(offset))
            tiebreak_best = (float(best.peak), float(best.action_max), -int(best.selected_offset or 0))
            if (score_now, tiebreak_now) > (score_best, tiebreak_best):
                best = decision
        if best is not None:
            return best

    return evaluate_candidate_scores(
        candidate_string_id=candidate_id,
        baseline_scores=baseline_scores,
        action_scores=action_scores,
        z_thr=z_thr,
        thr_peak=thr_peak,
        thr_duration_frames=thr_duration_frames,
        thr_impulse=thr_impulse,
        normalization_mode=normalization_mode,
        min_scale=min_scale,
        mad_scale_factor=mad_scale_factor,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        require_absolute_motion=require_absolute_motion,
        min_action_mean=min_action_mean,
        min_action_max=min_action_max,
        max_baseline_mean=max_baseline_mean,
        selected_action_frames=list(_series_value(candidate_series, "action_frames", [])),
    )


def evaluate_all_candidates(
    candidate_results: dict[int | str, Any] | list[Any],
    *,
    z_thr: float,
    thr_peak: float,
    thr_duration_frames: int,
    thr_impulse: float,
    normalization_mode: str = "zscore",
    min_scale: float = MIN_BASELINE_STD_FOR_Z,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    dynamic_select_metric: str = "max_impulse",
) -> dict[int | str, CandidateDecision]:
    decisions: dict[int | str, CandidateDecision] = {}

    if isinstance(candidate_results, dict):
        iterator = candidate_results.items()
    else:
        iterator = []
        for item in candidate_results:
            cid = _series_value(item, "candidate_string_id", None)
            if cid is None:
                cid = f"candidate_{len(decisions)}"
            iterator.append((cid, item))

    for cid, series in iterator:
        decisions[cid] = evaluate_candidate_series(
            series,
            z_thr=z_thr,
            thr_peak=thr_peak,
            thr_duration_frames=thr_duration_frames,
            thr_impulse=thr_impulse,
            normalization_mode=normalization_mode,
            min_scale=min_scale,
            mad_scale_factor=mad_scale_factor,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            require_absolute_motion=require_absolute_motion,
            min_action_mean=min_action_mean,
            min_action_max=min_action_max,
            max_baseline_mean=max_baseline_mean,
            dynamic_select_metric=dynamic_select_metric,
        )
    return decisions
