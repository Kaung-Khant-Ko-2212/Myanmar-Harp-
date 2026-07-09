from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .rules import CandidateDecision, evaluate_all_candidates
    from .strings import StringGeometry
    from .windows import EventWindowResult, TouchEvent, process_single_event_windows
except ImportError:  # pragma: no cover
    from src.rules import CandidateDecision, evaluate_all_candidates
    from src.strings import StringGeometry
    from src.windows import EventWindowResult, TouchEvent, process_single_event_windows


@dataclass
class DecisionMetrics:
    candidate_id: int | None
    candidate_score: float
    peak: float
    duration: int
    impulse: float
    vibrates: bool


@dataclass
class StrikeResult:
    event_time: float
    finger_type: str
    touched_id: int
    struck_id: int | None
    label: str
    best_metrics: DecisionMetrics
    second_metrics: DecisionMetrics
    debug: dict[str, Any]


def _to_int_string_id(value: Any) -> int:
    text = str(value).strip()
    try:
        f = float(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid string id: {value!r}") from exc
    i = int(round(f))
    if abs(f - i) > 1e-6:
        raise ValueError(f"Non-integer string id: {value!r}")
    return i


def _empty_metrics() -> DecisionMetrics:
    return DecisionMetrics(
        candidate_id=None,
        candidate_score=0.0,
        peak=0.0,
        duration=0,
        impulse=0.0,
        vibrates=False,
    )


def _decision_to_metrics(decision: CandidateDecision | None) -> DecisionMetrics:
    if decision is None:
        return _empty_metrics()
    return DecisionMetrics(
        candidate_id=_to_int_string_id(decision.candidate_string_id),
        candidate_score=float(decision.candidate_score),
        peak=float(decision.peak),
        duration=int(decision.duration),
        impulse=float(decision.impulse),
        vibrates=bool(decision.vibrates),
    )


def _decision_to_debug_dict(decision: CandidateDecision) -> dict[str, Any]:
    return {
        "candidate_string_id": _to_int_string_id(decision.candidate_string_id),
        "candidate_score": float(decision.candidate_score),
        "peak": float(decision.peak),
        "duration": int(decision.duration),
        "impulse": float(decision.impulse),
        "vibrates": bool(decision.vibrates),
        "baseline_mean": float(decision.baseline_mean),
        "baseline_std": float(decision.baseline_std),
        "baseline_center": float(decision.baseline_center),
        "baseline_scale": float(decision.baseline_scale),
        "baseline_median": float(decision.baseline_median),
        "baseline_mad": float(decision.baseline_mad),
        "baseline_max": float(decision.baseline_max),
        "baseline_count": int(decision.baseline_count),
        "action_count": int(decision.action_count),
        "action_mean": float(decision.action_mean),
        "action_max": float(decision.action_max),
        "normalization_mode": str(decision.normalization_mode),
        "relative_pass": bool(decision.relative_pass),
        "absolute_motion_pass": bool(decision.absolute_motion_pass),
        "baseline_pass": bool(decision.baseline_pass),
        "reject_reasons": list(decision.reject_reasons),
        "selected_offset": decision.selected_offset,
        "tested_offsets": list(decision.tested_offsets),
        "selected_action_frames": list(decision.selected_action_frames),
        "peak_action_index": int(decision.peak_action_index),
        "z_action": [float(v) for v in np.asarray(decision.z_action).reshape(-1)],
    }


def _select_probe_string_ids(
    *,
    all_string_ids: list[int],
    exclude_ids: set[int],
    shake_probe_count: int,
) -> list[int]:
    if shake_probe_count <= 0:
        return []
    pool = [sid for sid in sorted(all_string_ids) if sid not in exclude_ids]
    if not pool:
        return []
    n = min(int(shake_probe_count), len(pool))
    if n <= 0:
        return []
    if n == 1:
        return [pool[len(pool) // 2]]

    idxs = np.linspace(0, len(pool) - 1, num=n)
    chosen: list[int] = []
    seen: set[int] = set()
    for idx in idxs:
        i = int(round(float(idx)))
        i = max(0, min(i, len(pool) - 1))
        sid = int(pool[i])
        if sid not in seen:
            seen.add(sid)
            chosen.append(sid)
    return chosen


def _velocity_map_from_positions(
    positions_by_frame: dict[int, tuple[float, float]] | None,
    fps: float,
) -> dict[int, float]:
    if not positions_by_frame:
        return {}
    speed_by_frame: dict[int, float] = {}
    sorted_items = sorted((int(f), p) for f, p in positions_by_frame.items() if p is not None)
    prev_frame: int | None = None
    prev_point: tuple[float, float] | None = None
    fps_safe = max(float(fps), 1e-6)

    for frame_idx, point in sorted_items:
        if prev_frame is not None and prev_point is not None and frame_idx > prev_frame:
            dt = (frame_idx - prev_frame) / fps_safe
            if dt > 0:
                vx = float(point[0]) - float(prev_point[0])
                vy = float(point[1]) - float(prev_point[1])
                speed = float(np.hypot(vx, vy) / dt)
                speed_by_frame[frame_idx] = speed
        prev_frame = frame_idx
        prev_point = (float(point[0]), float(point[1]))
    return speed_by_frame


def _mean_from_velocity_map(
    velocity_by_frame: dict[int, float],
    frame_indices: list[int],
) -> tuple[float, int]:
    vals: list[float] = []
    for f in frame_indices:
        v = velocity_by_frame.get(int(f))
        if v is None:
            continue
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def _estimate_finger_velocity_metrics(
    *,
    event_window: EventWindowResult,
    peak_frame: int,
    vel_drop_frames: int,
    fps: float,
    finger_positions_by_frame: dict[int, tuple[float, float]] | None,
    finger_velocity_by_frame: dict[int, float] | None,
    event_velocity_stats: dict[str, float] | None,
) -> dict[str, Any]:
    if event_velocity_stats:
        mean_before = float(event_velocity_stats.get("mean_vel_before", float("nan")))
        mean_after = float(event_velocity_stats.get("mean_vel_after", float("nan")))
        return {
            "mean_vel_before": mean_before,
            "mean_vel_after": mean_after,
            "count_before": int(event_velocity_stats.get("count_before", 0)),
            "count_after": int(event_velocity_stats.get("count_after", 0)),
            "source": "event_velocity_stats",
        }

    if finger_velocity_by_frame:
        vel_map = {int(k): float(v) for k, v in finger_velocity_by_frame.items()}
        source = "finger_velocity_by_frame"
    else:
        vel_map = _velocity_map_from_positions(finger_positions_by_frame, fps=fps)
        source = "finger_positions_by_frame"

    before_frames = list(event_window.baseline_frames)
    after_frames = [peak_frame + i for i in range(1, max(0, int(vel_drop_frames)) + 1)]

    mean_before, count_before = _mean_from_velocity_map(vel_map, before_frames)
    mean_after, count_after = _mean_from_velocity_map(vel_map, after_frames)
    return {
        "mean_vel_before": float(mean_before),
        "mean_vel_after": float(mean_after),
        "count_before": int(count_before),
        "count_after": int(count_after),
        "source": source,
        "before_frames": before_frames,
        "after_frames": after_frames,
    }


def _apply_finger_gate(
    *,
    enable_finger_gate: bool,
    finger_type: str,
    peak_frame: int,
    event_window: EventWindowResult,
    fps: float,
    thumb_gate: dict[str, Any],
    index_gate: dict[str, Any],
    finger_positions_by_frame: dict[int, tuple[float, float]] | None,
    finger_velocity_by_frame: dict[int, float] | None,
    event_velocity_stats: dict[str, float] | None,
) -> tuple[bool, dict[str, Any]]:
    if not enable_finger_gate:
        return True, {
            "enabled": False,
            "pass": True,
            "reason": "disabled",
        }

    finger = str(finger_type).strip().lower()
    gate_cfg = thumb_gate if finger == "thumb" else index_gate
    vel_drop_frames = int(gate_cfg.get("vel_drop_frames", 6))
    min_vel_before = float(gate_cfg.get("min_vel_before", 2.0))
    max_vel_after = float(gate_cfg.get("max_vel_after", 1.2))

    metrics = _estimate_finger_velocity_metrics(
        event_window=event_window,
        peak_frame=peak_frame,
        vel_drop_frames=vel_drop_frames,
        fps=fps,
        finger_positions_by_frame=finger_positions_by_frame,
        finger_velocity_by_frame=finger_velocity_by_frame,
        event_velocity_stats=event_velocity_stats,
    )
    mean_before = float(metrics["mean_vel_before"])
    mean_after = float(metrics["mean_vel_after"])

    finite = np.isfinite(mean_before) and np.isfinite(mean_after)
    gate_pass = bool(
        finite
        and mean_before >= min_vel_before
        and mean_after <= max_vel_after
    )
    metrics.update(
        {
            "enabled": True,
            "finger_type": finger,
            "vel_drop_frames": vel_drop_frames,
            "min_vel_before": min_vel_before,
            "max_vel_after": max_vel_after,
            "pass": gate_pass,
            "reason": (
                "ok"
                if gate_pass
                else ("insufficient_velocity_data" if not finite else "velocity_threshold_failed")
            ),
        }
    )
    return gate_pass, metrics


def _best_and_second_vibrating(
    candidate_decisions: dict[int | str, CandidateDecision],
) -> tuple[CandidateDecision | None, CandidateDecision | None]:
    vibrating = [d for d in candidate_decisions.values() if bool(d.vibrates)]
    if not vibrating:
        return None, None
    vibrating.sort(
        key=lambda d: (float(d.candidate_score), float(d.peak), -int(_to_int_string_id(d.candidate_string_id))),
        reverse=True,
    )
    best = vibrating[0]
    second = vibrating[1] if len(vibrating) > 1 else None
    return best, second


def _compute_peak_frame(event_window: EventWindowResult, decision: CandidateDecision | None) -> int:
    if decision is None:
        return int(event_window.f0)
    action_frames = list(decision.selected_action_frames or event_window.action_frames)
    if len(action_frames) == 0:
        return int(event_window.f0)
    z = np.asarray(decision.z_action).reshape(-1)
    if z.size == 0:
        return int(event_window.f0)
    peak_idx = int(getattr(decision, "peak_action_index", int(np.argmax(z))))
    peak_idx = max(0, min(peak_idx, len(action_frames) - 1))
    return int(action_frames[peak_idx])


def _evaluate_probe_decisions(
    *,
    video_path: str | Path,
    event: TouchEvent,
    strings: list[StringGeometry],
    fps: float,
    probe_ids: list[int],
    baseline_len: int,
    action_len: int,
    action_start_frame_offset: int,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float,
    center_band_h: int,
    enable_hand_mask: bool,
    hand_mask_expand_px: float,
    farneback_params: dict[str, Any] | None,
    dy_thr: float,
    z_thr: float,
    thr_peak: float,
    thr_duration_frames: int,
    thr_impulse: float,
    baseline_gap_frames: int,
    dynamic_action_enabled: bool,
    dynamic_offsets: list[int] | None,
    dynamic_select_metric: str,
    geometry_enabled: bool,
    geometry_top_k: int,
    geometry_max_distance_px: float,
    always_include_touched_id: bool,
    include_id_radius_fallback: bool,
    missing_touched_id_fallback: str,
    fallback_top_k: int,
    fallback_max_distance_px: float,
    log_string_id_inconsistency: bool,
    adaptive_roi_enabled: bool,
    adaptive_height_ratio: float,
    min_roi_h: int,
    max_roi_h: int,
    min_neighbor_distance_px: float,
    border_mode: str,
    constant_border_value: int,
    reject_if_out_of_frame: bool,
    min_inside_fraction: float,
    hand_mask_mode: str,
    contact_band_exclusion_px: float,
    mask_contact_region: bool,
    allow_small_gaps: bool,
    max_gap_frames: int,
    normalize_by_gap: bool,
    normalization_mode: str,
    min_scale: float,
    mad_scale_factor: float,
    percentile_low: float,
    percentile_high: float,
    require_absolute_motion: bool,
    min_action_mean: float,
    min_action_max: float,
    max_baseline_mean: float | None,
    stabilize_enabled: bool,
    frames_for_event: dict[int, np.ndarray] | None = None,
) -> dict[int, CandidateDecision]:
    out: dict[int, CandidateDecision] = {}
    for probe_id in probe_ids:
        probe_event = replace(event, touched_string_id=int(probe_id))
        probe_window = process_single_event_windows(
            video_path=video_path,
            event=probe_event,
            strings=strings,
            fps=fps,
            baseline_len=baseline_len,
            action_len=action_len,
            action_start_frame_offset=action_start_frame_offset,
            roi_w=roi_w,
            roi_h=roi_h,
            trim_ends_ratio=trim_ends_ratio,
            center_band_h=center_band_h,
            candidate_radius_default=0,
            candidate_radius_close_contact=0,
            contact_dist_px_thr=0.0,
            enable_hand_mask=enable_hand_mask,
            hand_mask_expand_px=hand_mask_expand_px,
            farneback_params=farneback_params,
            dy_thr=dy_thr,
            stabilize_enabled=stabilize_enabled,
            preloaded_frames=frames_for_event,
            baseline_gap_frames=baseline_gap_frames,
            dynamic_action_enabled=dynamic_action_enabled,
            dynamic_offsets=dynamic_offsets,
            dynamic_select_metric=dynamic_select_metric,
            geometry_enabled=geometry_enabled,
            geometry_top_k=geometry_top_k,
            geometry_max_distance_px=geometry_max_distance_px,
            always_include_touched_id=always_include_touched_id,
            include_id_radius_fallback=include_id_radius_fallback,
            missing_touched_id_fallback=missing_touched_id_fallback,
            fallback_top_k=fallback_top_k,
            fallback_max_distance_px=fallback_max_distance_px,
            log_string_id_inconsistency=log_string_id_inconsistency,
            adaptive_roi_enabled=adaptive_roi_enabled,
            adaptive_height_ratio=adaptive_height_ratio,
            min_roi_h=min_roi_h,
            max_roi_h=max_roi_h,
            min_neighbor_distance_px=min_neighbor_distance_px,
            border_mode=border_mode,
            constant_border_value=constant_border_value,
            reject_if_out_of_frame=reject_if_out_of_frame,
            min_inside_fraction=min_inside_fraction,
            hand_mask_mode=hand_mask_mode,
            contact_band_exclusion_px=contact_band_exclusion_px,
            mask_contact_region=mask_contact_region,
            allow_small_gaps=allow_small_gaps,
            max_gap_frames=max_gap_frames,
            normalize_by_gap=normalize_by_gap,
        )
        probe_decisions = evaluate_all_candidates(
            probe_window.candidate_results,
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
        decision = probe_decisions.get(int(probe_id))
        if decision is not None:
            out[int(probe_id)] = decision
    return out


def decide_event_from_window(
    *,
    event_window: EventWindowResult,
    strings: list[StringGeometry],
    video_path: str | Path,
    fps: float,
    baseline_len: int,
    action_len: int,
    action_start_frame_offset: int,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float,
    center_band_h: int,
    enable_hand_mask: bool,
    hand_mask_expand_px: float,
    farneback_params: dict[str, Any] | None,
    dy_thr: float,
    z_thr: float,
    thr_peak: float,
    thr_duration_frames: int,
    thr_impulse: float,
    baseline_gap_frames: int = 0,
    dynamic_action_enabled: bool = False,
    dynamic_offsets: list[int] | None = None,
    dynamic_select_metric: str = "max_impulse",
    geometry_enabled: bool = False,
    geometry_top_k: int = 5,
    geometry_max_distance_px: float = 35.0,
    always_include_touched_id: bool = True,
    include_id_radius_fallback: bool = True,
    missing_touched_id_fallback: str = "none",
    fallback_top_k: int = 3,
    fallback_max_distance_px: float = 40.0,
    log_string_id_inconsistency: bool = True,
    adaptive_roi_enabled: bool = False,
    adaptive_height_ratio: float = 0.45,
    min_roi_h: int = 5,
    max_roi_h: int = 18,
    min_neighbor_distance_px: float = 4.0,
    border_mode: str = "replicate",
    constant_border_value: int = 0,
    reject_if_out_of_frame: bool = False,
    min_inside_fraction: float = 0.95,
    hand_mask_mode: str = "finger_point",
    contact_band_exclusion_px: float = 10.0,
    mask_contact_region: bool = True,
    allow_small_gaps: bool = False,
    max_gap_frames: int = 2,
    normalize_by_gap: bool = True,
    normalization_mode: str = "zscore",
    min_scale: float = 0.05,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    thr_domination_ratio: float = 1.3,
    thr_global_median_peak: float = 3.0,
    thr_many_strings_vibrating: int = 4,
    shake_probe_count: int = 6,
    enable_finger_gate: bool = True,
    thumb_gate: dict[str, Any] | None = None,
    index_gate: dict[str, Any] | None = None,
    stabilize_enabled: bool = False,
    finger_positions_by_frame: dict[int, tuple[float, float]] | None = None,
    finger_velocity_by_frame: dict[int, float] | None = None,
    event_velocity_stats: dict[str, float] | None = None,
) -> StrikeResult:
    candidate_decisions = evaluate_all_candidates(
        event_window.candidate_results,
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

    all_ids = sorted({_to_int_string_id(s.string_id) for s in strings})
    candidate_ids = {int(cid) for cid in event_window.candidates}
    probe_ids = _select_probe_string_ids(
        all_string_ids=all_ids,
        exclude_ids=candidate_ids,
        shake_probe_count=int(shake_probe_count),
    )
    probe_decisions = _evaluate_probe_decisions(
        video_path=video_path,
        event=event_window.event,
        strings=strings,
        fps=fps,
        probe_ids=probe_ids,
        baseline_len=baseline_len,
        action_len=action_len,
        action_start_frame_offset=action_start_frame_offset,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
        center_band_h=center_band_h,
        enable_hand_mask=enable_hand_mask,
        hand_mask_expand_px=hand_mask_expand_px,
        farneback_params=farneback_params,
        dy_thr=dy_thr,
        z_thr=z_thr,
        thr_peak=thr_peak,
        thr_duration_frames=thr_duration_frames,
        thr_impulse=thr_impulse,
        baseline_gap_frames=baseline_gap_frames,
        dynamic_action_enabled=dynamic_action_enabled,
        dynamic_offsets=dynamic_offsets,
        dynamic_select_metric=dynamic_select_metric,
        geometry_enabled=geometry_enabled,
        geometry_top_k=geometry_top_k,
        geometry_max_distance_px=geometry_max_distance_px,
        always_include_touched_id=always_include_touched_id,
        include_id_radius_fallback=include_id_radius_fallback,
        missing_touched_id_fallback=missing_touched_id_fallback,
        fallback_top_k=fallback_top_k,
        fallback_max_distance_px=fallback_max_distance_px,
        log_string_id_inconsistency=log_string_id_inconsistency,
        adaptive_roi_enabled=adaptive_roi_enabled,
        adaptive_height_ratio=adaptive_height_ratio,
        min_roi_h=min_roi_h,
        max_roi_h=max_roi_h,
        min_neighbor_distance_px=min_neighbor_distance_px,
        border_mode=border_mode,
        constant_border_value=constant_border_value,
        reject_if_out_of_frame=reject_if_out_of_frame,
        min_inside_fraction=min_inside_fraction,
        hand_mask_mode=hand_mask_mode,
        contact_band_exclusion_px=contact_band_exclusion_px,
        mask_contact_region=mask_contact_region,
        allow_small_gaps=allow_small_gaps,
        max_gap_frames=max_gap_frames,
        normalize_by_gap=normalize_by_gap,
        normalization_mode=normalization_mode,
        min_scale=min_scale,
        mad_scale_factor=mad_scale_factor,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        require_absolute_motion=require_absolute_motion,
        min_action_mean=min_action_mean,
        min_action_max=min_action_max,
        max_baseline_mean=max_baseline_mean,
        stabilize_enabled=stabilize_enabled,
        frames_for_event=event_window.frames_for_event,
    )

    best_decision, second_decision = _best_and_second_vibrating(candidate_decisions)
    best_metrics = _decision_to_metrics(best_decision)
    second_metrics = _decision_to_metrics(second_decision)

    probe_peaks = [float(d.peak) for d in probe_decisions.values()]
    median_probe_peak = float(np.median(probe_peaks)) if probe_peaks else 0.0
    all_decisions_for_shake = list(candidate_decisions.values()) + list(probe_decisions.values())
    vibrating_total = int(sum(1 for d in all_decisions_for_shake if bool(d.vibrates)))
    shake_trigger_by_global = median_probe_peak > float(thr_global_median_peak)
    shake_trigger_by_many = vibrating_total >= int(thr_many_strings_vibrating)
    # Avoid classifying pure non-vibration touch events as shake.
    shake_reject = bool(best_decision is not None and (shake_trigger_by_global or shake_trigger_by_many))

    if best_decision is None:
        domination_pass = False
    elif second_decision is None:
        domination_pass = True
    else:
        second_score = float(second_decision.candidate_score)
        if second_score <= 1e-12:
            domination_pass = float(best_decision.candidate_score) > 0.0
        else:
            domination_pass = float(best_decision.candidate_score) >= float(thr_domination_ratio) * second_score

    peak_frame = _compute_peak_frame(event_window, best_decision)
    gate_pass, gate_debug = _apply_finger_gate(
        enable_finger_gate=enable_finger_gate,
        finger_type=event_window.event.finger_type,
        peak_frame=peak_frame,
        event_window=event_window,
        fps=fps,
        thumb_gate=thumb_gate or {},
        index_gate=index_gate or {},
        finger_positions_by_frame=finger_positions_by_frame,
        finger_velocity_by_frame=finger_velocity_by_frame,
        event_velocity_stats=event_velocity_stats,
    )

    startup_guard_min_frame = max(1, int(baseline_len))
    startup_guard = int(event_window.f0) < startup_guard_min_frame

    if startup_guard:
        label = "touch_only"
        struck_id = None
    elif shake_reject:
        label = "shake_reject"
        struck_id = None
    elif best_decision is None:
        label = "touch_only"
        struck_id = None
    elif not gate_pass:
        label = "touch_only"
        struck_id = None
    elif not domination_pass:
        label = "touch_only"
        struck_id = None
    else:
        label = "strike"
        struck_id = _to_int_string_id(best_decision.candidate_string_id)

    debug = {
        "candidate_decisions": {int(k): _decision_to_debug_dict(v) for k, v in candidate_decisions.items()},
        "probe_ids": probe_ids,
        "probe_decisions": {int(k): _decision_to_debug_dict(v) for k, v in probe_decisions.items()},
        "median_probe_peak": float(median_probe_peak),
        "vibrating_count_total": int(vibrating_total),
        "shake_trigger_by_global": bool(shake_trigger_by_global),
        "shake_trigger_by_many": bool(shake_trigger_by_many),
        "shake_threshold_median_peak": float(thr_global_median_peak),
        "shake_threshold_many_vibrating": int(thr_many_strings_vibrating),
        "shake_reject": bool(shake_reject),
        "domination_pass": bool(domination_pass),
        "thr_domination_ratio": float(thr_domination_ratio),
        "startup_guard": bool(startup_guard),
        "startup_guard_min_frame": int(startup_guard_min_frame),
        "event_frame_index": int(event_window.f0),
        "peak_frame": int(peak_frame),
        "peak_relative_to_f0": int(peak_frame - int(event_window.f0)),
        "finger_gate": gate_debug,
        "event_window_debug": event_window.debug,
    }

    return StrikeResult(
        event_time=float(event_window.event.timestamp_sec),
        finger_type=str(event_window.event.finger_type),
        touched_id=int(event_window.event.touched_string_id),
        struck_id=struck_id,
        label=label,
        best_metrics=best_metrics,
        second_metrics=second_metrics,
        debug=debug,
    )


def decide_single_touch_event(
    *,
    video_path: str | Path,
    event: TouchEvent,
    strings: list[StringGeometry],
    fps: float,
    baseline_len: int,
    action_len: int,
    action_start_frame_offset: int,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float = 0.15,
    center_band_h: int = 10,
    candidate_radius_default: int = 2,
    candidate_radius_close_contact: int = 1,
    contact_dist_px_thr: float = 8.0,
    enable_hand_mask: bool = True,
    hand_mask_expand_px: float = 8.0,
    farneback_params: dict[str, Any] | None = None,
    dy_thr: float = 0.5,
    z_thr: float = 2.5,
    thr_peak: float = 4.0,
    thr_duration_frames: int = 3,
    thr_impulse: float = 8.0,
    baseline_gap_frames: int = 0,
    dynamic_action_enabled: bool = False,
    dynamic_offsets: list[int] | None = None,
    dynamic_select_metric: str = "max_impulse",
    geometry_enabled: bool = False,
    geometry_top_k: int = 5,
    geometry_max_distance_px: float = 35.0,
    always_include_touched_id: bool = True,
    include_id_radius_fallback: bool = True,
    missing_touched_id_fallback: str = "none",
    fallback_top_k: int = 3,
    fallback_max_distance_px: float = 40.0,
    log_string_id_inconsistency: bool = True,
    adaptive_roi_enabled: bool = False,
    adaptive_height_ratio: float = 0.45,
    min_roi_h: int = 5,
    max_roi_h: int = 18,
    min_neighbor_distance_px: float = 4.0,
    border_mode: str = "replicate",
    constant_border_value: int = 0,
    reject_if_out_of_frame: bool = False,
    min_inside_fraction: float = 0.95,
    hand_mask_mode: str = "finger_point",
    contact_band_exclusion_px: float = 10.0,
    mask_contact_region: bool = True,
    allow_small_gaps: bool = False,
    max_gap_frames: int = 2,
    normalize_by_gap: bool = True,
    normalization_mode: str = "zscore",
    min_scale: float = 0.05,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    thr_domination_ratio: float = 1.3,
    thr_global_median_peak: float = 3.0,
    thr_many_strings_vibrating: int = 4,
    shake_probe_count: int = 6,
    enable_finger_gate: bool = True,
    thumb_gate: dict[str, Any] | None = None,
    index_gate: dict[str, Any] | None = None,
    stabilize_enabled: bool = False,
    finger_positions_by_frame: dict[int, tuple[float, float]] | None = None,
    finger_velocity_by_frame: dict[int, float] | None = None,
    event_velocity_stats: dict[str, float] | None = None,
) -> StrikeResult:
    event_window = process_single_event_windows(
        video_path=video_path,
        event=event,
        strings=strings,
        fps=fps,
        baseline_len=baseline_len,
        action_len=action_len,
        action_start_frame_offset=action_start_frame_offset,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
        center_band_h=center_band_h,
        candidate_radius_default=candidate_radius_default,
        candidate_radius_close_contact=candidate_radius_close_contact,
        contact_dist_px_thr=contact_dist_px_thr,
        enable_hand_mask=enable_hand_mask,
        hand_mask_expand_px=hand_mask_expand_px,
        farneback_params=farneback_params,
        dy_thr=dy_thr,
        stabilize_enabled=stabilize_enabled,
        baseline_gap_frames=baseline_gap_frames,
        dynamic_action_enabled=dynamic_action_enabled,
        dynamic_offsets=dynamic_offsets,
        dynamic_select_metric=dynamic_select_metric,
        geometry_enabled=geometry_enabled,
        geometry_top_k=geometry_top_k,
        geometry_max_distance_px=geometry_max_distance_px,
        always_include_touched_id=always_include_touched_id,
        include_id_radius_fallback=include_id_radius_fallback,
        missing_touched_id_fallback=missing_touched_id_fallback,
        fallback_top_k=fallback_top_k,
        fallback_max_distance_px=fallback_max_distance_px,
        log_string_id_inconsistency=log_string_id_inconsistency,
        adaptive_roi_enabled=adaptive_roi_enabled,
        adaptive_height_ratio=adaptive_height_ratio,
        min_roi_h=min_roi_h,
        max_roi_h=max_roi_h,
        min_neighbor_distance_px=min_neighbor_distance_px,
        border_mode=border_mode,
        constant_border_value=constant_border_value,
        reject_if_out_of_frame=reject_if_out_of_frame,
        min_inside_fraction=min_inside_fraction,
        hand_mask_mode=hand_mask_mode,
        contact_band_exclusion_px=contact_band_exclusion_px,
        mask_contact_region=mask_contact_region,
        allow_small_gaps=allow_small_gaps,
        max_gap_frames=max_gap_frames,
        normalize_by_gap=normalize_by_gap,
    )
    return decide_event_from_window(
        event_window=event_window,
        strings=strings,
        video_path=video_path,
        fps=fps,
        baseline_len=baseline_len,
        action_len=action_len,
        action_start_frame_offset=action_start_frame_offset,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
        center_band_h=center_band_h,
        enable_hand_mask=enable_hand_mask,
        hand_mask_expand_px=hand_mask_expand_px,
        farneback_params=farneback_params,
        dy_thr=dy_thr,
        z_thr=z_thr,
        thr_peak=thr_peak,
        thr_duration_frames=thr_duration_frames,
        thr_impulse=thr_impulse,
        baseline_gap_frames=baseline_gap_frames,
        dynamic_action_enabled=dynamic_action_enabled,
        dynamic_offsets=dynamic_offsets,
        dynamic_select_metric=dynamic_select_metric,
        geometry_enabled=geometry_enabled,
        geometry_top_k=geometry_top_k,
        geometry_max_distance_px=geometry_max_distance_px,
        always_include_touched_id=always_include_touched_id,
        include_id_radius_fallback=include_id_radius_fallback,
        missing_touched_id_fallback=missing_touched_id_fallback,
        fallback_top_k=fallback_top_k,
        fallback_max_distance_px=fallback_max_distance_px,
        log_string_id_inconsistency=log_string_id_inconsistency,
        adaptive_roi_enabled=adaptive_roi_enabled,
        adaptive_height_ratio=adaptive_height_ratio,
        min_roi_h=min_roi_h,
        max_roi_h=max_roi_h,
        min_neighbor_distance_px=min_neighbor_distance_px,
        border_mode=border_mode,
        constant_border_value=constant_border_value,
        reject_if_out_of_frame=reject_if_out_of_frame,
        min_inside_fraction=min_inside_fraction,
        hand_mask_mode=hand_mask_mode,
        contact_band_exclusion_px=contact_band_exclusion_px,
        mask_contact_region=mask_contact_region,
        allow_small_gaps=allow_small_gaps,
        max_gap_frames=max_gap_frames,
        normalize_by_gap=normalize_by_gap,
        normalization_mode=normalization_mode,
        min_scale=min_scale,
        mad_scale_factor=mad_scale_factor,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        require_absolute_motion=require_absolute_motion,
        min_action_mean=min_action_mean,
        min_action_max=min_action_max,
        max_baseline_mean=max_baseline_mean,
        thr_domination_ratio=thr_domination_ratio,
        thr_global_median_peak=thr_global_median_peak,
        thr_many_strings_vibrating=thr_many_strings_vibrating,
        shake_probe_count=shake_probe_count,
        enable_finger_gate=enable_finger_gate,
        thumb_gate=thumb_gate,
        index_gate=index_gate,
        stabilize_enabled=stabilize_enabled,
        finger_positions_by_frame=finger_positions_by_frame,
        finger_velocity_by_frame=finger_velocity_by_frame,
        event_velocity_stats=event_velocity_stats,
    )


def decide_touch_events(
    *,
    video_path: str | Path,
    touch_events: list[TouchEvent],
    strings: list[StringGeometry],
    fps: float,
    baseline_len: int,
    action_len: int,
    action_start_frame_offset: int,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float = 0.15,
    center_band_h: int = 10,
    candidate_radius_default: int = 2,
    candidate_radius_close_contact: int = 1,
    contact_dist_px_thr: float = 8.0,
    enable_hand_mask: bool = True,
    hand_mask_expand_px: float = 8.0,
    farneback_params: dict[str, Any] | None = None,
    dy_thr: float = 0.5,
    z_thr: float = 2.5,
    thr_peak: float = 4.0,
    thr_duration_frames: int = 3,
    thr_impulse: float = 8.0,
    baseline_gap_frames: int = 0,
    dynamic_action_enabled: bool = False,
    dynamic_offsets: list[int] | None = None,
    dynamic_select_metric: str = "max_impulse",
    geometry_enabled: bool = False,
    geometry_top_k: int = 5,
    geometry_max_distance_px: float = 35.0,
    always_include_touched_id: bool = True,
    include_id_radius_fallback: bool = True,
    missing_touched_id_fallback: str = "none",
    fallback_top_k: int = 3,
    fallback_max_distance_px: float = 40.0,
    log_string_id_inconsistency: bool = True,
    adaptive_roi_enabled: bool = False,
    adaptive_height_ratio: float = 0.45,
    min_roi_h: int = 5,
    max_roi_h: int = 18,
    min_neighbor_distance_px: float = 4.0,
    border_mode: str = "replicate",
    constant_border_value: int = 0,
    reject_if_out_of_frame: bool = False,
    min_inside_fraction: float = 0.95,
    hand_mask_mode: str = "finger_point",
    contact_band_exclusion_px: float = 10.0,
    mask_contact_region: bool = True,
    allow_small_gaps: bool = False,
    max_gap_frames: int = 2,
    normalize_by_gap: bool = True,
    normalization_mode: str = "zscore",
    min_scale: float = 0.05,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
    require_absolute_motion: bool = False,
    min_action_mean: float = 0.02,
    min_action_max: float = 0.05,
    max_baseline_mean: float | None = None,
    thr_domination_ratio: float = 1.3,
    thr_global_median_peak: float = 3.0,
    thr_many_strings_vibrating: int = 4,
    shake_probe_count: int = 6,
    enable_finger_gate: bool = True,
    thumb_gate: dict[str, Any] | None = None,
    index_gate: dict[str, Any] | None = None,
    stabilize_enabled: bool = False,
    finger_positions_by_event: dict[int, dict[int, tuple[float, float]]] | None = None,
    finger_velocity_by_event: dict[int, dict[int, float]] | None = None,
    event_velocity_stats_by_event: dict[int, dict[str, float]] | None = None,
) -> list[StrikeResult]:
    results: list[StrikeResult] = []
    for ev in touch_events:
        row_key = int(ev.row_index)
        results.append(
            decide_single_touch_event(
                video_path=video_path,
                event=ev,
                strings=strings,
                fps=fps,
                baseline_len=baseline_len,
                action_len=action_len,
                action_start_frame_offset=action_start_frame_offset,
                roi_w=roi_w,
                roi_h=roi_h,
                trim_ends_ratio=trim_ends_ratio,
                center_band_h=center_band_h,
                candidate_radius_default=candidate_radius_default,
                candidate_radius_close_contact=candidate_radius_close_contact,
                contact_dist_px_thr=contact_dist_px_thr,
                enable_hand_mask=enable_hand_mask,
                hand_mask_expand_px=hand_mask_expand_px,
                farneback_params=farneback_params,
                dy_thr=dy_thr,
                z_thr=z_thr,
                thr_peak=thr_peak,
                thr_duration_frames=thr_duration_frames,
                thr_impulse=thr_impulse,
                baseline_gap_frames=baseline_gap_frames,
                dynamic_action_enabled=dynamic_action_enabled,
                dynamic_offsets=dynamic_offsets,
                dynamic_select_metric=dynamic_select_metric,
                geometry_enabled=geometry_enabled,
                geometry_top_k=geometry_top_k,
                geometry_max_distance_px=geometry_max_distance_px,
                always_include_touched_id=always_include_touched_id,
                include_id_radius_fallback=include_id_radius_fallback,
                missing_touched_id_fallback=missing_touched_id_fallback,
                fallback_top_k=fallback_top_k,
                fallback_max_distance_px=fallback_max_distance_px,
                log_string_id_inconsistency=log_string_id_inconsistency,
                adaptive_roi_enabled=adaptive_roi_enabled,
                adaptive_height_ratio=adaptive_height_ratio,
                min_roi_h=min_roi_h,
                max_roi_h=max_roi_h,
                min_neighbor_distance_px=min_neighbor_distance_px,
                border_mode=border_mode,
                constant_border_value=constant_border_value,
                reject_if_out_of_frame=reject_if_out_of_frame,
                min_inside_fraction=min_inside_fraction,
                hand_mask_mode=hand_mask_mode,
                contact_band_exclusion_px=contact_band_exclusion_px,
                mask_contact_region=mask_contact_region,
                allow_small_gaps=allow_small_gaps,
                max_gap_frames=max_gap_frames,
                normalize_by_gap=normalize_by_gap,
                normalization_mode=normalization_mode,
                min_scale=min_scale,
                mad_scale_factor=mad_scale_factor,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                require_absolute_motion=require_absolute_motion,
                min_action_mean=min_action_mean,
                min_action_max=min_action_max,
                max_baseline_mean=max_baseline_mean,
                thr_domination_ratio=thr_domination_ratio,
                thr_global_median_peak=thr_global_median_peak,
                thr_many_strings_vibrating=thr_many_strings_vibrating,
                shake_probe_count=shake_probe_count,
                enable_finger_gate=enable_finger_gate,
                thumb_gate=thumb_gate,
                index_gate=index_gate,
                stabilize_enabled=stabilize_enabled,
                finger_positions_by_frame=(
                    None if finger_positions_by_event is None else finger_positions_by_event.get(row_key)
                ),
                finger_velocity_by_frame=(
                    None if finger_velocity_by_event is None else finger_velocity_by_event.get(row_key)
                ),
                event_velocity_stats=(
                    None if event_velocity_stats_by_event is None else event_velocity_stats_by_event.get(row_key)
                ),
            )
        )
    return results
