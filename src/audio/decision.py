from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.audio.onset import detect_onset_in_window, prepare_onset_strength
from src.audio.pitch import estimate_pitch_with_fallbacks
from src.audio.tuning import TuningEntry, load_tuning_table, match_f0_to_tuning, select_candidate_string_ids
from src.io.json_utils import build_meta
from src.pipeline.config import confidence_label


@dataclass(frozen=True)
class AudioDecisionArtifacts:
    decision_payload: dict[str, Any]
    strike_payload: dict[str, Any]
    canonical_right_events: list[dict[str, Any]]


def _float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(round(float(v)))
    except Exception:
        return None


def _touch_time_sec(item: dict[str, Any]) -> float | None:
    return _float_or_none(item.get("timestamp_sec", item.get("time_sec")))


def _touch_frame_index(item: dict[str, Any], fps: float) -> int | None:
    frame_idx = _int_or_none(item.get("frame_index"))
    if frame_idx is not None:
        return frame_idx
    ts = _touch_time_sec(item)
    if ts is None:
        return None
    return int(round(float(ts) * max(float(fps), 1e-6)))


def _is_right_hand(item: dict[str, Any]) -> bool:
    return str(item.get("hand_side") or item.get("hand") or "").strip().lower() == "right"


def _normalize_touch_subset(item: dict[str, Any], fps: float) -> dict[str, Any]:
    ts = _touch_time_sec(item)
    return {
        "timestamp_sec": float(ts) if ts is not None else None,
        "frame_index": _touch_frame_index(item, fps),
        "hand_side": str(item.get("hand_side") or item.get("hand") or ""),
        "finger_type": str(item.get("finger_type") or ""),
        "touched_string_id": _int_or_none(item.get("touched_string_id", item.get("string_id"))),
        "touch_conf": _float_or_none(item.get("touch_conf")),
        "contact_x": _float_or_none(item.get("contact_x")),
        "contact_y": _float_or_none(item.get("contact_y")),
        "finger_x": _float_or_none(item.get("finger_x")),
        "finger_y": _float_or_none(item.get("finger_y")),
        "distance_px": _float_or_none(item.get("distance_px")),
    }


def _build_event_id(item: dict[str, Any], index_1based: int, fps: float) -> str:
    frame_idx = _touch_frame_index(item, fps)
    finger = str(item.get("finger_type") or "").strip().lower() or "unknown"
    touched = _int_or_none(item.get("touched_string_id", item.get("string_id")))
    if frame_idx is not None:
        return f"rh_{index_1based:05d}_f{frame_idx:06d}_{finger}_s{touched if touched is not None else 'na'}"
    ts = _touch_time_sec(item)
    if ts is not None:
        ms = int(round(float(ts) * 1000.0))
        return f"rh_{index_1based:05d}_t{ms:08d}_{finger}_s{touched if touched is not None else 'na'}"
    return f"rh_{index_1based:05d}_{finger}_s{touched if touched is not None else 'na'}"


def _sorted_right_touch_events(touch_events: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    items = [dict(ev) for ev in touch_events if isinstance(ev, dict) and _is_right_hand(ev)]
    items.sort(
        key=lambda ev: (
            _touch_frame_index(ev, fps) if _touch_frame_index(ev, fps) is not None else 10**9,
            _touch_time_sec(ev) if _touch_time_sec(ev) is not None else 10**9,
            str(ev.get("finger_type") or ""),
            _int_or_none(ev.get("touched_string_id", ev.get("string_id"))) or -1,
        )
    )
    for i, ev in enumerate(items, start=1):
        ev.setdefault("hand_side", "right")
        ts = _touch_time_sec(ev)
        if ts is not None:
            ev["timestamp_sec"] = float(ts)
            ev.setdefault("time_sec", float(ts))
        frame_idx = _touch_frame_index(ev, fps)
        if frame_idx is not None:
            ev["frame_index"] = int(frame_idx)
        ev["event_id"] = _build_event_id(ev, i, fps)
    return items


def _synthesize_touch_from_video_decision(vd: dict[str, Any], fps: float) -> dict[str, Any]:
    ts = _float_or_none(vd.get("time_sec"))
    frame_idx = _int_or_none(vd.get("frame_index"))
    if ts is None and frame_idx is not None:
        ts = float(frame_idx) / max(float(fps), 1e-6)
    return {
        "timestamp_sec": ts,
        "time_sec": ts,
        "frame_index": frame_idx if frame_idx is not None else (int(round(float(ts) * fps)) if ts is not None else None),
        "hand_side": "right",
        "finger_type": str(vd.get("finger_type") or ""),
        "touched_string_id": _int_or_none(vd.get("touched_id")),
        "touch_conf": None,
    }


def _align_touches_to_video_decisions(
    touch_events: list[dict[str, Any]],
    video_decision_events: list[dict[str, Any]],
    fps: float,
) -> list[dict[str, Any]]:
    right_touches = _sorted_right_touch_events(touch_events, fps)
    used: set[int] = set()
    aligned: list[dict[str, Any]] = []
    for vd in video_decision_events:
        if not isinstance(vd, dict):
            continue
        vd_time = _float_or_none(vd.get("time_sec"))
        vd_frame = _int_or_none(vd.get("frame_index"))
        vd_finger = str(vd.get("finger_type") or "").strip().lower()
        vd_touched = _int_or_none(vd.get("touched_id"))
        best_i: int | None = None
        best_key: tuple[float, float] | None = None
        for i, ev in enumerate(right_touches):
            if i in used:
                continue
            if vd_finger and str(ev.get("finger_type") or "").strip().lower() != vd_finger:
                continue
            ev_touched = _int_or_none(ev.get("touched_string_id", ev.get("string_id")))
            if vd_touched is not None and ev_touched is not None and ev_touched != vd_touched:
                continue
            ev_time = _touch_time_sec(ev)
            ev_frame = _touch_frame_index(ev, fps)
            dt = abs(float(ev_time) - float(vd_time)) if ev_time is not None and vd_time is not None else 9999.0
            df = abs(int(ev_frame) - int(vd_frame)) if ev_frame is not None and vd_frame is not None else 9999.0
            key = (dt, df)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
                if dt <= (1.0 / max(float(fps), 1e-6)) and df <= 1:
                    break
        if best_i is None:
            aligned.append(_synthesize_touch_from_video_decision(vd, fps))
        else:
            used.add(best_i)
            aligned.append(dict(right_touches[best_i]))
    aligned.sort(
        key=lambda ev: (
            _touch_frame_index(ev, fps) if _touch_frame_index(ev, fps) is not None else 10**9,
            _touch_time_sec(ev) if _touch_time_sec(ev) is not None else 10**9,
        )
    )
    for i, ev in enumerate(aligned, start=1):
        ev["event_id"] = _build_event_id(ev, i, fps)
        ts = _touch_time_sec(ev)
        if ts is not None:
            ev["timestamp_sec"] = float(ts)
            ev.setdefault("time_sec", float(ts))
        if _touch_frame_index(ev, fps) is not None:
            ev["frame_index"] = _touch_frame_index(ev, fps)
    return aligned


def build_canonical_right_events(
    *,
    touch_events: list[dict[str, Any]],
    fps: float,
    video_decision_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if video_decision_events:
        return _align_touches_to_video_decisions(touch_events, video_decision_events, fps)
    return _sorted_right_touch_events(touch_events, fps)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _compute_confidence(
    onset_z: float,
    pitch_conf: float,
    cents_error_abs: float,
    max_cents_error: float,
    weights: dict[str, Any],
) -> float:
    x = (
        float(weights.get("bias", -0.4))
        + float(weights.get("onset_z", 1.2)) * float(onset_z)
        + float(weights.get("pitch_conf", 1.5)) * float(pitch_conf)
        - float(weights.get("cents_penalty", 1.0)) * (float(cents_error_abs) / max(float(max_cents_error), 1e-6))
    )
    return max(0.0, min(1.0, _sigmoid(x)))


def _compute_onset_only_confidence(
    onset_z: float,
    touch_conf: float | None,
    weights: dict[str, Any],
) -> float:
    # Reuse the same weight keys for consistency, but ignore pitch/tuning terms.
    x = (
        float(weights.get("bias", -0.4))
        + float(weights.get("onset_z", 1.2)) * float(onset_z)
        + 0.35 * float(touch_conf if touch_conf is not None else 0.5)
    )
    return max(0.0, min(1.0, _sigmoid(x)))


def _audio_reject_event(
    *,
    ev: dict[str, Any],
    fps: float,
    thresholds: dict[str, Any],
    pitch_backend: str,
    status: str,
    audio_window: dict[str, float | None],
    onset_time_sec: float | None = None,
    onset_score: float | None = None,
    f0_hz: float | None = None,
    pitch_conf: float | None = None,
    candidate_strings: list[int] | None = None,
    cents_error: float | None = None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = 0.0
    return {
        "event_id": str(ev.get("event_id")),
        "touch": _normalize_touch_subset(ev, fps),
        "audio": {
            "status": status,
            "touch_time_sec": _touch_time_sec(ev),
            "audio_window": audio_window,
            "onset_time_sec": onset_time_sec,
            "onset_score": onset_score,
            "pitch_backend": pitch_backend,
            "f0_hz": f0_hz,
            "pitch_conf": pitch_conf,
            "matched_string_id": None,
            "cents_error": cents_error,
            "candidate_strings": candidate_strings or [],
            "debug": debug or {},
        },
        "decision": {
            "struck_string_id": None,
            "confidence": conf,
            "confidence_label": confidence_label(conf, thresholds),
            "reject_reason": status,
        },
    }


def _audio_strike_event(
    *,
    ev: dict[str, Any],
    fps: float,
    thresholds: dict[str, Any],
    pitch_backend: str,
    audio_window: dict[str, float | None],
    onset_time_sec: float,
    onset_score: float,
    f0_hz: float | None,
    pitch_conf: float | None,
    matched_string_id: int,
    cents_error: float | None,
    candidate_strings: list[int],
    debug: dict[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "event_id": str(ev.get("event_id")),
        "touch": _normalize_touch_subset(ev, fps),
        "audio": {
            "status": "strike",
            "touch_time_sec": _touch_time_sec(ev),
            "audio_window": audio_window,
            "onset_time_sec": float(onset_time_sec),
            "onset_score": float(onset_score),
            "pitch_backend": pitch_backend,
            "f0_hz": (float(f0_hz) if f0_hz is not None else None),
            "pitch_conf": (float(pitch_conf) if pitch_conf is not None else None),
            "matched_string_id": int(matched_string_id),
            "cents_error": (float(cents_error) if cents_error is not None else None),
            "candidate_strings": [int(s) for s in candidate_strings],
            "debug": debug,
        },
        "decision": {
            "struck_string_id": int(matched_string_id),
            "confidence": float(confidence),
            "confidence_label": confidence_label(float(confidence), thresholds),
            "reject_reason": None,
        },
    }


def _audio_strike_summary_row(event: dict[str, Any], fps: float) -> dict[str, Any]:
    touch_subset = event["touch"]
    audio = event["audio"]
    return {
        "event_id": str(event["event_id"]),
        "timestamp_sec": _float_or_none(touch_subset.get("timestamp_sec")),
        "frame_index": _int_or_none(touch_subset.get("frame_index")),
        "finger_type": str(touch_subset.get("finger_type") or ""),
        "touched_string_id": _int_or_none(touch_subset.get("touched_string_id")),
        "struck_string_id": _int_or_none(event["decision"].get("struck_string_id")),
        "onset_time_sec": _float_or_none(audio.get("onset_time_sec")),
        "onset_frame_index": (
            int(round(float(audio.get("onset_time_sec")) * max(float(fps), 1e-6)))
            if audio.get("onset_time_sec") is not None
            else None
        ),
        "f0_hz": _float_or_none(audio.get("f0_hz")),
        "pitch_conf": _float_or_none(audio.get("pitch_conf")),
        "cents_error": _float_or_none(audio.get("cents_error")),
        "confidence": float(event["decision"].get("confidence") or 0.0),
        "confidence_label": str(event["decision"].get("confidence_label") or ""),
    }


def run_audio_decision_for_right_events(
    *,
    touch_events: list[dict[str, Any]],
    fps: float,
    audio: np.ndarray | None,
    sr: int | None,
    config: dict[str, Any],
    source_video: str | None = None,
    touch_events_json_path: str | None = None,
    video_decision_events: list[dict[str, Any]] | None = None,
    tuning_by_string: dict[int, TuningEntry] | None = None,
) -> AudioDecisionArtifacts:
    fps_safe = max(float(fps), 1e-6)
    audio_cfg = dict(config.get("audio") or {})
    fusion_cfg = dict(config.get("fusion") or {})
    thresholds = dict(fusion_cfg.get("confidence_thresholds") or {})
    preferred_pitch_backend = str(audio_cfg.get("pitch_backend", "crepe"))
    decision_mode = str(audio_cfg.get("decision_mode", "onset_only")).strip().lower()
    use_pitch_tuning_match = decision_mode in {"onset_pitch_match", "pitch_match", "pitch", "pitch_tuning"}
    pitch_fallback_to_onset = bool(audio_cfg.get("pitch_fallback_to_onset", False))
    tuning_load_error: str | None = None
    if use_pitch_tuning_match and tuning_by_string is None:
        tuning_path = Path(str(audio_cfg.get("tuning_table_path", "configs/saung_tuning.json")))
        try:
            tuning_by_string = load_tuning_table(tuning_path)
        except Exception as exc:  # noqa: BLE001
            tuning_by_string = {}
            tuning_load_error = str(exc)
            # Degrade gracefully to onset-only if tuning is not available.
            use_pitch_tuning_match = False
            decision_mode = "onset_only"
    if tuning_by_string is None:
        tuning_by_string = {}

    canonical_events = build_canonical_right_events(
        touch_events=touch_events,
        fps=fps_safe,
        video_decision_events=video_decision_events,
    )

    events_out: list[dict[str, Any]] = []
    strike_out: list[dict[str, Any]] = []
    onset_cache = None
    onset_prep_error: str | None = None
    if audio is not None and sr is not None:
        try:
            onset_cache = prepare_onset_strength(
                np.asarray(audio, dtype=np.float32),
                int(sr),
                hop_length=int(audio_cfg.get("onset_strength_hop", 256)),
            )
        except Exception as exc:  # noqa: BLE001
            onset_prep_error = str(exc)

    if audio is None or sr is None or onset_cache is None:
        status = "no_audio"
        debug = {"reason": status}
        if onset_prep_error:
            debug["onset_prepare_error"] = onset_prep_error
        for ev in canonical_events:
            events_out.append(
                _audio_reject_event(
                    ev=ev,
                    fps=fps_safe,
                    thresholds=thresholds,
                    pitch_backend=preferred_pitch_backend,
                    status=status,
                    audio_window={"t0": None, "t1": None},
                    debug=debug,
                )
            )
    else:
        for ev in canonical_events:
            touch_time = _touch_time_sec(ev)
            audio_window = {
                "t0": float(touch_time) if touch_time is not None else None,
                "t1": (float(touch_time) + float(audio_cfg.get("onset_window_sec", 0.25))) if touch_time is not None else None,
            }
            if touch_time is None:
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=preferred_pitch_backend,
                        status="no_audio",
                        audio_window=audio_window,
                        debug={"reason": "missing_touch_timestamp"},
                    )
                )
                continue

            distance_px = _float_or_none(ev.get("distance_px"))
            touched_sid = _int_or_none(ev.get("touched_string_id", ev.get("string_id")))
            available_string_ids = list(tuning_by_string.keys()) if tuning_by_string else ([int(touched_sid)] if touched_sid is not None else [])
            candidate_strings = select_candidate_string_ids(
                touched_string_id=touched_sid,
                available_string_ids=available_string_ids,
                distance_px=distance_px,
                candidate_radius_default=int(audio_cfg.get("candidate_radius_default", 2)),
                candidate_radius_close_contact=int(audio_cfg.get("candidate_radius_close_contact", 1)),
                contact_dist_px_thr=float(audio_cfg.get("contact_dist_px_thr", 8)),
            )

            onset = detect_onset_in_window(
                onset_cache,
                touch_time_sec=float(touch_time),
                onset_window_sec=float(audio_cfg.get("onset_window_sec", 0.25)),
                baseline_window_sec=float(audio_cfg.get("baseline_window_sec", 0.20)),
                onset_threshold=float(audio_cfg.get("onset_threshold", 0.65)),
            )
            if onset.onset_time_sec is None:
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=preferred_pitch_backend,
                        status="no_onset",
                        audio_window=audio_window,
                        onset_score=_float_or_none(onset.onset_score),
                        candidate_strings=candidate_strings,
                        debug={"onset": onset.debug, "distance_px": distance_px},
                    )
                )
                continue

            if not use_pitch_tuning_match:
                if touched_sid is None:
                    events_out.append(
                        _audio_reject_event(
                            ev=ev,
                            fps=fps_safe,
                            thresholds=thresholds,
                            pitch_backend="none",
                            status="no_match",
                            audio_window=audio_window,
                            onset_time_sec=float(onset.onset_time_sec),
                            onset_score=_float_or_none(onset.onset_score),
                            candidate_strings=candidate_strings,
                            debug={
                                "mode": decision_mode,
                                "reason": "missing_touched_string_id",
                                "onset": onset.debug,
                                "distance_px": distance_px,
                            },
                        )
                    )
                    continue

                onset_z = float(onset.onset_score or 0.0)
                touch_conf = _float_or_none(ev.get("touch_conf"))
                conf = _compute_onset_only_confidence(
                    onset_z=onset_z,
                    touch_conf=touch_conf,
                    weights=dict(audio_cfg.get("confidence_weights") or {}),
                )
                onset_only_candidates = [int(touched_sid)] if touched_sid is not None else []
                event = _audio_strike_event(
                    ev=ev,
                    fps=fps_safe,
                    thresholds=thresholds,
                    pitch_backend="none",
                    audio_window=audio_window,
                    onset_time_sec=float(onset.onset_time_sec),
                    onset_score=onset_z,
                    f0_hz=None,
                    pitch_conf=None,
                    matched_string_id=int(touched_sid),
                    cents_error=None,
                    candidate_strings=onset_only_candidates,
                    confidence=conf,
                    debug={
                        "mode": decision_mode,
                        "rule": "onset_detected_at_touch",
                        "onset": onset.debug,
                        "distance_px": distance_px,
                        "touch_conf": touch_conf,
                    },
                )
                events_out.append(event)
                touch_subset = event["touch"]
                strike_out.append(
                    {
                        "event_id": str(event["event_id"]),
                        "timestamp_sec": _float_or_none(touch_subset.get("timestamp_sec")),
                        "frame_index": _int_or_none(touch_subset.get("frame_index")),
                        "finger_type": str(touch_subset.get("finger_type") or ""),
                        "touched_string_id": _int_or_none(touch_subset.get("touched_string_id")),
                        "struck_string_id": int(touched_sid),
                        "onset_time_sec": float(onset.onset_time_sec),
                        "onset_frame_index": int(round(float(onset.onset_time_sec) * fps_safe)),
                        "f0_hz": None,
                        "pitch_conf": None,
                        "cents_error": None,
                        "confidence": float(conf),
                        "confidence_label": str(event["decision"]["confidence_label"]),
                    }
                )
                continue

            def _append_pitch_fallback_strike(
                *,
                fallback_reason: str,
                pitch_backend: str,
                pitch_debug: dict[str, Any] | None = None,
                f0_hz: float | None = None,
                pitch_conf: float | None = None,
                cents_error: float | None = None,
                candidate_strings_override: list[int] | None = None,
            ) -> bool:
                if not pitch_fallback_to_onset or touched_sid is None:
                    return False
                onset_z = float(onset.onset_score or 0.0)
                touch_conf = _float_or_none(ev.get("touch_conf"))
                conf = _compute_onset_only_confidence(
                    onset_z=onset_z,
                    touch_conf=touch_conf,
                    weights=dict(audio_cfg.get("confidence_weights") or {}),
                )
                # Keep fallback confidence below "high" so true pitch matches win conflicts.
                conf = min(float(conf), 0.74)
                event = _audio_strike_event(
                    ev=ev,
                    fps=fps_safe,
                    thresholds=thresholds,
                    pitch_backend=pitch_backend,
                    audio_window=audio_window,
                    onset_time_sec=float(onset.onset_time_sec),
                    onset_score=onset_z,
                    f0_hz=f0_hz,
                    pitch_conf=pitch_conf,
                    matched_string_id=int(touched_sid),
                    cents_error=cents_error,
                    candidate_strings=candidate_strings_override or [int(touched_sid)],
                    confidence=conf,
                    debug={
                        "mode": decision_mode,
                        "rule": "pitch_failed_fallback_to_onset",
                        "fallback_reason": fallback_reason,
                        "onset": onset.debug,
                        "pitch": pitch_debug or {},
                        "distance_px": distance_px,
                        "touch_conf": touch_conf,
                    },
                )
                events_out.append(event)
                strike_out.append(_audio_strike_summary_row(event, fps_safe))
                return True

            pitch_window_t0 = float(onset.onset_time_sec)
            pitch_window_t1 = pitch_window_t0 + float(audio_cfg.get("pitch_window_sec", 0.12))
            pitch = estimate_pitch_with_fallbacks(
                np.asarray(audio, dtype=np.float32),
                int(sr),
                t0_sec=pitch_window_t0,
                t1_sec=pitch_window_t1,
                min_f0_hz=float(audio_cfg.get("min_f0_hz", 60)),
                max_f0_hz=float(audio_cfg.get("max_f0_hz", 2000)),
                preferred_backend=preferred_pitch_backend,
            )
            if pitch.f0_hz is None:
                if _append_pitch_fallback_strike(
                    fallback_reason="no_pitch",
                    pitch_backend=str(pitch.backend),
                    pitch_debug=pitch.debug,
                    candidate_strings_override=candidate_strings,
                ):
                    continue
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=str(pitch.backend),
                        status="low_pitch_conf",
                        audio_window=audio_window,
                        onset_time_sec=float(onset.onset_time_sec),
                        onset_score=_float_or_none(onset.onset_score),
                        candidate_strings=candidate_strings,
                        debug={"onset": onset.debug, "pitch": pitch.debug, "pitch_window": {"t0": pitch_window_t0, "t1": pitch_window_t1}},
                    )
                )
                continue

            pitch_conf = float(pitch.pitch_conf if pitch.pitch_conf is not None else 0.0)
            f0_hz = float(pitch.f0_hz)
            min_f0 = float(audio_cfg.get("min_f0_hz", 60))
            max_f0 = float(audio_cfg.get("max_f0_hz", 2000))
            if not (min_f0 <= f0_hz <= max_f0):
                if _append_pitch_fallback_strike(
                    fallback_reason="pitch_out_of_range",
                    pitch_backend=str(pitch.backend),
                    pitch_debug=pitch.debug,
                    f0_hz=f0_hz,
                    pitch_conf=pitch_conf,
                    candidate_strings_override=candidate_strings,
                ):
                    continue
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=str(pitch.backend),
                        status="pitch_out_of_range",
                        audio_window=audio_window,
                        onset_time_sec=float(onset.onset_time_sec),
                        onset_score=_float_or_none(onset.onset_score),
                        f0_hz=f0_hz,
                        pitch_conf=pitch_conf,
                        candidate_strings=candidate_strings,
                        debug={"onset": onset.debug, "pitch": pitch.debug},
                    )
                )
                continue

            min_pitch_conf = float(audio_cfg.get("min_pitch_conf", 0.50))
            if pitch_conf < min_pitch_conf:
                if _append_pitch_fallback_strike(
                    fallback_reason="low_pitch_conf",
                    pitch_backend=str(pitch.backend),
                    pitch_debug=pitch.debug,
                    f0_hz=f0_hz,
                    pitch_conf=pitch_conf,
                    candidate_strings_override=candidate_strings,
                ):
                    continue
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=str(pitch.backend),
                        status="low_pitch_conf",
                        audio_window=audio_window,
                        onset_time_sec=float(onset.onset_time_sec),
                        onset_score=_float_or_none(onset.onset_score),
                        f0_hz=f0_hz,
                        pitch_conf=pitch_conf,
                        candidate_strings=candidate_strings,
                        debug={"onset": onset.debug, "pitch": pitch.debug, "min_pitch_conf": min_pitch_conf},
                    )
                )
                continue

            match = match_f0_to_tuning(
                f0_hz=f0_hz,
                tuning_by_string=tuning_by_string,
                candidate_string_ids=candidate_strings,
                max_cents_error=float(audio_cfg.get("max_cents_error", 50)),
            )
            if match.matched_string_id is None:
                if _append_pitch_fallback_strike(
                    fallback_reason="no_tuning_match",
                    pitch_backend=str(pitch.backend),
                    pitch_debug={"pitch": pitch.debug, "match": match.debug},
                    f0_hz=f0_hz,
                    pitch_conf=pitch_conf,
                    cents_error=_float_or_none(match.cents_error),
                    candidate_strings_override=match.candidate_strings,
                ):
                    continue
                events_out.append(
                    _audio_reject_event(
                        ev=ev,
                        fps=fps_safe,
                        thresholds=thresholds,
                        pitch_backend=str(pitch.backend),
                        status="no_match",
                        audio_window=audio_window,
                        onset_time_sec=float(onset.onset_time_sec),
                        onset_score=_float_or_none(onset.onset_score),
                        f0_hz=f0_hz,
                        pitch_conf=pitch_conf,
                        cents_error=_float_or_none(match.cents_error),
                        candidate_strings=match.candidate_strings,
                        debug={"onset": onset.debug, "pitch": pitch.debug, "match": match.debug},
                    )
                )
                continue

            onset_z = float(onset.onset_score or 0.0)
            cents_err = float(match.cents_error or 0.0)
            conf = _compute_confidence(
                onset_z=onset_z,
                pitch_conf=pitch_conf,
                cents_error_abs=abs(cents_err),
                max_cents_error=float(audio_cfg.get("max_cents_error", 50)),
                weights=dict(audio_cfg.get("confidence_weights") or {}),
            )
            event = _audio_strike_event(
                ev=ev,
                fps=fps_safe,
                thresholds=thresholds,
                pitch_backend=str(pitch.backend),
                audio_window=audio_window,
                onset_time_sec=float(onset.onset_time_sec),
                onset_score=onset_z,
                f0_hz=f0_hz,
                pitch_conf=pitch_conf,
                matched_string_id=int(match.matched_string_id),
                cents_error=cents_err,
                candidate_strings=match.candidate_strings,
                confidence=conf,
                debug={
                    "onset": onset.debug,
                    "pitch": pitch.debug,
                    "match": match.debug,
                    "pitch_window": {"t0": pitch_window_t0, "t1": pitch_window_t1},
                    "distance_px": distance_px,
                },
            )
            events_out.append(event)
            touch_subset = event["touch"]
            strike_out.append(
                {
                    "event_id": str(event["event_id"]),
                    "timestamp_sec": _float_or_none(touch_subset.get("timestamp_sec")),
                    "frame_index": _int_or_none(touch_subset.get("frame_index")),
                    "finger_type": str(touch_subset.get("finger_type") or ""),
                    "touched_string_id": _int_or_none(touch_subset.get("touched_string_id")),
                    "struck_string_id": int(match.matched_string_id),
                    "onset_time_sec": float(onset.onset_time_sec),
                    "onset_frame_index": int(round(float(onset.onset_time_sec) * fps_safe)),
                    "f0_hz": f0_hz,
                    "pitch_conf": pitch_conf,
                    "cents_error": cents_err,
                    "confidence": float(conf),
                    "confidence_label": str(event["decision"]["confidence_label"]),
                }
            )

    status_counts = Counter(str(ev.get("audio", {}).get("status", "")) for ev in events_out)
    reject_counts = Counter(
        str(ev.get("decision", {}).get("reject_reason", ""))
        for ev in events_out
        if ev.get("decision", {}).get("reject_reason") is not None
    )
    conf_counts = Counter(str(ev.get("decision", {}).get("confidence_label", "")) for ev in events_out if ev.get("audio", {}).get("status") == "strike")
    meta = build_meta(
        source_video=source_video,
        touch_events_json_path=touch_events_json_path,
        fps=fps_safe,
        extra={
            "phase": "audio_first",
            "audio_enabled": bool(audio_cfg.get("enabled", True)),
            "decision_mode": decision_mode,
            "audio_sample_rate": (int(sr) if sr is not None else None),
            "pitch_backend_preference": preferred_pitch_backend,
            "tuning_table_size": int(len(tuning_by_string)),
        },
    )
    if tuning_load_error is not None:
        meta["tuning_load_error"] = tuning_load_error
    decision_payload = {
        "meta": meta,
        "events": events_out,
        "counts": {
            "events_total": int(len(events_out)),
            "strike_events_count": int(sum(1 for ev in events_out if ev.get("audio", {}).get("status") == "strike")),
            "status_counts": dict(status_counts),
            "reject_reason_counts": dict(reject_counts),
            "confidence_counts": dict(conf_counts),
        },
    }
    strike_payload = {
        "meta": meta,
        "events": strike_out,
        "counts": {
            "strike_events_count": int(len(strike_out)),
            "confidence_counts": dict(conf_counts),
        },
    }
    return AudioDecisionArtifacts(
        decision_payload=decision_payload,
        strike_payload=strike_payload,
        canonical_right_events=canonical_events,
    )
