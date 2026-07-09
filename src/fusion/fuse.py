from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.audio.decision import build_canonical_right_events
from src.io.json_utils import build_meta
from src.pipeline.config import confidence_label


@dataclass(frozen=True)
class FusionArtifacts:
    decision_payload: dict[str, Any]
    strike_payload: dict[str, Any]


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


def _video_confidence(v: dict[str, Any]) -> float:
    label = str(v.get("label") or "").strip().lower()
    if label != "strike" or v.get("struck_id") is None:
        return 0.0
    peak_z = float(v.get("peak_z", 0.0) or 0.0)
    candidate_score = float(v.get("candidate_score", 0.0) or 0.0)
    # Stable bounded heuristic based on existing metrics.
    peak_term = min(1.0, max(0.0, peak_z / 12.0))
    score_term = min(1.0, max(0.0, candidate_score / 24.0))
    return max(0.0, min(1.0, 0.55 * peak_term + 0.45 * score_term))


def _video_summary(v: dict[str, Any] | None, fps: float, thresholds: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(v, dict):
        return {
            "status": "missing",
            "struck_string_id": None,
            "confidence": 0.0,
            "confidence_label": confidence_label(0.0, thresholds),
            "frame_index": None,
            "time_sec": None,
            "peak_frame": None,
            "decision_reason": "missing",
            "raw": None,
        }
    conf = _video_confidence(v)
    frame_idx = _int_or_none(v.get("frame_index"))
    time_sec = _float_or_none(v.get("time_sec"))
    if time_sec is None and frame_idx is not None:
        time_sec = float(frame_idx) / max(float(fps), 1e-6)
    decision_debug = v.get("decision_debug") if isinstance(v.get("decision_debug"), dict) else {}
    status = "strike" if str(v.get("label") or "").strip().lower() == "strike" and v.get("struck_id") is not None else "touch_only"
    return {
        "status": status,
        "struck_string_id": _int_or_none(v.get("struck_id")),
        "confidence": float(conf),
        "confidence_label": confidence_label(float(conf), thresholds),
        "frame_index": frame_idx,
        "time_sec": time_sec,
        "peak_frame": _int_or_none(decision_debug.get("peak_frame")),
        "decision_reason": str(v.get("decision_reason") or ""),
        "raw": {
            "label": v.get("label"),
            "touched_id": v.get("touched_id"),
            "candidate_score": _float_or_none(v.get("candidate_score")),
            "peak_z": _float_or_none(v.get("peak_z")),
            "duration": _int_or_none(v.get("duration")),
            "impulse": _float_or_none(v.get("impulse")),
            "vibrates": bool(v.get("vibrates", False)),
            "decision_debug": decision_debug,
        },
    }


def _audio_summary(a: dict[str, Any] | None, thresholds: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(a, dict):
        return {
            "status": "missing",
            "struck_string_id": None,
            "confidence": 0.0,
            "confidence_label": confidence_label(0.0, thresholds),
            "onset_time_sec": None,
            "onset_frame": None,
            "pitch_backend": None,
            "f0_hz": None,
            "pitch_conf": None,
            "cents_error": None,
            "raw": None,
        }
    audio_part = a.get("audio") if isinstance(a.get("audio"), dict) else {}
    decision = a.get("decision") if isinstance(a.get("decision"), dict) else {}
    touch = a.get("touch") if isinstance(a.get("touch"), dict) else {}
    onset_time = _float_or_none(audio_part.get("onset_time_sec"))
    onset_frame = _int_or_none(audio_part.get("onset_frame_index"))
    if onset_frame is None and onset_time is not None:
        fps_guess = _float_or_none((a.get("meta") or {}).get("fps"))  # usually absent in event
        if fps_guess:
            onset_frame = int(round(float(onset_time) * float(fps_guess)))
    return {
        "status": str(audio_part.get("status") or "missing"),
        "struck_string_id": _int_or_none(decision.get("struck_string_id")),
        "confidence": float(decision.get("confidence") or 0.0),
        "confidence_label": str(decision.get("confidence_label") or confidence_label(0.0, thresholds)),
        "onset_time_sec": onset_time,
        "onset_frame": onset_frame,
        "pitch_backend": audio_part.get("pitch_backend"),
        "f0_hz": _float_or_none(audio_part.get("f0_hz")),
        "pitch_conf": _float_or_none(audio_part.get("pitch_conf")),
        "cents_error": _float_or_none(audio_part.get("cents_error")),
        "matched_string_id": _int_or_none(audio_part.get("matched_string_id")),
        "touch_frame_index": _int_or_none(touch.get("frame_index")),
        "raw": {
            "audio": audio_part,
            "decision": decision,
        },
    }


def _touch_key_from_touch_subset(t: dict[str, Any]) -> tuple[int | None, str, int | None]:
    return (
        _int_or_none(t.get("frame_index")),
        str(t.get("finger_type") or "").strip().lower(),
        _int_or_none(t.get("touched_string_id")),
    )


def _touch_key_from_video(v: dict[str, Any]) -> tuple[int | None, str, int | None]:
    return (
        _int_or_none(v.get("frame_index")),
        str(v.get("finger_type") or "").strip().lower(),
        _int_or_none(v.get("touched_id")),
    )


def _align_video_to_canonical(
    canonical_events: list[dict[str, Any]],
    video_decision_events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not canonical_events or not video_decision_events:
        return out

    # Fast path: same count, preserve order.
    if len(canonical_events) == len(video_decision_events):
        for ev, vd in zip(canonical_events, video_decision_events):
            out[str(ev.get("event_id"))] = vd
        return out

    by_key: dict[tuple[int | None, str, int | None], list[dict[str, Any]]] = {}
    for vd in video_decision_events:
        if not isinstance(vd, dict):
            continue
        by_key.setdefault(_touch_key_from_video(vd), []).append(vd)
    for ev in canonical_events:
        key = _touch_key_from_touch_subset(ev)
        choices = by_key.get(key) or []
        if choices:
            out[str(ev.get("event_id"))] = choices.pop(0)
    return out


def _align_audio_to_canonical(
    canonical_events: list[dict[str, Any]],
    audio_decision_events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not canonical_events or not audio_decision_events:
        return out
    by_id = {str(a.get("event_id")): a for a in audio_decision_events if isinstance(a, dict)}
    if by_id:
        for ev in canonical_events:
            eid = str(ev.get("event_id"))
            if eid in by_id:
                out[eid] = by_id[eid]
        if out:
            return out
    if len(canonical_events) == len(audio_decision_events):
        for ev, ad in zip(canonical_events, audio_decision_events):
            out[str(ev.get("event_id"))] = ad
    return out


def _select_timing(
    *,
    video: dict[str, Any],
    audio: dict[str, Any],
    fps: float,
    timing_source: str,
) -> dict[str, Any]:
    v_frame = _int_or_none(video.get("frame_index"))
    v_time = _float_or_none(video.get("time_sec"))
    v_peak = _int_or_none(video.get("peak_frame"))
    a_onset_time = _float_or_none(audio.get("onset_time_sec"))
    a_onset_frame = _int_or_none(audio.get("onset_frame"))
    if a_onset_frame is None and a_onset_time is not None:
        a_onset_frame = int(round(float(a_onset_time) * max(float(fps), 1e-6)))

    source = (timing_source or "hybrid").strip().lower()
    if source == "video":
        onset_frame = v_frame
        onset_time = v_time if v_time is not None else (float(onset_frame) / max(float(fps), 1e-6) if onset_frame is not None else None)
    elif source == "audio":
        onset_frame = a_onset_frame
        onset_time = a_onset_time
    else:
        onset_frame = v_frame if v_frame is not None else a_onset_frame
        onset_time = v_time if v_time is not None else a_onset_time

    return {
        "onset_frame": onset_frame,
        "onset_time_sec": onset_time,
        "peak_frame": v_peak,
    }


def _fuse_one(
    *,
    mode: str,
    video: dict[str, Any],
    audio: dict[str, Any],
    fusion_cfg: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_strike = video.get("status") == "strike" and video.get("struck_string_id") is not None
    audio_strike = audio.get("status") == "strike" and audio.get("struck_string_id") is not None
    video_conf = float(video.get("confidence") or 0.0)
    audio_conf = float(audio.get("confidence") or 0.0)
    prefer_audio_when_conf_ge = float(fusion_cfg.get("prefer_audio_when_conf_ge", 0.75))
    prefer_video_when_audio_missing = bool(fusion_cfg.get("prefer_video_when_audio_missing", True))
    medium_thr = float(thresholds.get("medium", 0.55))

    debug = {
        "video_status": video.get("status"),
        "video_confidence": video_conf,
        "video_struck_string_id": video.get("struck_string_id"),
        "audio_status": audio.get("status"),
        "audio_confidence": audio_conf,
        "audio_struck_string_id": audio.get("struck_string_id"),
    }
    fused: dict[str, Any] = {
        "mode": mode,
        "status": "touch_only",
        "struck_string_id": None,
        "confidence": 0.0,
        "confidence_label": confidence_label(0.0, thresholds),
        "strategy": "fallback_video",
        "debug": debug,
    }

    if mode == "audio_only":
        if audio_strike:
            conf = audio_conf
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": audio.get("struck_string_id"),
                    "confidence": conf,
                    "confidence_label": confidence_label(conf, thresholds),
                    "strategy": "fallback_audio",
                }
            )
        else:
            fused.update({"status": "touch_only", "strategy": "fallback_audio"})
        return fused, debug

    if mode == "video_only":
        if video_strike:
            conf = video_conf
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": video.get("struck_string_id"),
                    "confidence": conf,
                    "confidence_label": confidence_label(conf, thresholds),
                    "strategy": "fallback_video",
                }
            )
        else:
            fused.update({"status": "touch_only", "strategy": "fallback_video"})
        return fused, debug

    # av_fuse
    if video_strike and audio_strike:
        if int(video.get("struck_string_id")) == int(audio.get("struck_string_id")):
            conf = max(video_conf, audio_conf)
            conf = min(1.0, conf + 0.10 + 0.10 * min(video_conf, audio_conf))
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": int(audio.get("struck_string_id")),
                    "confidence": conf,
                    "confidence_label": confidence_label(conf, thresholds),
                    "strategy": "agree",
                }
            )
            return fused, debug

        # Conflict: strongly trust high-confidence audio for pitch/string identity.
        if audio_conf >= prefer_audio_when_conf_ge:
            conf = max(audio_conf, min(0.95, 0.5 * audio_conf + 0.3 * video_conf))
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": int(audio.get("struck_string_id")),
                    "confidence": conf,
                    "confidence_label": confidence_label(conf, thresholds),
                    "strategy": "audio_preferred",
                }
            )
            return fused, debug

        if video_conf < medium_thr and audio_conf < medium_thr:
            fused.update(
                {
                    "status": "reject",
                    "strategy": "video_preferred" if video_conf >= audio_conf else "audio_preferred",
                    "debug": {**debug, "reason": "low_conflict_both_low"},
                }
            )
            return fused, debug

        if audio_conf >= video_conf:
            conf = max(audio_conf, 0.5 * audio_conf + 0.2 * video_conf)
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": int(audio.get("struck_string_id")),
                    "confidence": min(1.0, conf),
                    "confidence_label": confidence_label(min(1.0, conf), thresholds),
                    "strategy": "audio_preferred",
                }
            )
        else:
            conf = max(video_conf, 0.5 * video_conf + 0.2 * audio_conf)
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": int(video.get("struck_string_id")),
                    "confidence": min(1.0, conf),
                    "confidence_label": confidence_label(min(1.0, conf), thresholds),
                    "strategy": "video_preferred",
                }
            )
        return fused, debug

    if audio_strike and not video_strike:
        conf = audio_conf
        fused.update(
            {
                "status": "strike",
                "struck_string_id": int(audio.get("struck_string_id")),
                "confidence": conf,
                "confidence_label": confidence_label(conf, thresholds),
                "strategy": "fallback_audio",
            }
        )
        return fused, debug

    if video_strike and not audio_strike:
        if prefer_video_when_audio_missing or str(audio.get("status")) in {"missing", "no_audio", "no_onset"}:
            conf = video_conf
            fused.update(
                {
                    "status": "strike",
                    "struck_string_id": int(video.get("struck_string_id")),
                    "confidence": conf,
                    "confidence_label": confidence_label(conf, thresholds),
                    "strategy": "fallback_video",
                }
            )
        else:
            fused.update({"status": "touch_only", "strategy": "fallback_video"})
        return fused, debug

    fused.update({"status": "touch_only", "strategy": "fallback_video"})
    return fused, debug


def fuse_audio_video_decisions(
    *,
    touch_events: list[dict[str, Any]],
    fps: float,
    config: dict[str, Any],
    video_decision_payload: dict[str, Any] | None,
    audio_decision_payload: dict[str, Any] | None,
    source_video: str | None = None,
    touch_events_json_path: str | None = None,
) -> FusionArtifacts:
    fps_safe = max(float(fps), 1e-6)
    fusion_cfg = dict(config.get("fusion") or {})
    mode = str(fusion_cfg.get("mode", "av_fuse")).strip().lower()
    thresholds = dict(fusion_cfg.get("confidence_thresholds") or {})

    video_decision_events = []
    if isinstance(video_decision_payload, dict):
        if isinstance(video_decision_payload.get("right_decision_events"), list):
            video_decision_events = [ev for ev in video_decision_payload.get("right_decision_events", []) if isinstance(ev, dict)]
        elif isinstance(video_decision_payload.get("events"), list):
            video_decision_events = [ev for ev in video_decision_payload.get("events", []) if isinstance(ev, dict)]

    audio_decision_events = []
    if isinstance(audio_decision_payload, dict) and isinstance(audio_decision_payload.get("events"), list):
        audio_decision_events = [ev for ev in audio_decision_payload.get("events", []) if isinstance(ev, dict)]

    if audio_decision_events:
        canonical_events = []
        for a in audio_decision_events:
            touch = a.get("touch") if isinstance(a.get("touch"), dict) else {}
            if not isinstance(touch, dict):
                touch = {}
            touch = dict(touch)
            touch["event_id"] = str(a.get("event_id"))
            canonical_events.append(touch)
    else:
        canonical_events = build_canonical_right_events(
            touch_events=touch_events,
            fps=fps_safe,
            video_decision_events=video_decision_events if video_decision_events else None,
        )

    video_by_event_id = _align_video_to_canonical(canonical_events, video_decision_events)
    audio_by_event_id = _align_audio_to_canonical(canonical_events, audio_decision_events)

    fused_events: list[dict[str, Any]] = []
    fused_strikes: list[dict[str, Any]] = []
    for ev in canonical_events:
        event_id = str(ev.get("event_id"))
        video_raw = video_by_event_id.get(event_id)
        audio_raw = audio_by_event_id.get(event_id)
        video = _video_summary(video_raw, fps_safe, thresholds)
        audio = _audio_summary(audio_raw, thresholds)
        fusion, _ = _fuse_one(
            mode=mode,
            video=video,
            audio=audio,
            fusion_cfg=fusion_cfg,
            thresholds=thresholds,
        )
        fusion["timing"] = _select_timing(
            video=video,
            audio=audio,
            fps=fps_safe,
            timing_source=str(fusion_cfg.get("timing_source", "hybrid")),
        )
        touch_subset = {
            "timestamp_sec": _float_or_none(ev.get("timestamp_sec", ev.get("time_sec"))),
            "frame_index": _int_or_none(ev.get("frame_index")),
            "hand_side": str(ev.get("hand_side") or ev.get("hand") or "right"),
            "finger_type": str(ev.get("finger_type") or ""),
            "touched_string_id": _int_or_none(ev.get("touched_string_id", ev.get("string_id"))),
            "touch_conf": _float_or_none(ev.get("touch_conf")),
            "contact_x": _float_or_none(ev.get("contact_x")),
            "contact_y": _float_or_none(ev.get("contact_y")),
        }
        row = {
            "event_id": event_id,
            "touch": touch_subset,
            "video": video,
            "audio": audio,
            "fusion": fusion,
        }
        fused_events.append(row)
        if fusion.get("status") == "strike" and fusion.get("struck_string_id") is not None:
            fused_strikes.append(
                {
                    "event_id": event_id,
                    "timestamp_sec": row["fusion"]["timing"].get("onset_time_sec") or touch_subset.get("timestamp_sec"),
                    "frame_index": row["fusion"]["timing"].get("onset_frame") or touch_subset.get("frame_index"),
                    "finger_type": touch_subset.get("finger_type"),
                    "touched_string_id": touch_subset.get("touched_string_id"),
                    "struck_string_id": _int_or_none(fusion.get("struck_string_id")),
                    "peak_frame": row["fusion"]["timing"].get("peak_frame"),
                    "confidence": float(fusion.get("confidence") or 0.0),
                    "confidence_label": str(fusion.get("confidence_label") or confidence_label(0.0, thresholds)),
                    "strategy": str(fusion.get("strategy") or ""),
                }
            )

    fusion_status_counts = Counter(str(row.get("fusion", {}).get("status", "")) for row in fused_events)
    fusion_strategy_counts = Counter(str(row.get("fusion", {}).get("strategy", "")) for row in fused_events)
    fusion_conf_counts = Counter(str(row.get("fusion", {}).get("confidence_label", "")) for row in fused_events if row.get("fusion", {}).get("status") == "strike")
    meta = build_meta(
        source_video=source_video,
        touch_events_json_path=touch_events_json_path,
        fps=fps_safe,
        extra={
            "phase": "av_fusion",
            "mode": mode,
            "timing_source": str(fusion_cfg.get("timing_source", "hybrid")),
        },
    )
    decision_payload = {
        "meta": meta,
        "events": fused_events,
        "counts": {
            "events_total": int(len(fused_events)),
            "fusion_status_counts": dict(fusion_status_counts),
            "fusion_strategy_counts": dict(fusion_strategy_counts),
            "strike_events_count": int(len(fused_strikes)),
            "confidence_counts": dict(fusion_conf_counts),
        },
    }
    strike_payload = {
        "meta": meta,
        "events": fused_strikes,
        "counts": {
            "strike_events_count": int(len(fused_strikes)),
            "confidence_counts": dict(fusion_conf_counts),
            "strategy_counts": dict(Counter(str(ev.get("strategy", "")) for ev in fused_strikes)),
        },
    }
    return FusionArtifacts(decision_payload=decision_payload, strike_payload=strike_payload)

