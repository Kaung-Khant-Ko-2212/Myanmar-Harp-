from __future__ import annotations

from bisect import bisect_left
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.audio.decision import run_audio_decision_for_right_events
from src.audio.extract import AudioExtractionResult, extract_audio_from_video
from src.audio.load import load_audio_mono
from src.audio.tuning import load_tuning_table
from src.fusion.fuse import fuse_audio_video_decisions
from src.io.json_utils import derive_output_stem_from_touch_json, safe_read_json, update_touch_events_bundle_paths, write_json
from src.pipeline.config import load_pipeline_config
from src.video.annotate import overlay_av_decisions_on_video


def _load_touch_payload_and_events(touch_events_json_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = safe_read_json(Path(touch_events_json_path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid touch events JSON: {touch_events_json_path}")
    touch_events = payload.get("touch_events")
    if not isinstance(touch_events, list):
        raise ValueError(f"Missing touch_events list: {touch_events_json_path}")
    return payload, [ev for ev in touch_events if isinstance(ev, dict)]


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    payload = safe_read_json(p)
    return payload if isinstance(payload, dict) else None


def _default_related_output_paths(touch_events_json_path: str | Path) -> dict[str, Path]:
    out_dir, base_stem = derive_output_stem_from_touch_json(touch_events_json_path)
    return {
        "right_video_decision_events": out_dir / f"{base_stem}_right_decision_events.json",
        "right_video_strike_events": out_dir / f"{base_stem}_right_strike_events.json",
        "right_audio_decision_events": out_dir / f"{base_stem}_right_audio_decision_events.json",
        "right_audio_strike_events": out_dir / f"{base_stem}_right_audio_strike_events.json",
        "right_av_decision_events": out_dir / f"{base_stem}_right_av_decision_events.json",
        "right_av_strike_events": out_dir / f"{base_stem}_right_av_strike_events.json",
        "right_av_alternating_on_off_slots": out_dir / f"{base_stem}_alternating_on_off_slots.json",
        "extracted_audio_wav": out_dir / f"{base_stem}_audio.wav",
    }


def _resolve_fps(
    *,
    touch_payload: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    fps_override: float | None,
) -> float:
    if fps_override is not None:
        return max(float(fps_override), 1e-6)
    try:
        fps = float(touch_payload.get("fps"))
        if fps > 0:
            return fps
    except Exception:
        pass
    try:
        fps = float((pipeline_cfg.get("general") or {}).get("fps"))
        if fps > 0:
            return fps
    except Exception:
        pass
    return 30.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _compute_beat_times(
    *,
    audio: np.ndarray | None,
    sr: int | None,
    pipeline_cfg: dict[str, Any],
) -> dict[str, Any]:
    beat_cfg = dict(pipeline_cfg.get("beat_alignment") or {})
    enabled = bool(beat_cfg.get("enabled", True))
    out: dict[str, Any] = {
        "enabled": enabled,
        "ok": False,
        "reason": None,
        "tempo_bpm": None,
        "beat_times_sec": [],
        "tolerance_sec": float(beat_cfg.get("tolerance_sec", 0.08)),
        "tolerance_ratio": float(beat_cfg.get("tolerance_ratio", 0.15)),
    }
    if not enabled:
        out["reason"] = "disabled"
        return out
    if audio is None or sr is None:
        out["reason"] = "no_audio"
        return out
    if not isinstance(audio, np.ndarray) or audio.size <= 0:
        out["reason"] = "empty_audio"
        return out

    try:
        import librosa  # type: ignore
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"librosa_missing:{exc}"
        return out

    try:
        hop_length = int(beat_cfg.get("hop_length", 512))
        start_bpm = _float_or_none(beat_cfg.get("start_bpm"))
        kwargs: dict[str, Any] = {
            "y": np.asarray(audio, dtype=np.float32),
            "sr": int(sr),
            "hop_length": hop_length,
        }
        if start_bpm is not None and start_bpm > 0:
            kwargs["start_bpm"] = float(start_bpm)
        tempo_bpm, beat_frames = librosa.beat.beat_track(**kwargs)
        beat_frames_arr = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
        if beat_frames_arr.size == 0:
            out["reason"] = "no_beats_detected"
            return out
        beat_times = librosa.frames_to_time(beat_frames_arr, sr=int(sr), hop_length=hop_length)
        beat_times_list = [float(t) for t in np.asarray(beat_times, dtype=np.float64).reshape(-1) if np.isfinite(t)]
        if not beat_times_list:
            out["reason"] = "no_beat_times"
            return out
        tempo_val = _float_or_none(np.asarray(tempo_bpm).reshape(-1)[0] if np.asarray(tempo_bpm).size else tempo_bpm)
        out.update(
            {
                "ok": True,
                "reason": None,
                "tempo_bpm": tempo_val,
                "beat_times_sec": beat_times_list,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"beat_track_failed:{exc}"
        return out


def _annotate_av_payloads_with_beat_alignment(
    *,
    decision_payload: dict[str, Any],
    strike_payload: dict[str, Any],
    beat_info: dict[str, Any],
) -> None:
    for payload in (decision_payload, strike_payload):
        payload.setdefault("meta", {})
        payload["meta"]["beat_alignment"] = {
            "enabled": bool(beat_info.get("enabled", False)),
            "ok": bool(beat_info.get("ok", False)),
            "reason": beat_info.get("reason"),
            "tempo_bpm": beat_info.get("tempo_bpm"),
            "beat_count": int(len(beat_info.get("beat_times_sec") or [])),
            "tolerance_sec": float(beat_info.get("tolerance_sec") or 0.0),
            "tolerance_ratio": float(beat_info.get("tolerance_ratio") or 0.0),
        }

    if not bool(beat_info.get("ok", False)):
        return

    beat_times = [float(t) for t in (beat_info.get("beat_times_sec") or []) if _float_or_none(t) is not None]
    if not beat_times:
        return
    beat_times = sorted(beat_times)

    intervals = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
    valid_intervals = [iv for iv in intervals if iv > 1e-6]
    median_interval = float(np.median(np.asarray(valid_intervals, dtype=np.float64))) if valid_intervals else None
    base_tol = float(beat_info.get("tolerance_sec") or 0.08)
    tol_ratio = float(beat_info.get("tolerance_ratio") or 0.15)
    dynamic_tol = (
        max(base_tol, float(median_interval) * tol_ratio) if median_interval is not None and median_interval > 0 else base_tol
    )

    def nearest_beat(t_sec: float) -> tuple[int, float]:
        idx = bisect_left(beat_times, t_sec)
        candidates: list[int] = []
        if idx < len(beat_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx = min(candidates, key=lambda i: abs(beat_times[i] - t_sec))
        return best_idx, beat_times[best_idx]

    strike_events = strike_payload.get("events")
    if not isinstance(strike_events, list):
        strike_events = []
    beat_by_event_id: dict[str, dict[str, Any]] = {}
    on_count = 0
    off_count = 0
    unknown_count = 0

    for ev in strike_events:
        if not isinstance(ev, dict):
            continue
        t_sec = _float_or_none(ev.get("timestamp_sec"))
        event_id = str(ev.get("event_id") or "")
        if t_sec is None:
            ev["beat_label"] = "unknown"
            ev["beat_alignment"] = {"ok": False, "reason": "missing_timestamp"}
            unknown_count += 1
            continue
        idx, beat_t = nearest_beat(t_sec)
        delta = float(t_sec - beat_t)
        label = "on_beat" if abs(delta) <= dynamic_tol else "off_beat"
        if label == "on_beat":
            on_count += 1
        else:
            off_count += 1
        alignment = {
            "ok": True,
            "label": label,
            "nearest_beat_index": int(idx),
            "nearest_beat_time_sec": float(beat_t),
            "delta_sec": float(delta),
            "abs_delta_sec": float(abs(delta)),
            "tolerance_sec": float(dynamic_tol),
            "tempo_bpm": beat_info.get("tempo_bpm"),
        }
        ev["beat_label"] = label
        ev["beat_alignment"] = alignment
        if event_id:
            beat_by_event_id[event_id] = alignment

    strike_payload.setdefault("counts", {})
    strike_payload["counts"]["beat_label_counts"] = {
        "on_beat": int(on_count),
        "off_beat": int(off_count),
        "unknown": int(unknown_count),
    }

    decision_events = decision_payload.get("events")
    if not isinstance(decision_events, list):
        return
    for row in decision_events:
        if not isinstance(row, dict):
            continue
        fusion = row.get("fusion")
        if not isinstance(fusion, dict):
            continue
        if str(fusion.get("status") or "") != "strike":
            continue
        event_id = str(row.get("event_id") or "")
        alignment = beat_by_event_id.get(event_id)
        if alignment is None:
            t_sec = _float_or_none((fusion.get("timing") or {}).get("onset_time_sec") if isinstance(fusion.get("timing"), dict) else None)
            if t_sec is None:
                continue
            idx, beat_t = nearest_beat(t_sec)
            delta = float(t_sec - beat_t)
            label = "on_beat" if abs(delta) <= dynamic_tol else "off_beat"
            alignment = {
                "ok": True,
                "label": label,
                "nearest_beat_index": int(idx),
                "nearest_beat_time_sec": float(beat_t),
                "delta_sec": float(delta),
                "abs_delta_sec": float(abs(delta)),
                "tolerance_sec": float(dynamic_tol),
                "tempo_bpm": beat_info.get("tempo_bpm"),
            }
        fusion["beat_label"] = alignment.get("label")
        fusion["beat_alignment"] = alignment


def _build_alternating_on_off_slots_summary(
    *,
    strike_payload: dict[str, Any],
    touch_events: list[dict[str, Any]],
    pipeline_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    strike_events_raw = strike_payload.get("events")
    if not isinstance(strike_events_raw, list):
        return None
    strike_events = [ev for ev in strike_events_raw if isinstance(ev, dict)]
    if not strike_events:
        return None

    beat_time_by_idx: dict[int, float] = {}
    tol_values: list[float] = []
    for ev in strike_events:
        ba = ev.get("beat_alignment") if isinstance(ev.get("beat_alignment"), dict) else {}
        try:
            beat_idx = int(ba.get("nearest_beat_index"))
            beat_time = float(ba.get("nearest_beat_time_sec"))
        except Exception:
            continue
        if np.isfinite(beat_time):
            beat_time_by_idx.setdefault(beat_idx, float(beat_time))
        tol = _float_or_none(ba.get("tolerance_sec"))
        if tol is not None and tol > 0:
            tol_values.append(float(tol))
    if not beat_time_by_idx:
        return None

    known_idxs = sorted(beat_time_by_idx)
    known_times = [float(beat_time_by_idx[i]) for i in known_idxs]
    intervals = [known_times[i + 1] - known_times[i] for i in range(len(known_times) - 1) if (known_times[i + 1] - known_times[i]) > 1e-6]
    beat_interval = float(np.median(np.asarray(intervals, dtype=np.float64))) if intervals else 0.35
    median_tol = float(np.median(np.asarray(tol_values, dtype=np.float64))) if tol_values else 0.08
    left_window = max(median_tol, min(0.20, beat_interval * 0.45))

    min_idx = int(min(known_idxs))
    max_idx = int(max(known_idxs))

    def estimate_beat_time(idx: int) -> float:
        if idx in beat_time_by_idx:
            return float(beat_time_by_idx[idx])
        pos = bisect_left(known_idxs, idx)
        if 0 < pos < len(known_idxs):
            left_idx = int(known_idxs[pos - 1])
            right_idx = int(known_idxs[pos])
            left_t = float(beat_time_by_idx[left_idx])
            right_t = float(beat_time_by_idx[right_idx])
            if right_idx != left_idx:
                frac = (idx - left_idx) / float(right_idx - left_idx)
                return float(left_t + frac * (right_t - left_t))
        anchor_idx = int(known_idxs[0] if pos <= 0 else known_idxs[-1])
        return float(beat_time_by_idx[anchor_idx] + (idx - anchor_idx) * beat_interval)

    # Aggregate right thumb/index fused strikes by beat slot.
    right_events_by_beat: dict[int, list[dict[str, Any]]] = {}
    for ev in strike_events:
        finger = str(ev.get("finger_type") or "").strip().lower()
        if finger not in {"thumb", "index"}:
            continue
        ba = ev.get("beat_alignment") if isinstance(ev.get("beat_alignment"), dict) else {}
        try:
            beat_idx = int(ba.get("nearest_beat_index"))
            struck_sid = int(ev.get("struck_string_id"))
        except Exception:
            continue
        conf = _float_or_none(ev.get("confidence"))
        row = {
            "string_id": int(struck_sid),
            "confidence": float(conf if conf is not None else 0.0),
            "finger_type": finger,
            "timestamp_sec": _float_or_none(ev.get("timestamp_sec")),
            "event_id": ev.get("event_id"),
        }
        right_events_by_beat.setdefault(int(beat_idx), []).append(row)

    # Infer left hand involvement from touch events near each beat slot time.
    left_touch_times: list[float] = []
    for ev in touch_events:
        hand = str(ev.get("hand_side") or ev.get("hand") or "").strip().lower()
        if hand != "left":
            continue
        t = _float_or_none(ev.get("timestamp_sec", ev.get("time_sec")))
        if t is not None:
            left_touch_times.append(float(t))
    left_touch_times.sort()

    grid_pairs = [(int(i), float(estimate_beat_time(int(i)))) for i in range(min_idx, max_idx + 1)]
    grid_idxs = [i for i, _ in grid_pairs]
    grid_times = [t for _, t in grid_pairs]
    left_touch_count_by_beat: dict[int, int] = {}
    for t in left_touch_times:
        pos = bisect_left(grid_times, t)
        candidate_pos: list[int] = []
        if pos < len(grid_times):
            candidate_pos.append(pos)
        if pos > 0:
            candidate_pos.append(pos - 1)
        if not candidate_pos:
            continue
        best_pos = min(candidate_pos, key=lambda j: abs(grid_times[j] - t))
        if abs(grid_times[best_pos] - t) <= left_window:
            beat_idx = int(grid_idxs[best_pos])
            left_touch_count_by_beat[beat_idx] = int(left_touch_count_by_beat.get(beat_idx, 0) + 1)

    def top_strings(items: list[dict[str, Any]], topn: int = 2) -> list[dict[str, Any]]:
        agg: dict[int, dict[str, Any]] = {}
        for it in items:
            sid = int(it.get("string_id"))
            conf = float(_float_or_none(it.get("confidence")) or 0.0)
            if sid not in agg:
                agg[sid] = {
                    "string_id": sid,
                    "count": 0,
                    "confidence_sum": 0.0,
                    "max_confidence": 0.0,
                }
            agg[sid]["count"] = int(agg[sid]["count"] + 1)
            agg[sid]["confidence_sum"] = float(agg[sid]["confidence_sum"] + conf)
            agg[sid]["max_confidence"] = float(max(float(agg[sid]["max_confidence"]), conf))
        ranked = sorted(
            agg.values(),
            key=lambda row: (
                float(row["confidence_sum"]),
                int(row["count"]),
                float(row["max_confidence"]),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(topn))]

    beat_cfg = dict(pipeline_cfg.get("beat_alignment") or {})
    slot_start = str(beat_cfg.get("alternating_slot_start", "on_beat")).strip().lower()
    start_as_on = slot_start != "off_beat"

    sequence: list[dict[str, Any]] = []
    for offset, beat_idx in enumerate(range(min_idx, max_idx + 1)):
        slot_name = "on_beat" if ((offset % 2 == 0) == start_as_on) else "off_beat"
        beat_time_sec = float(estimate_beat_time(int(beat_idx)))
        top = top_strings(right_events_by_beat.get(int(beat_idx), []), topn=2)
        left_count = int(left_touch_count_by_beat.get(int(beat_idx), 0))
        slot_payload: dict[str, Any] = {
            "beat_index": int(beat_idx),
            "beat_time_sec": round(float(beat_time_sec), 3),
            "strings": [int(row["string_id"]) for row in top],
            "left_hand_involved": bool(left_count > 0),
        }
        if left_count > 0:
            slot_payload["left_hand_note"] = "left hand is involved"
            slot_payload["left_hand_touch_count_near_slot"] = int(left_count)
        if top:
            slot_payload["string_candidates"] = top
        sequence.append({slot_name: slot_payload})

    strike_meta = strike_payload.get("meta") if isinstance(strike_payload.get("meta"), dict) else {}
    return {
        "source_right_av_strike_events_json": None,  # populated by caller after path is known
        "source_touch_events_json": strike_meta.get("touch_events_json_path"),
        "format": "alternating_on_off_sequence",
        "notes": [
            "Alternating on/off slots are generated from beat index order (not per-event beat_label threshold).",
            "strings are top 1-2 fused AV string candidates from right thumb/index events nearest each beat slot.",
            "left_hand_involved is inferred from left-hand touch events near each beat slot time.",
        ],
        "slot_start": "on_beat" if start_as_on else "off_beat",
        "slot_time_step_estimate_sec": round(float(beat_interval), 6),
        "left_hand_assignment_window_sec": round(float(left_window), 6),
        "sequence_length": int(len(sequence)),
        "sequence": sequence,
    }


def _maybe_extract_or_load_audio(
    *,
    video_path: str | Path | None,
    audio_input_path: str | Path | None,
    output_wav_path: Path,
    pipeline_cfg: dict[str, Any],
    audio_enabled: bool,
) -> dict[str, Any]:
    audio_cfg = dict(pipeline_cfg.get("audio") or {})
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    out: dict[str, Any] = {
        "audio": None,
        "sr": None,
        "audio_path": None,
        "extract_result": None,
        "load_result": None,
        "error": None,
    }
    if not audio_enabled:
        out["error"] = "audio_disabled"
        return out

    source_audio_path: Path | None = Path(audio_input_path) if audio_input_path else None
    extract_result: AudioExtractionResult | None = None
    if source_audio_path is None:
        if not bool(audio_cfg.get("extract_audio", True)):
            out["error"] = "audio_extract_disabled"
            return out
        if video_path is None:
            out["error"] = "missing_video_for_audio_extract"
            return out
        extract_result = extract_audio_from_video(
            video_path=Path(video_path),
            out_wav_path=output_wav_path,
            sample_rate=sample_rate,
        )
        out["extract_result"] = {
            "ok": extract_result.ok,
            "wav_path": str(extract_result.wav_path) if extract_result.wav_path else None,
            "sample_rate": int(extract_result.sample_rate),
            "backend": extract_result.backend,
            "error": extract_result.error,
        }
        if not extract_result.ok or extract_result.wav_path is None:
            out["error"] = extract_result.error or "audio_extract_failed"
            return out
        source_audio_path = extract_result.wav_path

    load_result = load_audio_mono(source_audio_path, sample_rate=sample_rate)
    out["load_result"] = {
        "ok": load_result.ok,
        "sr": load_result.sr,
        "backend": load_result.backend,
        "error": load_result.error,
        "debug": load_result.debug,
    }
    if not load_result.ok:
        out["error"] = load_result.error or "audio_load_failed"
        return out

    out["audio"] = np.asarray(load_result.audio, dtype=np.float32) if load_result.audio is not None else None
    out["sr"] = int(load_result.sr) if load_result.sr is not None else None
    out["audio_path"] = str(source_audio_path)
    return out


def run_audio_fusion_postprocess(
    *,
    video_path: str | Path | None,
    touch_events_json_path: str | Path,
    right_video_decision_events_json_path: str | Path | None = None,
    annotated_video_path: str | Path | None = None,
    string_geometries: list[dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
    audio_input_path: str | Path | None = None,
    audio_enabled: bool | None = None,
    fusion_mode: str | None = None,
    audio_decision_mode: str | None = None,
    enable_overlay: bool = True,
    fps_override: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    touch_payload, touch_events = _load_touch_payload_and_events(touch_events_json_path)
    pipeline_cfg = load_pipeline_config(config_path)

    if audio_enabled is not None:
        pipeline_cfg.setdefault("audio", {})
        pipeline_cfg["audio"]["enabled"] = bool(audio_enabled)
    if fusion_mode is not None:
        pipeline_cfg.setdefault("fusion", {})
        pipeline_cfg["fusion"]["mode"] = str(fusion_mode)
    if audio_decision_mode is not None:
        pipeline_cfg.setdefault("audio", {})
        pipeline_cfg["audio"]["decision_mode"] = str(audio_decision_mode)
        if str(audio_decision_mode).strip().lower() in {"onset_pitch_match", "pitch_match", "pitch", "pitch_tuning"}:
            pipeline_cfg["audio"].setdefault("pitch_fallback_to_onset", True)

    fps = _resolve_fps(touch_payload=touch_payload, pipeline_cfg=pipeline_cfg, fps_override=fps_override)
    paths = _default_related_output_paths(touch_events_json_path)
    if right_video_decision_events_json_path is None and paths["right_video_decision_events"].exists():
        right_video_decision_events_json_path = paths["right_video_decision_events"]

    video_decision_payload = _load_optional_json(right_video_decision_events_json_path)
    video_decision_events: list[dict[str, Any]] | None = None
    if isinstance(video_decision_payload, dict):
        raw = video_decision_payload.get("right_decision_events")
        if isinstance(raw, list):
            video_decision_events = [ev for ev in raw if isinstance(ev, dict)]

    audio_cfg = dict(pipeline_cfg.get("audio") or {})
    audio_is_enabled = bool(audio_cfg.get("enabled", True))
    t_audio_io = time.perf_counter()
    audio_io = _maybe_extract_or_load_audio(
        video_path=video_path,
        audio_input_path=audio_input_path,
        output_wav_path=paths["extracted_audio_wav"],
        pipeline_cfg=pipeline_cfg,
        audio_enabled=audio_is_enabled,
    )
    timings["audio_io_sec"] = round(float(time.perf_counter() - t_audio_io), 3)

    tuning_by_string = None
    tuning_info: dict[str, Any] | None = None
    decision_mode = str(audio_cfg.get("decision_mode", "onset_only")).strip().lower()
    if decision_mode in {"onset_pitch_match", "pitch_match", "pitch", "pitch_tuning"}:
        tuning_path = Path(str(audio_cfg.get("tuning_table_path", "configs/saung_tuning.json")))
        try:
            tuning_by_string = load_tuning_table(tuning_path)
            tuning_info = {"path": str(tuning_path), "loaded": True, "string_count": len(tuning_by_string)}
        except Exception as exc:  # noqa: BLE001
            tuning_info = {"path": str(tuning_path), "loaded": False, "error": str(exc)}

    t_audio_decision = time.perf_counter()
    audio_artifacts = run_audio_decision_for_right_events(
        touch_events=touch_events,
        fps=fps,
        audio=audio_io.get("audio"),
        sr=audio_io.get("sr"),
        config=pipeline_cfg,
        source_video=str(video_path) if video_path is not None else None,
        touch_events_json_path=str(touch_events_json_path),
        video_decision_events=video_decision_events,
        tuning_by_string=tuning_by_string,
    )
    timings["audio_decision_sec"] = round(float(time.perf_counter() - t_audio_decision), 3)
    # Enrich audio meta/debug with extraction/load status.
    for payload in (audio_artifacts.decision_payload, audio_artifacts.strike_payload):
        payload.setdefault("meta", {})
        payload["meta"]["audio_source_path"] = audio_io.get("audio_path")
        payload["meta"]["audio_extract"] = audio_io.get("extract_result")
        payload["meta"]["audio_load"] = audio_io.get("load_result")
        payload["meta"]["tuning"] = tuning_info
        if audio_io.get("error"):
            payload["meta"]["audio_error"] = audio_io.get("error")

    t_write_audio = time.perf_counter()
    write_json(paths["right_audio_decision_events"], audio_artifacts.decision_payload)
    write_json(paths["right_audio_strike_events"], audio_artifacts.strike_payload)
    timings["write_audio_json_sec"] = round(float(time.perf_counter() - t_write_audio), 3)

    t_fusion = time.perf_counter()
    fusion_artifacts = fuse_audio_video_decisions(
        touch_events=touch_events,
        fps=fps,
        config=pipeline_cfg,
        video_decision_payload=video_decision_payload,
        audio_decision_payload=audio_artifacts.decision_payload,
        source_video=str(video_path) if video_path is not None else None,
        touch_events_json_path=str(touch_events_json_path),
    )
    timings["fusion_sec"] = round(float(time.perf_counter() - t_fusion), 3)

    t_beat_align = time.perf_counter()
    beat_info = _compute_beat_times(
        audio=audio_io.get("audio"),
        sr=audio_io.get("sr"),
        pipeline_cfg=pipeline_cfg,
    )
    _annotate_av_payloads_with_beat_alignment(
        decision_payload=fusion_artifacts.decision_payload,
        strike_payload=fusion_artifacts.strike_payload,
        beat_info=beat_info,
    )
    timings["beat_alignment_sec"] = round(float(time.perf_counter() - t_beat_align), 3)

    alternating_slots_payload = _build_alternating_on_off_slots_summary(
        strike_payload=fusion_artifacts.strike_payload,
        touch_events=touch_events,
        pipeline_cfg=pipeline_cfg,
    )
    if isinstance(alternating_slots_payload, dict):
        alternating_slots_payload["source_right_av_strike_events_json"] = str(paths["right_av_strike_events"])

    t_write_fusion = time.perf_counter()
    write_json(paths["right_av_decision_events"], fusion_artifacts.decision_payload)
    write_json(paths["right_av_strike_events"], fusion_artifacts.strike_payload)
    if isinstance(alternating_slots_payload, dict):
        write_json(paths["right_av_alternating_on_off_slots"], alternating_slots_payload)
    timings["write_fusion_json_sec"] = round(float(time.perf_counter() - t_write_fusion), 3)

    overlay_info: dict[str, Any] | None = None
    final_annotated_video_path = str(annotated_video_path) if annotated_video_path else None
    if enable_overlay and annotated_video_path and Path(str(annotated_video_path)).exists():
        t_overlay = time.perf_counter()
        out_overlay = Path(str(annotated_video_path)).with_name(f"{Path(str(annotated_video_path)).stem}_av{Path(str(annotated_video_path)).suffix}")
        overlay_string_geometries = string_geometries
        if not isinstance(overlay_string_geometries, list):
            payload_geometries = touch_payload.get("string_geometries")
            if isinstance(payload_geometries, list):
                overlay_string_geometries = payload_geometries
        overlay_info = overlay_av_decisions_on_video(
            input_video_path=annotated_video_path,
            output_video_path=out_overlay,
            fusion_decision_payload=fusion_artifacts.decision_payload,
            fusion_strike_payload=fusion_artifacts.strike_payload,
            string_geometries=overlay_string_geometries,
            source_with_audio=video_path,
            hold_frames=max(8, int(round(fps * 0.20))),
        )
        timings["overlay_total_sec"] = round(float(time.perf_counter() - t_overlay), 3)
        if overlay_info.get("ok"):
            final_annotated_video_path = str(overlay_info.get("output_video_path"))

    t_bundle_update = time.perf_counter()
    update_touch_events_bundle_paths(
        touch_events_json_path,
        path_updates={
            "right_audio_decision_events_json_path": str(paths["right_audio_decision_events"]),
            "right_audio_strike_events_json_path": str(paths["right_audio_strike_events"]),
            "right_av_decision_events_json_path": str(paths["right_av_decision_events"]),
            "right_av_strike_events_json_path": str(paths["right_av_strike_events"]),
            "right_av_alternating_on_off_slots_json_path": (
                str(paths["right_av_alternating_on_off_slots"]) if isinstance(alternating_slots_payload, dict) else None
            ),
        },
        count_updates={
            "right_audio_decision_events_count": int(audio_artifacts.decision_payload.get("counts", {}).get("events_total", 0)),
            "right_audio_strike_events_count": int(audio_artifacts.strike_payload.get("counts", {}).get("strike_events_count", 0)),
            "right_av_decision_events_count": int(fusion_artifacts.decision_payload.get("counts", {}).get("events_total", 0)),
            "right_av_strike_events_count": int(fusion_artifacts.strike_payload.get("counts", {}).get("strike_events_count", 0)),
        },
    )
    timings["bundle_update_sec"] = round(float(time.perf_counter() - t_bundle_update), 3)

    elapsed = round(float(time.perf_counter() - started), 3)
    return {
        "fps": float(fps),
        "pipeline_config_path": str(config_path) if config_path is not None else None,
        "audio_enabled": bool(audio_is_enabled),
        "audio_decision_mode": str((pipeline_cfg.get("audio") or {}).get("decision_mode", "onset_only")),
        "fusion_mode": str((pipeline_cfg.get("fusion") or {}).get("mode", "av_fuse")),
        "audio_error": audio_io.get("error"),
        "audio_source_path": audio_io.get("audio_path"),
        "audio_extract": audio_io.get("extract_result"),
        "audio_load": audio_io.get("load_result"),
        "right_audio_decision_events_json_path": str(paths["right_audio_decision_events"]),
        "right_audio_strike_events_json_path": str(paths["right_audio_strike_events"]),
        "right_av_decision_events_json_path": str(paths["right_av_decision_events"]),
        "right_av_strike_events_json_path": str(paths["right_av_strike_events"]),
        "right_av_alternating_on_off_slots_json_path": (
            str(paths["right_av_alternating_on_off_slots"]) if isinstance(alternating_slots_payload, dict) else None
        ),
        "right_audio_decision_events_count": int(audio_artifacts.decision_payload.get("counts", {}).get("events_total", 0)),
        "right_audio_strike_events_count": int(audio_artifacts.strike_payload.get("counts", {}).get("strike_events_count", 0)),
        "right_av_decision_events_count": int(fusion_artifacts.decision_payload.get("counts", {}).get("events_total", 0)),
        "right_av_strike_events_count": int(fusion_artifacts.strike_payload.get("counts", {}).get("strike_events_count", 0)),
        "right_av_alternating_on_off_slots_count": (
            int(alternating_slots_payload.get("sequence_length", 0)) if isinstance(alternating_slots_payload, dict) else 0
        ),
        "beat_alignment": {
            "enabled": bool(beat_info.get("enabled", False)),
            "ok": bool(beat_info.get("ok", False)),
            "reason": beat_info.get("reason"),
            "tempo_bpm": beat_info.get("tempo_bpm"),
            "beat_count": int(len(beat_info.get("beat_times_sec") or [])),
        },
        "annotated_video_path": final_annotated_video_path,
        "overlay_info": overlay_info,
        "timings": timings,
        "elapsed_sec": elapsed,
    }


def run_pipeline(
    *,
    video_path: str | Path,
    model_path: str | Path | None = None,
    expected_strings: int = 16,
    tag: str = "best",
    config_path: str | Path | None = None,
    fusion_mode: str | None = None,
    audio_enabled: bool | None = None,
    audio_input_path: str | Path | None = None,
    enable_hand_tracking: bool = True,
    draw_hand_labels: bool = False,
    hand_pipeline_enabled: bool | None = None,
    run_video_stage: bool = True,
    run_vibration_stage: bool = False,
    vibration_stage_runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    End-to-end wrapper with thin integrations:
    1) video detection + touch generation (existing backend)
    2) optional vibration stage wrapper (callback or precomputed JSON fallback)
    3) audio-first strike
    4) AV fusion
    5) post-overlay annotated video

    `run_vibration_stage` is a wrapper hook because the current vibration logic already
    lives in `backend/app.py`; callers can pass a callback or rely on precomputed JSONs.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    result: dict[str, Any] = {}
    if run_video_stage:
        try:
            try:
                from backend.post_processing import run_video_predict  # type: ignore
            except Exception:
                from post_processing import run_video_predict  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Cannot import run_video_predict: {exc}") from exc

        if model_path is None:
            raise ValueError("model_path is required when run_video_stage=True")

        result = run_video_predict(
            tag=tag,
            model_path=Path(model_path),
            video_path=video_path,
            expected_strings=int(expected_strings),
            save_video=True,
            show_preview=False,
            transcode_output=False,
            transcode_preset="veryfast",
            enable_hand_tracking=bool(enable_hand_tracking),
            draw_hand_labels=bool(draw_hand_labels),
            hand_pipeline_enabled=hand_pipeline_enabled,
        )
    else:
        # TODO: Support fully precomputed inputs in a single manifest if needed.
        result = {
            "source_video": str(video_path),
            "touch_events_json_path": None,
            "out_video_path": None,
        }

    touch_events_json_path = result.get("touch_events_json_path")
    if not touch_events_json_path:
        raise RuntimeError("Missing touch_events_json_path after video stage.")

    if run_vibration_stage and vibration_stage_runner is not None:
        # TODO: The existing vibration decision pipeline is implemented inline in backend/app.py.
        # This callback lets callers reuse that logic without duplicating it here.
        result.update(vibration_stage_runner(result, load_pipeline_config(config_path)))

    av_outputs = run_audio_fusion_postprocess(
        video_path=video_path,
        touch_events_json_path=touch_events_json_path,
        right_video_decision_events_json_path=result.get("right_decision_events_json_path"),
        annotated_video_path=result.get("out_video_path"),
        string_geometries=result.get("string_geometries"),
        config_path=config_path,
        audio_input_path=audio_input_path,
        audio_enabled=audio_enabled,
        fusion_mode=fusion_mode,
        fps_override=_float_or_none(result.get("fps")),
    )

    result["av_pipeline"] = av_outputs
    if av_outputs.get("annotated_video_path"):
        result["out_video_path_av"] = av_outputs["annotated_video_path"]
    return result
