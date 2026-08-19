#uvicorn app:app --host 0.0.0.0 --port 8000
from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import importlib
import json
import sys
from threading import Lock, Thread
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
PREDICTIONS_DIR = PROJECT_DIR / "predict_postprocessed"
CACHE_DIR = PROJECT_DIR / "cache"
DEBUG_REPORTS_DIR = PROJECT_DIR / "debug_reports"
BEST_MODEL_PATH = REPO_DIR / "harp_pose_v11m_prepped" / "weights" / "best.pt"
STRIKE_CONFIG_PATH = REPO_DIR / "saung_strike_video_farneback_rules" / "configs" / "config.yaml"
PIPELINE_CONFIG_PATH = REPO_DIR / "configs" / "config.yaml"
CACHE_SCHEMA_VERSION = 1
CACHE_HIT_DELAY_SEC = 0.0
ACCURATE_MODE_STRING_INFER_EVERY_N = 1
ACCURATE_MODE_HAND_PROCESS_WIDTH = 0
ACCURATE_MODE_HAND_MODEL_COMPLEXITY = 1
ACCURATE_MODE_MAX_STRIKE_EVENTS = 500
ACCURATE_MODE_STRIKE_MIN_EVENT_GAP_FRAMES = 6
ACCURATE_MODE_SHAKE_PROBE_COUNT = 0
ACCURATE_MODE_MIN_TOUCH_CONFIDENCE = 0.20
CACHE_OUTPUT_PATH_KEYS = (
    "annotated_video_path",
    "touch_events_json_path",
    "left_touch_events_json_path",
    "right_decision_events_json_path",
    "right_strike_events_json_path",
    "right_audio_decision_events_json_path",
    "right_audio_strike_events_json_path",
    "right_av_decision_events_json_path",
    "right_av_strike_events_json_path",
    "right_av_alternating_on_off_slots_json_path",
)
CACHE_SIGNATURE_PATHS = (
    PROJECT_DIR / "app.py",
    PROJECT_DIR / "post_processing.py",
    REPO_DIR / "src" / "pipeline" / "run.py",
    REPO_DIR / "src" / "audio" / "decision.py",
    REPO_DIR / "src" / "fusion" / "fuse.py",
    REPO_DIR / "saung_strike_video_farneback_rules" / "src" / "decision.py",
    STRIKE_CONFIG_PATH,
    PIPELINE_CONFIG_PATH,
)

# Ensure project-root imports work even when launching from `backend/`.
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Myanmar Harp Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/predictions", StaticFiles(directory=str(PREDICTIONS_DIR)), name="predictions")
app.mount("/analysis-debug", StaticFiles(directory=str(DEBUG_REPORTS_DIR)), name="analysis-debug")

PREDICTION_JOBS: dict[str, dict[str, Any]] = {}
PREDICTION_JOBS_LOCK = Lock()


@app.on_event("startup")
def log_strike_module_import_status() -> None:
    _, err = _import_strike_symbol("saung_strike_video_farneback_rules.src.decision", "decide_touch_events")
    if err is None:
        print("[STARTUP] Strike module import OK: saung_strike_video_farneback_rules.src.decision")
    else:
        print(f"[STARTUP][WARN] Strike module import failed: saung_strike_video_farneback_rules.src.decision -> {err}")


def _import_strike_symbol(module_name: str, symbol_name: str) -> tuple[Any | None, str | None]:
    # Keep REPO_DIR ahead of the legacy strike-package path because both expose a top-level `src`
    # package. Our audio/fusion pipeline lives in REPO_DIR/src, while the strike module has
    # saung_strike_video_farneback_rules/src. Wrong ordering can shadow `src.pipeline`.
    search_paths = [
        str(REPO_DIR / "saung_strike_video_farneback_rules"),
        str(REPO_DIR),
    ]
    last_exc: Exception | None = None

    for p in search_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    for _ in range(2):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol_name), None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    strike_pkg_dir = REPO_DIR / "saung_strike_video_farneback_rules"
    details = (
        f"{last_exc}; repo_dir_exists={REPO_DIR.exists()}; "
        f"strike_pkg_dir_exists={strike_pkg_dir.exists()}; "
        f"cwd={Path.cwd()}"
    )
    return None, details


def _ensure_repo_src_precedence() -> None:
    """Ensure repo-root `src/` package wins over legacy strike `src/` package."""
    repo_path = str(REPO_DIR)
    strike_path = str(REPO_DIR / "saung_strike_video_farneback_rules")
    try:
        while repo_path in sys.path:
            sys.path.remove(repo_path)
    except Exception:
        pass
    sys.path.insert(0, repo_path)
    # Keep strike path available, but behind repo-root.
    if strike_path in sys.path:
        try:
            sys.path.remove(strike_path)
        except Exception:
            pass
        sys.path.insert(1, strike_path)


def _purge_shadowed_src_modules() -> None:
    """Remove a previously-imported legacy `src` package that shadows repo `src.pipeline`."""
    strike_root = str((REPO_DIR / "saung_strike_video_farneback_rules").resolve()).lower()
    src_mod = sys.modules.get("src")
    if src_mod is None:
        return

    candidate_paths: list[str] = []
    mod_file = getattr(src_mod, "__file__", None)
    if isinstance(mod_file, str):
        candidate_paths.append(mod_file)
    mod_path = getattr(src_mod, "__path__", None)
    if mod_path is not None:
        try:
            candidate_paths.extend([str(p) for p in list(mod_path)])
        except Exception:
            pass

    is_shadowed = any(strike_root in str(p).lower() for p in candidate_paths)
    if not is_shadowed:
        return

    doomed = [name for name in list(sys.modules.keys()) if name == "src" or name.startswith("src.")]
    for name in doomed:
        sys.modules.pop(name, None)
    print(
        "[INFO] Purged shadowed legacy `src` modules before audio/AV import: "
        f"count={len(doomed)} src_paths={candidate_paths}"
    )


def _load_yaml_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _tip_to_finger_type(tip_name: str) -> str:
    tip = tip_name.strip().lower()
    if tip.startswith("thumb"):
        return "thumb"
    if tip.startswith("index"):
        return "index"
    return ""


def _build_string_geometries_for_decision(raw: Any) -> list[Any]:
    StringGeometry, err = _import_strike_symbol(
        "saung_strike_video_farneback_rules.src.strings",
        "StringGeometry",
    )
    if StringGeometry is None:
        if err:
            print(f"[WARN] Could not import StringGeometry: {err}")
        return []

    if not isinstance(raw, list):
        return []

    out: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid_raw = item.get("string_id")
        endpoints = item.get("endpoints")
        if sid_raw is None or not isinstance(endpoints, list) or len(endpoints) < 2:
            continue
        p1 = endpoints[0]
        p2 = endpoints[1]
        if not isinstance(p1, (list, tuple)) or not isinstance(p2, (list, tuple)) or len(p1) < 2 or len(p2) < 2:
            continue
        try:
            sid = int(round(float(sid_raw)))
            geom = StringGeometry(
                string_id=sid,
                mode="endpoints",
                points=(
                    (float(p1[0]), float(p1[1])),
                    (float(p2[0]), float(p2[1])),
                ),
            )
            out.append(geom)
        except Exception:
            continue
    return out


def _build_touch_events_for_decision(
    raw_touch_events: Any,
    *,
    fps: float,
    max_events: int,
    min_gap_frames: int,
    allowed_finger_types: set[str] | None = None,
    min_touch_confidence: float = 0.0,
) -> tuple[list[Any], dict[str, Any]]:
    TouchEvent, err = _import_strike_symbol(
        "saung_strike_video_farneback_rules.src.windows",
        "TouchEvent",
    )
    if TouchEvent is None:
        return [], {"error": f"decision modules unavailable: {err}"}

    if not isinstance(raw_touch_events, list):
        return [], {"error": "invalid touch events payload"}

    fps_safe = max(float(fps), 1e-6)
    allowed_fingers = {
        str(f).strip().lower()
        for f in (allowed_finger_types or {"thumb", "index"})
        if str(f).strip()
    }
    if not allowed_fingers:
        allowed_fingers = {"thumb", "index"}
    candidates: list[tuple[int, Any]] = []
    total_rows = 0
    dropped_by_touch_conf = 0
    min_touch_conf = max(0.0, float(min_touch_confidence))

    for i, item in enumerate(raw_touch_events, start=2):
        if not isinstance(item, dict):
            continue
        total_rows += 1
        hand_side = str(item.get("hand_side") or item.get("hand") or "").strip().lower()
        tip_name = str(item.get("fingertip") or "").strip().lower()
        finger_type = str(item.get("finger_type") or _tip_to_finger_type(tip_name)).strip().lower()
        if hand_side != "right" or finger_type not in allowed_fingers:
            continue

        touched_id_raw = item.get("touched_string_id", item.get("string_id"))
        if touched_id_raw is None:
            continue
        try:
            touched_id = int(round(float(touched_id_raw)))
        except Exception:
            continue

        ts_raw = item.get("timestamp_sec", item.get("time_sec"))
        frame_idx_raw = item.get("frame_index")
        if ts_raw is None and frame_idx_raw is None:
            continue
        try:
            timestamp_sec = float(ts_raw) if ts_raw is not None else float(frame_idx_raw) / fps_safe
        except Exception:
            continue
        try:
            frame_idx = int(frame_idx_raw) if frame_idx_raw is not None else int(round(timestamp_sec * fps_safe))
        except Exception:
            frame_idx = int(round(timestamp_sec * fps_safe))

        try:
            contact_x = float(item.get("contact_x", item.get("finger_x", 0.0)))
            contact_y = float(item.get("contact_y", item.get("finger_y", 0.0)))
        except Exception:
            contact_x = 0.0
            contact_y = 0.0

        finger_x = item.get("finger_x")
        finger_y = item.get("finger_y")
        try:
            finger_x_val = float(finger_x) if finger_x is not None else None
            finger_y_val = float(finger_y) if finger_y is not None else None
        except Exception:
            finger_x_val = None
            finger_y_val = None

        try:
            hand_bbox_x1 = float(item.get("hand_bbox_x1")) if item.get("hand_bbox_x1") is not None else None
            hand_bbox_y1 = float(item.get("hand_bbox_y1")) if item.get("hand_bbox_y1") is not None else None
            hand_bbox_x2 = float(item.get("hand_bbox_x2")) if item.get("hand_bbox_x2") is not None else None
            hand_bbox_y2 = float(item.get("hand_bbox_y2")) if item.get("hand_bbox_y2") is not None else None
        except Exception:
            hand_bbox_x1 = None
            hand_bbox_y1 = None
            hand_bbox_x2 = None
            hand_bbox_y2 = None

        try:
            touch_conf = float(item.get("touch_conf", 1.0))
        except Exception:
            touch_conf = 1.0
        if touch_conf < min_touch_conf:
            dropped_by_touch_conf += 1
            continue
        try:
            distance_px = float(item.get("distance_px")) if item.get("distance_px") is not None else None
        except Exception:
            distance_px = None

        event = TouchEvent(
            timestamp_sec=float(timestamp_sec),
            hand_side=hand_side,
            finger_type=finger_type,
            touched_string_id=int(touched_id),
            touch_conf=float(touch_conf),
            contact_x=float(contact_x),
            contact_y=float(contact_y),
            finger_x=finger_x_val,
            finger_y=finger_y_val,
            row_index=i,
            distance_px=distance_px,
            hand_bbox_x1=hand_bbox_x1,
            hand_bbox_y1=hand_bbox_y1,
            hand_bbox_x2=hand_bbox_x2,
            hand_bbox_y2=hand_bbox_y2,
        )
        candidates.append((frame_idx, event))

    candidates.sort(key=lambda pair: pair[0])

    accepted: list[Any] = []
    last_frame_by_key: dict[tuple[str, int], int] = {}
    last_frame_by_finger: dict[str, int] = {}
    min_gap = max(1, int(min_gap_frames))
    dropped_by_gap = 0
    dropped_by_finger_gap = 0
    for frame_idx, event in candidates:
        finger_key = str(event.finger_type).strip().lower()
        prev_finger = last_frame_by_finger.get(finger_key)
        if prev_finger is not None and (frame_idx - prev_finger) < min_gap:
            dropped_by_finger_gap += 1
            continue

        key = (finger_key, int(event.touched_string_id))
        prev = last_frame_by_key.get(key)
        if prev is not None and (frame_idx - prev) < min_gap:
            dropped_by_gap += 1
            continue
        accepted.append(event)
        last_frame_by_key[key] = frame_idx
        last_frame_by_finger[finger_key] = frame_idx

    accepted_before_cap = len(accepted)
    max_events_int = int(max_events)
    sampled_down = 0
    cap_strategy = "none"
    if max_events_int > 0 and accepted_before_cap > max_events_int:
        priority_slots = max(1, int(round(max_events_int * 0.75)))
        priority_slots = min(priority_slots, max_events_int)
        ranked_indices = sorted(
            range(accepted_before_cap),
            key=lambda idx: (
                float(getattr(accepted[idx], "touch_conf", 0.0) or 0.0),
                -int(getattr(accepted[idx], "row_index", idx)),
            ),
            reverse=True,
        )
        selected_indices: set[int] = set(ranked_indices[:priority_slots])
        remaining_slots = max_events_int - len(selected_indices)
        if remaining_slots > 0:
            if remaining_slots == 1:
                coverage_indices = [accepted_before_cap - 1]
            else:
                step = (accepted_before_cap - 1) / float(remaining_slots - 1)
                coverage_indices = [
                    max(0, min(accepted_before_cap - 1, int(round(i * step))))
                    for i in range(remaining_slots)
                ]
            for idx in coverage_indices:
                selected_indices.add(idx)
                if len(selected_indices) >= max_events_int:
                    break
        if len(selected_indices) < max_events_int:
            for idx in ranked_indices:
                selected_indices.add(idx)
                if len(selected_indices) >= max_events_int:
                    break
        accepted = [accepted[idx] for idx in sorted(selected_indices)[:max_events_int]]
        sampled_down = accepted_before_cap - len(accepted)
        cap_strategy = "touch_confidence_priority_with_time_coverage"

    return accepted, {
        "raw_touch_events": len(raw_touch_events),
        "candidate_touch_events": len(candidates),
        "accepted_events": len(accepted),
        "accepted_before_cap": int(accepted_before_cap),
        "dropped_by_gap": dropped_by_gap,
        "dropped_by_finger_gap": dropped_by_finger_gap,
        "dropped_by_touch_confidence": int(dropped_by_touch_conf),
        "min_touch_confidence": float(min_touch_conf),
        "max_events": int(max_events),
        "min_gap_frames": int(min_gap_frames),
        "sampled_down": int(sampled_down),
        "cap_applied": bool(sampled_down > 0),
        "cap_strategy": cap_strategy,
        "allowed_finger_types": sorted(allowed_fingers),
    }


def _build_event_velocity_stats_by_event(touch_events: list[Any], fps: float) -> dict[int, dict[str, float]]:
    fps_safe = max(float(fps), 1e-6)
    groups: dict[str, list[Any]] = {"thumb": [], "index": []}
    for ev in touch_events:
        finger_type = str(getattr(ev, "finger_type", "")).lower()
        if finger_type in groups:
            groups[finger_type].append(ev)
    for finger in groups:
        groups[finger].sort(key=lambda ev: float(getattr(ev, "timestamp_sec", 0.0)))

    out: dict[int, dict[str, float]] = {}
    for finger, events in groups.items():
        n = len(events)
        for i, ev in enumerate(events):
            row_key = int(getattr(ev, "row_index"))
            before = float("nan")
            after = float("nan")
            count_before = 0
            count_after = 0

            if i > 0:
                prev_ev = events[i - 1]
                if (
                    getattr(prev_ev, "finger_x", None) is not None
                    and getattr(prev_ev, "finger_y", None) is not None
                    and getattr(ev, "finger_x", None) is not None
                    and getattr(ev, "finger_y", None) is not None
                ):
                    dt = float(getattr(ev, "timestamp_sec")) - float(getattr(prev_ev, "timestamp_sec"))
                    if dt > 0:
                        dx = float(getattr(ev, "finger_x")) - float(getattr(prev_ev, "finger_x"))
                        dy = float(getattr(ev, "finger_y")) - float(getattr(prev_ev, "finger_y"))
                        before = float((dx * dx + dy * dy) ** 0.5 / max(dt, 1e-6))
                        count_before = 1

            if i + 1 < n:
                next_ev = events[i + 1]
                if (
                    getattr(next_ev, "finger_x", None) is not None
                    and getattr(next_ev, "finger_y", None) is not None
                    and getattr(ev, "finger_x", None) is not None
                    and getattr(ev, "finger_y", None) is not None
                ):
                    dt = float(getattr(next_ev, "timestamp_sec")) - float(getattr(ev, "timestamp_sec"))
                    if dt > 0:
                        dx = float(getattr(next_ev, "finger_x")) - float(getattr(ev, "finger_x"))
                        dy = float(getattr(next_ev, "finger_y")) - float(getattr(ev, "finger_y"))
                        after = float((dx * dx + dy * dy) ** 0.5 / max(dt, 1e-6))
                        count_after = 1

            out[row_key] = {
                "mean_vel_before": before,
                "mean_vel_after": after,
                "count_before": float(count_before),
                "count_after": float(count_after),
                "fps": float(fps_safe),
            }
    return out


def _serialize_strike_result(item: Any, include_debug: bool = False) -> dict[str, Any]:
    best_metrics = dict(vars(getattr(item, "best_metrics")))
    debug = getattr(item, "debug", {})
    debug = debug if isinstance(debug, dict) else {}
    finger_gate = debug.get("finger_gate", {})
    finger_gate = finger_gate if isinstance(finger_gate, dict) else {}

    candidate_ranking: list[dict[str, Any]] = []
    for bucket_name, source_name in (("candidate_decisions", "event_candidates"), ("probe_decisions", "shake_probes")):
        bucket = debug.get(bucket_name, {})
        if not isinstance(bucket, dict):
            continue
        for raw_candidate in bucket.values():
            if not isinstance(raw_candidate, dict):
                continue
            candidate_ranking.append(
                {
                    "source": source_name,
                    "candidate_string_id": raw_candidate.get("candidate_string_id"),
                    "candidate_score": float(raw_candidate.get("candidate_score", 0.0)),
                    "peak": float(raw_candidate.get("peak", 0.0)),
                    "duration": int(raw_candidate.get("duration", 0)),
                    "impulse": float(raw_candidate.get("impulse", 0.0)),
                    "vibrates": bool(raw_candidate.get("vibrates", False)),
                    "baseline_mean": float(raw_candidate.get("baseline_mean", 0.0)),
                    "baseline_std": float(raw_candidate.get("baseline_std", 0.0)),
                }
            )
    candidate_ranking.sort(
        key=lambda row: (
            bool(row.get("vibrates", False)),
            float(row.get("candidate_score", 0.0)),
            float(row.get("peak", 0.0)),
        ),
        reverse=True,
    )

    label = str(getattr(item, "label", "")).strip().lower()
    if label == "strike":
        decision_reason = "strike"
    elif label == "shake_reject":
        decision_reason = "shake_reject"
    elif bool(debug.get("startup_guard", False)):
        decision_reason = "startup_guard"
    elif not bool(best_metrics.get("vibrates", False)):
        decision_reason = "no_vibrating_candidate"
    elif not bool(debug.get("domination_pass", True)):
        decision_reason = "domination_failed"
    elif not bool(finger_gate.get("pass", True)):
        gate_reason = str(finger_gate.get("reason", "velocity_threshold_failed")).strip().lower()
        decision_reason = f"finger_gate:{gate_reason}"
    else:
        decision_reason = "touch_only"

    payload = {
        "event_time": float(getattr(item, "event_time", 0.0)),
        "finger_type": str(getattr(item, "finger_type", "")),
        "touched_id": int(getattr(item, "touched_id", 0)),
        "struck_id": getattr(item, "struck_id", None),
        "label": str(getattr(item, "label", "")),
        "touch_conf": getattr(item, "touch_conf", None),
        "distance_px": getattr(item, "distance_px", None),
        "contact_x": getattr(item, "contact_x", None),
        "contact_y": getattr(item, "contact_y", None),
        "finger_x": getattr(item, "finger_x", None),
        "finger_y": getattr(item, "finger_y", None),
        "best_metrics": best_metrics,
        "second_metrics": dict(vars(getattr(item, "second_metrics"))),
        "decision_debug": {
            "decision_reason": decision_reason,
            "domination_pass": bool(debug.get("domination_pass", True)),
            "shake_reject": bool(debug.get("shake_reject", False)),
            "finger_gate_enabled": bool(finger_gate.get("enabled", False)),
            "finger_gate_pass": bool(finger_gate.get("pass", True)),
            "finger_gate_reason": str(finger_gate.get("reason", "")),
            "mean_vel_before": finger_gate.get("mean_vel_before"),
            "mean_vel_after": finger_gate.get("mean_vel_after"),
            "startup_guard": bool(debug.get("startup_guard", False)),
            "startup_guard_min_frame": debug.get("startup_guard_min_frame"),
            "event_frame_index": debug.get("event_frame_index"),
            "peak_frame": debug.get("peak_frame"),
            "median_probe_peak": debug.get("median_probe_peak"),
            "vibrating_count_total": debug.get("vibrating_count_total"),
            "winning_candidate_id": best_metrics.get("candidate_id"),
            "second_candidate_id": None,
            "score_margin_vs_second": None,
            "candidate_ranking": candidate_ranking,
            "event_candidate_count": int(len(debug.get("candidate_decisions", {}))) if isinstance(debug.get("candidate_decisions"), dict) else 0,
            "probe_candidate_count": int(len(debug.get("probe_decisions", {}))) if isinstance(debug.get("probe_decisions"), dict) else 0,
            "probe_string_ids": list(debug.get("probe_ids", [])) if isinstance(debug.get("probe_ids"), list) else [],
        },
    }
    second_candidate_id = payload["second_metrics"].get("candidate_id")
    payload["decision_debug"]["second_candidate_id"] = second_candidate_id
    if second_candidate_id is not None:
        payload["decision_debug"]["score_margin_vs_second"] = float(best_metrics.get("candidate_score", 0.0)) - float(
            payload["second_metrics"].get("candidate_score", 0.0)
        )
    if include_debug:
        payload["debug"] = debug
    return payload


def _strike_confidence_level(candidate_score: float, peak_z: float) -> str:
    score = float(candidate_score)
    peak = float(peak_z)
    if score >= 20.0 and peak >= 10.0:
        return "high"
    if score >= 9.0 and peak >= 6.0:
        return "medium"
    return "low"


def _save_split_event_jsons(
    *,
    touch_events_json_path: str | None,
    touch_events: list[dict[str, Any]],
    strike_results: list[dict[str, Any]],
    fps: float,
    video_name: str,
    frames_processed: int,
    strike_algorithm_applied: bool = False,
    strike_algorithm_error: str | None = None,
) -> dict[str, Any]:
    if not touch_events_json_path:
        return {}

    src_path = Path(str(touch_events_json_path))
    out_dir = src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_name = src_path.name
    suffix = "_touch_events.json"
    base_stem = stem_name[: -len(suffix)] if stem_name.endswith(suffix) else src_path.stem
    tag = out_dir.name
    fps_safe = max(float(fps), 1e-6)

    left_touch_events = [
        ev
        for ev in touch_events
        if str(ev.get("hand_side") or ev.get("hand") or "").strip().lower() == "left"
    ]
    left_payload = {
        "video_name": video_name,
        "tag": tag,
        "fps": float(fps_safe),
        "frames_processed": int(frames_processed),
        "left_touch_events_count": int(len(left_touch_events)),
        "left_touch_events": left_touch_events,
    }
    left_path = out_dir / f"{base_stem}_left_touch_events.json"
    with left_path.open("w", encoding="utf-8") as f:
        json.dump(left_payload, f, indent=2, ensure_ascii=False)

    right_strikes: list[dict[str, Any]] = []
    right_decisions: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    reject_reason_counts: Counter[str] = Counter()
    strike_confidence_counts: Counter[str] = Counter()
    for item in strike_results:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        label_counts[label] += 1
        decision_debug = item.get("decision_debug", {}) if isinstance(item.get("decision_debug"), dict) else {}
        decision_reason = str(decision_debug.get("decision_reason", "")).strip().lower() or label
        if label != "strike":
            reject_reason_counts[decision_reason] += 1

        event_time = float(item.get("event_time", 0.0))
        best_metrics = item.get("best_metrics", {}) if isinstance(item.get("best_metrics"), dict) else {}
        right_decisions.append(
            {
                "time_sec": float(event_time),
                "frame_index": int(round(event_time * fps_safe)),
                "finger_type": str(item.get("finger_type", "")),
                "touched_id": item.get("touched_id"),
                "struck_id": item.get("struck_id"),
                "touch_conf": item.get("touch_conf"),
                "distance_px": item.get("distance_px"),
                "contact_x": item.get("contact_x"),
                "contact_y": item.get("contact_y"),
                "finger_x": item.get("finger_x"),
                "finger_y": item.get("finger_y"),
                "label": label,
                "decision_reason": decision_reason,
                "candidate_score": float(best_metrics.get("candidate_score", 0.0)),
                "peak_z": float(best_metrics.get("peak", 0.0)),
                "duration": int(best_metrics.get("duration", 0)),
                "impulse": float(best_metrics.get("impulse", 0.0)),
                "vibrates": bool(best_metrics.get("vibrates", False)),
                "decision_debug": decision_debug,
            }
        )

        struck_id = item.get("struck_id")
        if label != "strike" or struck_id is None:
            continue

        candidate_score = float(best_metrics.get("candidate_score", 0.0))
        peak_z = float(best_metrics.get("peak", 0.0))
        strike_confidence = _strike_confidence_level(candidate_score, peak_z)
        strike_confidence_counts[strike_confidence] += 1
        right_strikes.append(
            {
                "time_sec": float(event_time),
                "frame_index": int(round(event_time * fps_safe)),
                "finger_type": str(item.get("finger_type", "")),
                "touched_id": item.get("touched_id"),
                "struck_id": struck_id,
                "touch_conf": item.get("touch_conf"),
                "distance_px": item.get("distance_px"),
                "contact_x": item.get("contact_x"),
                "contact_y": item.get("contact_y"),
                "finger_x": item.get("finger_x"),
                "finger_y": item.get("finger_y"),
                "candidate_score": candidate_score,
                "peak_z": peak_z,
                "strike_confidence": strike_confidence,
                "decision_reason": decision_reason,
            }
        )

    right_decision_payload = {
        "video_name": video_name,
        "tag": tag,
        "fps": float(fps_safe),
        "frames_processed": int(frames_processed),
        "strike_algorithm_applied": bool(strike_algorithm_applied),
        "strike_algorithm_error": strike_algorithm_error,
        "right_decision_events_count": int(len(right_decisions)),
        "label_counts": dict(label_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "right_decision_events": right_decisions,
    }
    right_decision_path = out_dir / f"{base_stem}_right_decision_events.json"
    with right_decision_path.open("w", encoding="utf-8") as f:
        json.dump(right_decision_payload, f, indent=2, ensure_ascii=False)

    right_payload = {
        "video_name": video_name,
        "tag": tag,
        "fps": float(fps_safe),
        "frames_processed": int(frames_processed),
        "strike_algorithm_applied": bool(strike_algorithm_applied),
        "strike_algorithm_error": strike_algorithm_error,
        "right_decision_events_count": int(len(right_decisions)),
        "label_counts": dict(label_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "strike_confidence_counts": dict(strike_confidence_counts),
        "right_strike_events_count": int(len(right_strikes)),
        "right_strike_events": right_strikes,
    }
    right_path = out_dir / f"{base_stem}_right_strike_events.json"
    with right_path.open("w", encoding="utf-8") as f:
        json.dump(right_payload, f, indent=2, ensure_ascii=False)

    # Also annotate the source touch-events JSON for easier debugging.
    try:
        source_payload: dict[str, Any] = {}
        if src_path.exists():
            with src_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                source_payload = existing
        source_payload["left_touch_events_json_path"] = str(left_path)
        source_payload["right_strike_events_json_path"] = str(right_path)
        source_payload["right_decision_events_json_path"] = str(right_decision_path)
        source_payload["left_touch_events_count"] = int(len(left_touch_events))
        source_payload["right_decision_events_count"] = int(len(right_decisions))
        source_payload["right_strike_events_count"] = int(len(right_strikes))
        source_payload["strike_algorithm_applied"] = bool(strike_algorithm_applied)
        source_payload["strike_algorithm_error"] = strike_algorithm_error
        with src_path.open("w", encoding="utf-8") as f:
            json.dump(source_payload, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return {
        "left_touch_events_json_path": str(left_path),
        "right_strike_events_json_path": str(right_path),
        "right_decision_events_json_path": str(right_decision_path),
        "left_touch_events_count": int(len(left_touch_events)),
        "right_decision_events_count": int(len(right_decisions)),
        "right_strike_events_count": int(len(right_strikes)),
    }


def _predicted_video_url_for_path(base_url: str, video_path: str | Path | None) -> str | None:
    if video_path is None:
        return None
    out_video_path = Path(str(video_path))
    if not out_video_path.exists():
        return None
    try:
        relative_output_path = out_video_path.relative_to(PREDICTIONS_DIR).as_posix()
    except ValueError:
        relative_output_path = out_video_path.name
    return f"{base_url.rstrip('/')}/predictions/{relative_output_path}"


def _debug_file_url_for_path(base_url: str, file_path: str | Path | None) -> str | None:
    if file_path is None:
        return None
    p = Path(str(file_path))
    if not p.exists():
        return None
    try:
        relative_path = p.relative_to(DEBUG_REPORTS_DIR).as_posix()
    except ValueError:
        return None
    return f"{base_url.rstrip('/')}/analysis-debug/{relative_path}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_optional(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_cache_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_cache_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(v) for v in value]
    return value


def _build_cache_code_signatures() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for path in CACHE_SIGNATURE_PATHS:
        try:
            rel = str(path.relative_to(REPO_DIR)).replace("\\", "/")
        except Exception:
            rel = str(path)
        out[rel] = _sha256_file_optional(path)
    return out


def _build_request_cache_key(
    *,
    video_bytes: bytes,
    audio_bytes: bytes | None,
    request_options: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    metadata = {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "video_sha256": _sha256_bytes(video_bytes),
        "audio_sha256": (_sha256_bytes(audio_bytes) if audio_bytes is not None else None),
        "request_options": _normalize_cache_value(request_options),
        "code_signatures": _build_cache_code_signatures(),
    }
    cache_key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return cache_key, metadata


def _cache_manifest_path(cache_key: str) -> Path:
    return CACHE_DIR / cache_key / "response.json"


def _cache_output_paths_exist(response: dict[str, Any]) -> bool:
    annotated_path = response.get("annotated_video_path")
    if not isinstance(annotated_path, str) or not annotated_path.strip() or not Path(annotated_path).exists():
        return False
    for key in CACHE_OUTPUT_PATH_KEYS:
        value = response.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if not Path(value).exists():
            return False
    return True


def _prepare_cached_response(base_url: str, response: dict[str, Any], cache_key: str) -> dict[str, Any]:
    cached = json.loads(json.dumps(response))
    predicted_video_url = _predicted_video_url_for_path(base_url, cached.get("annotated_video_path"))
    if predicted_video_url is not None:
        cached["predicted_video_url"] = predicted_video_url
    debug_report_url = _debug_file_url_for_path(base_url, cached.get("analysis_debug_report_path"))
    if debug_report_url is not None:
        cached["analysis_debug_report_url"] = debug_report_url
    cached["cache_hit"] = True
    cached["cache_key"] = cache_key
    return cached


def _load_cached_response(base_url: str, cache_key: str) -> dict[str, Any] | None:
    manifest_path = _cache_manifest_path(cache_key)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    response = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response, dict):
        return None
    if not _cache_output_paths_exist(response):
        return None
    return _prepare_cached_response(base_url, response, cache_key)


def _store_cached_response(cache_key: str, metadata: dict[str, Any], response: dict[str, Any]) -> None:
    cache_dir = CACHE_DIR / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "response.json"
    payload = {
        "cache_key": cache_key,
        "metadata": metadata,
        "created_at_epoch_sec": time.time(),
        "response": json.loads(json.dumps(response)),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_audio_fusion_postprocess_safe(
    *,
    upload_path: Path,
    result: dict[str, Any],
    split_event_files: dict[str, Any],
    fusion_mode: str | None,
    audio_enabled: bool | None,
    audio_decision_mode: str | None,
    audio_input_path: Path | None = None,
) -> dict[str, Any]:
    touch_events_json_path = result.get("touch_events_json_path")
    if not touch_events_json_path:
        return {"enabled": False, "reason": "missing_touch_events_json_path"}
    try:
        print(
            "[INFO] Audio/AV postprocess started: "
            f"touch_json={touch_events_json_path}, "
            f"video_decision_json={split_event_files.get('right_decision_events_json_path')}, "
            f"audio_enabled={audio_enabled}, fusion_mode={fusion_mode}, "
            f"audio_decision_mode={audio_decision_mode}"
        )
        _ensure_repo_src_precedence()
        _purge_shadowed_src_modules()
        importlib.invalidate_caches()
        src_mod = sys.modules.get("src")
        if src_mod is not None:
            print(
                "[INFO] Existing `src` module before audio import: "
                f"file={getattr(src_mod, '__file__', None)} path={getattr(src_mod, '__path__', None)}"
            )
        from src.pipeline.run import run_audio_fusion_postprocess
        try:
            imported_src = importlib.import_module("src")
            print(
                "[INFO] Audio/AV import resolved `src` package: "
                f"file={getattr(imported_src, '__file__', None)} path={getattr(imported_src, '__path__', None)}"
            )
        except Exception:
            pass

        outputs = run_audio_fusion_postprocess(
            video_path=upload_path,
            touch_events_json_path=touch_events_json_path,
            right_video_decision_events_json_path=split_event_files.get("right_decision_events_json_path"),
            annotated_video_path=result.get("out_video_path"),
            string_geometries=result.get("string_geometries"),
            config_path=PIPELINE_CONFIG_PATH if PIPELINE_CONFIG_PATH.exists() else None,
            audio_input_path=audio_input_path,
            audio_enabled=audio_enabled,
            fusion_mode=fusion_mode,
            audio_decision_mode=audio_decision_mode,
            enable_overlay=True,
            fps_override=float(result.get("fps") or 0.0) or None,
        )
        print(
            "[INFO] Audio/AV postprocess finished: "
            f"audio_decisions={outputs.get('right_audio_decision_events_count', 0)}, "
            f"audio_strikes={outputs.get('right_audio_strike_events_count', 0)}, "
            f"av_decisions={outputs.get('right_av_decision_events_count', 0)}, "
            f"av_strikes={outputs.get('right_av_strike_events_count', 0)}, "
            f"audio_error={outputs.get('audio_error')}"
        )
        return {"enabled": True, **outputs}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Audio/AV postprocess failed: {exc}")
        print(traceback.format_exc())
        return {"enabled": True, "error": str(exc)}


def _load_json_payload_optional(path_value: Any) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _events_from_payload(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("events", "right_decision_events", "right_strike_events", "touch_events", "left_touch_events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _debug_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _debug_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out
    except Exception:
        return None


def _counter_from_events(events: list[dict[str, Any]], *path: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        value: Any = event
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        counts[str(value if value is not None else "missing")] += 1
    return dict(counts)


def _write_debug_snapshots(
    *,
    source_video_path: Path,
    debug_dir: Path,
    rows: list[dict[str, Any]],
    string_geometries: list[dict[str, Any]] | None = None,
    max_snapshots: int = 60,
) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        import cv2  # type: ignore
    except Exception:
        return out

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        return out
    strings_by_id: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for geom in string_geometries or []:
        if not isinstance(geom, dict):
            continue
        sid = _debug_int(geom.get("string_id"))
        endpoints = geom.get("endpoints")
        if sid is None or not isinstance(endpoints, list) or len(endpoints) < 2:
            continue
        try:
            p1 = endpoints[0]
            p2 = endpoints[1]
            strings_by_id[int(sid)] = (
                (int(round(float(p1[0]))), int(round(float(p1[1])))),
                (int(round(float(p2[0]))), int(round(float(p2[1])))),
            )
        except Exception:
            continue

    def draw_string(frame: Any, sid: int | None, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
        if sid is None:
            return
        endpoints = strings_by_id.get(int(sid))
        if endpoints is None:
            return
        p1, p2 = endpoints
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        mid = (int(round((p1[0] + p2[0]) * 0.5)), int(round((p1[1] + p2[1]) * 0.5)))
        cv2.putText(frame, f"{label}s{sid}", (mid[0] + 4, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def draw_point(frame: Any, x: float | None, y: float | None, color: tuple[int, int, int], label: str) -> None:
        if x is None or y is None:
            return
        p = (int(round(float(x))), int(round(float(y))))
        cv2.circle(frame, p, 7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    try:
        snapshot_dir = debug_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        selected = [
            row
            for row in rows
            if _debug_int(row.get("frame_index")) is not None and row.get("debug_priority", 0) > 0
        ][:max_snapshots]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for row in selected:
            row_index = int(row.get("row_index", 0))
            frame_index = int(row.get("frame_index", 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            raw_debug = row.get("raw_debug") if isinstance(row.get("raw_debug"), dict) else {}
            raw_video = raw_debug.get("video") if isinstance(raw_debug.get("video"), dict) else {}
            decision_debug = raw_video.get("decision_debug") if isinstance(raw_video.get("decision_debug"), dict) else {}
            ranking = decision_debug.get("candidate_ranking") if isinstance(decision_debug.get("candidate_ranking"), list) else []
            for rank, candidate in enumerate(ranking[:3], start=1):
                if not isinstance(candidate, dict):
                    continue
                candidate_sid = _debug_int(candidate.get("candidate_string_id"))
                draw_string(frame, candidate_sid, (190, 190, 190), f"c{rank}:", thickness=1)
            draw_string(frame, _debug_int(row.get("touched_string_id")), (0, 215, 255), "touch:", thickness=2)
            draw_string(frame, _debug_int(row.get("fusion_struck_string_id")), (80, 230, 80), "fused:", thickness=3)
            draw_string(frame, _debug_int(row.get("video_struck_string_id")), (255, 160, 80), "video:", thickness=2)
            finger_x = _debug_float(row.get("finger_x"))
            finger_y = _debug_float(row.get("finger_y"))
            contact_x = _debug_float(row.get("contact_x"))
            contact_y = _debug_float(row.get("contact_y"))
            if finger_x is not None and finger_y is not None and contact_x is not None and contact_y is not None:
                cv2.line(
                    frame,
                    (int(round(finger_x)), int(round(finger_y))),
                    (int(round(contact_x)), int(round(contact_y))),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            draw_point(frame, finger_x, finger_y, (255, 255, 0), "finger")
            draw_point(frame, contact_x, contact_y, (0, 0, 255), "contact")

            overlay = frame.copy()
            header_h = min(h, 118)
            cv2.rectangle(overlay, (0, 0), (w, header_h), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
            lines = [
                f"event {row_index} frame {frame_index} time {row.get('timestamp_sec')}",
                f"finger {row.get('finger_type')} touched s{row.get('touched_string_id')} fused s{row.get('fusion_struck_string_id')} conf {row.get('touch_conf')}",
                f"video {row.get('video_status')} s{row.get('video_struck_string_id')} | audio {row.get('audio_status')} s{row.get('audio_struck_string_id')}",
                f"fusion {row.get('fusion_status')} {row.get('fusion_confidence_label')} {row.get('fusion_strategy')}",
                "flags: " + ", ".join(row.get("flags", [])[:5]),
            ]
            y = 22
            for line in lines:
                cv2.putText(frame, str(line)[:150], (12, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 21
            out_path = snapshot_dir / f"event_{row_index:04d}_frame_{frame_index:06d}.jpg"
            if cv2.imwrite(str(out_path), frame):
                out[row_index] = str(out_path)
    finally:
        cap.release()
    return out


def _write_analysis_debug_report(
    *,
    base_url: str,
    upload_path: Path,
    result: dict[str, Any],
    strike_inference: dict[str, Any],
    split_event_files: dict[str, Any],
    av_inference: dict[str, Any],
    right_audio_decision_payload: dict[str, Any] | None,
    right_av_decision_payload: dict[str, Any] | None,
    right_av_strike_payload: dict[str, Any] | None,
    pipeline_timings: dict[str, Any],
    run_profile: dict[str, Any],
) -> dict[str, Any]:
    debug_dir = DEBUG_REPORTS_DIR / upload_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    av_events = _events_from_payload(right_av_decision_payload)
    audio_events = _events_from_payload(right_audio_decision_payload)
    av_strikes = _events_from_payload(right_av_strike_payload)

    rows: list[dict[str, Any]] = []
    for index, event in enumerate(av_events):
        touch = event.get("touch") if isinstance(event.get("touch"), dict) else {}
        video = event.get("video") if isinstance(event.get("video"), dict) else {}
        audio = event.get("audio") if isinstance(event.get("audio"), dict) else {}
        fusion = event.get("fusion") if isinstance(event.get("fusion"), dict) else {}
        timing = fusion.get("timing") if isinstance(fusion.get("timing"), dict) else {}

        frame_index = _debug_int(touch.get("frame_index"))
        if frame_index is None:
            frame_index = _debug_int(video.get("frame_index"))
        if frame_index is None:
            frame_index = _debug_int(timing.get("onset_frame"))

        touched_sid = _debug_int(touch.get("touched_string_id"))
        video_sid = _debug_int(video.get("struck_string_id"))
        audio_sid = _debug_int(audio.get("struck_string_id"))
        fusion_sid = _debug_int(fusion.get("struck_string_id"))
        fusion_status = str(fusion.get("status") or "")
        audio_status = str(audio.get("status") or "")
        video_status = str(video.get("status") or "")
        confidence = _debug_float(fusion.get("confidence")) or 0.0
        touch_conf = _debug_float(touch.get("touch_conf"))
        pitch_conf = _debug_float(audio.get("pitch_conf"))
        cents_error = _debug_float(audio.get("cents_error"))

        flags: list[str] = []
        if fusion_status != "strike":
            flags.append("not_fused_strike")
        if audio_status not in {"strike", "missing"}:
            flags.append(f"audio_{audio_status}")
        if video_status not in {"strike", "missing"}:
            flags.append(f"video_{video_status}")
        if touched_sid is not None and fusion_sid is not None and touched_sid != fusion_sid:
            flags.append("touched_fused_mismatch")
        if video_sid is not None and audio_sid is not None and video_sid != audio_sid:
            flags.append("audio_video_conflict")
        if confidence < 0.55:
            flags.append("low_fusion_confidence")
        if touch_conf is not None and touch_conf < 0.30:
            flags.append("low_touch_confidence")
        if pitch_conf is not None and pitch_conf < 0.50:
            flags.append("low_pitch_confidence")
        if cents_error is not None and abs(cents_error) > 50:
            flags.append("large_pitch_cents_error")

        debug_priority = 0
        if "audio_video_conflict" in flags or "touched_fused_mismatch" in flags:
            debug_priority = 3
        elif fusion_status != "strike" or audio_status not in {"strike", "missing"}:
            debug_priority = 2
        elif flags:
            debug_priority = 1

        rows.append(
            {
                "row_index": index,
                "event_id": event.get("event_id"),
                "timestamp_sec": _debug_float(touch.get("timestamp_sec", touch.get("time_sec"))),
                "frame_index": frame_index,
                "finger_type": touch.get("finger_type"),
                "touched_string_id": touched_sid,
                "touch_conf": touch_conf,
                "distance_px": _debug_float(touch.get("distance_px")),
                "contact_x": _debug_float(touch.get("contact_x")),
                "contact_y": _debug_float(touch.get("contact_y")),
                "finger_x": _debug_float(touch.get("finger_x")),
                "finger_y": _debug_float(touch.get("finger_y")),
                "video_status": video_status,
                "video_struck_string_id": video_sid,
                "video_confidence": _debug_float(video.get("confidence")),
                "video_peak_frame": _debug_int(video.get("peak_frame")),
                "audio_status": audio_status,
                "audio_struck_string_id": audio_sid,
                "audio_confidence": _debug_float(audio.get("confidence")),
                "audio_confidence_label": audio.get("confidence_label"),
                "audio_onset_time_sec": _debug_float(audio.get("onset_time_sec")),
                "audio_onset_frame": _debug_int(audio.get("onset_frame")),
                "audio_pitch_backend": audio.get("pitch_backend"),
                "audio_f0_hz": _debug_float(audio.get("f0_hz")),
                "audio_pitch_conf": pitch_conf,
                "audio_cents_error": cents_error,
                "fusion_status": fusion_status,
                "fusion_struck_string_id": fusion_sid,
                "fusion_confidence": confidence,
                "fusion_confidence_label": fusion.get("confidence_label"),
                "fusion_strategy": fusion.get("strategy"),
                "fusion_timing": timing,
                "flags": flags,
                "debug_priority": debug_priority,
                "raw_debug": {
                    "video": video.get("raw") if isinstance(video.get("raw"), dict) else None,
                    "audio": audio.get("raw") if isinstance(audio.get("raw"), dict) else None,
                    "fusion": fusion.get("debug") if isinstance(fusion.get("debug"), dict) else None,
                },
            }
        )

    rows.sort(key=lambda row: (-int(row.get("debug_priority", 0)), _debug_int(row.get("frame_index")) or 0, int(row.get("row_index", 0))))
    snapshot_paths = _write_debug_snapshots(
        source_video_path=upload_path,
        debug_dir=debug_dir,
        rows=rows,
        string_geometries=result.get("string_geometries") if isinstance(result.get("string_geometries"), list) else [],
    )
    for row in rows:
        snapshot_path = snapshot_paths.get(int(row.get("row_index", -1)))
        if snapshot_path:
            row["snapshot_path"] = snapshot_path
            row["snapshot_url"] = _debug_file_url_for_path(base_url, snapshot_path)

    summary = {
        "frames_processed": int(result.get("frames_processed", 0)),
        "touch_events_count": int(result.get("touch_events_count", 0)),
        "strike_events_for_decision": int(strike_inference.get("events_for_decision", 0) or 0),
        "video_decision_events_count": int(split_event_files.get("right_decision_events_count", 0) or 0),
        "video_strike_events_count": int(split_event_files.get("right_strike_events_count", 0) or 0),
        "audio_decision_events_count": len(audio_events),
        "av_decision_events_count": len(av_events),
        "av_strike_events_count": len(av_strikes),
        "debug_rows_count": len(rows),
        "snapshot_count": len(snapshot_paths),
        "fusion_status_counts": _counter_from_events(av_events, "fusion", "status"),
        "fusion_strategy_counts": _counter_from_events(av_events, "fusion", "strategy"),
        "audio_status_counts": _counter_from_events(av_events, "audio", "status"),
        "audio_backend_counts": _counter_from_events(av_events, "audio", "pitch_backend"),
        "flag_counts": dict(Counter(flag for row in rows for flag in row.get("flags", []))),
        "timings": pipeline_timings,
        "run_profile": run_profile,
        "touch_detection": result.get("touch_detection"),
        "strike_touch_filter": strike_inference.get("touch_events_debug"),
        "audio_config": {
            "decision_mode": av_inference.get("audio_decision_mode"),
            "tuning": (
                ((right_audio_decision_payload or {}).get("meta") or {}).get("tuning")
                if isinstance((right_audio_decision_payload or {}).get("meta"), dict)
                else None
            ),
        },
    }
    report = {
        "video_name": upload_path.name,
        "source_video_path": str(upload_path),
        "generated_at_epoch_sec": time.time(),
        "summary": summary,
        "paths": {
            "touch_events_json_path": result.get("touch_events_json_path"),
            "right_video_decision_events_json_path": split_event_files.get("right_decision_events_json_path"),
            "right_video_strike_events_json_path": split_event_files.get("right_strike_events_json_path"),
            "right_audio_decision_events_json_path": av_inference.get("right_audio_decision_events_json_path"),
            "right_audio_strike_events_json_path": av_inference.get("right_audio_strike_events_json_path"),
            "right_av_decision_events_json_path": av_inference.get("right_av_decision_events_json_path"),
            "right_av_strike_events_json_path": av_inference.get("right_av_strike_events_json_path"),
        },
        "debug_rows": rows,
    }
    report_path = debug_dir / "analysis_debug_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "analysis_debug_report_path": str(report_path),
        "analysis_debug_report_url": _debug_file_url_for_path(base_url, report_path),
        "analysis_debug_snapshot_count": len(snapshot_paths),
        "analysis_debug_dir_path": str(debug_dir),
        "analysis_debug_dir_url": _debug_file_url_for_path(base_url, debug_dir) if debug_dir.exists() else None,
        "analysis_debug_summary": summary,
    }


def _run_prediction_from_saved_video(
    *,
    base_url: str,
    upload_path: Path,
    expected_strings: int,
    enable_hand_tracking: bool,
    draw_hand_labels: bool,
    hand_pipeline_enabled: bool | None,
    enable_strike_decision: bool,
    max_strike_events: int,
    strike_min_event_gap_frames: int,
    include_strike_debug: bool,
    fusion_mode: str | None = None,
    audio_enabled: bool | None = None,
    audio_decision_mode: str | None = None,
    enable_debug_report: bool = True,
    audio_input_path: Path | None = None,
) -> dict[str, object]:
    pipeline_started_at = time.perf_counter()
    if not BEST_MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model not found: {BEST_MODEL_PATH}",
        )

    string_infer_every_n = ACCURATE_MODE_STRING_INFER_EVERY_N
    max_strike_events = min(int(max_strike_events), ACCURATE_MODE_MAX_STRIKE_EVENTS)
    strike_min_event_gap_frames = max(int(strike_min_event_gap_frames), ACCURATE_MODE_STRIKE_MIN_EVENT_GAP_FRAMES)

    hand_process_width = ACCURATE_MODE_HAND_PROCESS_WIDTH
    hand_model_complexity = ACCURATE_MODE_HAND_MODEL_COMPLEXITY
    print(
        "[INFO] Effective run profile: "
        "profile=accurate, "
        f"string_infer_every_n={int(string_infer_every_n)}, "
        f"max_strike_events={int(max_strike_events)}, "
        f"strike_min_event_gap_frames={int(strike_min_event_gap_frames)}, "
        f"hand_process_width={int(hand_process_width)}, "
        f"hand_model_complexity={int(hand_model_complexity)}, "
        f"audio_decision_mode={audio_decision_mode}, "
        f"enable_debug_report={bool(enable_debug_report)}"
    )

    try:
        try:
            from .post_processing import detect_audio_tag, detect_video_codec_tag, run_video_predict, transcode_to_h264
        except ImportError:
            from post_processing import detect_audio_tag, detect_video_codec_tag, run_video_predict, transcode_to_h264

        input_has_audio_track = bool(detect_audio_tag(upload_path))
        result = run_video_predict(
            tag="best",
            model_path=BEST_MODEL_PATH,
            video_path=upload_path,
            expected_strings=expected_strings,
            string_infer_every_n=string_infer_every_n,
            save_video=True,
            show_preview=False,
            transcode_output=False,
            transcode_preset="veryfast",
            enable_hand_tracking=enable_hand_tracking,
            hand_process_width=hand_process_width,
            hand_model_complexity=hand_model_complexity,
            draw_hand_labels=draw_hand_labels,
            hand_pipeline_enabled=hand_pipeline_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    out_video_path = result.get("out_video_path")
    if out_video_path is None:
        raise HTTPException(status_code=500, detail="Prediction completed without an output video.")

    out_video_path = Path(out_video_path)
    if not out_video_path.exists():
        raise HTTPException(status_code=500, detail=f"Output video missing: {out_video_path}")

    predicted_video_url = _predicted_video_url_for_path(base_url, out_video_path)
    if predicted_video_url is None:
        predicted_video_url = f"{base_url.rstrip('/')}/predictions/{out_video_path.name}"

    strike_results: list[dict[str, Any]] = []
    strike_inference: dict[str, Any] = {
        "enabled": bool(enable_strike_decision),
        "config_path": str(STRIKE_CONFIG_PATH),
        "pipeline_config_path": str(PIPELINE_CONFIG_PATH) if PIPELINE_CONFIG_PATH.exists() else None,
    }
    if enable_strike_decision:
        strike_started_at = time.perf_counter()
        try:
            decide_touch_events, import_err = _import_strike_symbol(
                "saung_strike_video_farneback_rules.src.decision",
                "decide_touch_events",
            )
            if decide_touch_events is None:
                raise RuntimeError(f"Cannot import decide_touch_events: {import_err}")

            strike_cfg = _load_yaml_optional(STRIKE_CONFIG_PATH)
            pipeline_cfg = _load_yaml_optional(PIPELINE_CONFIG_PATH)
            if pipeline_cfg:
                try:
                    from src.pipeline.config import apply_video_vibration_overrides_to_legacy_strike_config

                    strike_cfg = apply_video_vibration_overrides_to_legacy_strike_config(strike_cfg, pipeline_cfg)
                except Exception as cfg_exc:  # noqa: BLE001
                    strike_inference["pipeline_cfg_override_error"] = str(cfg_exc)

            windows_cfg = strike_cfg.get("windows", {}) if isinstance(strike_cfg.get("windows"), dict) else {}
            candidates_cfg = strike_cfg.get("candidates", {}) if isinstance(strike_cfg.get("candidates"), dict) else {}
            roi_cfg = strike_cfg.get("ROI", {}) if isinstance(strike_cfg.get("ROI"), dict) else {}
            farneback_cfg = strike_cfg.get("farneback_params", {}) if isinstance(strike_cfg.get("farneback_params"), dict) else {}
            vibration_cfg = strike_cfg.get("vibration", {}) if isinstance(strike_cfg.get("vibration"), dict) else {}
            rules_cfg = strike_cfg.get("rules", {}) if isinstance(strike_cfg.get("rules"), dict) else {}
            global_shake_cfg = strike_cfg.get("global_shake", {}) if isinstance(strike_cfg.get("global_shake"), dict) else {}
            finger_gate_cfg = strike_cfg.get("finger_gating", {}) if isinstance(strike_cfg.get("finger_gating"), dict) else {}
            masking_cfg = strike_cfg.get("masking", {}) if isinstance(strike_cfg.get("masking"), dict) else {}
            frame_transitions_cfg = strike_cfg.get("frame_transitions", {}) if isinstance(strike_cfg.get("frame_transitions"), dict) else {}
            decision_cfg = strike_cfg.get("decision", {}) if isinstance(strike_cfg.get("decision"), dict) else {}
            stabilization_cfg = strike_cfg.get("stabilization", {}) if isinstance(strike_cfg.get("stabilization"), dict) else {}
            right_finger_types_cfg = decision_cfg.get("right_finger_types")
            if isinstance(right_finger_types_cfg, list):
                allowed_finger_types = {
                    str(v).strip().lower()
                    for v in right_finger_types_cfg
                    if str(v).strip()
                }
            else:
                allowed_finger_types = {"thumb", "index"}
            if not allowed_finger_types:
                allowed_finger_types = {"thumb", "index"}
            dy_thr = float(
                vibration_cfg.get(
                    "dy_thr",
                    strike_cfg.get("dy_thr", farneback_cfg.get("dy_thr", 0.40)),
                )
            )
            stabilize_enabled = _coerce_bool(
                stabilization_cfg.get(
                    "enabled",
                    decision_cfg.get("stabilize_enabled", strike_cfg.get("stabilize_enabled", True)),
                ),
                default=True,
            )
            fps_used = float(result.get("fps") or strike_cfg.get("fps") or 30.0)
            baseline_len = max(1, int(round(float(windows_cfg.get("baseline_sec", 0.25)) * fps_used)))
            action_len = max(1, int(round(float(windows_cfg.get("action_sec", 0.25)) * fps_used)))
            action_start_offset = int(windows_cfg.get("action_start_frame_offset", 1))
            shake_probe_count = int(global_shake_cfg.get("shake_probe_count", 6))
            shake_probe_count = min(shake_probe_count, ACCURATE_MODE_SHAKE_PROBE_COUNT)

            strings_for_decision = _build_string_geometries_for_decision(result.get("string_geometries", []))
            touch_events_for_decision, touch_events_debug = _build_touch_events_for_decision(
                result.get("touch_events", []),
                fps=fps_used,
                max_events=max(1, int(max_strike_events)),
                min_gap_frames=max(1, int(strike_min_event_gap_frames)),
                allowed_finger_types=allowed_finger_types,
                min_touch_confidence=ACCURATE_MODE_MIN_TOUCH_CONFIDENCE,
            )
            velocity_stats_by_event = _build_event_velocity_stats_by_event(touch_events_for_decision, fps=fps_used)

            strike_inference.update(
                {
                    "fps_used": fps_used,
                    "baseline_len": baseline_len,
                    "action_len": action_len,
                    "action_start_frame_offset": action_start_offset,
                    "dy_thr": float(dy_thr),
                    "stabilize_enabled": bool(stabilize_enabled),
                    "shake_probe_count": int(shake_probe_count),
                    "touch_events_debug": touch_events_debug,
                    "strings_for_decision": len(strings_for_decision),
                    "fusion_mode_requested": fusion_mode,
                    "audio_enabled_requested": audio_enabled,
                    "audio_decision_mode_requested": audio_decision_mode,
                }
            )
            strike_inference["events_for_decision"] = len(touch_events_for_decision)
            print(
                "[INFO] Strike inference started: "
                f"events={len(touch_events_for_decision)}, "
                f"strings={len(strings_for_decision)}, "
                f"shake_probe_count={shake_probe_count}, "
                f"stabilize={bool(stabilize_enabled)}"
            )

            if strings_for_decision and touch_events_for_decision:
                strike_objects = decide_touch_events(
                    video_path=upload_path,
                    touch_events=touch_events_for_decision,
                    strings=strings_for_decision,
                    fps=fps_used,
                    baseline_len=baseline_len,
                    action_len=action_len,
                    action_start_frame_offset=action_start_offset,
                    roi_w=int(roi_cfg.get("roi_w", 160)),
                    roi_h=int(roi_cfg.get("roi_h", 32)),
                    trim_ends_ratio=float(roi_cfg.get("trim_ends_ratio", 0.15)),
                    center_band_h=int(roi_cfg.get("center_band_h", 10)),
                    candidate_radius_default=int(candidates_cfg.get("candidate_radius_default", 2)),
                    candidate_radius_close_contact=int(candidates_cfg.get("candidate_radius_close_contact", 1)),
                    contact_dist_px_thr=float(candidates_cfg.get("contact_dist_px_thr", 8.0)),
                    enable_hand_mask=_coerce_bool(masking_cfg.get("enable_hand_mask", True), default=True),
                    hand_mask_expand_px=float(masking_cfg.get("hand_mask_expand_px", 8.0)),
                    farneback_params=farneback_cfg,
                    dy_thr=float(dy_thr),
                    z_thr=float(rules_cfg.get("z_thr", 2.35)),
                    thr_peak=float(rules_cfg.get("thr_peak", 4.0)),
                    thr_duration_frames=int(rules_cfg.get("thr_duration_frames", 2)),
                    thr_impulse=float(rules_cfg.get("thr_impulse", 7.0)),
                    baseline_gap_frames=int(windows_cfg.get("baseline_gap_frames", 0)),
                    dynamic_action_enabled=_coerce_bool(windows_cfg.get("dynamic_action_enabled", False), default=False),
                    dynamic_offsets=windows_cfg.get("dynamic_offsets"),
                    dynamic_select_metric=str(windows_cfg.get("dynamic_select_metric", "max_impulse")),
                    geometry_enabled=_coerce_bool(candidates_cfg.get("geometry_enabled", False), default=False),
                    geometry_top_k=int(candidates_cfg.get("geometry_top_k", 5)),
                    geometry_max_distance_px=float(candidates_cfg.get("geometry_max_distance_px", 35.0)),
                    always_include_touched_id=_coerce_bool(candidates_cfg.get("always_include_touched_id", True), default=True),
                    include_id_radius_fallback=_coerce_bool(candidates_cfg.get("include_id_radius_fallback", True), default=True),
                    missing_touched_id_fallback=str(candidates_cfg.get("missing_touched_id_fallback", "none")),
                    fallback_top_k=int(candidates_cfg.get("fallback_top_k", 3)),
                    fallback_max_distance_px=float(candidates_cfg.get("fallback_max_distance_px", 40.0)),
                    log_string_id_inconsistency=_coerce_bool(candidates_cfg.get("log_string_id_inconsistency", True), default=True),
                    adaptive_roi_enabled=_coerce_bool(roi_cfg.get("adaptive_enabled", False), default=False),
                    adaptive_height_ratio=float(roi_cfg.get("adaptive_height_ratio", 0.45)),
                    min_roi_h=int(roi_cfg.get("min_roi_h", 5)),
                    max_roi_h=int(roi_cfg.get("max_roi_h", 18)),
                    min_neighbor_distance_px=float(roi_cfg.get("min_neighbor_distance_px", 4.0)),
                    border_mode=str(roi_cfg.get("border_mode", "replicate")),
                    constant_border_value=int(roi_cfg.get("constant_border_value", 0)),
                    reject_if_out_of_frame=_coerce_bool(roi_cfg.get("reject_if_out_of_frame", False), default=False),
                    min_inside_fraction=float(roi_cfg.get("min_inside_fraction", 0.95)),
                    hand_mask_mode=str(masking_cfg.get("mode", "finger_point")),
                    contact_band_exclusion_px=float(masking_cfg.get("contact_band_exclusion_px", 10.0)),
                    mask_contact_region=_coerce_bool(masking_cfg.get("mask_contact_region", True), default=True),
                    allow_small_gaps=_coerce_bool(frame_transitions_cfg.get("allow_small_gaps", False), default=False),
                    max_gap_frames=int(frame_transitions_cfg.get("max_gap_frames", 2)),
                    normalize_by_gap=_coerce_bool(frame_transitions_cfg.get("normalize_by_gap", True), default=True),
                    normalization_mode=str(rules_cfg.get("normalization_mode", "zscore")),
                    min_scale=float(rules_cfg.get("min_scale", 0.05)),
                    mad_scale_factor=float(rules_cfg.get("mad_scale_factor", 1.4826)),
                    percentile_low=float(rules_cfg.get("percentile_low", 25)),
                    percentile_high=float(rules_cfg.get("percentile_high", 75)),
                    require_absolute_motion=_coerce_bool(rules_cfg.get("require_absolute_motion", False), default=False),
                    min_action_mean=float(rules_cfg.get("min_action_mean", 0.02)),
                    min_action_max=float(rules_cfg.get("min_action_max", 0.05)),
                    max_baseline_mean=(
                        None if rules_cfg.get("max_baseline_mean") is None else float(rules_cfg.get("max_baseline_mean"))
                    ),
                    thr_domination_ratio=float(rules_cfg.get("thr_domination_ratio", 1.2)),
                    thr_global_median_peak=float(global_shake_cfg.get("thr_global_median_peak", 5.0)),
                    thr_many_strings_vibrating=int(global_shake_cfg.get("thr_many_strings_vibrating", 6)),
                    shake_probe_count=int(shake_probe_count),
                    enable_finger_gate=_coerce_bool(finger_gate_cfg.get("enable_finger_gate", True), default=True),
                    thumb_gate=finger_gate_cfg.get("thumb_gate", {}),
                    index_gate=finger_gate_cfg.get("index_gate", {}),
                    stabilize_enabled=bool(stabilize_enabled),
                    event_velocity_stats_by_event=velocity_stats_by_event,
                )
                strike_results = [
                    _serialize_strike_result(item, include_debug=include_strike_debug)
                    for item in strike_objects
                ]
                strike_inference["processed_events"] = len(strike_results)
            else:
                strike_inference["processed_events"] = 0
                strike_inference["reason"] = "missing_strings_or_events"
            strike_elapsed = time.perf_counter() - strike_started_at
            strike_inference["elapsed_sec"] = round(float(strike_elapsed), 3)
            print(
                "[INFO] Strike inference finished: "
                f"processed_events={strike_inference.get('processed_events', 0)} "
                f"in {strike_inference['elapsed_sec']}s"
            )
        except Exception as exc:  # noqa: BLE001
            strike_inference["error"] = str(exc)
            strike_elapsed = time.perf_counter() - strike_started_at
            strike_inference["elapsed_sec"] = round(float(strike_elapsed), 3)
            print(
                "[WARN] Strike inference failed: "
                f"{exc} after {strike_inference['elapsed_sec']}s"
            )

    strike_highlight_info: dict[str, Any] | None = None
    # Strike-string highlighting is intentionally disabled.
    # if strike_results:
    #     strings_by_frame_path = result.get("strings_by_frame_jsonl_path")
    #     if isinstance(strings_by_frame_path, str) and strings_by_frame_path.strip():
    #         try:
    #             try:
    #                 from .post_processing import highlight_strikes_on_video
    #             except ImportError:
    #                 from post_processing import highlight_strikes_on_video
    #
    #             strings_path = Path(strings_by_frame_path)
    #             fps_for_highlight = float(result.get("fps") or strike_inference.get("fps_used") or 30.0)
    #             hold_frames = max(8, int(round(fps_for_highlight * 0.20)))
    #             highlight_out_path = out_video_path.with_name(
    #                 f"{out_video_path.stem}_strike{out_video_path.suffix}"
    #             )
    #             strike_highlight_info = highlight_strikes_on_video(
    #                 input_video_path=out_video_path,
    #                 output_video_path=highlight_out_path,
    #                 strings_by_frame_jsonl_path=strings_path,
    #                 strike_events=strike_results,
    #                 fps=fps_for_highlight,
    #                 hold_frames=hold_frames,
    #                 source_with_audio=upload_path,
    #                 transcode_output=True,
    #                 transcode_preset="veryfast",
    #             )
    #             if isinstance(strike_highlight_info, dict) and strike_highlight_info.get("ok"):
    #                 new_path = Path(str(strike_highlight_info.get("output_video_path")))
    #                 if new_path.exists():
    #                     out_video_path = new_path
    #                     result["out_video_path"] = str(new_path)
    #                     predicted_video_url = _predicted_video_url_for_path(base_url, new_path) or predicted_video_url
    #         except Exception as exc:  # noqa: BLE001
    #             strike_highlight_info = {"ok": False, "error": str(exc)}

    split_event_files = _save_split_event_jsons(
        touch_events_json_path=result.get("touch_events_json_path"),
        touch_events=result.get("touch_events", []),
        strike_results=strike_results,
        fps=float(result.get("fps") or 30.0),
        video_name=Path(str(result.get("source_video", upload_path))).name,
        frames_processed=int(result.get("frames_processed", 0)),
        strike_algorithm_applied=bool(
            enable_strike_decision and "error" not in strike_inference
        ),
        strike_algorithm_error=(
            None if "error" not in strike_inference else str(strike_inference.get("error"))
        ),
    )

    av_inference = _run_audio_fusion_postprocess_safe(
        upload_path=upload_path,
        result=result,
        split_event_files=split_event_files,
        fusion_mode=fusion_mode,
        audio_enabled=audio_enabled,
        audio_decision_mode=audio_decision_mode,
        audio_input_path=audio_input_path,
    )
    right_audio_decision_payload = _load_json_payload_optional(av_inference.get("right_audio_decision_events_json_path"))
    right_audio_strike_payload = _load_json_payload_optional(av_inference.get("right_audio_strike_events_json_path"))
    right_av_decision_payload = _load_json_payload_optional(av_inference.get("right_av_decision_events_json_path"))
    right_av_strike_payload = _load_json_payload_optional(av_inference.get("right_av_strike_events_json_path"))
    right_av_alternating_slots_payload = _load_json_payload_optional(
        av_inference.get("right_av_alternating_on_off_slots_json_path")
    )
    recognized_notes = [
        str(event.get("note_name"))
        for event in _events_from_payload(right_audio_strike_payload)
        if event.get("note_name")
    ]
    final_annotated_video_path = av_inference.get("annotated_video_path") or str(out_video_path)
    final_video_path = Path(str(final_annotated_video_path))

    overlay_info = av_inference.get("overlay_info") if isinstance(av_inference, dict) else None
    final_codec = str(result.get("final_codec") or "")
    final_transcoded = bool(result.get("transcoded", False))
    final_audio_muxed = bool(result.get("audio_muxed", False))
    final_has_audio_track = bool(result.get("has_audio_track", False))
    if isinstance(overlay_info, dict):
        final_transcoded = bool(overlay_info.get("transcoded", final_transcoded))
        final_audio_muxed = bool(overlay_info.get("audio_muxed", final_audio_muxed))

    fallback_transcode_elapsed_sec = 0.0
    if final_video_path.exists():
        detected_codec = detect_video_codec_tag(final_video_path)
        if detected_codec is not None:
            final_codec = detected_codec
        final_has_audio_track = bool(detect_audio_tag(final_video_path))

        if final_codec == "mp4v" and not final_transcoded:
            # Audio/AV postprocess may fail before the final overlay transcode runs.
            # Ensure the returned video is still browser-playable.
            t_fallback_transcode = time.perf_counter()
            final_video_path, final_transcoded, fallback_audio_muxed = transcode_to_h264(
                final_video_path,
                source_with_audio=upload_path,
                preset="veryfast",
            )
            fallback_transcode_elapsed_sec = round(float(time.perf_counter() - t_fallback_transcode), 3)
            final_audio_muxed = bool(final_audio_muxed or fallback_audio_muxed)
            final_annotated_video_path = str(final_video_path)
            detected_codec = detect_video_codec_tag(final_video_path)
            if detected_codec is not None:
                final_codec = detected_codec
            final_has_audio_track = bool(detect_audio_tag(final_video_path))

    if final_codec == "mp4v" and not final_transcoded:
        raise HTTPException(
            status_code=500,
            detail=(
                "Predicted video was encoded as mp4v, which is not browser-compatible in many clients. "
                "Install ffmpeg (or imageio-ffmpeg) on the backend and retry."
            ),
        )

    predicted_video_url = _predicted_video_url_for_path(base_url, final_annotated_video_path) or predicted_video_url

    pipeline_timings = {
        "video_stage_total_sec": float(result.get("elapsed_sec") or 0.0),
        "video_processing_sec": float(result.get("processing_elapsed_sec") or 0.0),
        "video_transcode_sec": float(result.get("transcode_elapsed_sec") or 0.0),
        "strike_stage_sec": float(strike_inference.get("elapsed_sec") or 0.0),
        "av_stage_sec": float(av_inference.get("elapsed_sec") or 0.0) if isinstance(av_inference, dict) else 0.0,
        "fallback_transcode_sec": float(fallback_transcode_elapsed_sec),
        "total_sec": round(float(time.perf_counter() - pipeline_started_at), 3),
    }

    run_profile = {
        "profile": "accurate",
        "string_infer_every_n": int(string_infer_every_n),
        "max_strike_events": int(max_strike_events),
        "strike_min_event_gap_frames": int(strike_min_event_gap_frames),
        "min_touch_confidence": float(ACCURATE_MODE_MIN_TOUCH_CONFIDENCE),
        "audio_decision_mode": audio_decision_mode,
        "debug_report_enabled": bool(enable_debug_report),
    }
    debug_artifacts: dict[str, Any] = {}
    if enable_debug_report:
        try:
            debug_artifacts = _write_analysis_debug_report(
                base_url=base_url,
                upload_path=upload_path,
                result=result,
                strike_inference=strike_inference,
                split_event_files=split_event_files,
                av_inference=av_inference,
                right_audio_decision_payload=right_audio_decision_payload,
                right_av_decision_payload=right_av_decision_payload,
                right_av_strike_payload=right_av_strike_payload,
                pipeline_timings=pipeline_timings,
                run_profile=run_profile,
            )
        except Exception as exc:  # noqa: BLE001
            debug_artifacts = {"analysis_debug_error": str(exc)}

    return {
        "predicted_video_url": predicted_video_url,
        "annotated_video_path": final_annotated_video_path,
        "ksy_notes": recognized_notes,
        "frames_processed": int(result.get("frames_processed", 0)),
        "video_codec": result.get("writer_codec"),
        "final_codec": final_codec,
        "transcoded": final_transcoded,
        "audio_muxed": final_audio_muxed,
        "input_has_audio_track": input_has_audio_track,
        "has_audio_track": final_has_audio_track,
        "yolo_runtime": result.get("yolo_runtime"),
        "hand_tracking_enabled": result.get("hand_tracking_enabled"),
        "hand_pipeline_enabled": result.get("hand_pipeline_enabled"),
        "hand_frames_detected": int(result.get("hand_frames_detected", 0)),
        "hand_fingertips_drawn": int(result.get("hand_fingertips_drawn", 0)),
        "touch_events_count": int(result.get("touch_events_count", 0)),
        "touch_events": result.get("touch_events", []),
        "touch_detection": result.get("touch_detection"),
        "touch_events_json_path": result.get("touch_events_json_path"),
        "strike_results_count": len(strike_results),
        "strike_results": strike_results,
        "strike_inference": strike_inference,
        "left_touch_events_count": int(split_event_files.get("left_touch_events_count", 0)),
        "right_decision_events_count": int(split_event_files.get("right_decision_events_count", 0)),
        "right_strike_events_count": int(split_event_files.get("right_strike_events_count", 0)),
        "left_touch_events_json_path": split_event_files.get("left_touch_events_json_path"),
        "right_decision_events_json_path": split_event_files.get("right_decision_events_json_path"),
        "right_strike_events_json_path": split_event_files.get("right_strike_events_json_path"),
        # Explicit video aliases (backward compatible additions).
        "right_video_decision_events_json_path": split_event_files.get("right_decision_events_json_path"),
        "right_video_strike_events_json_path": split_event_files.get("right_strike_events_json_path"),
        # New audio outputs.
        "right_audio_decision_events_json_path": av_inference.get("right_audio_decision_events_json_path"),
        "right_audio_strike_events_json_path": av_inference.get("right_audio_strike_events_json_path"),
        "right_audio_decision_events_count": int(av_inference.get("right_audio_decision_events_count", 0)),
        "right_audio_strike_events_count": int(av_inference.get("right_audio_strike_events_count", 0)),
        "right_audio_decision_events": _events_from_payload(right_audio_decision_payload),
        "right_audio_strike_events": _events_from_payload(right_audio_strike_payload),
        # New AV fusion outputs.
        "right_av_decision_events_json_path": av_inference.get("right_av_decision_events_json_path"),
        "right_av_strike_events_json_path": av_inference.get("right_av_strike_events_json_path"),
        "right_av_alternating_on_off_slots_json_path": av_inference.get("right_av_alternating_on_off_slots_json_path"),
        "right_av_decision_events_count": int(av_inference.get("right_av_decision_events_count", 0)),
        "right_av_strike_events_count": int(av_inference.get("right_av_strike_events_count", 0)),
        "right_av_alternating_on_off_slots_count": int(av_inference.get("right_av_alternating_on_off_slots_count", 0)),
        "right_av_decision_events": _events_from_payload(right_av_decision_payload),
        "right_av_strike_events": _events_from_payload(right_av_strike_payload),
        "right_av_alternating_on_off_slots": right_av_alternating_slots_payload,
        "av_inference": av_inference,
        "strike_highlight_info": strike_highlight_info,
        "performance_timings": pipeline_timings,
        "run_profile": run_profile,
        "pipeline_config_path": (str(PIPELINE_CONFIG_PATH) if PIPELINE_CONFIG_PATH.exists() else None),
        **debug_artifacts,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _prediction_request_options(
    *,
    expected_strings: int,
    enable_hand_tracking: bool,
    draw_hand_labels: bool,
    hand_pipeline_enabled: bool | None,
    enable_strike_decision: bool,
    max_strike_events: int,
    strike_min_event_gap_frames: int,
    include_strike_debug: bool,
    fusion_mode: str | None,
    audio_enabled: bool | None,
    audio_decision_mode: str | None,
    enable_debug_report: bool,
    audio_input: bool = False,
) -> dict[str, Any]:
    return {
        "expected_strings": int(expected_strings),
        "profile": "accurate",
        "string_infer_every_n": ACCURATE_MODE_STRING_INFER_EVERY_N,
        "enable_hand_tracking": bool(enable_hand_tracking),
        "draw_hand_labels": bool(draw_hand_labels),
        "hand_pipeline_enabled": hand_pipeline_enabled,
        "enable_strike_decision": bool(enable_strike_decision),
        "max_strike_events": int(max_strike_events),
        "strike_min_event_gap_frames": int(strike_min_event_gap_frames),
        "include_strike_debug": bool(include_strike_debug),
        "fusion_mode": fusion_mode,
        "audio_enabled": audio_enabled,
        "audio_decision_mode": audio_decision_mode,
        "enable_debug_report": bool(enable_debug_report),
        "audio_input": bool(audio_input),
    }


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with PREDICTION_JOBS_LOCK:
        job = PREDICTION_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Prediction job not found.")
        return dict(job)


def _update_prediction_job(job_id: str, **updates: Any) -> None:
    with PREDICTION_JOBS_LOCK:
        job = PREDICTION_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _prediction_job_worker(
    *,
    job_id: str,
    base_url: str,
    upload_path: Path,
    cache_key: str,
    cache_metadata: dict[str, Any],
    expected_strings: int,
    enable_hand_tracking: bool,
    draw_hand_labels: bool,
    hand_pipeline_enabled: bool | None,
    enable_strike_decision: bool,
    max_strike_events: int,
    strike_min_event_gap_frames: int,
    include_strike_debug: bool,
    fusion_mode: str | None,
    audio_enabled: bool | None,
    audio_decision_mode: str | None,
    enable_debug_report: bool,
) -> None:
    _update_prediction_job(job_id, status="processing", stage="running_prediction")
    started_at = time.perf_counter()
    try:
        response = _run_prediction_from_saved_video(
            base_url=base_url,
            upload_path=upload_path,
            expected_strings=expected_strings,
            enable_hand_tracking=enable_hand_tracking,
            draw_hand_labels=draw_hand_labels,
            hand_pipeline_enabled=hand_pipeline_enabled,
            enable_strike_decision=enable_strike_decision,
            max_strike_events=max_strike_events,
            strike_min_event_gap_frames=strike_min_event_gap_frames,
            include_strike_debug=include_strike_debug,
            fusion_mode=fusion_mode,
            audio_enabled=audio_enabled,
            audio_decision_mode=audio_decision_mode,
            enable_debug_report=enable_debug_report,
            audio_input_path=None,
        )
        response["cache_hit"] = False
        response["cache_key"] = cache_key
        _store_cached_response(cache_key, cache_metadata, response)
        _update_prediction_job(
            job_id,
            status="complete",
            stage="complete",
            result=response,
            elapsed_sec=round(float(time.perf_counter() - started_at), 3),
        )
    except HTTPException as exc:
        _update_prediction_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc.detail),
            elapsed_sec=round(float(time.perf_counter() - started_at), 3),
        )
    except Exception as exc:  # noqa: BLE001
        _update_prediction_job(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
            elapsed_sec=round(float(time.perf_counter() - started_at), 3),
        )


@app.get("/api/jobs/{job_id}")
def get_prediction_job(job_id: str) -> dict[str, Any]:
    return _job_snapshot(job_id)


@app.post("/api/jobs/predict-video")
async def start_predict_video_job(
    request: Request,
    expected_strings: int = 16,
    enable_hand_tracking: bool = True,
    draw_hand_labels: bool = False,
    hand_pipeline_enabled: bool | None = None,
    enable_strike_decision: bool = True,
    max_strike_events: int = 400,
    strike_min_event_gap_frames: int = 4,
    include_strike_debug: bool = False,
    fusion_mode: str | None = None,
    audio_enabled: bool | None = None,
    audio_decision_mode: str | None = None,
    enable_debug_report: bool = True,
) -> dict[str, Any]:
    if not BEST_MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model not found: {BEST_MODEL_PATH}")

    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.startswith("video/") and "application/octet-stream" not in content_type:
        raise HTTPException(status_code=400, detail="Only video uploads are supported.")

    file_name = request.headers.get("x-file-name", "upload.mp4")
    extension = Path(file_name).suffix.lower() or ".mp4"
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty request body.")

    base_url = str(request.base_url).rstrip("/")
    request_options = _prediction_request_options(
        expected_strings=expected_strings,
        enable_hand_tracking=enable_hand_tracking,
        draw_hand_labels=draw_hand_labels,
        hand_pipeline_enabled=hand_pipeline_enabled,
        enable_strike_decision=enable_strike_decision,
        max_strike_events=max_strike_events,
        strike_min_event_gap_frames=strike_min_event_gap_frames,
        include_strike_debug=include_strike_debug,
        fusion_mode=fusion_mode,
        audio_enabled=audio_enabled,
        audio_decision_mode=audio_decision_mode,
        enable_debug_report=enable_debug_report,
        audio_input=False,
    )
    cache_key, cache_metadata = _build_request_cache_key(
        video_bytes=payload,
        audio_bytes=None,
        request_options=request_options,
    )
    job_id = uuid4().hex
    now = time.time()

    cached_response = _load_cached_response(base_url, cache_key)
    if cached_response is not None:
        with PREDICTION_JOBS_LOCK:
            PREDICTION_JOBS[job_id] = {
                "job_id": job_id,
                "status": "complete",
                "stage": "cache_hit",
                "file_name": file_name,
                "cache_key": cache_key,
                "cache_hit": True,
                "created_at": now,
                "updated_at": now,
                "result": cached_response,
            }
        return _job_snapshot(job_id)

    upload_path = UPLOAD_DIR / f"{job_id}{extension}"
    with upload_path.open("wb") as buffer:
        buffer.write(payload)

    with PREDICTION_JOBS_LOCK:
        PREDICTION_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "file_name": file_name,
            "cache_key": cache_key,
            "cache_hit": False,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

    worker = Thread(
        target=_prediction_job_worker,
        kwargs={
            "job_id": job_id,
            "base_url": base_url,
            "upload_path": upload_path,
            "cache_key": cache_key,
            "cache_metadata": cache_metadata,
            "expected_strings": expected_strings,
            "enable_hand_tracking": enable_hand_tracking,
            "draw_hand_labels": draw_hand_labels,
            "hand_pipeline_enabled": hand_pipeline_enabled,
            "enable_strike_decision": enable_strike_decision,
            "max_strike_events": max_strike_events,
            "strike_min_event_gap_frames": strike_min_event_gap_frames,
            "include_strike_debug": include_strike_debug,
            "fusion_mode": fusion_mode,
            "audio_enabled": audio_enabled,
            "audio_decision_mode": audio_decision_mode,
            "enable_debug_report": enable_debug_report,
        },
        daemon=True,
    )
    worker.start()
    return _job_snapshot(job_id)


@app.post("/api/predict-video")
async def predict_video(
    request: Request,
    expected_strings: int = 16,
    enable_hand_tracking: bool = True,
    draw_hand_labels: bool = False,
    hand_pipeline_enabled: bool | None = None,
    enable_strike_decision: bool = True,
    max_strike_events: int = 400,
    strike_min_event_gap_frames: int = 4,
    include_strike_debug: bool = False,
    fusion_mode: str | None = None,
    audio_enabled: bool | None = None,
    audio_decision_mode: str | None = None,
    enable_debug_report: bool = True,
) -> dict[str, object]:
    if not BEST_MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model not found: {BEST_MODEL_PATH}",
        )

    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.startswith("video/") and "application/octet-stream" not in content_type:
        raise HTTPException(status_code=400, detail="Only video uploads are supported.")

    file_name = request.headers.get("x-file-name", "upload.mp4")
    extension = Path(file_name).suffix.lower()
    if extension == "":
        extension = ".mp4"

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty request body.")

    base_url = str(request.base_url).rstrip("/")
    request_options = {
        "expected_strings": int(expected_strings),
        "profile": "accurate",
        "string_infer_every_n": ACCURATE_MODE_STRING_INFER_EVERY_N,
        "enable_hand_tracking": bool(enable_hand_tracking),
        "draw_hand_labels": bool(draw_hand_labels),
        "hand_pipeline_enabled": hand_pipeline_enabled,
        "enable_strike_decision": bool(enable_strike_decision),
        "max_strike_events": int(max_strike_events),
        "strike_min_event_gap_frames": int(strike_min_event_gap_frames),
        "include_strike_debug": bool(include_strike_debug),
        "fusion_mode": fusion_mode,
        "audio_enabled": audio_enabled,
        "audio_decision_mode": audio_decision_mode,
        "enable_debug_report": bool(enable_debug_report),
        "audio_input": False,
    }
    cache_key, cache_metadata = _build_request_cache_key(
        video_bytes=payload,
        audio_bytes=None,
        request_options=request_options,
    )
    cached_response = _load_cached_response(base_url, cache_key)
    if cached_response is not None:
        if CACHE_HIT_DELAY_SEC > 0:
            await asyncio.sleep(CACHE_HIT_DELAY_SEC)
        return cached_response

    upload_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    with upload_path.open("wb") as buffer:
        buffer.write(payload)

    response = _run_prediction_from_saved_video(
        base_url=base_url,
        upload_path=upload_path,
        expected_strings=expected_strings,
        enable_hand_tracking=enable_hand_tracking,
        draw_hand_labels=draw_hand_labels,
        hand_pipeline_enabled=hand_pipeline_enabled,
        enable_strike_decision=enable_strike_decision,
        max_strike_events=max_strike_events,
        strike_min_event_gap_frames=strike_min_event_gap_frames,
        include_strike_debug=include_strike_debug,
        fusion_mode=fusion_mode,
        audio_enabled=audio_enabled,
        audio_decision_mode=audio_decision_mode,
        enable_debug_report=enable_debug_report,
        audio_input_path=None,
    )
    response["cache_hit"] = False
    response["cache_key"] = cache_key
    _store_cached_response(cache_key, cache_metadata, response)
    return response

    try:
        try:
            from .post_processing import run_video_predict
        except ImportError:
            from post_processing import run_video_predict

        result = run_video_predict(
            tag="best",
            model_path=BEST_MODEL_PATH,
            video_path=upload_path,
            expected_strings=expected_strings,
            save_video=True,
            show_preview=False,
            enable_hand_tracking=enable_hand_tracking,
            draw_hand_labels=draw_hand_labels,
            hand_pipeline_enabled=hand_pipeline_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    out_video_path = result.get("out_video_path")
    if out_video_path is None:
        raise HTTPException(status_code=500, detail="Prediction completed without an output video.")

    out_video_path = Path(out_video_path)
    if not out_video_path.exists():
        raise HTTPException(status_code=500, detail=f"Output video missing: {out_video_path}")

    try:
        relative_output_path = out_video_path.relative_to(PREDICTIONS_DIR).as_posix()
    except ValueError:
        relative_output_path = out_video_path.name

    final_codec = str(result.get("final_codec") or "")
    transcoded = bool(result.get("transcoded", False))
    if final_codec == "mp4v" and not transcoded:
        raise HTTPException(
            status_code=500,
            detail=(
                "Predicted video was encoded as mp4v, which is not browser-compatible in many clients. "
                "Install ffmpeg (or imageio-ffmpeg) on the backend and retry."
            ),
        )

    base_url = str(request.base_url).rstrip("/")
    predicted_video_url = f"{base_url}/predictions/{relative_output_path}"

    strike_results: list[dict[str, Any]] = []
    strike_inference: dict[str, Any] = {
        "enabled": bool(enable_strike_decision),
        "config_path": str(STRIKE_CONFIG_PATH),
    }
    if enable_strike_decision:
        strike_started_at = time.perf_counter()
        try:
            decide_touch_events, import_err = _import_strike_symbol(
                "saung_strike_video_farneback_rules.src.decision",
                "decide_touch_events",
            )
            if decide_touch_events is None:
                raise RuntimeError(f"Cannot import decide_touch_events: {import_err}")

            strike_cfg = _load_yaml_optional(STRIKE_CONFIG_PATH)
            windows_cfg = strike_cfg.get("windows", {}) if isinstance(strike_cfg.get("windows"), dict) else {}
            candidates_cfg = strike_cfg.get("candidates", {}) if isinstance(strike_cfg.get("candidates"), dict) else {}
            roi_cfg = strike_cfg.get("ROI", {}) if isinstance(strike_cfg.get("ROI"), dict) else {}
            farneback_cfg = strike_cfg.get("farneback_params", {}) if isinstance(strike_cfg.get("farneback_params"), dict) else {}
            vibration_cfg = strike_cfg.get("vibration", {}) if isinstance(strike_cfg.get("vibration"), dict) else {}
            rules_cfg = strike_cfg.get("rules", {}) if isinstance(strike_cfg.get("rules"), dict) else {}
            global_shake_cfg = strike_cfg.get("global_shake", {}) if isinstance(strike_cfg.get("global_shake"), dict) else {}
            finger_gate_cfg = strike_cfg.get("finger_gating", {}) if isinstance(strike_cfg.get("finger_gating"), dict) else {}
            masking_cfg = strike_cfg.get("masking", {}) if isinstance(strike_cfg.get("masking"), dict) else {}
            frame_transitions_cfg = strike_cfg.get("frame_transitions", {}) if isinstance(strike_cfg.get("frame_transitions"), dict) else {}
            decision_cfg = strike_cfg.get("decision", {}) if isinstance(strike_cfg.get("decision"), dict) else {}
            stabilization_cfg = strike_cfg.get("stabilization", {}) if isinstance(strike_cfg.get("stabilization"), dict) else {}
            right_finger_types_cfg = decision_cfg.get("right_finger_types")
            if isinstance(right_finger_types_cfg, list):
                allowed_finger_types = {
                    str(v).strip().lower()
                    for v in right_finger_types_cfg
                    if str(v).strip()
                }
            else:
                allowed_finger_types = {"thumb", "index"}
            if not allowed_finger_types:
                allowed_finger_types = {"thumb", "index"}
            dy_thr = float(
                vibration_cfg.get(
                    "dy_thr",
                    strike_cfg.get("dy_thr", farneback_cfg.get("dy_thr", 0.40)),
                )
            )
            stabilize_enabled = _coerce_bool(
                stabilization_cfg.get(
                    "enabled",
                    decision_cfg.get("stabilize_enabled", strike_cfg.get("stabilize_enabled", True)),
                ),
                default=True,
            )
            fps_used = float(result.get("fps") or strike_cfg.get("fps") or 30.0)
            baseline_len = max(1, int(round(float(windows_cfg.get("baseline_sec", 0.25)) * fps_used)))
            action_len = max(1, int(round(float(windows_cfg.get("action_sec", 0.25)) * fps_used)))
            action_start_offset = int(windows_cfg.get("action_start_frame_offset", 1))
            shake_probe_count = int(global_shake_cfg.get("shake_probe_count", 6))
            shake_probe_count = min(shake_probe_count, ACCURATE_MODE_SHAKE_PROBE_COUNT)

            strings_for_decision = _build_string_geometries_for_decision(result.get("string_geometries", []))
            touch_events_for_decision, touch_events_debug = _build_touch_events_for_decision(
                result.get("touch_events", []),
                fps=fps_used,
                max_events=max(1, int(max_strike_events)),
                min_gap_frames=max(1, int(strike_min_event_gap_frames)),
                allowed_finger_types=allowed_finger_types,
                min_touch_confidence=ACCURATE_MODE_MIN_TOUCH_CONFIDENCE,
            )
            velocity_stats_by_event = _build_event_velocity_stats_by_event(touch_events_for_decision, fps=fps_used)

            strike_inference.update(
                {
                    "fps_used": fps_used,
                    "baseline_len": baseline_len,
                    "action_len": action_len,
                    "action_start_frame_offset": action_start_offset,
                    "dy_thr": float(dy_thr),
                    "stabilize_enabled": bool(stabilize_enabled),
                    "shake_probe_count": int(shake_probe_count),
                    "touch_events_debug": touch_events_debug,
                    "strings_for_decision": len(strings_for_decision),
                }
            )
            strike_inference["events_for_decision"] = len(touch_events_for_decision)
            print(
                "[INFO] Strike inference started: "
                f"events={len(touch_events_for_decision)}, "
                f"strings={len(strings_for_decision)}, "
                f"shake_probe_count={shake_probe_count}, "
                f"stabilize={bool(stabilize_enabled)}"
            )

            if strings_for_decision and touch_events_for_decision:
                strike_objects = decide_touch_events(
                    video_path=upload_path,
                    touch_events=touch_events_for_decision,
                    strings=strings_for_decision,
                    fps=fps_used,
                    baseline_len=baseline_len,
                    action_len=action_len,
                    action_start_frame_offset=action_start_offset,
                    roi_w=int(roi_cfg.get("roi_w", 160)),
                    roi_h=int(roi_cfg.get("roi_h", 32)),
                    trim_ends_ratio=float(roi_cfg.get("trim_ends_ratio", 0.15)),
                    center_band_h=int(roi_cfg.get("center_band_h", 10)),
                    candidate_radius_default=int(candidates_cfg.get("candidate_radius_default", 2)),
                    candidate_radius_close_contact=int(candidates_cfg.get("candidate_radius_close_contact", 1)),
                    contact_dist_px_thr=float(candidates_cfg.get("contact_dist_px_thr", 8.0)),
                    enable_hand_mask=_coerce_bool(masking_cfg.get("enable_hand_mask", True), default=True),
                    hand_mask_expand_px=float(masking_cfg.get("hand_mask_expand_px", 8.0)),
                    farneback_params=farneback_cfg,
                    dy_thr=float(dy_thr),
                    z_thr=float(rules_cfg.get("z_thr", 2.35)),
                    thr_peak=float(rules_cfg.get("thr_peak", 4.0)),
                    thr_duration_frames=int(rules_cfg.get("thr_duration_frames", 2)),
                    thr_impulse=float(rules_cfg.get("thr_impulse", 7.0)),
                    baseline_gap_frames=int(windows_cfg.get("baseline_gap_frames", 0)),
                    dynamic_action_enabled=_coerce_bool(windows_cfg.get("dynamic_action_enabled", False), default=False),
                    dynamic_offsets=windows_cfg.get("dynamic_offsets"),
                    dynamic_select_metric=str(windows_cfg.get("dynamic_select_metric", "max_impulse")),
                    geometry_enabled=_coerce_bool(candidates_cfg.get("geometry_enabled", False), default=False),
                    geometry_top_k=int(candidates_cfg.get("geometry_top_k", 5)),
                    geometry_max_distance_px=float(candidates_cfg.get("geometry_max_distance_px", 35.0)),
                    always_include_touched_id=_coerce_bool(candidates_cfg.get("always_include_touched_id", True), default=True),
                    include_id_radius_fallback=_coerce_bool(candidates_cfg.get("include_id_radius_fallback", True), default=True),
                    missing_touched_id_fallback=str(candidates_cfg.get("missing_touched_id_fallback", "none")),
                    fallback_top_k=int(candidates_cfg.get("fallback_top_k", 3)),
                    fallback_max_distance_px=float(candidates_cfg.get("fallback_max_distance_px", 40.0)),
                    log_string_id_inconsistency=_coerce_bool(candidates_cfg.get("log_string_id_inconsistency", True), default=True),
                    adaptive_roi_enabled=_coerce_bool(roi_cfg.get("adaptive_enabled", False), default=False),
                    adaptive_height_ratio=float(roi_cfg.get("adaptive_height_ratio", 0.45)),
                    min_roi_h=int(roi_cfg.get("min_roi_h", 5)),
                    max_roi_h=int(roi_cfg.get("max_roi_h", 18)),
                    min_neighbor_distance_px=float(roi_cfg.get("min_neighbor_distance_px", 4.0)),
                    border_mode=str(roi_cfg.get("border_mode", "replicate")),
                    constant_border_value=int(roi_cfg.get("constant_border_value", 0)),
                    reject_if_out_of_frame=_coerce_bool(roi_cfg.get("reject_if_out_of_frame", False), default=False),
                    min_inside_fraction=float(roi_cfg.get("min_inside_fraction", 0.95)),
                    hand_mask_mode=str(masking_cfg.get("mode", "finger_point")),
                    contact_band_exclusion_px=float(masking_cfg.get("contact_band_exclusion_px", 10.0)),
                    mask_contact_region=_coerce_bool(masking_cfg.get("mask_contact_region", True), default=True),
                    allow_small_gaps=_coerce_bool(frame_transitions_cfg.get("allow_small_gaps", False), default=False),
                    max_gap_frames=int(frame_transitions_cfg.get("max_gap_frames", 2)),
                    normalize_by_gap=_coerce_bool(frame_transitions_cfg.get("normalize_by_gap", True), default=True),
                    normalization_mode=str(rules_cfg.get("normalization_mode", "zscore")),
                    min_scale=float(rules_cfg.get("min_scale", 0.05)),
                    mad_scale_factor=float(rules_cfg.get("mad_scale_factor", 1.4826)),
                    percentile_low=float(rules_cfg.get("percentile_low", 25)),
                    percentile_high=float(rules_cfg.get("percentile_high", 75)),
                    require_absolute_motion=_coerce_bool(rules_cfg.get("require_absolute_motion", False), default=False),
                    min_action_mean=float(rules_cfg.get("min_action_mean", 0.02)),
                    min_action_max=float(rules_cfg.get("min_action_max", 0.05)),
                    max_baseline_mean=(
                        None if rules_cfg.get("max_baseline_mean") is None else float(rules_cfg.get("max_baseline_mean"))
                    ),
                    thr_domination_ratio=float(rules_cfg.get("thr_domination_ratio", 1.2)),
                    thr_global_median_peak=float(global_shake_cfg.get("thr_global_median_peak", 5.0)),
                    thr_many_strings_vibrating=int(global_shake_cfg.get("thr_many_strings_vibrating", 6)),
                    shake_probe_count=int(shake_probe_count),
                    enable_finger_gate=_coerce_bool(finger_gate_cfg.get("enable_finger_gate", True), default=True),
                    thumb_gate=finger_gate_cfg.get("thumb_gate", {}),
                    index_gate=finger_gate_cfg.get("index_gate", {}),
                    stabilize_enabled=bool(stabilize_enabled),
                    event_velocity_stats_by_event=velocity_stats_by_event,
                )
                strike_results = [
                    _serialize_strike_result(item, include_debug=include_strike_debug)
                    for item in strike_objects
                ]
                strike_inference["processed_events"] = len(strike_results)
            else:
                strike_inference["processed_events"] = 0
                strike_inference["reason"] = "missing_strings_or_events"
            strike_elapsed = time.perf_counter() - strike_started_at
            strike_inference["elapsed_sec"] = round(float(strike_elapsed), 3)
            print(
                "[INFO] Strike inference finished: "
                f"processed_events={strike_inference.get('processed_events', 0)} "
                f"in {strike_inference['elapsed_sec']}s"
            )
        except Exception as exc:  # noqa: BLE001
            strike_inference["error"] = str(exc)
            strike_elapsed = time.perf_counter() - strike_started_at
            strike_inference["elapsed_sec"] = round(float(strike_elapsed), 3)
            print(
                "[WARN] Strike inference failed: "
                f"{exc} after {strike_inference['elapsed_sec']}s"
            )

    split_event_files = _save_split_event_jsons(
        touch_events_json_path=result.get("touch_events_json_path"),
        touch_events=result.get("touch_events", []),
        strike_results=strike_results,
        fps=float(result.get("fps") or 30.0),
        video_name=Path(str(result.get("source_video", upload_path))).name,
        frames_processed=int(result.get("frames_processed", 0)),
        strike_algorithm_applied=bool(
            enable_strike_decision and "error" not in strike_inference
        ),
        strike_algorithm_error=(
            None if "error" not in strike_inference else str(strike_inference.get("error"))
        ),
    )

    return {
        "predicted_video_url": predicted_video_url,
        "ksy_notes": [],
        "frames_processed": int(result.get("frames_processed", 0)),
        "video_codec": result.get("writer_codec"),
        "final_codec": result.get("final_codec"),
        "transcoded": result.get("transcoded"),
        "audio_muxed": result.get("audio_muxed"),
        "has_audio_track": result.get("has_audio_track"),
        "hand_tracking_enabled": result.get("hand_tracking_enabled"),
        "hand_pipeline_enabled": result.get("hand_pipeline_enabled"),
        "hand_frames_detected": int(result.get("hand_frames_detected", 0)),
        "hand_fingertips_drawn": int(result.get("hand_fingertips_drawn", 0)),
        "touch_events_count": int(result.get("touch_events_count", 0)),
        "touch_events": result.get("touch_events", []),
        "touch_detection": result.get("touch_detection"),
        "touch_events_json_path": result.get("touch_events_json_path"),
        "strike_results_count": len(strike_results),
        "strike_results": strike_results,
        "strike_inference": strike_inference,
        "left_touch_events_count": int(split_event_files.get("left_touch_events_count", 0)),
        "right_decision_events_count": int(split_event_files.get("right_decision_events_count", 0)),
        "right_strike_events_count": int(split_event_files.get("right_strike_events_count", 0)),
        "left_touch_events_json_path": split_event_files.get("left_touch_events_json_path"),
        "right_decision_events_json_path": split_event_files.get("right_decision_events_json_path"),
        "right_strike_events_json_path": split_event_files.get("right_strike_events_json_path"),
    }


@app.post("/predict")
async def predict_multipart(
    request: Request,
) -> dict[str, object]:
    if not BEST_MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model not found: {BEST_MODEL_PATH}")

    content_type_header = str(request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type_header:
        raise HTTPException(
            status_code=400,
            detail="`/predict` expects multipart/form-data. Use `/api/predict-video` for raw video body uploads.",
        )

    try:
        form = await request.form()
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            raise HTTPException(
                status_code=500,
                detail='Multipart support not installed. Run: pip install python-multipart',
            ) from exc
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

    def _get_upload(*keys: str):
        for key in keys:
            v = form.get(key)
            if v is not None:
                return v
        return None

    def _get_text(name: str, default: str | None = None) -> str | None:
        v = form.get(name, default)
        if v is None:
            return None
        return str(v)

    def _get_int(name: str, default: int) -> int:
        v = form.get(name, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def _get_opt_int(name: str) -> int | None:
        v = form.get(name)
        if v is None or str(v).strip() == "":
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _get_opt_bool(name: str) -> bool | None:
        v = form.get(name)
        if v is None or str(v).strip() == "":
            return None
        return _coerce_bool(v, default=False)

    video_file = _get_upload("video_file", "video", "file")
    if video_file is None:
        raise HTTPException(status_code=400, detail="Missing multipart file field `video_file` (also accepts `video` or `file`).")

    expected_strings = _get_int("expected_strings", 16)
    enable_hand_tracking = _coerce_bool(form.get("enable_hand_tracking", True), default=True)
    draw_hand_labels = _coerce_bool(form.get("draw_hand_labels", False), default=False)
    hand_pipeline_enabled = _get_opt_bool("hand_pipeline_enabled")
    enable_strike_decision = _coerce_bool(form.get("enable_strike_decision", True), default=True)
    max_strike_events = _get_int("max_strike_events", 400)
    strike_min_event_gap_frames = _get_int("strike_min_event_gap_frames", 4)
    include_strike_debug = _coerce_bool(form.get("include_strike_debug", False), default=False)
    fusion_mode = _get_text("fusion_mode", None)
    audio_enabled = _get_opt_bool("audio_enabled")
    audio_file = _get_upload("audio_file", "audio")

    content_type = str(getattr(video_file, "content_type", "") or "")
    if content_type and not content_type.startswith("video/") and "application/octet-stream" not in content_type:
        raise HTTPException(status_code=400, detail=f"Invalid video content type: {content_type}")

    video_bytes = await video_file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video upload.")

    audio_bytes: bytes | None = None
    audio_name = "audio.wav"
    audio_ext = ".wav"
    if audio_file is not None:
        audio_name = str(getattr(audio_file, "filename", None) or "audio.wav")
        audio_ext = Path(audio_name).suffix.lower() or ".wav"
        audio_bytes = await audio_file.read()
        if audio_bytes is not None and len(audio_bytes) == 0:
            audio_bytes = None

    base_url = str(request.base_url).rstrip("/")
    request_options = {
        "expected_strings": int(expected_strings),
        "profile": "accurate",
        "string_infer_every_n": ACCURATE_MODE_STRING_INFER_EVERY_N,
        "enable_hand_tracking": bool(enable_hand_tracking),
        "draw_hand_labels": bool(draw_hand_labels),
        "hand_pipeline_enabled": hand_pipeline_enabled,
        "enable_strike_decision": bool(enable_strike_decision),
        "max_strike_events": int(max_strike_events),
        "strike_min_event_gap_frames": int(strike_min_event_gap_frames),
        "include_strike_debug": bool(include_strike_debug),
        "fusion_mode": fusion_mode,
        "audio_enabled": audio_enabled,
        "audio_input": bool(audio_bytes is not None),
    }
    cache_key, cache_metadata = _build_request_cache_key(
        video_bytes=video_bytes,
        audio_bytes=audio_bytes,
        request_options=request_options,
    )
    cached_response = _load_cached_response(base_url, cache_key)
    if cached_response is not None:
        if CACHE_HIT_DELAY_SEC > 0:
            await asyncio.sleep(CACHE_HIT_DELAY_SEC)
        return cached_response

    extension = Path(str(getattr(video_file, "filename", None) or "upload.mp4")).suffix.lower() or ".mp4"
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    with upload_path.open("wb") as f:
        f.write(video_bytes)

    audio_path: Path | None = None
    if audio_bytes is not None:
        audio_path = UPLOAD_DIR / f"{uuid4().hex}{audio_ext}"
        with audio_path.open("wb") as f:
            f.write(audio_bytes)

    response = _run_prediction_from_saved_video(
        base_url=base_url,
        upload_path=upload_path,
        expected_strings=int(expected_strings),
        enable_hand_tracking=bool(enable_hand_tracking),
        draw_hand_labels=bool(draw_hand_labels),
        hand_pipeline_enabled=hand_pipeline_enabled,
        enable_strike_decision=bool(enable_strike_decision),
        max_strike_events=int(max_strike_events),
        strike_min_event_gap_frames=int(strike_min_event_gap_frames),
        include_strike_debug=bool(include_strike_debug),
        fusion_mode=fusion_mode,
        audio_enabled=audio_enabled,
        audio_input_path=audio_path,
    )

    response["audio_input_upload_path"] = str(audio_path) if audio_path is not None else None
    response["cache_hit"] = False
    response["cache_key"] = cache_key
    _store_cached_response(cache_key, cache_metadata, response)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
