from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import csv
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from .farneback import DEFAULT_DY_THR, compute_vibration_frame
    from .mask import create_roi_mask
    from .roi import extract_rotated_roi
    from .strings import (
        StringGeometry,
        closest_point_and_distance_px,
        estimate_neighbor_spacing_px,
    )
    from .video_reader import VideoReader
except ImportError:  # pragma: no cover
    from src.farneback import DEFAULT_DY_THR, compute_vibration_frame
    from src.mask import create_roi_mask
    from src.roi import extract_rotated_roi
    from src.strings import (
        StringGeometry,
        closest_point_and_distance_px,
        estimate_neighbor_spacing_px,
    )
    from src.video_reader import VideoReader


REQUIRED_TOUCH_COLUMNS = [
    "timestamp_sec",
    "hand_side",
    "finger_type",
    "touched_string_id",
    "touch_conf",
    "contact_x",
    "contact_y",
    "finger_x",
    "finger_y",
]

ALLOWED_HAND_SIDE = {"right"}
ALLOWED_FINGER_TYPES = {"thumb", "index"}
FRAME_CACHE_MAX_FRAMES = max(0, int(os.getenv("SAUNG_FRAME_CACHE_MAX_FRAMES", "256")))
_FRAME_CACHE: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
_FRAME_CACHE_LOCK = Lock()


def _frame_cache_get(video_id: str, frame_idx: int) -> np.ndarray | None:
    if FRAME_CACHE_MAX_FRAMES <= 0:
        return None
    key = (video_id, int(frame_idx))
    with _FRAME_CACHE_LOCK:
        frame = _FRAME_CACHE.get(key)
        if frame is None:
            return None
        _FRAME_CACHE.move_to_end(key)
        return frame.copy()


def _frame_cache_put(video_id: str, frame_idx: int, frame: np.ndarray) -> None:
    if FRAME_CACHE_MAX_FRAMES <= 0:
        return
    key = (video_id, int(frame_idx))
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = frame.copy()
        _FRAME_CACHE.move_to_end(key)
        while len(_FRAME_CACHE) > FRAME_CACHE_MAX_FRAMES:
            _FRAME_CACHE.popitem(last=False)


@dataclass(frozen=True)
class TouchEvent:
    timestamp_sec: float
    hand_side: str
    finger_type: str
    touched_string_id: int
    touch_conf: float
    contact_x: float
    contact_y: float
    finger_x: float | None
    finger_y: float | None
    row_index: int
    hand_bbox_x1: float | None = None
    hand_bbox_y1: float | None = None
    hand_bbox_x2: float | None = None
    hand_bbox_y2: float | None = None


@dataclass
class CandidateScoreSeries:
    candidate_string_id: int
    frame_indices: list[int]
    score_seq: list[float]
    baseline_seq: list[float]
    action_seq: list[float]
    score_by_frame: dict[int, float]
    action_frames: list[int]
    action_windows: dict[int, list[int]] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventWindowResult:
    event: TouchEvent
    f0: int
    baseline_frames: list[int]
    action_frames: list[int]
    candidates: list[int]
    candidate_results: dict[int, CandidateScoreSeries]
    frames_for_event: dict[int, np.ndarray] | None
    debug: dict[str, Any]


def _parse_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid float for {field_name}: {value!r}") from exc


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_int_like(value: Any, field_name: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Missing integer value for {field_name}.")
    try:
        f = float(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid integer-like value for {field_name}: {value!r}") from exc
    i = int(round(f))
    if abs(f - i) > 1e-6:
        raise ValueError(f"Non-integer value for {field_name}: {value!r}")
    return i


def parse_touch_events_csv(path: str | Path) -> list[TouchEvent]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"touch_events.csv not found: {csv_path}")

    events: list[TouchEvent] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = [col for col in REQUIRED_TOUCH_COLUMNS if col not in headers]
        if missing:
            raise ValueError(f"touch_events.csv missing columns: {missing}")

        for row_index, row in enumerate(reader, start=2):
            hand_side = str(row.get("hand_side", "")).strip().lower()
            finger_type = str(row.get("finger_type", "")).strip().lower()
            if hand_side not in ALLOWED_HAND_SIDE:
                continue
            if finger_type not in ALLOWED_FINGER_TYPES:
                continue

            event = TouchEvent(
                timestamp_sec=_parse_float(row.get("timestamp_sec"), "timestamp_sec"),
                hand_side=hand_side,
                finger_type=finger_type,
                touched_string_id=_parse_int_like(row.get("touched_string_id"), "touched_string_id"),
                touch_conf=_parse_float(row.get("touch_conf"), "touch_conf"),
                contact_x=_parse_float(row.get("contact_x"), "contact_x"),
                contact_y=_parse_float(row.get("contact_y"), "contact_y"),
                finger_x=_parse_optional_float(row.get("finger_x")),
                finger_y=_parse_optional_float(row.get("finger_y")),
                row_index=row_index,
                hand_bbox_x1=_parse_optional_float(row.get("hand_bbox_x1")),
                hand_bbox_y1=_parse_optional_float(row.get("hand_bbox_y1")),
                hand_bbox_x2=_parse_optional_float(row.get("hand_bbox_x2")),
                hand_bbox_y2=_parse_optional_float(row.get("hand_bbox_y2")),
            )
            events.append(event)
    return events


def event_frame_index(timestamp_sec: float, fps: float) -> int:
    return int(round(float(timestamp_sec) * float(fps)))


def build_frame_windows(
    *,
    f0: int,
    baseline_len: int,
    action_len: int,
    action_start_frame_offset: int,
    baseline_gap_frames: int = 0,
    max_frame_index: int | None = None,
) -> tuple[list[int], list[int]]:
    if baseline_len < 0:
        raise ValueError("baseline_len must be >= 0.")
    if action_len < 0:
        raise ValueError("action_len must be >= 0.")

    baseline_end = f0 - int(baseline_gap_frames)
    baseline_start = baseline_end - int(baseline_len)
    baseline_frames = [f for f in range(baseline_start, baseline_end)]

    action_start = f0 + int(action_start_frame_offset)
    action_end_inclusive = f0 + int(action_len)
    action_frames: list[int] = []
    if action_end_inclusive >= action_start:
        action_frames = [f for f in range(action_start, action_end_inclusive + 1)]

    if max_frame_index is not None:
        baseline_frames = [f for f in baseline_frames if 0 <= f <= max_frame_index]
        action_frames = [f for f in action_frames if 0 <= f <= max_frame_index]
    else:
        baseline_frames = [f for f in baseline_frames if f >= 0]
        action_frames = [f for f in action_frames if f >= 0]

    return baseline_frames, action_frames


def build_dynamic_action_windows(
    *,
    f0: int,
    baseline_len: int,
    action_len: int,
    candidate_offsets: list[int],
    baseline_gap: int = 0,
    max_frame_index: int | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
    baseline_end = f0 - int(baseline_gap)
    baseline_start = baseline_end - int(baseline_len)
    baseline_frames = [f for f in range(baseline_start, baseline_end)]
    if max_frame_index is not None:
        baseline_frames = [f for f in baseline_frames if 0 <= f <= max_frame_index]
    else:
        baseline_frames = [f for f in baseline_frames if f >= 0]

    out: dict[int, list[int]] = {}
    for raw_offset in candidate_offsets:
        offset = int(raw_offset)
        start = f0 + offset
        frames = [start + i for i in range(max(0, int(action_len)))]
        if max_frame_index is not None:
            frames = [f for f in frames if 0 <= f <= max_frame_index]
        else:
            frames = [f for f in frames if f >= 0]
        out[offset] = frames
    return baseline_frames, out


def _strings_by_int_id(strings: list[StringGeometry]) -> dict[int, StringGeometry]:
    out: dict[int, StringGeometry] = {}
    for geom in strings:
        sid = _parse_int_like(geom.string_id, "string_id")
        out[sid] = geom
    return out


def _id_radius_candidates(
    *,
    event: TouchEvent,
    strings_by_id: dict[int, StringGeometry],
    candidate_radius_default: int,
    candidate_radius_close_contact: int,
    contact_dist_px_thr: float,
) -> tuple[list[int], dict[str, Any]]:
    if not strings_by_id:
        return [], {
            "contact_dist_px": math.inf,
            "radius_used": int(candidate_radius_default),
            "candidate_count": 0,
        }

    touched_geom = strings_by_id.get(event.touched_string_id)
    if touched_geom is None:
        return [], {
            "contact_dist_px": math.inf,
            "radius_used": int(candidate_radius_default),
            "candidate_count": 0,
            "reason": "touched_string_missing",
        }

    _, contact_dist_px = closest_point_and_distance_px(
        touched_geom,
        x=event.contact_x,
        y=event.contact_y,
    )
    use_close = contact_dist_px <= float(contact_dist_px_thr)
    radius = int(candidate_radius_close_contact if use_close else candidate_radius_default)
    radius = max(0, radius)

    valid_ids = sorted(strings_by_id.keys())
    min_id = valid_ids[0]
    max_id = valid_ids[-1]

    candidates: list[int] = []
    for sid in range(event.touched_string_id - radius, event.touched_string_id + radius + 1):
        if sid < min_id or sid > max_id:
            continue
        if sid in strings_by_id:
            candidates.append(sid)
    if event.touched_string_id in strings_by_id and event.touched_string_id not in candidates:
        candidates.append(event.touched_string_id)
    candidates = sorted(set(candidates))

    return candidates, {
        "contact_dist_px": float(contact_dist_px),
        "radius_used": int(radius),
        "candidate_count": len(candidates),
        "touched_string_id": int(event.touched_string_id),
    }


def select_event_candidates_by_geometry(
    *,
    event: TouchEvent,
    strings_by_id: dict[int, StringGeometry],
    top_k: int,
    max_distance_px: float,
    always_include_touched_id: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    ranked: list[tuple[float, int]] = []
    distance_rows: list[dict[str, Any]] = []
    for sid, geom in strings_by_id.items():
        _, dist = closest_point_and_distance_px(
            geom,
            x=event.contact_x,
            y=event.contact_y,
        )
        ranked.append((float(dist), int(sid)))
        distance_rows.append({"string_id": int(sid), "distance_px": float(dist)})

    ranked.sort(key=lambda item: (item[0], item[1]))
    top_k_int = max(0, int(top_k))
    filtered = [
        int(sid)
        for dist, sid in ranked
        if dist <= float(max_distance_px)
    ]
    candidates = filtered[:top_k_int] if top_k_int > 0 else []

    if always_include_touched_id and event.touched_string_id in strings_by_id and event.touched_string_id not in candidates:
        candidates.append(int(event.touched_string_id))
    candidates = sorted(set(candidates))
    return candidates, {
        "geometry_candidates": (
            [
            row for row in sorted(distance_rows, key=lambda item: (item["distance_px"], item["string_id"]))
            if float(row["distance_px"]) <= float(max_distance_px)
            ][:top_k_int]
            if top_k_int > 0
            else []
        ),
        "top_k": int(top_k_int),
        "max_distance_px": float(max_distance_px),
        "always_include_touched_id": bool(always_include_touched_id),
    }


def select_event_candidates(
    *,
    event: TouchEvent,
    strings_by_id: dict[int, StringGeometry],
    candidate_radius_default: int,
    candidate_radius_close_contact: int,
    contact_dist_px_thr: float,
    geometry_enabled: bool = False,
    geometry_top_k: int = 5,
    geometry_max_distance_px: float = 35.0,
    always_include_touched_id: bool = True,
    include_id_radius_fallback: bool = True,
    missing_touched_id_fallback: str = "none",
    fallback_top_k: int = 3,
    fallback_max_distance_px: float = 40.0,
    log_string_id_inconsistency: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    if not strings_by_id:
        return [], {
            "candidate_method": "none",
            "final_candidates": [],
            "candidate_count": 0,
            "reason": "no_strings",
        }

    id_radius_candidates, id_debug = _id_radius_candidates(
        event=event,
        strings_by_id=strings_by_id,
        candidate_radius_default=candidate_radius_default,
        candidate_radius_close_contact=candidate_radius_close_contact,
        contact_dist_px_thr=contact_dist_px_thr,
    )
    touched_geom = strings_by_id.get(event.touched_string_id)

    warnings: list[str] = []
    valid_ids = sorted(strings_by_id.keys())
    if log_string_id_inconsistency and valid_ids:
        expected = list(range(valid_ids[0], valid_ids[-1] + 1))
        if expected != valid_ids:
            warnings.append("non_contiguous_string_ids")
        if event.touched_string_id < valid_ids[0] or event.touched_string_id > valid_ids[-1]:
            warnings.append("touched_id_outside_detected_range")

    geometry_candidates: list[int] = []
    geometry_debug: dict[str, Any] = {"geometry_candidates": []}
    if geometry_enabled:
        geometry_candidates, geometry_debug = select_event_candidates_by_geometry(
            event=event,
            strings_by_id=strings_by_id,
            top_k=geometry_top_k,
            max_distance_px=geometry_max_distance_px,
            always_include_touched_id=always_include_touched_id,
        )

    candidate_method = "id_radius"
    final_candidates: list[int] = list(id_radius_candidates)
    reason: str | None = None

    if touched_geom is None:
        fallback_mode = str(missing_touched_id_fallback).strip().lower()
        if fallback_mode == "nearest_geometry":
            final_candidates, geometry_debug = select_event_candidates_by_geometry(
                event=event,
                strings_by_id=strings_by_id,
                top_k=fallback_top_k,
                max_distance_px=fallback_max_distance_px,
                always_include_touched_id=always_include_touched_id,
            )
            candidate_method = "geometry_fallback"
            reason = "touched_id_missing_geometry_fallback_used"
            warnings.append("touched_string_missing")
        else:
            final_candidates = []
            candidate_method = "none"
            reason = "touched_string_missing"
    elif geometry_enabled:
        candidate_method = "geometry_only"
        final_candidates = list(geometry_candidates)
        if include_id_radius_fallback:
            candidate_method = "geometry_plus_id_radius"
            final_candidates = sorted(set(final_candidates + id_radius_candidates))
        if not final_candidates:
            final_candidates = list(id_radius_candidates)
            candidate_method = "id_radius_fallback"

    if always_include_touched_id and event.touched_string_id in strings_by_id and event.touched_string_id not in final_candidates:
        final_candidates.append(int(event.touched_string_id))
    final_candidates = sorted(set(final_candidates))

    if log_string_id_inconsistency and geometry_debug.get("geometry_candidates"):
        nearest = geometry_debug["geometry_candidates"][0]["string_id"]
        if int(nearest) != int(event.touched_string_id):
            warnings.append("nearest_geometry_differs_from_touched_id")
    if not final_candidates:
        warnings.append("candidate_set_empty")

    return final_candidates, {
        "candidate_method": candidate_method,
        "geometry_candidates": geometry_debug.get("geometry_candidates", []),
        "id_radius_candidates": id_radius_candidates,
        "final_candidates": final_candidates,
        "candidate_count": len(final_candidates),
        "touched_string_id": int(event.touched_string_id),
        "reason": reason,
        "warnings": warnings,
        "id_radius_debug": id_debug,
    }


def load_frames_for_indices(
    *,
    video_path: str | Path,
    frame_indices: list[int] | set[int],
    stabilize_enabled: bool = False,
    preloaded_frames: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], float]:
    needed = sorted(set(int(i) for i in frame_indices if int(i) >= 0))
    if not needed:
        return {}, 0.0

    out: dict[int, np.ndarray] = {}
    if preloaded_frames:
        for idx in needed:
            frame = preloaded_frames.get(idx)
            if frame is not None:
                out[idx] = frame.copy()

    video_id = str(Path(video_path).resolve())
    if not stabilize_enabled:
        for idx in needed:
            if idx in out:
                continue
            frame = _frame_cache_get(video_id, idx)
            if frame is not None:
                out[idx] = frame

    missing = [idx for idx in needed if idx not in out]
    if not missing:
        return out, 0.0

    if not stabilize_enabled and cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        missing_set = set(missing)
        ranges: list[tuple[int, int]] = []
        start = missing[0]
        prev = start
        for idx in missing[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            ranges.append((start, prev))
            start = idx
            prev = idx
        ranges.append((start, prev))

        for start_idx, end_idx in ranges:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
            for idx in range(start_idx, end_idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in missing_set:
                    frame_copy = frame.copy()
                    out[idx] = frame_copy
                    _frame_cache_put(video_id, idx, frame_copy)
        cap.release()
        return out, fps

    needed_set = set(missing)
    max_needed = missing[-1]
    with VideoReader(video_path=video_path, stabilize_enabled=stabilize_enabled) as reader:
        while True:
            packet = reader.read()
            if packet is None:
                break
            idx = int(packet.frame_index)
            if idx in needed_set:
                out[idx] = packet.frame.copy()
            if idx >= max_needed:
                break
        return out, float(reader.fps)


def _event_finger_point(event: TouchEvent) -> tuple[float, float] | None:
    if event.finger_x is None or event.finger_y is None:
        return None
    return float(event.finger_x), float(event.finger_y)


def _event_contact_point(event: TouchEvent) -> tuple[float, float]:
    return float(event.contact_x), float(event.contact_y)


def _event_hand_bbox(event: TouchEvent) -> tuple[float, float, float, float] | None:
    vals = (
        event.hand_bbox_x1,
        event.hand_bbox_y1,
        event.hand_bbox_x2,
        event.hand_bbox_y2,
    )
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _resolve_roi_height(
    *,
    candidate_geom: StringGeometry,
    strings: list[StringGeometry],
    roi_h: int,
    trim_ends_ratio: float,
    adaptive_enabled: bool,
    adaptive_height_ratio: float,
    min_roi_h: int,
    max_roi_h: int,
    min_neighbor_distance_px: float,
) -> tuple[int, dict[str, Any]]:
    roi_h_used = int(roi_h)
    neighbor_spacing_px = None
    if adaptive_enabled:
        neighbor_spacing_px = estimate_neighbor_spacing_px(
            candidate_geom,
            strings,
            sample_count=5,
            trim_ends_ratio=trim_ends_ratio,
        )
        if neighbor_spacing_px is not None and neighbor_spacing_px >= float(min_neighbor_distance_px):
            adaptive_h = int(round(float(neighbor_spacing_px) * float(adaptive_height_ratio)))
            roi_h_used = max(int(min_roi_h), min(int(max_roi_h), adaptive_h))
    return int(max(2, roi_h_used)), {
        "roi_h_original": int(roi_h),
        "roi_h_used": int(max(2, roi_h_used)),
        "adaptive_roi_enabled": bool(adaptive_enabled),
        "adaptive_height_ratio": float(adaptive_height_ratio),
        "neighbor_spacing_px": None if neighbor_spacing_px is None else float(neighbor_spacing_px),
        "min_neighbor_distance_px": float(min_neighbor_distance_px),
    }


def _compute_candidate_scores(
    *,
    event: TouchEvent,
    candidate_id: int,
    candidate_geom: StringGeometry,
    strings: list[StringGeometry],
    frames_for_event: dict[int, np.ndarray],
    frame_indices: list[int],
    baseline_frames: list[int],
    action_frames: list[int],
    action_windows: dict[int, list[int]] | None,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float,
    center_band_h: int,
    enable_hand_mask: bool,
    hand_mask_mode: str,
    contact_band_exclusion_px: float,
    mask_contact_region: bool,
    hand_mask_expand_px: float,
    farneback_params: dict[str, Any] | None,
    dy_thr: float,
    adaptive_roi_enabled: bool,
    adaptive_height_ratio: float,
    min_roi_h: int,
    max_roi_h: int,
    min_neighbor_distance_px: float,
    border_mode: str,
    constant_border_value: int,
    reject_if_out_of_frame: bool,
    min_inside_fraction: float,
    allow_small_gaps: bool,
    max_gap_frames: int,
    normalize_by_gap: bool,
) -> CandidateScoreSeries:
    score_by_frame: dict[int, float] = {}
    mean_abs_list: list[float] = []
    p90_list: list[float] = []
    frac_high_list: list[float] = []
    valid_frac_list: list[float] = []
    missing_frames: list[int] = []
    rejected_roi_frames: list[int] = []
    roi_frame_debug: dict[int, dict[str, Any]] = {}
    transitions_scored = 0
    non_consecutive_transitions_scored = 0
    transitions_zeroed_due_to_gap = 0
    max_gap_seen = 0

    roi_h_used, roi_debug = _resolve_roi_height(
        candidate_geom=candidate_geom,
        strings=strings,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
        adaptive_enabled=adaptive_roi_enabled,
        adaptive_height_ratio=adaptive_height_ratio,
        min_roi_h=min_roi_h,
        max_roi_h=max_roi_h,
        min_neighbor_distance_px=min_neighbor_distance_px,
    )

    prev_roi: np.ndarray | None = None
    prev_frame_idx: int | None = None

    for frame_idx in frame_indices:
        frame = frames_for_event.get(frame_idx)
        if frame is None:
            score_by_frame[frame_idx] = 0.0
            missing_frames.append(frame_idx)
            prev_roi = None
            prev_frame_idx = None
            continue

        roi, mat_img_to_roi, roi_extract_debug = extract_rotated_roi(
            frame=frame,
            string_geom=candidate_geom,
            roi_w=roi_w,
            roi_h=roi_h_used,
            trim_ends_ratio=trim_ends_ratio,
            border_mode=border_mode,
            constant_border_value=constant_border_value,
            reject_if_out_of_frame=reject_if_out_of_frame,
            min_inside_fraction=min_inside_fraction,
            return_debug=True,
        )
        roi_frame_debug[int(frame_idx)] = roi_extract_debug
        if roi is None:
            score_by_frame[frame_idx] = 0.0
            rejected_roi_frames.append(frame_idx)
            prev_roi = None
            prev_frame_idx = None
            continue

        if enable_hand_mask:
            mask = create_roi_mask(
                roi_h=roi_h_used,
                roi_w=roi_w,
                center_band_h=center_band_h,
                mat_img_to_roi=mat_img_to_roi,
                hand_bbox_img_xyxy=_event_hand_bbox(event),
                finger_point_img=_event_finger_point(event),
                contact_point_img=_event_contact_point(event),
                hand_mask_expand_px=hand_mask_expand_px,
                mode=hand_mask_mode,
                contact_band_exclusion_px=contact_band_exclusion_px,
                mask_contact_region=mask_contact_region,
            )
        else:
            mask = np.ones((roi_h_used, roi_w), dtype=np.uint8)

        gap = None if prev_frame_idx is None else int(frame_idx - prev_frame_idx)
        if prev_roi is None or prev_frame_idx is None or gap is None or gap <= 0:
            score_by_frame[frame_idx] = 0.0
        elif gap == 1 or (allow_small_gaps and gap <= int(max_gap_frames)):
            vib = compute_vibration_frame(
                prev_roi=prev_roi,
                roi=roi,
                mask=mask,
                farneback_params=farneback_params,
                dy_thr=dy_thr,
            )
            score = float(vib.vib_score_frame)
            if gap > 1:
                non_consecutive_transitions_scored += 1
                max_gap_seen = max(max_gap_seen, int(gap))
                if normalize_by_gap:
                    score /= float(gap)
            score_by_frame[frame_idx] = float(score)
            mean_abs_list.append(float(vib.mean_abs_dy))
            p90_list.append(float(vib.p90_abs_dy))
            frac_high_list.append(float(vib.frac_high_dy))
            valid_frac_list.append(float(vib.valid_frac))
            transitions_scored += 1
        else:
            score_by_frame[frame_idx] = 0.0
            transitions_zeroed_due_to_gap += 1
            max_gap_seen = max(max_gap_seen, int(gap))

        prev_roi = roi
        prev_frame_idx = frame_idx

    score_seq = [float(score_by_frame.get(f, 0.0)) for f in frame_indices]
    baseline_seq = [float(score_by_frame.get(f, 0.0)) for f in baseline_frames]
    action_seq = [float(score_by_frame.get(f, 0.0)) for f in action_frames]
    action_windows_out = {
        int(offset): [int(f) for f in frames]
        for offset, frames in (action_windows or {}).items()
    }

    return CandidateScoreSeries(
        candidate_string_id=int(candidate_id),
        frame_indices=list(frame_indices),
        score_seq=score_seq,
        baseline_seq=baseline_seq,
        action_seq=action_seq,
        score_by_frame={int(k): float(v) for k, v in score_by_frame.items()},
        action_frames=list(action_frames),
        action_windows=action_windows_out,
        debug={
            "missing_frames": missing_frames,
            "missing_count": len(missing_frames),
            "rejected_roi_frames": rejected_roi_frames,
            "rejected_roi_count": len(rejected_roi_frames),
            "transitions_scored": int(transitions_scored),
            "non_consecutive_transitions_scored": int(non_consecutive_transitions_scored),
            "transitions_zeroed_due_to_gap": int(transitions_zeroed_due_to_gap),
            "max_gap_seen": int(max_gap_seen),
            "mean_abs_dy_mean": float(np.mean(mean_abs_list)) if mean_abs_list else 0.0,
            "p90_abs_dy_mean": float(np.mean(p90_list)) if p90_list else 0.0,
            "frac_high_dy_mean": float(np.mean(frac_high_list)) if frac_high_list else 0.0,
            "valid_frac_mean": float(np.mean(valid_frac_list)) if valid_frac_list else 0.0,
            "baseline_mean": float(np.mean(baseline_seq)) if baseline_seq else 0.0,
            "action_mean": float(np.mean(action_seq)) if action_seq else 0.0,
            "roi_debug": roi_debug,
            "roi_frame_debug": roi_frame_debug,
            "hand_mask_enabled": bool(enable_hand_mask),
            "hand_mask_mode": str(hand_mask_mode),
            "hand_mask_expand_px": float(hand_mask_expand_px),
            "contact_band_exclusion_px": float(contact_band_exclusion_px),
            "mask_contact_region": bool(mask_contact_region),
        },
    )


def process_single_event_windows(
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
    dy_thr: float = DEFAULT_DY_THR,
    stabilize_enabled: bool = False,
    preloaded_frames: dict[int, np.ndarray] | None = None,
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
) -> EventWindowResult:
    strings_by_id = _strings_by_int_id(strings)

    f0 = event_frame_index(event.timestamp_sec, fps)
    dynamic_offsets_list = [int(v) for v in (dynamic_offsets or [-2, -1, 0, 1, 2, 3, 4, 5, 6])]
    action_windows: dict[int, list[int]] | None = None
    if dynamic_action_enabled:
        baseline_frames, action_windows = build_dynamic_action_windows(
            f0=f0,
            baseline_len=baseline_len,
            action_len=action_len,
            candidate_offsets=dynamic_offsets_list,
            baseline_gap=baseline_gap_frames,
            max_frame_index=None,
        )
        default_action_frames = list(action_windows.get(int(action_start_frame_offset), []))
        if not default_action_frames and action_windows:
            default_action_frames = list(action_windows[sorted(action_windows.keys())[0]])
        action_frames = default_action_frames
    else:
        baseline_frames, action_frames = build_frame_windows(
            f0=f0,
            baseline_len=baseline_len,
            action_len=action_len,
            action_start_frame_offset=action_start_frame_offset,
            baseline_gap_frames=baseline_gap_frames,
            max_frame_index=None,
        )

    frame_indices = sorted(set(
        baseline_frames
        + ([] if not dynamic_action_enabled or action_windows is None else [f for frames in action_windows.values() for f in frames])
        + action_frames
    ))
    frames_for_event, fps_video = load_frames_for_indices(
        video_path=video_path,
        frame_indices=frame_indices,
        stabilize_enabled=stabilize_enabled,
        preloaded_frames=preloaded_frames,
    )

    candidates, candidate_debug = select_event_candidates(
        event=event,
        strings_by_id=strings_by_id,
        candidate_radius_default=candidate_radius_default,
        candidate_radius_close_contact=candidate_radius_close_contact,
        contact_dist_px_thr=contact_dist_px_thr,
        geometry_enabled=geometry_enabled,
        geometry_top_k=geometry_top_k,
        geometry_max_distance_px=geometry_max_distance_px,
        always_include_touched_id=always_include_touched_id,
        include_id_radius_fallback=include_id_radius_fallback,
        missing_touched_id_fallback=missing_touched_id_fallback,
        fallback_top_k=fallback_top_k,
        fallback_max_distance_px=fallback_max_distance_px,
        log_string_id_inconsistency=log_string_id_inconsistency,
    )

    candidate_results: dict[int, CandidateScoreSeries] = {}
    for cid in candidates:
        geom = strings_by_id[cid]
        candidate_results[cid] = _compute_candidate_scores(
            event=event,
            candidate_id=cid,
            candidate_geom=geom,
            strings=strings,
            frames_for_event=frames_for_event,
            frame_indices=frame_indices,
            baseline_frames=baseline_frames,
            action_frames=action_frames,
            action_windows=action_windows,
            roi_w=roi_w,
            roi_h=roi_h,
            trim_ends_ratio=trim_ends_ratio,
            center_band_h=center_band_h,
            enable_hand_mask=enable_hand_mask,
            hand_mask_mode=hand_mask_mode,
            contact_band_exclusion_px=contact_band_exclusion_px,
            mask_contact_region=mask_contact_region,
            hand_mask_expand_px=hand_mask_expand_px,
            farneback_params=farneback_params,
            dy_thr=dy_thr,
            adaptive_roi_enabled=adaptive_roi_enabled,
            adaptive_height_ratio=adaptive_height_ratio,
            min_roi_h=min_roi_h,
            max_roi_h=max_roi_h,
            min_neighbor_distance_px=min_neighbor_distance_px,
            border_mode=border_mode,
            constant_border_value=constant_border_value,
            reject_if_out_of_frame=reject_if_out_of_frame,
            min_inside_fraction=min_inside_fraction,
            allow_small_gaps=allow_small_gaps,
            max_gap_frames=max_gap_frames,
            normalize_by_gap=normalize_by_gap,
        )

    return EventWindowResult(
        event=event,
        f0=f0,
        baseline_frames=baseline_frames,
        action_frames=action_frames,
        candidates=candidates,
        candidate_results=candidate_results,
        frames_for_event=frames_for_event,
        debug={
            "event_row_index": int(event.row_index),
            "fps_input": float(fps),
            "fps_video": float(fps_video),
            "baseline_len": int(baseline_len),
            "action_len": int(action_len),
            "action_start_frame_offset": int(action_start_frame_offset),
            "baseline_gap_frames": int(baseline_gap_frames),
            "dynamic_action_enabled": bool(dynamic_action_enabled),
            "dynamic_select_metric": str(dynamic_select_metric),
            "tested_offsets": dynamic_offsets_list if dynamic_action_enabled else [],
            "baseline_frames": baseline_frames,
            "action_frames": action_frames,
            "action_windows": {} if action_windows is None else {int(k): list(v) for k, v in action_windows.items()},
            "frame_indices": frame_indices,
            "loaded_frame_count": len(frames_for_event),
            "candidate_selection": candidate_debug,
        },
    )


def process_touch_events_windows(
    *,
    video_path: str | Path,
    strings: list[StringGeometry],
    touch_events: list[TouchEvent],
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
    dy_thr: float = DEFAULT_DY_THR,
    stabilize_enabled: bool = False,
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
) -> list[EventWindowResult]:
    results: list[EventWindowResult] = []
    for event in touch_events:
        results.append(
            process_single_event_windows(
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
        )
    return results
