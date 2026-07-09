from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
import json
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import shutil
import subprocess

try:
    import mediapipe as mp
except ImportError:
    mp = None

# -------- Paths / Defaults --------

PROJECT_DIR = Path(__file__).resolve().parent   # folder where this script is located
REPO_DIR = PROJECT_DIR.parent
DEFAULT_SRC = PROJECT_DIR / "images"            # input images folder
WEIGHTS_DIR = PROJECT_DIR / "weights"           # legacy weights folder
HARP_WEIGHTS_DIR = REPO_DIR / "harp_pose_v11m_prepped" / "weights"
OUT_PROJECT = PROJECT_DIR / "predict_postprocessed" # output folder
TOUCH_EVENTS_DIR = PROJECT_DIR / "touch_events"


IMGSZ = 960
DET_CONF = 0.25
KPT_CONF_THRES = 0.10
DEFAULT_YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto").strip() or "auto"

KPT_RADIUS = 5
LINE_THICKNESS = 1

MODELS = {
    "best": HARP_WEIGHTS_DIR / "best.pt" if (HARP_WEIGHTS_DIR / "best.pt").exists() else WEIGHTS_DIR / "best.pt",
    "last": HARP_WEIGHTS_DIR / "last.pt" if (HARP_WEIGHTS_DIR / "last.pt").exists() else WEIGHTS_DIR / "last.pt",
}

WEB_FRIENDLY_CODECS = ("avc1", "H264", "X264")
WRITER_CODEC_PREFERENCE = ("mp4v", "avc1", "H264", "X264")
HAND_PIPELINE_QUEUE_SIZE = 8
# Enable MediaPipe worker-thread pipeline by default.
# Override with environment variable HAND_PIPELINE_ENABLED=0 to force sync mode.
HAND_PIPELINE_ENABLED = os.getenv("HAND_PIPELINE_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _resolve_yolo_runtime() -> dict[str, object]:
    requested_device = DEFAULT_YOLO_DEVICE
    cuda_available = False
    gpu_name: str | None = None

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        cuda_available = False

    requested_lower = requested_device.lower()
    if requested_lower in {"auto", ""}:
        device = "cuda:0" if cuda_available else "cpu"
    elif requested_lower == "cuda":
        device = "cuda:0"
    elif requested_lower.isdigit():
        device = f"cuda:{requested_lower}"
    else:
        device = requested_device

    uses_cuda = str(device).lower().startswith("cuda")
    if uses_cuda and not cuda_available:
        print(f"[WARN] YOLO device requested as {requested_device!r}, but CUDA is not available. Falling back to CPU.")
        device = "cpu"
        uses_cuda = False

    half = _env_bool("YOLO_HALF", uses_cuda)
    if not uses_cuda:
        half = False

    return {
        "requested_device": requested_device,
        "device": device,
        "half": bool(half),
        "cuda_available": bool(cuda_available),
        "gpu_name": gpu_name,
    }


def _load_yolo_model(model_path: Path) -> tuple[YOLO, dict[str, object]]:
    runtime = _resolve_yolo_runtime()
    model = YOLO(str(model_path))
    device = str(runtime["device"])
    try:
        model.to(device)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not move YOLO model to {device}: {exc}")
    return model, runtime


def _yolo_predict_kwargs(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "task": "pose",
        "imgsz": IMGSZ,
        "conf": DET_CONF,
        "verbose": False,
        "device": runtime["device"],
        "half": bool(runtime.get("half", False)),
    }


def _log_yolo_runtime(runtime: dict[str, object]) -> None:
    gpu_name = runtime.get("gpu_name")
    gpu_suffix = f", gpu={gpu_name}" if gpu_name else ""
    print(
        "[INFO] YOLO runtime: "
        f"device={runtime.get('device')}, "
        f"half={runtime.get('half')}, "
        f"cuda_available={runtime.get('cuda_available')}"
        f"{gpu_suffix}"
    )

FINGERTIP_IDS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

FINGERTIP_INDEX_SET = set(FINGERTIP_IDS.values())
TIP_NAME_BY_INDEX = {idx: name for name, idx in FINGERTIP_IDS.items()}

TIP_COLORS = {
    "thumb_tip": (60, 180, 75),
    "index_tip": (0, 255, 255),
    "middle_tip": (255, 0, 0),
    "ring_tip": (0, 128, 255),
    "pinky_tip": (255, 0, 255),
}

HAND_SIDE_COLORS = {
    "left": (80, 220, 120),
    "right": (80, 140, 255),
    "unknown": (180, 180, 180),
}

HAND_SIDE_CENTER_HYSTERESIS_RATIO = 0.08
HAND_TRACK_EMA_ALPHA = 0.35
# Keep fallback disabled in production: repeated ROI passes on the same
# MediaPipe graph can trigger fatal internal packet errors on some builds.
HAND_ROI_FALLBACK_ENABLED = True
LEFT_HAND_ROI_START_RATIO = 0.45
RIGHT_HAND_ROI_END_RATIO = 0.55
ROI_DUPLICATE_MIN_DIST_RATIO = 0.18
HARP_HAND_ROI_PAD_X_RATIO = 0.08
HARP_HAND_ROI_PAD_Y_RATIO = 0.10
HARP_HAND_SPLIT_ROI_ENABLED = True
HARP_HAND_SPLIT_OVERLAP_RATIO = 0.12
RIGHT_HAND_TOUCH_TIPS = {"thumb_tip", "index_tip"}
LEFT_HAND_TOUCH_TIPS = {"thumb_tip"}
TOUCH_DISTANCE_RATIO = 0.01


# ---------------------------
# Utility: nice distinct colors
# ---------------------------
def make_palette(n: int) -> list[tuple[int, int, int]]:
    hues = np.linspace(0, 179, n, endpoint=False, dtype=np.uint8)
    palette = []
    for h in hues:
        hsv = np.uint8([[[h, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


def safe_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y


def clamp_roi(
    roi: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    x0 = max(0, min(int(x0), frame_w - 1))
    y0 = max(0, min(int(y0), frame_h - 1))
    x1 = max(x0 + 1, min(int(x1), frame_w))
    y1 = max(y0 + 1, min(int(y1), frame_h))
    return x0, y0, x1, y1


def expand_roi(
    roi: tuple[int, int, int, int],
    *,
    frame_w: int,
    frame_h: int,
    pad_x_ratio: float = HARP_HAND_ROI_PAD_X_RATIO,
    pad_y_ratio: float = HARP_HAND_ROI_PAD_Y_RATIO,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = clamp_roi(roi, frame_w=frame_w, frame_h=frame_h)
    roi_w = max(1, x1 - x0)
    roi_h = max(1, y1 - y0)
    pad_x = int(round(roi_w * pad_x_ratio))
    pad_y = int(round(roi_h * pad_y_ratio))
    return clamp_roi((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), frame_w=frame_w, frame_h=frame_h)


def extract_harp_roi_from_result(result) -> tuple[int, int, int, int] | None:
    if result is None or result.boxes is None or len(result.boxes) <= 0:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()[0]
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    h, w = result.orig_img.shape[:2]
    return expand_roi((x0, y0, x1, y1), frame_w=w, frame_h=h)


def split_harp_roi_for_hands(
    harp_roi: tuple[int, int, int, int],
    *,
    frame_w: int,
    frame_h: int,
    overlap_ratio: float = HARP_HAND_SPLIT_OVERLAP_RATIO,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    x0, y0, x1, y1 = clamp_roi(harp_roi, frame_w=frame_w, frame_h=frame_h)
    roi_w = max(1, x1 - x0)
    mid_x = int(round((x0 + x1) * 0.5))
    overlap_px = int(round(roi_w * overlap_ratio))
    left_roi = clamp_roi((x0, y0, mid_x + overlap_px, y1), frame_w=frame_w, frame_h=frame_h)
    right_roi = clamp_roi((mid_x - overlap_px, y0, x1, y1), frame_w=frame_w, frame_h=frame_h)
    return left_roi, right_roi


def run_hand_detector_on_region(
    source_frame,
    hands_detector,
    process_width: int,
    roi: tuple[int, int, int, int] | None = None,
) -> list[dict]:
    frame_h, frame_w = source_frame.shape[:2]
    if roi is None:
        x0, y0, x1, y1 = 0, 0, frame_w, frame_h
    else:
        x0, y0, x1, y1 = roi
        x0 = max(0, min(x0, frame_w - 1))
        y0 = max(0, min(y0, frame_h - 1))
        x1 = max(x0 + 1, min(x1, frame_w))
        y1 = max(y0 + 1, min(y1, frame_h))

    region = source_frame[y0:y1, x0:x1]
    region_h, region_w = region.shape[:2]
    if region_h <= 0 or region_w <= 0:
        return []

    proc_frame = region
    if process_width > 0 and region_w > process_width:
        scale = process_width / region_w
        proc_w = process_width
        proc_h = max(1, int(region_h * scale))
        proc_frame = cv2.resize(region, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
    else:
        proc_h, proc_w = region_h, region_w

    scale_x = region_w / proc_w
    scale_y = region_h / proc_h

    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)
    rgb.flags.writeable = True

    landmarks_list = results.multi_hand_landmarks or []
    handedness_list = results.multi_handedness or []
    candidates = []

    for hand_i, hand_landmarks in enumerate(landmarks_list):
        landmark_points: dict[int, tuple[int, int]] = {}
        xs = []
        ys = []
        for lm_idx, lm in enumerate(hand_landmarks.landmark):
            x = int(lm.x * proc_w * scale_x) + x0
            y = int(lm.y * proc_h * scale_y) + y0
            x, y = safe_point(x, y, frame_w, frame_h)
            landmark_points[lm_idx] = (x, y)
            xs.append(x)
            ys.append(y)

        if not landmark_points:
            continue

        mp_label = "unknown"
        mp_score = 0.0
        if hand_i < len(handedness_list):
            handedness = handedness_list[hand_i].classification[0]
            mp_label = handedness.label.lower()
            mp_score = float(handedness.score)

        bbox = (min(xs), min(ys), max(xs), max(ys))
        center_x = 0.5 * (bbox[0] + bbox[2])
        center_y = 0.5 * (bbox[1] + bbox[3])
        candidates.append(
            {
                "landmark_points": landmark_points,
                "bbox": bbox,
                "center": (center_x, center_y),
                "mp_label": mp_label,
                "mp_score": mp_score,
            }
        )

    return candidates


def is_duplicate_hand_candidate(candidate: dict, existing_candidates: list[dict], min_center_dist_px: float) -> bool:
    cx, cy = candidate["center"]
    for other in existing_candidates:
        ox, oy = other["center"]
        if np.hypot(cx - ox, cy - oy) < min_center_dist_px:
            return True
    return False


def pick_hand_side_by_position(
    center_x: float,
    frame_w: int,
    tracking_state: dict,
    reference_roi: tuple[int, int, int, int] | None = None,
) -> str:
    if reference_roi is not None:
        roi_x0, _, roi_x1, _ = reference_roi
        ref_w = max(1.0, float(roi_x1 - roi_x0))
        mid_x = 0.5 * (roi_x0 + roi_x1)
        margin = ref_w * HAND_SIDE_CENTER_HYSTERESIS_RATIO
    else:
        mid_x = frame_w * 0.5
        margin = frame_w * HAND_SIDE_CENTER_HYSTERESIS_RATIO
    if center_x < mid_x - margin:
        return "right"
    if center_x > mid_x + margin:
        return "left"

    left_center = tracking_state.get("left_center")
    right_center = tracking_state.get("right_center")
    if left_center is not None and right_center is not None:
        dist_to_left = abs(center_x - left_center[0])
        dist_to_right = abs(center_x - right_center[0])
        return "left" if dist_to_left <= dist_to_right else "right"
    if left_center is not None:
        return "left"
    if right_center is not None:
        return "right"
    return "right" if center_x < mid_x else "left"


def update_hand_track_center(tracking_state: dict, hand_label: str, center: tuple[float, float]) -> None:
    if hand_label not in ("left", "right"):
        return
    key = f"{hand_label}_center"
    prev_center = tracking_state.get(key)
    if prev_center is None:
        tracking_state[key] = center
        return
    alpha = HAND_TRACK_EMA_ALPHA
    tracking_state[key] = (
        (1.0 - alpha) * prev_center[0] + alpha * center[0],
        (1.0 - alpha) * prev_center[1] + alpha * center[1],
    )


def point_to_line_segment_distance(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    return point_to_line_segment_projection(point, seg_start, seg_end)[2]


def point_to_line_segment_projection(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[float, float, float]:
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return float(x1), float(y1), float(np.hypot(px - x1, py - y1))

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return float(closest_x), float(closest_y), float(np.hypot(px - closest_x, py - closest_y))


def detect_fingertip_string_touches(
    strings: list[tuple[int, tuple[float, float], tuple[float, float]]],
    fingertip_points: list[dict],
    max_touch_distance_px: float,
) -> dict[tuple[str, str], dict]:
    touches: dict[tuple[str, str], dict] = {}
    for fingertip in fingertip_points:
        hand_label = str(fingertip.get("hand_label", "unknown"))
        tip_name = str(fingertip.get("tip_name", ""))

        if hand_label == "right":
            if tip_name not in RIGHT_HAND_TOUCH_TIPS:
                continue
        elif hand_label == "left":
            if tip_name not in LEFT_HAND_TOUCH_TIPS:
                continue
        else:
            continue

        point = (float(fingertip.get("x", 0.0)), float(fingertip.get("y", 0.0)))
        nearest_sid = None
        nearest_dist = float("inf")
        nearest_contact_point = None

        for sid, left_pt, right_pt in strings:
            contact_x, contact_y, dist = point_to_line_segment_projection(point, left_pt, right_pt)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_sid = sid
                nearest_contact_point = (contact_x, contact_y)

        if nearest_sid is None or nearest_dist > max_touch_distance_px:
            continue

        key = (hand_label, tip_name)
        prev = touches.get(key)
        if prev is None or nearest_dist < float(prev.get("distance_px", float("inf"))):
            touches[key] = {
                "string_id": int(nearest_sid),
                "distance_px": float(nearest_dist),
                "finger_point": (float(point[0]), float(point[1])),
                "contact_point": (
                    float(nearest_contact_point[0]),
                    float(nearest_contact_point[1]),
                ) if nearest_contact_point is not None else (float(point[0]), float(point[1])),
            }

    return touches


def detect_and_draw_hand_landmarks(
    source_frame,
    draw_frame,
    hands_detector,
    hands_detector_left=None,
    hands_detector_right=None,
    process_width: int = 640,
    draw_labels: bool = False,
    tracking_state: dict | None = None,
    harp_roi: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, list[dict]]:
    frame_h, frame_w = source_frame.shape[:2]
    if tracking_state is None:
        tracking_state = {}

    if harp_roi is not None:
        harp_roi = clamp_roi(harp_roi, frame_w=frame_w, frame_h=frame_h)

    hand_candidates: list[dict] = []
    used_split_harp_rois = False

    if (
        HARP_HAND_SPLIT_ROI_ENABLED
        and harp_roi is not None
        and hands_detector_left is not None
        and hands_detector_right is not None
    ):
        used_split_harp_rois = True
        left_roi, right_roi = split_harp_roi_for_hands(harp_roi, frame_w=frame_w, frame_h=frame_h)
        roi_w = max(1, harp_roi[2] - harp_roi[0])
        min_dist_px = roi_w * ROI_DUPLICATE_MIN_DIST_RATIO

        left_zone_candidates = run_hand_detector_on_region(
            source_frame=source_frame,
            hands_detector=hands_detector_left,
            process_width=process_width,
            roi=left_roi,
        )
        left_zone_candidates = sorted(
            left_zone_candidates,
            key=lambda c: c.get("mp_score", 0.0),
            reverse=True,
        )
        for cand in left_zone_candidates:
            cand["assigned_label"] = "right"
            hand_candidates.append(cand)
            break

        right_zone_candidates = run_hand_detector_on_region(
            source_frame=source_frame,
            hands_detector=hands_detector_right,
            process_width=process_width,
            roi=right_roi,
        )
        right_zone_candidates = sorted(
            right_zone_candidates,
            key=lambda c: c.get("mp_score", 0.0),
            reverse=True,
        )
        for cand in right_zone_candidates:
            if not is_duplicate_hand_candidate(cand, hand_candidates, min_center_dist_px=min_dist_px):
                cand["assigned_label"] = "left"
                hand_candidates.append(cand)
                break

        if len(hand_candidates) < 2 and hands_detector is not None:
            fallback_candidates = run_hand_detector_on_region(
                source_frame=source_frame,
                hands_detector=hands_detector,
                process_width=process_width,
                roi=harp_roi,
            )
            fallback_candidates = sorted(
                fallback_candidates,
                key=lambda c: c.get("mp_score", 0.0),
                reverse=True,
            )
            for cand in fallback_candidates:
                if is_duplicate_hand_candidate(cand, hand_candidates, min_center_dist_px=min_dist_px):
                    continue
                existing_labels = {str(c.get("assigned_label", "")) for c in hand_candidates}
                missing_labels = [label for label in ("right", "left") if label not in existing_labels]
                if len(missing_labels) == 1:
                    cand["assigned_label"] = missing_labels[0]
                else:
                    cand["assigned_label"] = pick_hand_side_by_position(
                        center_x=cand["center"][0],
                        frame_w=frame_w,
                        tracking_state=tracking_state,
                        reference_roi=harp_roi,
                    )
                hand_candidates.append(cand)
                if len(hand_candidates) >= 2:
                    break

    if not used_split_harp_rois:
        hand_candidates = run_hand_detector_on_region(
            source_frame=source_frame,
            hands_detector=hands_detector,
            process_width=process_width,
            roi=harp_roi,
        )

    if (not used_split_harp_rois) and HAND_ROI_FALLBACK_ENABLED and len(hand_candidates) < 2:
        if harp_roi is None:
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, frame_w, frame_h
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = harp_roi
        roi_w = max(1, roi_x1 - roi_x0)
        roi_mid_x = 0.5 * (roi_x0 + roi_x1)

        centers_x = [cand["center"][0] for cand in hand_candidates]
        need_left_side_of_player = not any(x >= roi_mid_x for x in centers_x)
        need_right_side_of_player = not any(x < roi_mid_x for x in centers_x)
        min_dist_px = roi_w * ROI_DUPLICATE_MIN_DIST_RATIO

        if need_left_side_of_player:
            right_roi = (
                int(roi_x0 + roi_w * LEFT_HAND_ROI_START_RATIO),
                roi_y0,
                roi_x1,
                roi_y1,
            )
            roi_candidates = run_hand_detector_on_region(
                source_frame=source_frame,
                hands_detector=hands_detector,
                process_width=process_width,
                roi=right_roi,
            )
            for cand in roi_candidates:
                if not is_duplicate_hand_candidate(cand, hand_candidates, min_center_dist_px=min_dist_px):
                    hand_candidates.append(cand)
                    break

        if len(hand_candidates) < 2 and need_right_side_of_player:
            left_roi = (
                roi_x0,
                roi_y0,
                int(roi_x0 + roi_w * RIGHT_HAND_ROI_END_RATIO),
                roi_y1,
            )
            roi_candidates = run_hand_detector_on_region(
                source_frame=source_frame,
                hands_detector=hands_detector,
                process_width=process_width,
                roi=left_roi,
            )
            for cand in roi_candidates:
                if not is_duplicate_hand_candidate(cand, hand_candidates, min_center_dist_px=min_dist_px):
                    hand_candidates.append(cand)
                    break

    if len(hand_candidates) > 2:
        hand_candidates = sorted(hand_candidates, key=lambda c: c.get("mp_score", 0.0), reverse=True)[:2]

    if len(hand_candidates) >= 2:
        ordered = sorted(hand_candidates, key=lambda c: c["center"][0])
        ordered[0]["assigned_label"] = "right"
        ordered[-1]["assigned_label"] = "left"
        hand_candidates = ordered[:2]
    elif len(hand_candidates) == 1:
        only = hand_candidates[0]
        if "assigned_label" not in only:
            only["assigned_label"] = pick_hand_side_by_position(
                center_x=only["center"][0],
                frame_w=frame_w,
                tracking_state=tracking_state,
                reference_roi=harp_roi,
            )

    hands_detected = len(hand_candidates)
    hand_landmarks_drawn = 0
    fingertips_drawn = 0
    fingertip_points: list[dict] = []

    for hand_candidate in hand_candidates:
        hand_label = hand_candidate.get("assigned_label", "unknown")
        hand_color = HAND_SIDE_COLORS.get(hand_label, HAND_SIDE_COLORS["unknown"])
        landmark_points = hand_candidate["landmark_points"]
        draw_full_hand_landmarks = hand_label == "right"
        if hand_label == "right":
            drawable_tip_names = RIGHT_HAND_TOUCH_TIPS
        elif hand_label == "left":
            drawable_tip_names = LEFT_HAND_TOUCH_TIPS
        else:
            drawable_tip_names = set(FINGERTIP_IDS.keys())

        update_hand_track_center(tracking_state, hand_label, hand_candidate["center"])

        if draw_full_hand_landmarks and mp is not None:
            for start_idx, end_idx in mp.solutions.hands.HAND_CONNECTIONS:
                p0 = landmark_points.get(int(start_idx))
                p1 = landmark_points.get(int(end_idx))
                if p0 is None or p1 is None:
                    continue
                cv2.line(draw_frame, p0, p1, hand_color, 1, lineType=cv2.LINE_AA)

        for lm_idx, point in landmark_points.items():
            x, y = point
            is_fingertip = lm_idx in FINGERTIP_INDEX_SET

            if not draw_full_hand_landmarks and not is_fingertip:
                continue

            color = hand_color
            radius = 2
            if is_fingertip:
                tip_name = TIP_NAME_BY_INDEX[lm_idx]
                if tip_name in drawable_tip_names:
                    fingertips_drawn += 1
                    fingertip_points.append(
                        {
                            "hand_label": hand_label,
                            "tip_name": tip_name,
                            "x": int(x),
                            "y": int(y),
                        }
                    )
                elif not draw_full_hand_landmarks:
                    # Left/unknown hands only draw selected fingertips.
                    continue
                color = TIP_COLORS.get(tip_name, hand_color)
                radius = 4

            cv2.circle(draw_frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            if draw_labels:
                cv2.putText(
                    draw_frame,
                    f"{lm_idx}",
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            hand_landmarks_drawn += 1

        # Right hand draws full landmarks + skeleton; left hand remains selected-tip only.

        wrist_point = landmark_points.get(0)
        if wrist_point is not None:
            label_text = hand_label
            if draw_labels:
                mp_label = hand_candidate.get("mp_label", "unknown")
                mp_score = float(hand_candidate.get("mp_score", 0.0))
                if mp_score > 0:
                    label_text = f"{hand_label} [mp:{mp_label} {mp_score:.2f}]"
            label_anchor = (wrist_point[0] + 8, max(20, wrist_point[1] - 12))
            cv2.putText(
                draw_frame,
                label_text,
                label_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                hand_color,
                2,
                cv2.LINE_AA,
            )

    return hands_detected, hand_landmarks_drawn, fingertips_drawn, fingertip_points


def hand_tracking_worker_loop(
    input_queue: Queue,
    output_queue: Queue,
    *,
    process_width: int,
    draw_labels: bool,
    hand_max_hands: int,
    hand_model_complexity: int,
    hand_min_detection_confidence: float,
    hand_min_tracking_confidence: float,
    left_hand_min_detection_confidence: float,
    right_hand_min_detection_confidence: float,
    left_hand_min_tracking_confidence: float,
    right_hand_min_tracking_confidence: float,
) -> None:
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=hand_max_hands,
        model_complexity=hand_model_complexity,
        min_detection_confidence=hand_min_detection_confidence,
        min_tracking_confidence=hand_min_tracking_confidence,
    )
    hands_detector_left = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=hand_model_complexity,
        min_detection_confidence=left_hand_min_detection_confidence,
        min_tracking_confidence=left_hand_min_tracking_confidence,
    )
    hands_detector_right = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=hand_model_complexity,
        min_detection_confidence=right_hand_min_detection_confidence,
        min_tracking_confidence=right_hand_min_tracking_confidence,
    )
    tracking_state = {
        "left_center": None,
        "right_center": None,
    }
    try:
        while True:
            item = input_queue.get()
            if item is None:
                break

            frame_idx, source_frame, annotated_frame, harp_roi = item
            try:
                hands_detected, landmarks_drawn, fingertips_drawn, fingertip_points = detect_and_draw_hand_landmarks(
                    source_frame=source_frame,
                    draw_frame=annotated_frame,
                    hands_detector=hands_detector,
                    hands_detector_left=hands_detector_left,
                    hands_detector_right=hands_detector_right,
                    process_width=process_width,
                    draw_labels=draw_labels,
                    tracking_state=tracking_state,
                    harp_roi=harp_roi,
                )
                output_queue.put(
                    (
                        frame_idx,
                        annotated_frame,
                        hands_detected,
                        landmarks_drawn,
                        fingertips_drawn,
                        fingertip_points,
                        None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                output_queue.put((frame_idx, annotated_frame, 0, 0, 0, [], exc))
    finally:
        hands_detector.close()
        hands_detector_left.close()
        hands_detector_right.close()
        output_queue.put(None)

def typical_gap_mode(gaps, bin_size=3):
    gaps = np.array(gaps, dtype=float)
    if len(gaps) == 0:
        return None

    bins = np.round(gaps / bin_size).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    best_bin = values[np.argmax(counts)]
    return best_bin * bin_size



# ---------------------------------------
# Robust endpoint correction + pairing
# ---------------------------------------
def correct_string_endpoints(left_pts, right_pts, expected_n=None):
    left_pts = np.array(left_pts, dtype=float)
    right_pts = np.array(right_pts, dtype=float)

    if len(left_pts) == 0 or len(right_pts) == 0:
        return []

    # sort bottom->top using Y
    left_pts = left_pts[np.argsort(-left_pts[:, 1])]
    right_pts = right_pts[np.argsort(-right_pts[:, 1])]

    # enforce equal count
    n = min(len(left_pts), len(right_pts))
    if expected_n is not None:
        n = min(n, expected_n)

    left_pts = left_pts[:n]
    right_pts = right_pts[:n]

    # ----------------------------------------------------
    # Step 0: Fix Y spacing using "mode gap" rule
    # ----------------------------------------------------
    gaps = []
    for i in range(n - 1):
        gaps.append(right_pts[i, 1] - right_pts[i + 1, 1])  # bottom->top

    gap_est = typical_gap_mode(gaps, bin_size=3)

    if gap_est is not None and gap_est > 0:
        corrected_y = right_pts[:, 1].copy()

        for i in range(1, n):
            expected_y = corrected_y[i - 1] - gap_est
            actual_y = corrected_y[i]

            # if gap becomes too large => force correction
            if abs(actual_y - expected_y) > 1.8 * gap_est:
                corrected_y[i] = expected_y

        right_pts[:, 1] = corrected_y

    # ----------------------------------------------------
    # Step 1: Fit rim curve using only reliable bottom points
    # ----------------------------------------------------
    use_k = min(10, len(right_pts))
    anchor_pts = right_pts[:use_k]

    y_anchor = anchor_pts[:, 1]
    x_anchor = anchor_pts[:, 0]

    coeff = np.polyfit(y_anchor, x_anchor, deg=2)
    poly = np.poly1d(coeff)

    # ----------------------------------------------------
    # Step 2: Replace suspicious X points using curve snapping
    # ----------------------------------------------------
    corrected_right = right_pts.copy()

    for i in range(len(corrected_right)):
        y = corrected_right[i, 1]
        x = corrected_right[i, 0]

        x_expected = poly(y)
        dist = abs(x - x_expected)

        max_allowed = 40 + (i * 3)

        if dist > max_allowed:
            corrected_right[i, 0] = x_expected

    # ----------------------------------------------------
    # Step 3: Smooth right endpoints with final curve fit
    # ----------------------------------------------------
    y_all = corrected_right[:, 1]
    x_all = corrected_right[:, 0]

    coeff2 = np.polyfit(y_all, x_all, deg=2)
    poly2 = np.poly1d(coeff2)

    corrected_right[:, 0] = poly2(y_all)

    # ----------------------------------------------------
    # Step 4: Return pairs bottom->top
    # ----------------------------------------------------
    pairs = []
    for i in range(n):
        pairs.append((tuple(left_pts[i]), tuple(corrected_right[i])))

    return pairs




# ---------------------------------------
# Extract corrected strings from YOLO result
# ---------------------------------------
def extract_corrected_strings(result, kpt_conf_thres=0.05, expected_strings=16):
    """
    Returns list of (sid, left_point, right_point) sorted bottom->top
    sid starts from 1 (bottom-most)
    """

    if result.keypoints is None:
        return []

    kpts_xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        kpts_conf = result.keypoints.conf.cpu().numpy()
    else:
        kpts_conf = None

    if len(kpts_xy) == 0:
        return []

    # We assume ONE detection = harp
    pts = kpts_xy[0]
    confs = kpts_conf[0] if kpts_conf is not None else np.ones(len(pts), dtype=float)

    left_pts = []
    right_pts = []

    # Even indices = left, Odd indices = right (based on your dataset)
    for i in range(0, len(pts) - 1, 2):
        if confs[i] >= kpt_conf_thres:
            left_pts.append((pts[i][0], pts[i][1]))

        if confs[i + 1] >= kpt_conf_thres:
            right_pts.append((pts[i + 1][0], pts[i + 1][1]))

    # Correct + pair them robustly
    pairs = correct_string_endpoints(left_pts, right_pts, expected_n=expected_strings)

    output = []
    for idx, (lp, rp) in enumerate(pairs):
        sid = idx + 1
        output.append((sid, lp, rp))

    return output


# ---------------------------------------
# Draw corrected strings
# ---------------------------------------
def draw_corrected_strings(
    result,
    expected_strings=16,
    strings: list[tuple[int, tuple[float, float], tuple[float, float]]] | None = None,
    base_frame=None,
    draw_strings: bool = True,
    draw_harp_annotation: bool = True,
):
    im = base_frame.copy() if base_frame is not None else result.orig_img.copy()
    h, w = im.shape[:2]

    # Draw bounding box (if exists)
    if draw_harp_annotation and result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()[0]
        conf = float(result.boxes.conf.cpu().numpy()[0])

        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            im,
            f"harp {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if not draw_strings:
        return im

    if strings is None:
        strings = extract_corrected_strings(result, kpt_conf_thres=KPT_CONF_THRES, expected_strings=expected_strings)

    palette = make_palette(max(1, len(strings)))

    for i, (sid, lp, rp) in enumerate(strings):
        color = palette[i]

        lx, ly = int(lp[0]), int(lp[1])
        rx, ry = int(rp[0]), int(rp[1])

        # draw endpoints
        cv2.circle(im, (lx, ly), KPT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(im, (rx, ry), KPT_RADIUS, color, -1, cv2.LINE_AA)

        # draw string line
        cv2.line(im, (lx, ly), (rx, ry), color, LINE_THICKNESS, cv2.LINE_AA)

        # label near midpoint
        cx = int((lx + rx) / 2)
        cy = int((ly + ry) / 2)

        cv2.putText(
            im,
            f"s{sid}",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            1,
            cv2.LINE_AA,
        )

    return im


def _serialize_frame_strings(
    frame_strings: list[tuple[int, tuple[float, float], tuple[float, float]]],
) -> list[list[float]]:
    out: list[list[float]] = []
    for sid, left_pt, right_pt in frame_strings:
        out.append(
            [
                int(sid),
                float(left_pt[0]),
                float(left_pt[1]),
                float(right_pt[0]),
                float(right_pt[1]),
            ]
        )
    return out


def _write_strings_by_frame_line(
    file_handle,
    *,
    frame_index: int,
    frame_strings: list[tuple[int, tuple[float, float], tuple[float, float]]],
) -> None:
    payload = {
        "frame_index": int(frame_index),
        "strings": _serialize_frame_strings(frame_strings),
    }
    file_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def save_prediction_outputs(result, corrected_img, strings, out_img_path: Path, out_txt_path: Path):
    # Save corrected image
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), corrected_img)

    # Prepare text content
    lines = []
    stem = out_img_path.stem
    h, w = corrected_img.shape[:2]
    lines.append(f"image: {out_img_path.name}")
    lines.append(f"image_size: {w},{h}")

    # bounding box if available
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()[0]
        conf = float(result.boxes.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        lines.append(f"bbox: {x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},conf:{conf:.3f}")

    # Per-string lines
    for sid, lp, rp in strings:
        lx, ly = float(lp[0]), float(lp[1])
        rx, ry = float(rp[0]), float(rp[1])
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        label = f"s{sid}"
        lines.append(
            f"label:{label},sid:{sid},left:{lx:.2f},{ly:.2f},right:{rx:.2f},{ry:.2f},mid:{cx:.2f},{cy:.2f}"
        )

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines))


def save_touch_events_json(
    *,
    tag: str,
    video_path: Path,
    touch_events: list[dict],
    fps: float,
    frames_processed: int,
) -> Path:
    import json

    out_dir = TOUCH_EVENTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_touch_events.json"

    payload = {
        "video_name": video_path.name,
        "tag": tag,
        "fps": float(fps),
        "frames_processed": int(frames_processed),
        "touch_events_count": int(len(touch_events)),
        "touch_events": touch_events,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path



# ---------------------------------------
# Run prediction
# ---------------------------------------
def run_predict(tag: str, model_path: Path, source: Path, expected_strings: int = 16):
    if not model_path.exists():
        print(f"[SKIP] {tag}: missing {model_path}")
        return

    model, yolo_runtime = _load_yolo_model(model_path)
    yolo_kwargs = _yolo_predict_kwargs(yolo_runtime)
    _log_yolo_runtime(yolo_runtime)

    run_dir = OUT_PROJECT / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(source),
        save=False,       # disable default save
        **yolo_kwargs,
    )

    for r in results:
        corrected_img = draw_corrected_strings(r, expected_strings=expected_strings)

        stem = Path(r.path).stem if getattr(r, "path", None) else "image"
        out_img_path = run_dir / f"{stem}_corrected.jpg"
        out_txt_path = run_dir / f"{stem}_corrected.txt"

        strings = extract_corrected_strings(r, kpt_conf_thres=KPT_CONF_THRES, expected_strings=expected_strings)

        save_prediction_outputs(r, corrected_img, strings, out_img_path, out_txt_path)

        print(f"[OK] Saved corrected image -> {out_img_path}")
        print(f"[OK] Saved annotations -> {out_txt_path}")

    print(f"[DONE] {tag} saved in {run_dir}")


def create_video_writer(
    out_video_path: Path,
    fps: float,
    frame_size: tuple[int, int],
):
    """
    Create a VideoWriter using codec preference order.
    Returns (writer, codec_used, actual_output_path).
    """
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in WRITER_CODEC_PREFERENCE:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"[INFO] VideoWriter codec selected: {codec}")
            return writer, codec, out_video_path
        writer.release()

    raise RuntimeError(
        "Failed to initialize video writer with codecs: "
        + ", ".join(WRITER_CODEC_PREFERENCE)
    )


def _find_ffmpeg_executable() -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_media_stream_lines(video_path: Path) -> str | None:
    ffmpeg_exe = _find_ffmpeg_executable()
    if ffmpeg_exe is None or not video_path.exists():
        return None

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-i",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except Exception:
        return None

    text = b"\n".join([proc.stdout or b"", proc.stderr or b""])
    return text.decode("utf-8", errors="ignore")


def transcode_to_h264(
    in_path: Path,
    source_with_audio: Path | None = None,
    preset: str | None = "veryfast",
) -> tuple[Path, bool, bool]:
    """
    Transcode to browser-friendly H.264 when ffmpeg is available.
    Returns (final_path, transcoded_flag, output_has_audio_track).
    """
    ffmpeg_exe = _find_ffmpeg_executable()
    if ffmpeg_exe is None:
        return in_path, False, False

    temp_out = in_path.with_name(f"{in_path.stem}_h264{in_path.suffix}")
    use_audio_input = source_with_audio is not None and source_with_audio.exists()

    if use_audio_input:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(in_path),
            "-i",
            str(source_with_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
        ]
        if preset:
            cmd.extend(["-preset", str(preset)])
        cmd.extend([
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-shortest",
            str(temp_out),
        ])
    else:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(in_path),
            "-c:v",
            "libx264",
        ]
        if preset:
            cmd.extend(["-preset", str(preset)])
        cmd.extend([
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_out),
        ])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        if temp_out.exists():
            temp_out.unlink(missing_ok=True)
        return in_path, False, False

    shutil.move(str(temp_out), str(in_path))
    return in_path, True, detect_audio_tag(in_path)


def highlight_strikes_on_video(
    *,
    input_video_path: Path,
    output_video_path: Path,
    strings_by_frame_jsonl_path: Path,
    strike_events: list[dict],
    fps: float,
    hold_frames: int = 10,
    source_with_audio: Path | None = None,
    transcode_output: bool = True,
    transcode_preset: str | None = "veryfast",
) -> dict[str, object]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"opencv_missing:{exc}"}

    if not input_video_path.exists():
        return {"ok": False, "error": f"input_video_missing:{input_video_path}"}
    if not strings_by_frame_jsonl_path.exists():
        return {"ok": False, "error": f"strings_by_frame_missing:{strings_by_frame_jsonl_path}"}

    def _event_frame_index(ev: dict) -> int | None:
        debug = ev.get("decision_debug") if isinstance(ev.get("decision_debug"), dict) else {}
        if isinstance(debug, dict):
            peak = debug.get("peak_frame")
            if peak is not None:
                try:
                    return int(peak)
                except Exception:
                    pass
            event_frame = debug.get("event_frame_index")
            if event_frame is not None:
                try:
                    return int(event_frame)
                except Exception:
                    pass
        frame_idx_raw = ev.get("frame_index")
        if frame_idx_raw is not None:
            try:
                return int(frame_idx_raw)
            except Exception:
                pass
        t = ev.get("timestamp_sec")
        if t is None:
            t = ev.get("event_time")
        if t is None:
            t = ev.get("time_sec")
        if t is None:
            return None
        try:
            return int(round(float(t) * max(float(fps), 1e-6)))
        except Exception:
            return None

    def _struck_id(ev: dict) -> int | None:
        for key in ("struck_id", "struck_string_id"):
            sid = ev.get(key)
            if sid is None:
                continue
            try:
                return int(round(float(sid)))
            except Exception:
                continue
        return None

    highlights: dict[int, list[int]] = {}
    for ev in strike_events:
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("label") or ev.get("status") or "").strip().lower()
        if label and label != "strike":
            continue
        sid = _struck_id(ev)
        if sid is None:
            continue
        frame_idx = _event_frame_index(ev)
        if frame_idx is None:
            continue
        for f in range(int(frame_idx), int(frame_idx) + max(1, int(hold_frames))):
            highlights.setdefault(int(f), []).append(int(sid))

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        return {"ok": False, "error": f"cannot_open_video:{input_video_path}"}

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps_in <= 0:
        fps_in = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return {"ok": False, "error": "invalid_video_dimensions"}

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"ok": False, "error": f"cannot_open_writer:{output_video_path}"}

    frame_idx = 0
    strike_frames_drawn = 0
    last_strings_by_id: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    try:
        with strings_by_frame_jsonl_path.open("r", encoding="utf-8") as strings_file:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                record = None
                line = strings_file.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                        except Exception:
                            record = None
                strings_by_id: dict[int, tuple[tuple[int, int], tuple[int, int]]] = last_strings_by_id
                if isinstance(record, dict):
                    rec_idx = record.get("frame_index")
                    try:
                        rec_idx = int(rec_idx)
                    except Exception:
                        rec_idx = None
                    if rec_idx == frame_idx:
                        strings_by_id = {}
                        raw_strings = record.get("strings", [])
                        if isinstance(raw_strings, list):
                            for row in raw_strings:
                                if not isinstance(row, (list, tuple)) or len(row) < 5:
                                    continue
                                try:
                                    sid = int(row[0])
                                    p1 = (int(round(float(row[1]))), int(round(float(row[2]))))
                                    p2 = (int(round(float(row[3]))), int(round(float(row[4]))))
                                except Exception:
                                    continue
                                strings_by_id[sid] = (p1, p2)
                        last_strings_by_id = strings_by_id

                struck_sids = highlights.get(frame_idx, [])
                if struck_sids and strings_by_id:
                    highlight_overlay = frame.copy()
                    drawn_any = False
                    for sid in sorted(set(int(s) for s in struck_sids)):
                        endpoints = strings_by_id.get(int(sid))
                        if endpoints is None:
                            continue
                        p1, p2 = endpoints
                        cv2.line(highlight_overlay, p1, p2, (0, 120, 255), 8, cv2.LINE_AA)
                        cv2.line(highlight_overlay, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
                        cv2.circle(highlight_overlay, p1, 6, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(highlight_overlay, p2, 6, (0, 255, 255), -1, cv2.LINE_AA)
                        drawn_any = True
                    if drawn_any:
                        cv2.addWeighted(highlight_overlay, 0.75, frame, 0.25, 0.0, frame)
                        strike_frames_drawn += 1

                writer.write(frame)
                frame_idx += 1
    finally:
        cap.release()
        writer.release()

    transcoded = False
    audio_muxed = False
    final_path = output_video_path
    if transcode_output:
        try:
            final_path, transcoded, audio_muxed = transcode_to_h264(
                final_path,
                source_with_audio=source_with_audio,
                preset=transcode_preset,
            )
        except Exception:
            pass

    return {
        "ok": True,
        "output_video_path": str(final_path),
        "frames_annotated": int(frame_idx),
        "strike_highlight_frames": int(strike_frames_drawn),
        "transcoded": bool(transcoded),
        "audio_muxed": bool(audio_muxed),
    }


def detect_video_codec_tag(video_path: Path) -> str | None:
    """
    Best-effort video codec detection. Prefer FFmpeg's stream parser, then
    fall back to a shallow container tag scan for environments without FFmpeg.
    """
    if not video_path.exists():
        return None

    probe_text = _probe_media_stream_lines(video_path)
    if probe_text is not None:
        for line in probe_text.splitlines():
            if "Stream #" not in line or "Video:" not in line:
                continue
            codec_text = line.split("Video:", 1)[1].split(",", 1)[0].lower()
            if "h264" in codec_text or "avc1" in codec_text:
                return "avc1"
            if "mpeg4" in codec_text or "mp4v" in codec_text:
                return "mp4v"
            if "hevc" in codec_text or "h265" in codec_text or "hvc1" in codec_text or "hev1" in codec_text:
                return "hvc1"
            if "vp9" in codec_text or "vp09" in codec_text:
                return "vp09"
            if "av1" in codec_text or "av01" in codec_text:
                return "av01"
            return codec_text.strip().split(" ", 1)[0] or None
        return None

    try:
        data = video_path.read_bytes()
    except Exception:
        return None

    for tag in (b"avc1", b"hvc1", b"hev1", b"vp09", b"av01", b"mp4v"):
        if data.find(tag) != -1:
            return tag.decode("ascii")

    return None


def detect_audio_tag(video_path: Path) -> bool:
    """
    Best-effort audio stream detection. Prefer FFmpeg's stream parser, then
    fall back to a shallow container tag scan for environments without FFmpeg.
    """
    if not video_path.exists():
        return False

    probe_text = _probe_media_stream_lines(video_path)
    if probe_text is not None:
        for line in probe_text.splitlines():
            if "Stream #" in line and "Audio:" in line:
                return True
        return False

    try:
        data = video_path.read_bytes()
    except Exception:
        return False

    for tag in (b"mp4a", b"soun", b"ac-3", b"ec-3"):
        if data.find(tag) != -1:
            return True

    return False

def run_video_predict(
    tag: str,
    model_path: Path,
    video_path: Path,
    expected_strings: int = 16,
    string_infer_every_n: int = 1,
    save_video: bool = True,
    show_preview: bool = False,
    transcode_output: bool = True,
    transcode_preset: str | None = "veryfast",
    enable_hand_tracking: bool = True,
    hand_process_width: int = 0,
    hand_model_complexity: int = 1,
    hand_max_hands: int = 2,
    hand_min_detection_confidence: float = 0.5,
    hand_min_tracking_confidence: float = 0.4,
    left_hand_min_detection_confidence: float = 0.5,
    right_hand_min_detection_confidence: float = 0.5,
    left_hand_min_tracking_confidence: float = 0.4,
    right_hand_min_tracking_confidence: float = 0.4,
    draw_hand_labels: bool = False,
    hand_pipeline_enabled: bool | None = None,
):
    started_total = time.perf_counter()
    if not model_path.exists():
        print(f"[SKIP] {tag}: missing {model_path}")
        raise FileNotFoundError(f"Missing model: {model_path}")

    if enable_hand_tracking and mp is None:
        raise RuntimeError("Missing dependency: mediapipe. Install with: pip install mediapipe")

    if hand_model_complexity not in (0, 1):
        print(
            f"[WARN] Invalid hand_model_complexity={hand_model_complexity} for MediaPipe Hands. "
            "Using 1 (supported values: 0 or 1)."
        )
        hand_model_complexity = 1
    if string_infer_every_n < 1:
        print(
            f"[WARN] Invalid string_infer_every_n={string_infer_every_n}. "
            "Using 1 (run YOLO every frame)."
        )
        string_infer_every_n = 1
    left_hand_min_detection_confidence = float(
        hand_min_detection_confidence
        if left_hand_min_detection_confidence is None
        else left_hand_min_detection_confidence
    )
    right_hand_min_detection_confidence = float(
        hand_min_detection_confidence
        if right_hand_min_detection_confidence is None
        else right_hand_min_detection_confidence
    )
    left_hand_min_tracking_confidence = float(
        hand_min_tracking_confidence
        if left_hand_min_tracking_confidence is None
        else left_hand_min_tracking_confidence
    )
    right_hand_min_tracking_confidence = float(
        hand_min_tracking_confidence
        if right_hand_min_tracking_confidence is None
        else right_hand_min_tracking_confidence
    )

    model, yolo_runtime = _load_yolo_model(model_path)
    yolo_kwargs = _yolo_predict_kwargs(yolo_runtime)
    _log_yolo_runtime(yolo_runtime)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    out_video_path = None
    if save_video:
        out_video_path = OUT_PROJECT / tag / f"{video_path.stem}_annotated.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0

    writer = None
    writer_codec = None
    if save_video:
        writer, writer_codec, out_video_path = create_video_writer(
            out_video_path=out_video_path,
            fps=fps,
            frame_size=(w, h),
        )
    strings_by_frame_path = None
    strings_by_frame_file = None
    if save_video:
        strings_dir = TOUCH_EVENTS_DIR / tag
        strings_dir.mkdir(parents=True, exist_ok=True)
        strings_by_frame_path = strings_dir / f"{video_path.stem}_strings_by_frame.jsonl"
        strings_by_frame_file = strings_by_frame_path.open("w", encoding="utf-8")

    frame_count = 0
    input_frame_idx = 0
    string_inference_frames = 0
    hand_frames_detected = 0
    hand_landmarks_drawn = 0
    hand_fingertips_drawn = 0
    touch_events: list[dict] = []
    string_endpoint_samples: dict[int, list[tuple[float, float, float, float]]] = {}
    active_touches_by_fingertip: dict[tuple[str, str], int | None] = {}
    touch_distance_px = max(6.0, min(w, h) * TOUCH_DISTANCE_RATIO)
    print(f"[INFO] Running prediction on video -> {video_path}")
    if string_infer_every_n > 1:
        print(f"[INFO] String inference cadence enabled: YOLO runs every {string_infer_every_n} frames.")

    hand_input_queue = None
    hand_output_queue = None
    hand_worker_thread = None
    hands_detector_sync = None
    hands_detector_sync_left = None
    hands_detector_sync_right = None
    sync_tracking_state = {
        "left_center": None,
        "right_center": None,
    }
    pending_hand_results: dict[int, tuple] = {}
    pending_strings_by_idx: dict[int, list[tuple[int, tuple[float, float], tuple[float, float]]]] = {}
    next_output_idx = 0
    produced_frames = 0
    stop_requested = False
    hand_worker_error = None
    use_hand_pipeline = HAND_PIPELINE_ENABLED if hand_pipeline_enabled is None else hand_pipeline_enabled
    last_yolo_result = None
    last_frame_strings: list[tuple[int, tuple[float, float], tuple[float, float]]] = []
    last_harp_roi: tuple[int, int, int, int] | None = None

    if enable_hand_tracking and use_hand_pipeline:
        hand_input_queue = Queue(maxsize=HAND_PIPELINE_QUEUE_SIZE)
        hand_output_queue = Queue(maxsize=HAND_PIPELINE_QUEUE_SIZE)
        hand_worker_thread = Thread(
            target=hand_tracking_worker_loop,
            kwargs={
                "input_queue": hand_input_queue,
                "output_queue": hand_output_queue,
                "process_width": hand_process_width,
                "draw_labels": draw_hand_labels,
                "hand_max_hands": hand_max_hands,
                "hand_model_complexity": hand_model_complexity,
                "hand_min_detection_confidence": hand_min_detection_confidence,
                "hand_min_tracking_confidence": hand_min_tracking_confidence,
                "left_hand_min_detection_confidence": left_hand_min_detection_confidence,
                "right_hand_min_detection_confidence": right_hand_min_detection_confidence,
                "left_hand_min_tracking_confidence": left_hand_min_tracking_confidence,
                "right_hand_min_tracking_confidence": right_hand_min_tracking_confidence,
            },
            daemon=True,
        )
        hand_worker_thread.start()
        print("[INFO] Hand tracking pipeline enabled (YOLO GPU + MediaPipe worker thread).")
    elif enable_hand_tracking:
        hands_detector_sync = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=hand_max_hands,
            model_complexity=hand_model_complexity,
            min_detection_confidence=hand_min_detection_confidence,
            min_tracking_confidence=hand_min_tracking_confidence,
        )
        hands_detector_sync_left = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=hand_model_complexity,
            min_detection_confidence=left_hand_min_detection_confidence,
            min_tracking_confidence=left_hand_min_tracking_confidence,
        )
        hands_detector_sync_right = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=hand_model_complexity,
            min_detection_confidence=right_hand_min_detection_confidence,
            min_tracking_confidence=right_hand_min_tracking_confidence,
        )
        print("[INFO] Hand tracking running in safe mode (single-thread MediaPipe).")

    def collect_string_geometry_samples(
        frame_strings: list[tuple[int, tuple[float, float], tuple[float, float]]],
    ) -> None:
        for sid, left_pt, right_pt in frame_strings:
            key = int(sid)
            if key not in string_endpoint_samples:
                string_endpoint_samples[key] = []
            string_endpoint_samples[key].append(
                (
                    float(left_pt[0]),
                    float(left_pt[1]),
                    float(right_pt[0]),
                    float(right_pt[1]),
                )
            )

    def register_touch_events(
        frame_index: int,
        frame_strings: list[tuple[int, tuple[float, float], tuple[float, float]]],
        fingertip_points: list[dict],
    ) -> None:
        touches = detect_fingertip_string_touches(
            strings=frame_strings,
            fingertip_points=fingertip_points,
            max_touch_distance_px=touch_distance_px,
        )

        current_touch_map = {key: int(value["string_id"]) for key, value in touches.items()}
        all_keys = set(active_touches_by_fingertip.keys()) | set(current_touch_map.keys())
        timestamp_sec = round(frame_index / max(fps, 1e-6), 3)

        for key in all_keys:
            prev_sid = active_touches_by_fingertip.get(key)
            cur_touch = touches.get(key)
            cur_sid = current_touch_map.get(key)
            if prev_sid == cur_sid:
                continue

            active_touches_by_fingertip[key] = cur_sid
            if cur_sid is None:
                continue

            hand_label, tip_name = key
            finger_type = tip_name.replace("_tip", "") if tip_name.endswith("_tip") else tip_name
            touch_distance = float(cur_touch.get("distance_px", touch_distance_px)) if cur_touch is not None else touch_distance_px
            touch_conf = max(0.0, 1.0 - (touch_distance / max(touch_distance_px, 1e-6)))
            finger_point = cur_touch.get("finger_point", (0.0, 0.0)) if cur_touch is not None else (0.0, 0.0)
            contact_point = cur_touch.get("contact_point", finger_point) if cur_touch is not None else finger_point
            touch_events.append(
                {
                    "time_sec": timestamp_sec,
                    "timestamp_sec": timestamp_sec,
                    "frame_index": int(frame_index),
                    "hand": hand_label,
                    "hand_side": hand_label,
                    "fingertip": tip_name,
                    "finger_type": finger_type,
                    "string_id": int(cur_sid),
                    "touched_string_id": int(cur_sid),
                    "distance_px": touch_distance,
                    "touch_conf": float(touch_conf),
                    "contact_x": float(contact_point[0]),
                    "contact_y": float(contact_point[1]),
                    "finger_x": float(finger_point[0]),
                    "finger_y": float(finger_point[1]),
                }
            )

    def flush_ready_frames() -> None:
        nonlocal next_output_idx
        nonlocal frame_count
        nonlocal hand_frames_detected
        nonlocal hand_landmarks_drawn
        nonlocal hand_fingertips_drawn
        nonlocal stop_requested
        nonlocal hand_worker_error

        while next_output_idx in pending_hand_results:
            (
                output_frame,
                hands_detected,
                landmarks_drawn,
                fingertips_drawn,
                frame_strings,
                fingertip_points,
                worker_error,
            ) = pending_hand_results.pop(next_output_idx)

            if worker_error is not None and hand_worker_error is None:
                hand_worker_error = worker_error

            if hands_detected > 0:
                hand_frames_detected += 1
            hand_landmarks_drawn += landmarks_drawn
            hand_fingertips_drawn += fingertips_drawn
            register_touch_events(
                frame_index=next_output_idx,
                frame_strings=frame_strings,
                fingertip_points=fingertip_points,
            )

            if show_preview:
                cv2.imshow(f"Live Harp Strings + Hands - {tag}", output_frame)

            if writer is not None:
                writer.write(output_frame)
            if strings_by_frame_file is not None:
                _write_strings_by_frame_line(
                    strings_by_frame_file,
                    frame_index=next_output_idx,
                    frame_strings=frame_strings,
                )

            frame_count += 1
            next_output_idx += 1

            if show_preview and cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                stop_requested = True
                break

    loop_started_at = time.perf_counter()
    try:
        while not stop_requested:
            ret, frame = cap.read()
            if not ret:
                break

            run_string_inference = (
                last_yolo_result is None
                or (input_frame_idx % string_infer_every_n == 0)
            )

            if run_string_inference:
                results = model.predict(
                    source=frame,
                    **yolo_kwargs,
                )

                r = results[0]
                last_yolo_result = r
                last_harp_roi = extract_harp_roi_from_result(r)
                last_frame_strings = extract_corrected_strings(
                    r,
                    kpt_conf_thres=KPT_CONF_THRES,
                    expected_strings=expected_strings,
                )
                collect_string_geometry_samples(last_frame_strings)
                string_inference_frames += 1
            else:
                r = last_yolo_result

            harp_roi = last_harp_roi
            frame_strings = last_frame_strings
            annotated = draw_corrected_strings(
                r,
                expected_strings=expected_strings,
                strings=frame_strings,
                base_frame=frame,
                draw_strings=False,
                draw_harp_annotation=False,
            )

            if hand_worker_thread is not None:
                pending_strings_by_idx[produced_frames] = frame_strings
                hand_input_queue.put((produced_frames, frame.copy(), annotated, harp_roi))
                produced_frames += 1

                while True:
                    try:
                        worker_item = hand_output_queue.get_nowait()
                    except Empty:
                        break

                    if worker_item is None:
                        continue

                    (
                        output_idx,
                        output_frame,
                        hands_detected,
                        landmarks_drawn,
                        fingertips_drawn,
                        fingertip_points,
                        worker_error,
                    ) = worker_item
                    strings_for_idx = pending_strings_by_idx.pop(output_idx, [])
                    pending_hand_results[output_idx] = (
                        output_frame,
                        hands_detected,
                        landmarks_drawn,
                        fingertips_drawn,
                        strings_for_idx,
                        fingertip_points,
                        worker_error,
                    )
                    flush_ready_frames()

                    if stop_requested or hand_worker_error is not None:
                        break

                if hand_worker_error is not None:
                    raise RuntimeError(f"Hand tracking worker failed: {hand_worker_error}") from hand_worker_error
            else:
                if hands_detector_sync is not None:
                    hands_detected, landmarks_drawn, fingertips_drawn, fingertip_points = detect_and_draw_hand_landmarks(
                        source_frame=frame,
                        draw_frame=annotated,
                        hands_detector=hands_detector_sync,
                        hands_detector_left=hands_detector_sync_left,
                        hands_detector_right=hands_detector_sync_right,
                        process_width=hand_process_width,
                        draw_labels=draw_hand_labels,
                        tracking_state=sync_tracking_state,
                        harp_roi=harp_roi,
                    )
                    if hands_detected > 0:
                        hand_frames_detected += 1
                    hand_landmarks_drawn += landmarks_drawn
                    hand_fingertips_drawn += fingertips_drawn
                    register_touch_events(
                        frame_index=frame_count,
                        frame_strings=frame_strings,
                        fingertip_points=fingertip_points,
                    )

                if show_preview:
                    cv2.imshow(f"Live Harp Strings + Hands - {tag}", annotated)

                if writer is not None:
                    writer.write(annotated)
                if strings_by_frame_file is not None:
                    _write_strings_by_frame_line(
                        strings_by_frame_file,
                        frame_index=frame_count,
                        frame_strings=frame_strings,
                    )
                frame_count += 1

                if show_preview and cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                    stop_requested = True

            input_frame_idx += 1

        if hand_worker_thread is not None:
            hand_input_queue.put(None)
            while next_output_idx < produced_frames:
                worker_item = hand_output_queue.get()
                if worker_item is None:
                    continue

                (
                    output_idx,
                    output_frame,
                    hands_detected,
                    landmarks_drawn,
                    fingertips_drawn,
                    fingertip_points,
                    worker_error,
                ) = worker_item
                strings_for_idx = pending_strings_by_idx.pop(output_idx, [])
                pending_hand_results[output_idx] = (
                    output_frame,
                    hands_detected,
                    landmarks_drawn,
                    fingertips_drawn,
                    strings_for_idx,
                    fingertip_points,
                    worker_error,
                )
                flush_ready_frames()

                if hand_worker_error is not None:
                    break

            hand_worker_thread.join()
            if hand_worker_error is not None:
                raise RuntimeError(f"Hand tracking worker failed: {hand_worker_error}") from hand_worker_error
    finally:
        if hand_worker_thread is not None and hand_worker_thread.is_alive():
            try:
                hand_input_queue.put_nowait(None)
            except Full:
                pass
            hand_worker_thread.join(timeout=2.0)
        if hands_detector_sync is not None:
            hands_detector_sync.close()
        if hands_detector_sync_left is not None:
            hands_detector_sync_left.close()
        if hands_detector_sync_right is not None:
            hands_detector_sync_right.close()
        if strings_by_frame_file is not None:
            strings_by_frame_file.close()

    cap.release()
    if writer is not None:
        writer.release()

    if show_preview:
        cv2.destroyAllWindows()

    processing_elapsed_sec = round(float(time.perf_counter() - loop_started_at), 3)

    print(f"[DONE] Video finished. Frames processed: {frame_count}")
    transcoded = False
    audio_muxed = False
    final_codec = writer_codec
    has_audio_track = False
    transcode_elapsed_sec = 0.0
    if save_video and out_video_path is not None:
        if transcode_output:
            # Optional ffmpeg post-processing:
            # 1) browser-safe H.264 output
            # 2) preserve source audio where available
            transcode_started_at = time.perf_counter()
            out_video_path, transcoded, audio_muxed = transcode_to_h264(
                out_video_path,
                source_with_audio=video_path,
                preset=transcode_preset,
            )
            transcode_elapsed_sec = round(float(time.perf_counter() - transcode_started_at), 3)

        detected = detect_video_codec_tag(out_video_path)
        if detected is not None:
            final_codec = detected
        has_audio_track = detect_audio_tag(out_video_path)

        print(f"[SAVED] Annotated video -> {out_video_path}")
        if transcoded:
            print("[INFO] Output transcoded to H.264 for browser playback.")
        elif transcode_output and writer_codec not in WEB_FRIENDLY_CODECS:
            print("[WARN] H.264 transcoding not available; browser playback may fail for some clients.")
        elif not transcode_output:
            print("[INFO] Initial H.264 transcode deferred for later pipeline stages.")
        if audio_muxed and has_audio_track:
            print("[INFO] Source audio preserved in output video.")
        elif not has_audio_track and not transcode_output:
            print("[INFO] Intermediate OpenCV output has no audio; final pipeline may remux source audio.")
        elif not has_audio_track:
            print("[WARN] Output video has no audio track.")

    print(f"[INFO] Frames with hands detected: {hand_frames_detected}")
    print(f"[INFO] Frames with string YOLO inference: {string_inference_frames}")
    print(f"[INFO] Hand landmarks drawn: {hand_landmarks_drawn}")
    print(f"[INFO] Hand fingertips drawn: {hand_fingertips_drawn}")
    print(f"[INFO] Touch events detected: {len(touch_events)}")
    touch_events_json_path = save_touch_events_json(
        tag=tag,
        video_path=video_path,
        touch_events=touch_events,
        fps=fps,
        frames_processed=frame_count,
    )
    print(f"[INFO] Touch events JSON saved: {touch_events_json_path}")

    string_geometries: list[dict] = []
    for sid in sorted(string_endpoint_samples.keys()):
        samples = string_endpoint_samples[sid]
        if not samples:
            continue
        arr = np.asarray(samples, dtype=np.float32)
        med = np.median(arr, axis=0)
        string_geometries.append(
            {
                "string_id": int(sid),
                "endpoints": [
                    [float(med[0]), float(med[1])],
                    [float(med[2]), float(med[3])],
                ],
                "sample_count": int(len(samples)),
            }
        )

    total_elapsed_sec = round(float(time.perf_counter() - started_total), 3)
    return {
        "out_video_path": out_video_path,
        "frames_processed": frame_count,
        "fps": float(fps),
        "source_video": video_path,
        "model_path": model_path,
        "writer_codec": writer_codec,
        "final_codec": final_codec,
        "transcoded": transcoded,
        "audio_muxed": audio_muxed,
        "has_audio_track": has_audio_track,
        "yolo_runtime": dict(yolo_runtime),
        "hand_tracking_enabled": enable_hand_tracking,
        "hand_pipeline_enabled": enable_hand_tracking and use_hand_pipeline,
        "string_infer_every_n": int(string_infer_every_n),
        "string_inference_frames": int(string_inference_frames),
        "hand_frames_detected": hand_frames_detected,
        "hand_landmarks_drawn": hand_landmarks_drawn,
        "hand_fingertips_drawn": hand_fingertips_drawn,
        "touch_events_count": len(touch_events),
        "touch_events": touch_events,
        "touch_events_json_path": str(touch_events_json_path),
        "string_geometries": string_geometries,
        "strings_by_frame_jsonl_path": str(strings_by_frame_path) if strings_by_frame_path is not None else None,
        "processing_elapsed_sec": float(processing_elapsed_sec),
        "transcode_elapsed_sec": float(transcode_elapsed_sec),
        "elapsed_sec": float(total_elapsed_sec),
    }


def run_video_live(
    tag: str,
    model_path: Path,
    video_path: Path,
    expected_strings: int = 16,
    string_infer_every_n: int = 1,
    left_hand_min_detection_confidence: float | None = None,
    right_hand_min_detection_confidence: float | None = None,
    left_hand_min_tracking_confidence: float | None = None,
    right_hand_min_tracking_confidence: float | None = None,
    hand_pipeline_enabled: bool | None = None,
):
    run_video_predict(
        tag=tag,
        model_path=model_path,
        video_path=video_path,
        expected_strings=expected_strings,
        string_infer_every_n=string_infer_every_n,
        left_hand_min_detection_confidence=left_hand_min_detection_confidence,
        right_hand_min_detection_confidence=right_hand_min_detection_confidence,
        left_hand_min_tracking_confidence=left_hand_min_tracking_confidence,
        right_hand_min_tracking_confidence=right_hand_min_tracking_confidence,
        save_video=True,
        show_preview=True,
        hand_pipeline_enabled=hand_pipeline_enabled,
    )



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live harp string detection on video")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--out", type=str, default=str(OUT_PROJECT), help="Output folder root")
    p.add_argument("--model", type=str, default="best", choices=list(MODELS.keys()))
    p.add_argument("--expected_strings", type=int, default=16)
    p.add_argument(
        "--string_skip",
        type=int,
        default=1,
        help="Run YOLO string inference every N frames (1 = every frame).",
    )
    p.add_argument("--left_hand_min_detection_confidence", type=float, default=None)
    p.add_argument("--right_hand_min_detection_confidence", type=float, default=None)
    p.add_argument("--left_hand_min_tracking_confidence", type=float, default=None)
    p.add_argument("--right_hand_min_tracking_confidence", type=float, default=None)
    p.add_argument(
        "--hand_pipeline",
        dest="hand_pipeline_enabled",
        action="store_true",
        help="Enable parallel MediaPipe worker-thread pipeline.",
    )
    p.add_argument(
        "--no_hand_pipeline",
        dest="hand_pipeline_enabled",
        action="store_false",
        help="Disable MediaPipe worker-thread pipeline and use sync mode.",
    )
    p.set_defaults(hand_pipeline_enabled=None)

    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUT_PROJECT = Path(args.out)
    OUT_PROJECT.mkdir(parents=True, exist_ok=True)

    run_video_live(
        args.model,
        MODELS[args.model],
        video_path,
        expected_strings=args.expected_strings,
        string_infer_every_n=args.string_skip,
        left_hand_min_detection_confidence=args.left_hand_min_detection_confidence,
        right_hand_min_detection_confidence=args.right_hand_min_detection_confidence,
        left_hand_min_tracking_confidence=args.left_hand_min_tracking_confidence,
        right_hand_min_tracking_confidence=args.right_hand_min_tracking_confidence,
        hand_pipeline_enabled=args.hand_pipeline_enabled,
    )
