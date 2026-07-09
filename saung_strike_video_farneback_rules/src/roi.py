from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from .strings import sample_mid_segment
except ImportError:  # pragma: no cover
    from src.strings import sample_mid_segment


Point = tuple[float, float]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


def _to_gray_uint8(frame: np.ndarray) -> np.ndarray:
    _require_cv2()
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def _roi_frame_axes(
    string_geom: dict[str, Any] | Any,
    trim_ends_ratio: float,
) -> tuple[Point, Point, Point]:
    p1, p2 = sample_mid_segment(string_geom, trim_ends_ratio=trim_ends_ratio)
    cx = 0.5 * (p1[0] + p2[0])
    cy = 0.5 * (p1[1] + p2[1])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    mag = float(np.hypot(dx, dy))
    if mag <= 1e-6:
        raise ValueError("Degenerate mid-segment: cannot compute ROI axes.")
    direction = (dx / mag, dy / mag)
    normal = (-direction[1], direction[0])
    center = (cx, cy)
    return center, direction, normal


def roi_box_corners(
    string_geom: dict[str, Any] | Any,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float = 0.15,
) -> np.ndarray:
    if roi_w <= 1 or roi_h <= 1:
        raise ValueError("roi_w and roi_h must be > 1.")

    center, direction, normal = _roi_frame_axes(string_geom, trim_ends_ratio=trim_ends_ratio)
    half_w = 0.5 * (float(roi_w) - 1.0)
    half_h = 0.5 * (float(roi_h) - 1.0)

    dx, dy = direction
    nx, ny = normal
    cx, cy = center

    top_left = (cx - dx * half_w - nx * half_h, cy - dy * half_w - ny * half_h)
    top_right = (cx + dx * half_w - nx * half_h, cy + dy * half_w - ny * half_h)
    bottom_right = (cx + dx * half_w + nx * half_h, cy + dy * half_w + ny * half_h)
    bottom_left = (cx - dx * half_w + nx * half_h, cy - dy * half_w + ny * half_h)

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def image_to_roi_affine(
    string_geom: dict[str, Any] | Any,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float = 0.15,
) -> np.ndarray:
    _require_cv2()
    corners = roi_box_corners(
        string_geom=string_geom,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
    )

    src = np.array([corners[0], corners[1], corners[3]], dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(roi_w - 1), 0.0],
            [0.0, float(roi_h - 1)],
        ],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)


def roi_inside_fraction(
    corners: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0

    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    bbox_w = max(0.0, x_max - x_min)
    bbox_h = max(0.0, y_max - y_min)
    bbox_area = bbox_w * bbox_h
    if bbox_area <= 1e-6:
        return 0.0

    clip_x0 = max(0.0, x_min)
    clip_y0 = max(0.0, y_min)
    clip_x1 = min(float(frame_w - 1), x_max)
    clip_y1 = min(float(frame_h - 1), y_max)
    clip_w = max(0.0, clip_x1 - clip_x0)
    clip_h = max(0.0, clip_y1 - clip_y0)
    return float((clip_w * clip_h) / bbox_area)


def extract_rotated_roi(
    frame: np.ndarray,
    string_geom: dict[str, Any] | Any,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float = 0.15,
    border_mode: str = "replicate",
    constant_border_value: int = 0,
    reject_if_out_of_frame: bool = False,
    min_inside_fraction: float = 0.95,
    return_debug: bool = False,
) -> tuple[np.ndarray | None, np.ndarray] | tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    _require_cv2()
    if frame is None:
        raise ValueError("frame cannot be None.")
    if roi_w <= 1 or roi_h <= 1:
        raise ValueError("roi_w and roi_h must be > 1.")

    gray = _to_gray_uint8(frame)
    corners = roi_box_corners(
        string_geom=string_geom,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
    )
    inside_fraction = roi_inside_fraction(
        corners,
        frame_w=int(gray.shape[1]),
        frame_h=int(gray.shape[0]),
    )
    mat_img_to_roi = image_to_roi_affine(
        string_geom=string_geom,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
    )
    border_mode_text = str(border_mode).strip().lower()
    reject_for_border = border_mode_text == "reject"
    debug: dict[str, Any] = {
        "border_mode": border_mode_text,
        "constant_border_value": int(constant_border_value),
        "inside_fraction": float(inside_fraction),
        "min_inside_fraction": float(min_inside_fraction),
        "roi_rejected": False,
        "reject_reason": None,
    }
    if (reject_if_out_of_frame or reject_for_border) and inside_fraction < float(min_inside_fraction):
        debug["roi_rejected"] = True
        debug["reject_reason"] = "roi_out_of_frame"
        if return_debug:
            return None, mat_img_to_roi, debug
        return None, mat_img_to_roi

    if border_mode_text == "constant":
        border_mode_cv = cv2.BORDER_CONSTANT
        border_value = int(constant_border_value)
    else:
        border_mode_cv = cv2.BORDER_REPLICATE
        border_value = 0

    roi = cv2.warpAffine(
        gray,
        mat_img_to_roi,
        (int(roi_w), int(roi_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode_cv,
        borderValue=border_value,
    )
    if roi.dtype != np.uint8:
        roi = np.clip(roi, 0, 255).astype(np.uint8)
    if return_debug:
        return roi, mat_img_to_roi, debug
    return roi, mat_img_to_roi
