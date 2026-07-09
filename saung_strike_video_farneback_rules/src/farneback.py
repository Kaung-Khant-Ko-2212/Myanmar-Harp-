from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


DEFAULT_DY_THR = 0.5


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    _require_cv2()
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("ROI must be grayscale (H,W) or BGR (H,W,3).")

    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def _coerce_mask(mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.ones((h, w), dtype=np.uint8)

    m = np.asarray(mask)
    if m.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {m.shape} does not match ROI shape {(h, w)}.")
    if m.ndim > 2:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _resolve_params(farneback_params: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(farneback_params or {})
    return {
        "pyr_scale": float(p.get("pyr_scale", 0.5)),
        "levels": int(p.get("levels", 3)),
        "winsize": int(p.get("winsize", 15)),
        "iterations": int(p.get("iterations", 3)),
        "poly_n": int(p.get("poly_n", 5)),
        "poly_sigma": float(p.get("poly_sigma", 1.2)),
        "flags": int(p.get("flags", 0)),
        "pre_blur": bool(p.get("pre_blur", True)),
    }


def preprocess_roi_for_flow(roi: np.ndarray, pre_blur: bool) -> np.ndarray:
    gray = _to_gray_uint8(roi)
    if pre_blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def compute_farneback_flow(
    prev_roi: np.ndarray,
    roi: np.ndarray,
    *,
    farneback_params: dict[str, Any] | None = None,
) -> np.ndarray:
    _require_cv2()
    params = _resolve_params(farneback_params)
    prev = preprocess_roi_for_flow(prev_roi, pre_blur=params["pre_blur"])
    curr = preprocess_roi_for_flow(roi, pre_blur=params["pre_blur"])
    if prev.shape != curr.shape:
        raise ValueError(f"ROI shapes must match. Got prev={prev.shape}, curr={curr.shape}")

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        params["pyr_scale"],
        params["levels"],
        params["winsize"],
        params["iterations"],
        params["poly_n"],
        params["poly_sigma"],
        params["flags"],
    )
    return flow.astype(np.float32)


def masked_dy_stats(
    flow: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    dy_thr: float = DEFAULT_DY_THR,
) -> dict[str, float]:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Flow must have shape (H, W, 2). Got {flow.shape}.")

    h, w = flow.shape[:2]
    mask_u8 = _coerce_mask(mask, (h, w))
    valid = mask_u8 > 0
    valid_count = int(np.count_nonzero(valid))
    total_count = int(h * w)

    if valid_count == 0:
        return {
            "mean_abs_dy": 0.0,
            "p90_abs_dy": 0.0,
            "frac_high_dy": 0.0,
            "valid_count": 0.0,
            "valid_frac": 0.0,
            "dy_thr": float(dy_thr),
            "vib_score_frame": 0.0,
        }

    dy = flow[..., 1]
    abs_dy = np.abs(dy[valid])
    mean_abs_dy = float(np.mean(abs_dy))
    p90_abs_dy = float(np.percentile(abs_dy, 90))
    frac_high_dy = float(np.mean(abs_dy >= float(dy_thr)))

    vib_score_frame = float(0.6 * mean_abs_dy + 0.3 * p90_abs_dy + 0.1 * frac_high_dy)
    return {
        "mean_abs_dy": mean_abs_dy,
        "p90_abs_dy": p90_abs_dy,
        "frac_high_dy": frac_high_dy,
        "valid_count": float(valid_count),
        "valid_frac": float(valid_count / max(1, total_count)),
        "dy_thr": float(dy_thr),
        "vib_score_frame": vib_score_frame,
    }


@dataclass
class FarnebackFrameResult:
    flow: np.ndarray
    dy: np.ndarray
    mask: np.ndarray
    mean_abs_dy: float
    p90_abs_dy: float
    frac_high_dy: float
    vib_score_frame: float
    valid_count: int
    valid_frac: float
    dy_thr: float


def compute_vibration_frame(
    prev_roi: np.ndarray,
    roi: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    farneback_params: dict[str, Any] | None = None,
    dy_thr: float = DEFAULT_DY_THR,
) -> FarnebackFrameResult:
    flow = compute_farneback_flow(
        prev_roi=prev_roi,
        roi=roi,
        farneback_params=farneback_params,
    )
    stats = masked_dy_stats(flow, mask=mask, dy_thr=dy_thr)
    dy = flow[..., 1].astype(np.float32)
    m = _coerce_mask(mask, dy.shape[:2])
    return FarnebackFrameResult(
        flow=flow,
        dy=dy,
        mask=m,
        mean_abs_dy=float(stats["mean_abs_dy"]),
        p90_abs_dy=float(stats["p90_abs_dy"]),
        frac_high_dy=float(stats["frac_high_dy"]),
        vib_score_frame=float(stats["vib_score_frame"]),
        valid_count=int(stats["valid_count"]),
        valid_frac=float(stats["valid_frac"]),
        dy_thr=float(stats["dy_thr"]),
    )

