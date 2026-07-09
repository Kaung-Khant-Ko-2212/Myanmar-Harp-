from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


Point = tuple[float, float]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


def _ensure_affine(mat_img_to_roi: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat_img_to_roi, dtype=np.float32)
    if mat.shape != (2, 3):
        raise ValueError(f"Expected affine shape (2, 3), got {mat.shape}.")
    return mat


def _parse_hand_bbox_xyxy(hand_bbox: Any | None) -> tuple[float, float, float, float] | None:
    if hand_bbox is None:
        return None

    if isinstance(hand_bbox, dict):
        if all(k in hand_bbox for k in ("x1", "y1", "x2", "y2")):
            x1 = float(hand_bbox["x1"])
            y1 = float(hand_bbox["y1"])
            x2 = float(hand_bbox["x2"])
            y2 = float(hand_bbox["y2"])
        elif all(k in hand_bbox for k in ("left", "top", "right", "bottom")):
            x1 = float(hand_bbox["left"])
            y1 = float(hand_bbox["top"])
            x2 = float(hand_bbox["right"])
            y2 = float(hand_bbox["bottom"])
        else:
            raise ValueError("hand_bbox dict must contain x1/y1/x2/y2 or left/top/right/bottom.")
    elif isinstance(hand_bbox, (list, tuple)) and len(hand_bbox) >= 4:
        x1, y1, x2, y2 = float(hand_bbox[0]), float(hand_bbox[1]), float(hand_bbox[2]), float(hand_bbox[3])
    else:
        raise ValueError("hand_bbox must be dict or [x1, y1, x2, y2].")

    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    return x_lo, y_lo, x_hi, y_hi


def _parse_finger_point(finger_point_img: Any | None) -> Point | None:
    if finger_point_img is None:
        return None
    if isinstance(finger_point_img, dict):
        if "x" in finger_point_img and "y" in finger_point_img:
            return float(finger_point_img["x"]), float(finger_point_img["y"])
        raise ValueError("finger_point_img dict must contain x and y.")
    if isinstance(finger_point_img, (list, tuple)) and len(finger_point_img) >= 2:
        return float(finger_point_img[0]), float(finger_point_img[1])
    raise ValueError("finger_point_img must be dict or [x, y].")


def transform_points_image_to_roi(
    points_img_xy: np.ndarray | list[Point] | tuple[Point, ...],
    mat_img_to_roi: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_img_xy, dtype=np.float32).reshape(-1, 2)
    mat = _ensure_affine(mat_img_to_roi)
    a = mat[:, :2]
    b = mat[:, 2]
    return points @ a.T + b


def _roi_scale_from_affine(mat_img_to_roi: np.ndarray) -> float:
    mat = _ensure_affine(mat_img_to_roi)
    sx = float(np.hypot(mat[0, 0], mat[1, 0]))
    sy = float(np.hypot(mat[0, 1], mat[1, 1]))
    return max(1e-6, 0.5 * (sx + sy))


def create_roi_mask(
    *,
    roi_h: int,
    roi_w: int,
    center_band_h: int,
    mat_img_to_roi: np.ndarray,
    hand_bbox_img_xyxy: Any | None = None,
    finger_point_img: Any | None = None,
    contact_point_img: Any | None = None,
    hand_mask_expand_px: float = 8.0,
    finger_radius_px: float = 0.0,
    mode: str = "finger_point",
    contact_band_exclusion_px: float = 10.0,
    mask_contact_region: bool = True,
) -> np.ndarray:
    _require_cv2()
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError("roi_h and roi_w must be positive.")
    if center_band_h <= 0:
        raise ValueError("center_band_h must be positive.")

    _ensure_affine(mat_img_to_roi)
    mask = np.zeros((int(roi_h), int(roi_w)), dtype=np.uint8)

    band_h = min(int(center_band_h), int(roi_h))
    y0 = int(max(0, (roi_h - band_h) // 2))
    y1 = int(min(roi_h, y0 + band_h))
    mask[y0:y1, :] = 1

    # Exclusion mask in ROI coordinates: 1 means remove pixel.
    exclusion = np.zeros_like(mask, dtype=np.uint8)
    mode_text = str(mode).strip().lower()
    roi_scale = _roi_scale_from_affine(mat_img_to_roi)

    parsed_bbox = _parse_hand_bbox_xyxy(hand_bbox_img_xyxy)
    if parsed_bbox is not None and mode_text == "hand_bbox":
        x1, y1, x2, y2 = parsed_bbox
        corners_img = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        corners_roi = transform_points_image_to_roi(corners_img, mat_img_to_roi)
        poly = np.round(corners_roi).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(exclusion, poly, color=1)
    else:
        parsed_finger = _parse_finger_point(finger_point_img)
        if parsed_finger is not None and mode_text in {"finger_point", "finger_point_plus_contact_band", "hand_bbox"}:
            finger_roi = transform_points_image_to_roi(np.array([parsed_finger], dtype=np.float32), mat_img_to_roi)[0]
            radius_roi = (float(finger_radius_px) + float(hand_mask_expand_px)) * roi_scale
            radius_roi_int = max(1, int(round(radius_roi)))
            cx = int(round(float(finger_roi[0])))
            cy = int(round(float(finger_roi[1])))
            cv2.circle(exclusion, (cx, cy), radius_roi_int, color=1, thickness=-1, lineType=cv2.LINE_AA)

    parsed_contact = _parse_finger_point(contact_point_img)
    if (
        parsed_contact is not None
        and mask_contact_region
        and mode_text in {"finger_point_plus_contact_band", "hand_bbox"}
    ):
        contact_roi = transform_points_image_to_roi(np.array([parsed_contact], dtype=np.float32), mat_img_to_roi)[0]
        band_half_w = max(1, int(round(float(contact_band_exclusion_px) * roi_scale)))
        cx = int(round(float(contact_roi[0])))
        x0 = max(0, cx - band_half_w)
        x1 = min(int(roi_w), cx + band_half_w + 1)
        exclusion[:, x0:x1] = 1

    mask[exclusion > 0] = 0
    return mask.astype(np.uint8)


def overlay_mask_on_roi(
    roi: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.45,
    keep_color_bgr: tuple[int, int, int] = (0, 220, 0),
    drop_color_bgr: tuple[int, int, int] = (0, 0, 220),
) -> np.ndarray:
    _require_cv2()
    if roi.ndim == 2:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    elif roi.ndim == 3 and roi.shape[2] == 3:
        roi_bgr = roi.copy()
    else:
        raise ValueError("roi must be grayscale (H,W) or BGR (H,W,3).")

    m = np.asarray(mask, dtype=np.uint8)
    if m.shape[:2] != roi_bgr.shape[:2]:
        raise ValueError(f"mask shape {m.shape} does not match ROI shape {roi_bgr.shape[:2]}.")

    out = roi_bgr.copy()
    keep_idx = m > 0
    drop_idx = ~keep_idx

    keep_layer = np.zeros_like(out, dtype=np.uint8)
    keep_layer[:, :] = keep_color_bgr
    drop_layer = np.zeros_like(out, dtype=np.uint8)
    drop_layer[:, :] = drop_color_bgr

    if np.any(keep_idx):
        out[keep_idx] = cv2.addWeighted(
            out[keep_idx],
            1.0 - float(alpha),
            keep_layer[keep_idx],
            float(alpha),
            0.0,
        )
    if np.any(drop_idx):
        out[drop_idx] = cv2.addWeighted(
            out[drop_idx],
            1.0 - float(alpha),
            drop_layer[drop_idx],
            float(alpha),
            0.0,
        )
    return out


def save_mask_debug_images(
    *,
    roi: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    stem: str,
    alpha: float = 0.45,
) -> dict[str, Path]:
    _require_cv2()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_path = out_dir / f"{stem}_roi.png"
    mask_path = out_dir / f"{stem}_mask.png"
    overlay_path = out_dir / f"{stem}_overlay.png"

    mask_vis = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    overlay = overlay_mask_on_roi(roi=roi, mask=mask, alpha=alpha)

    if roi.ndim == 2:
        roi_to_save = roi
    elif roi.ndim == 3 and roi.shape[2] == 3:
        roi_to_save = roi
    else:
        raise ValueError("roi must be grayscale (H,W) or BGR (H,W,3).")

    cv2.imwrite(str(roi_path), roi_to_save)
    cv2.imwrite(str(mask_path), mask_vis)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "roi_path": roi_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
    }
