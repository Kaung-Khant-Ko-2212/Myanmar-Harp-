from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.farneback import DEFAULT_DY_THR, compute_vibration_frame
from src.mask import create_roi_mask
from src.roi import extract_rotated_roi, roi_box_corners
from src.strings import StringGeometry, load_strings_json
from src.video_reader import VideoReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Farneback dy heatmap around one event."
    )
    p.add_argument("--video", type=Path, default=REPO_ROOT / "video.mp4")
    p.add_argument("--strings", type=Path, default=REPO_ROOT / "strings.json")
    p.add_argument(
        "--events",
        type=Path,
        default=REPO_ROOT / "touch_events.csv",
        help="Event file (.csv from this repo or backend touch_events .json).",
    )
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    p.add_argument("--event-index", type=int, default=0)
    p.add_argument("--string-id", type=str, default="", help="Override event string id.")
    p.add_argument("--window-before", type=int, default=10)
    p.add_argument("--window-after", type=int, default=10)
    p.add_argument("--dy-thr", type=float, default=DEFAULT_DY_THR)
    p.add_argument("--heatmap-clip", type=float, default=3.0)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "flow_debug.mp4",
    )
    p.add_argument(
        "--enable-stabilization",
        dest="enable_stabilization",
        action="store_true",
        help="Use stabilized frames before ROI extraction.",
    )
    p.add_argument(
        "--no-stabilization",
        dest="enable_stabilization",
        action="store_false",
        help="Use raw frames without stabilization.",
    )
    p.set_defaults(enable_stabilization=False)
    return p.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _normalize_string_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                ts_raw = row.get("timestamp_sec") or row.get("time_sec")
                sid_raw = row.get("touched_string_id") or row.get("string_id")
                if ts_raw is None or sid_raw is None:
                    continue
                try:
                    timestamp_sec = float(ts_raw)
                except Exception:
                    continue

                finger_point = None
                if row.get("finger_x") not in (None, "") and row.get("finger_y") not in (None, ""):
                    try:
                        finger_point = (float(row["finger_x"]), float(row["finger_y"]))
                    except Exception:
                        finger_point = None

                out.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "string_id": _normalize_string_id(sid_raw),
                        "finger_point_img": finger_point,
                        "raw": row,
                    }
                )
        return out

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw_events = data.get("touch_events", []) if isinstance(data, dict) else data
        out: list[dict[str, Any]] = []
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            ts_raw = item.get("timestamp_sec", item.get("time_sec"))
            sid_raw = item.get("touched_string_id", item.get("string_id"))
            if ts_raw is None or sid_raw is None:
                continue
            try:
                timestamp_sec = float(ts_raw)
            except Exception:
                continue
            out.append(
                {
                    "timestamp_sec": timestamp_sec,
                    "string_id": _normalize_string_id(sid_raw),
                    "finger_point_img": None,
                    "raw": item,
                }
            )
        return out

    raise ValueError(f"Unsupported events file type: {path.suffix}")


def _dy_heatmap(dy: np.ndarray, mask: np.ndarray, clip_abs_px: float) -> np.ndarray:
    clip_abs_px = max(1e-6, float(clip_abs_px))
    abs_dy = np.abs(np.asarray(dy, dtype=np.float32))
    norm = np.clip(abs_dy / clip_abs_px, 0.0, 1.0)
    gray = np.round(norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    color[m == 0] = (15, 15, 15)
    return color


def _draw_roi_box(
    frame: np.ndarray,
    string_geom: StringGeometry,
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float,
) -> None:
    corners = roi_box_corners(
        string_geom=string_geom,
        roi_w=roi_w,
        roi_h=roi_h,
        trim_ends_ratio=trim_ends_ratio,
    )
    pts = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    cfg = _load_yaml(args.config)
    roi_cfg = cfg.get("ROI", {}) if isinstance(cfg.get("ROI"), dict) else {}
    fb_cfg = cfg.get("farneback_params", {}) if isinstance(cfg.get("farneback_params"), dict) else {}
    masking_cfg = cfg.get("masking", {}) if isinstance(cfg.get("masking"), dict) else {}

    roi_w = int(roi_cfg.get("roi_w", 160))
    roi_h = int(roi_cfg.get("roi_h", 32))
    trim_ends_ratio = float(roi_cfg.get("trim_ends_ratio", 0.15))
    center_band_h = int(roi_cfg.get("center_band_h", 10))
    hand_mask_expand_px = float(masking_cfg.get("hand_mask_expand_px", 8))
    enable_hand_mask = bool(masking_cfg.get("enable_hand_mask", True))

    strings = load_strings_json(args.strings)
    string_by_id = {str(s.string_id): s for s in strings}
    events = _load_events(args.events)
    if not events:
        raise ValueError(f"No events found in {args.events}")

    if args.event_index < 0 or args.event_index >= len(events):
        raise IndexError(f"event-index out of range: {args.event_index} (events={len(events)})")
    event = events[args.event_index]

    selected_sid = _normalize_string_id(args.string_id) if args.string_id.strip() else event["string_id"]
    if selected_sid not in string_by_id:
        raise KeyError(f"String id {selected_sid} not found in {args.strings}")
    string_geom = string_by_id[selected_sid]

    reader = VideoReader(
        video_path=args.video,
        stabilize_enabled=bool(args.enable_stabilization),
    )
    event_frame = int(round(float(event["timestamp_sec"]) * float(reader.fps)))
    start_frame = max(0, event_frame - int(args.window_before))
    end_frame = max(start_frame, event_frame + int(args.window_after))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(reader.fps),
        (int(reader.width) * 2, int(reader.height)),
    )
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Cannot open writer: {args.output}")

    prev_roi: np.ndarray | None = None
    vib_scores: list[float] = []
    processed = 0
    try:
        while True:
            packet = reader.read()
            if packet is None:
                break
            if packet.frame_index < start_frame:
                continue
            if packet.frame_index > end_frame:
                break

            frame_for_roi = packet.frame
            roi, mat_img_to_roi = extract_rotated_roi(
                frame=frame_for_roi,
                string_geom=string_geom,
                roi_w=roi_w,
                roi_h=roi_h,
                trim_ends_ratio=trim_ends_ratio,
            )

            mask = create_roi_mask(
                roi_h=roi_h,
                roi_w=roi_w,
                center_band_h=center_band_h,
                mat_img_to_roi=mat_img_to_roi,
                finger_point_img=event.get("finger_point_img") if enable_hand_mask else None,
                hand_mask_expand_px=hand_mask_expand_px,
            )

            if prev_roi is None:
                dy = np.zeros((roi_h, roi_w), dtype=np.float32)
                vib_score = 0.0
                mean_abs_dy = 0.0
                p90_abs_dy = 0.0
                frac_high = 0.0
            else:
                fb = compute_vibration_frame(
                    prev_roi=prev_roi,
                    roi=roi,
                    mask=mask,
                    farneback_params=fb_cfg,
                    dy_thr=float(args.dy_thr),
                )
                dy = fb.dy
                vib_score = fb.vib_score_frame
                mean_abs_dy = fb.mean_abs_dy
                p90_abs_dy = fb.p90_abs_dy
                frac_high = fb.frac_high_dy
                vib_scores.append(vib_score)

            left = frame_for_roi.copy()
            _draw_roi_box(
                frame=left,
                string_geom=string_geom,
                roi_w=roi_w,
                roi_h=roi_h,
                trim_ends_ratio=trim_ends_ratio,
            )
            cv2.putText(
                left,
                f"f={packet.frame_index} event_f={event_frame} sid={selected_sid}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                left,
                f"stabilization={packet.stabilization_method} motion={packet.motion_score:.3f}",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            heat = _dy_heatmap(dy=dy, mask=mask, clip_abs_px=float(args.heatmap_clip))
            right = cv2.resize(heat, (int(reader.width), int(reader.height)), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                right,
                f"vib={vib_score:.3f} mean={mean_abs_dy:.3f} p90={p90_abs_dy:.3f} frac={frac_high:.3f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                right,
                f"|dy| clip={float(args.heatmap_clip):.2f} thr={float(args.dy_thr):.2f}",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            writer.write(np.hstack([left, right]))
            prev_roi = roi
            processed += 1
    finally:
        reader.release()
        writer.release()

    print(f"Events loaded: {len(events)}")
    print(f"Selected event index: {args.event_index}")
    print(f"Selected string id: {selected_sid}")
    print(f"Frame window: [{start_frame}, {end_frame}]")
    print(f"Frames written: {processed}")
    if vib_scores:
        arr = np.asarray(vib_scores, dtype=np.float64)
        print(f"vib_score mean: {arr.mean():.6f}")
        print(f"vib_score p95: {np.percentile(arr, 95):.6f}")
        print(f"vib_score max: {arr.max():.6f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
