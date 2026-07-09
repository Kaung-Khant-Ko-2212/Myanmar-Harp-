from __future__ import annotations

import argparse
import math
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

from src.roi import extract_rotated_roi, roi_box_corners
from src.strings import StringGeometry, load_strings_json
from src.video_reader import VideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ROI preview video: left=original with ROI boxes, right=ROI crops."
    )
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "video.mp4")
    parser.add_argument("--strings", type=Path, default=REPO_ROOT / "strings.json")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "roi_preview.mp4")
    parser.add_argument(
        "--string-ids",
        type=str,
        default="",
        help="Comma-separated string ids to preview. Empty uses all strings.",
    )
    parser.add_argument("--roi-w", type=int, default=None, help="Override ROI width.")
    parser.add_argument("--roi-h", type=int, default=None, help="Override ROI height.")
    parser.add_argument(
        "--trim-ends-ratio",
        type=float,
        default=None,
        help="Trim ratio used for stable mid-segment sampling.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process full video.")
    parser.add_argument(
        "--enable-stabilization",
        dest="enable_stabilization",
        action="store_true",
        help="Use globally stabilized frames for ROI extraction.",
    )
    parser.add_argument(
        "--no-stabilization",
        dest="enable_stabilization",
        action="store_false",
        help="Disable stabilization and use raw frames.",
    )
    parser.set_defaults(enable_stabilization=False)
    return parser.parse_args()


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


def _string_color(string_id: int | str) -> tuple[int, int, int]:
    key = str(string_id)
    seed = sum(ord(c) for c in key) % 255
    b = 60 + (37 * seed) % 180
    g = 60 + (73 * seed) % 180
    r = 60 + (109 * seed) % 180
    return int(b), int(g), int(r)


def _parse_selected_ids(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _choose_strings(all_strings: list[StringGeometry], selected_ids: set[str]) -> list[StringGeometry]:
    if not selected_ids:
        return all_strings
    out = [geom for geom in all_strings if str(geom.string_id) in selected_ids]
    if not out:
        raise ValueError(f"No matching string_ids found. Requested: {sorted(selected_ids)}")
    return out


def _draw_left_panel(
    frame: np.ndarray,
    selected_strings: list[StringGeometry],
    roi_w: int,
    roi_h: int,
    trim_ends_ratio: float,
) -> np.ndarray:
    left = frame.copy()
    for geom in selected_strings:
        color = _string_color(geom.string_id)
        corners = roi_box_corners(
            string_geom=geom,
            roi_w=roi_w,
            roi_h=roi_h,
            trim_ends_ratio=trim_ends_ratio,
        )
        pts = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(left, [pts], True, color, 2, cv2.LINE_AA)
        center = tuple(np.round(corners.mean(axis=0)).astype(int))
        cv2.putText(
            left,
            f"s{geom.string_id}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return left


def _build_right_panel(
    rois_gray: list[np.ndarray],
    labels: list[str],
    colors: list[tuple[int, int, int]],
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    count = len(rois_gray)
    if count == 0:
        cv2.putText(panel, "No ROIs", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
        return panel

    cols = 1 if count == 1 else int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    cell_w = max(1, panel_w // cols)
    cell_h = max(1, panel_h // rows)

    for i, roi_gray in enumerate(rois_gray):
        row = i // cols
        col = i % cols
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = panel_w if col == cols - 1 else (col + 1) * cell_w
        y1 = panel_h if row == rows - 1 else (row + 1) * cell_h
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)

        roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(roi_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        panel[y0:y1, x0:x1] = resized
        cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), colors[i], 1)
        cv2.putText(
            panel,
            labels[i],
            (x0 + 6, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[i],
            1,
            cv2.LINE_AA,
        )
    return panel


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    config = _load_yaml(args.config)
    roi_cfg = config.get("ROI", {}) if isinstance(config.get("ROI"), dict) else {}
    roi_w = int(args.roi_w if args.roi_w is not None else roi_cfg.get("roi_w", 160))
    roi_h = int(args.roi_h if args.roi_h is not None else roi_cfg.get("roi_h", 32))
    trim_ends_ratio = float(
        args.trim_ends_ratio if args.trim_ends_ratio is not None else roi_cfg.get("trim_ends_ratio", 0.15)
    )

    all_strings = load_strings_json(args.strings)
    selected_ids = _parse_selected_ids(args.string_ids)
    selected_strings = _choose_strings(all_strings, selected_ids)
    if not selected_strings:
        raise ValueError("No strings available in strings.json.")

    reader = VideoReader(
        video_path=args.video,
        stabilize_enabled=bool(args.enable_stabilization),
    )
    fps = float(reader.fps)
    frame_w = int(reader.width)
    frame_h = int(reader.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w * 2, frame_h),
    )
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    processed = 0
    try:
        while True:
            packet = reader.read()
            if packet is None:
                break
            processed += 1
            frame = packet.frame

            left = _draw_left_panel(
                frame=frame,
                selected_strings=selected_strings,
                roi_w=roi_w,
                roi_h=roi_h,
                trim_ends_ratio=trim_ends_ratio,
            )
            cv2.putText(
                left,
                f"stabilization={packet.stabilization_method} motion={packet.motion_score:.3f}",
                (20, max(30, frame_h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            rois_gray: list[np.ndarray] = []
            labels: list[str] = []
            colors: list[tuple[int, int, int]] = []
            for geom in selected_strings:
                roi_gray, _ = extract_rotated_roi(
                    frame=frame,
                    string_geom=geom,
                    roi_w=roi_w,
                    roi_h=roi_h,
                    trim_ends_ratio=trim_ends_ratio,
                )
                rois_gray.append(roi_gray)
                labels.append(f"s{geom.string_id}")
                colors.append(_string_color(geom.string_id))

            right = _build_right_panel(
                rois_gray=rois_gray,
                labels=labels,
                colors=colors,
                panel_h=frame_h,
                panel_w=frame_w,
            )

            combined = np.hstack([left, right])
            writer.write(combined)

            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        reader.release()
        writer.release()

    print(f"Strings loaded: {len(all_strings)}")
    print(f"Strings selected: {len(selected_strings)}")
    print(f"Frames written: {processed}")
    print(f"ROI size: {roi_w}x{roi_h}")
    print(f"Stabilization enabled: {bool(args.enable_stabilization)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
