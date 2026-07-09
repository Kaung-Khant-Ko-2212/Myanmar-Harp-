from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.video_reader import VideoReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write side-by-side stabilization preview and print motion-score stats."
    )
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "video.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "stabilization_before_after.mp4",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process full video.")
    return parser.parse_args()


def _draw_label(frame: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    reader = VideoReader(
        video_path=args.video,
        stabilize_enabled=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(reader.fps),
        (int(reader.width) * 2, int(reader.height)),
    )
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    motion_scores: list[float] = []
    methods: Counter[str] = Counter()
    processed = 0
    try:
        while True:
            packet = reader.read()
            if packet is None:
                break
            processed += 1

            raw = packet.raw_frame.copy()
            stab = packet.frame.copy()

            _draw_label(raw, "Before (raw)", 20, 32)
            _draw_label(stab, "After (stabilized)", 20, 32)
            _draw_label(
                stab,
                f"method={packet.stabilization_method} motion={packet.motion_score:.3f}",
                20,
                62,
            )

            combined = np.hstack([raw, stab])
            writer.write(combined)

            motion_scores.append(float(packet.motion_score))
            methods[packet.stabilization_method] += 1

            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        reader.release()
        writer.release()

    if not motion_scores:
        print("No frames processed.")
        print(f"Output: {args.output}")
        return

    arr = np.asarray(motion_scores, dtype=np.float64)
    print(f"Frames processed: {processed}")
    print(f"Method counts: {dict(methods)}")
    print(f"motion_score mean: {arr.mean():.6f}")
    print(f"motion_score median: {np.median(arr):.6f}")
    print(f"motion_score p95: {np.percentile(arr, 95):.6f}")
    print(f"motion_score max: {arr.max():.6f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
