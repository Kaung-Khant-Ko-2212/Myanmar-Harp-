from pathlib import Path
import argparse
import cv2


def segment_video(
    src: Path,
    out_dir: Path,
    clip_minutes: float = 1.0,
) -> None:
    """
    Split a single video file into fixed-length clips.

    Each output clip is approximately `clip_minutes` long.
    """
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS from video: {src}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_len_frames = int(round(fps * 60 * clip_minutes))
    if clip_len_frames <= 0:
        cap.release()
        raise ValueError("clip_minutes is too small; results in zero frames per clip.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stem = src.stem

    writer = None
    current_clip_idx = -1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clip_idx = frame_idx // clip_len_frames
        if writer is None or clip_idx != current_clip_idx:
            if writer is not None:
                writer.release()

            current_clip_idx = clip_idx
            out_path = out_dir / f"{stem}_part{clip_idx:03d}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            print(f"[NEW CLIP] Writing to {out_path}")

        writer.write(frame)
        frame_idx += 1

    if writer is not None:
        writer.release()

    cap.release()

    print(
        f"[DONE] Created {current_clip_idx + 1 if current_clip_idx >= 0 else 0} "
        f"clip(s) from {src} (total frames: {total_frames})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment a video into fixed-length clips (default: 1 minute each)."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="video_segments",
        help="Output directory where clips will be saved",
    )
    parser.add_argument(
        "--clip_minutes",
        type=float,
        default=1.0,
        help="Length of each clip, in minutes (default: 1.0)",
    )

    args = parser.parse_args()

    segment_video(
        src=Path(args.src),
        out_dir=Path(args.out_dir),
        clip_minutes=args.clip_minutes,
    )

