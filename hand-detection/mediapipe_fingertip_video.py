import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional, Union

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install with: pip install opencv-python mediapipe"
    ) from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: mediapipe. Install with: pip install mediapipe"
    ) from exc

FINGERTIP_IDS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

TIP_COLORS = {
    "thumb_tip": (60, 180, 75),
    "index_tip": (0, 255, 255),
    "middle_tip": (255, 0, 0),
    "ring_tip": (0, 128, 255),
    "pinky_tip": (255, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect hand landmarks in a video using MediaPipe Hands, optimized for "
            "fingertip tracking speed."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path to input video (quotes recommended if path has spaces)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        nargs="+",
        help="CSV output path (default: <input_stem>_fingertips.csv)",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        nargs="+",
        help="Optional path to save annotated output video",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Inference width for speed/latency tradeoff (0 = full resolution)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to track",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="MediaPipe model complexity (0 is fastest)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence for hand detection",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence for hand tracking",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between processed frames for higher throughput",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live preview while processing",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw fingertip names next to points (slightly slower)",
    )
    parser.add_argument(
        "--cv-threads",
        type=int,
        default=0,
        help="OpenCV thread count (0 keeps OpenCV default)",
    )
    return parser.parse_args()


def path_arg(value: Optional[Union[str, List[str]]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(value)
    return value


def safe_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y


def main() -> None:
    args = parse_args()

    input_path = Path(path_arg(args.input))
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_csv_raw = path_arg(args.output_csv)
    output_video_raw = path_arg(args.output_video)

    output_csv = Path(output_csv_raw) if output_csv_raw else input_path.with_name(f"{input_path.stem}_fingertips.csv")

    if args.cv_threads > 0:
        cv2.setNumThreads(args.cv_threads)
    cv2.setUseOptimized(True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 1e-6:
        source_fps = 30.0

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video is empty or unreadable")

    frame_h, frame_w = frame.shape[:2]

    writer = None
    if output_video_raw:
        output_video = Path(output_video_raw)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, source_fps, (frame_w, frame_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output video for writing: {output_video}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands

    processed_frames = 0
    detected_frames = 0
    fingertip_rows = 0
    total_infer_seconds = 0.0
    frame_index = 0
    t_start = time.perf_counter()

    with output_csv.open("w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            [
                "frame_index",
                "timestamp_ms",
                "hand_index",
                "hand_label",
                "tip_name",
                "x_px",
                "y_px",
                "z_norm",
                "x_norm",
                "y_norm",
            ]
        )

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as hands:
            while ok:
                process_this_frame = args.skip_frames == 0 or frame_index % (args.skip_frames + 1) == 0

                if process_this_frame:
                    proc_frame = frame
                    if args.process_width > 0 and frame_w > args.process_width:
                        scale = args.process_width / frame_w
                        proc_w = args.process_width
                        proc_h = max(1, int(frame_h * scale))
                        proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        proc_h, proc_w = frame_h, frame_w

                    scale_x = frame_w / proc_w
                    scale_y = frame_h / proc_h

                    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False

                    infer_start = time.perf_counter()
                    results = hands.process(rgb)
                    total_infer_seconds += time.perf_counter() - infer_start

                    rgb.flags.writeable = True
                    processed_frames += 1

                    landmarks_list = results.multi_hand_landmarks or []
                    handedness_list = results.multi_handedness or []

                    if landmarks_list:
                        detected_frames += 1

                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                    for hand_i, hand_landmarks in enumerate(landmarks_list):
                        hand_label = "unknown"
                        if hand_i < len(handedness_list):
                            hand_label = handedness_list[hand_i].classification[0].label.lower()

                        for tip_name, tip_idx in FINGERTIP_IDS.items():
                            lm = hand_landmarks.landmark[tip_idx]
                            x = int(lm.x * proc_w * scale_x)
                            y = int(lm.y * proc_h * scale_y)
                            x, y = safe_point(x, y, frame_w, frame_h)

                            csv_writer.writerow(
                                [
                                    frame_index,
                                    round(timestamp_ms, 3),
                                    hand_i,
                                    hand_label,
                                    tip_name,
                                    x,
                                    y,
                                    float(lm.z),
                                    float(lm.x),
                                    float(lm.y),
                                ]
                            )
                            fingertip_rows += 1
                            print(f"Frame {frame_index}: Hand {hand_i} ({hand_label}) {tip_name} at ({x}, {y}), z={lm.z:.4f}")
                            if writer is not None or args.display:
                                color = TIP_COLORS[tip_name]
                                cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
                                if args.draw_labels:
                                    cv2.putText(
                                        frame,
                                        tip_name,
                                        (x + 6, y - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        color,
                                        1,
                                        cv2.LINE_AA,
                                    )

                if writer is not None:
                    writer.write(frame)

                if args.display:
                    cv2.imshow("MediaPipe Fingertip Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_index += 1
                ok, frame = cap.read()

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    overall_fps = processed_frames / elapsed if elapsed > 0 else 0.0
    infer_fps = processed_frames / total_infer_seconds if total_infer_seconds > 0 else 0.0

    print("Processing complete")
    print(f"Input video: {input_path}")
    print(f"Fingertip CSV: {output_csv}")
    if output_video_raw:
        print(f"Annotated video: {output_video_raw}")
    print(f"Frames decoded: {frame_index}")
    print(f"Frames processed by model: {processed_frames}")
    print(f"Frames with hands detected: {detected_frames}")
    print(f"Fingertip rows written: {fingertip_rows}")
    print(f"Total time: {elapsed:.2f} s")
    print(f"End-to-end FPS: {overall_fps:.2f}")
    print(f"Model inference FPS: {infer_fps:.2f}")


if __name__ == "__main__":
    main()
