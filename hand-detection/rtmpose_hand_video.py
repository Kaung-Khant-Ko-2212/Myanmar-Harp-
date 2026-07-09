import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional, Union

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install with: pip install opencv-python"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install with: pip install numpy"
    ) from exc

try:
    # Prefer narrow inferencer import to avoid loading full dataset stack.
    from mmpose.apis.inferencers import MMPoseInferencer
except ImportError:
    try:
        # Fallback for older mmpose layouts.
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: mmpose. "
            "Install OpenMMLab runtime + mmpose first, then retry."
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
            "Run RTMPose hand keypoint inference on a video and export fingertip points to CSV."
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
        help="CSV output path (default: <input_stem>_rtmpose_fingertips.csv)",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        nargs="+",
        help="Optional path to save annotated output video",
    )
    parser.add_argument(
        "--pose2d",
        default="hand",
        help=(
            "MMPose pose2d alias/config for hand inference. "
            "Default 'hand' maps to RTMPose hand model aliases in MMPose."
        ),
    )
    parser.add_argument(
        "--pose2d-weights",
        default=None,
        nargs="+",
        help="Optional checkpoint path/URL overriding default weights",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu (default: cuda:0)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=0,
        help="Inference width for speed/latency tradeoff (0 = full resolution)",
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.25,
        help="Bounding-box confidence threshold (best effort; inferencer-version dependent)",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.15,
        help="Keypoint score threshold for writing/drawing fingertip points",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to keep per frame",
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


def path_arg(value: Optional[Union[str, list[str]]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(value)
    return value


def safe_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return as_float(value[0], default=default)
    if isinstance(value, np.ndarray) and value.size > 0:
        return float(value.reshape(-1)[0])
    return default


def collect_instances(node: Any) -> list[dict[str, Any]]:
    if node is None:
        return []

    if isinstance(node, dict):
        if "keypoints" in node:
            return [node]

        found: list[dict[str, Any]] = []
        for key in ("predictions", "instances", "pose_instances"):
            if key in node:
                found.extend(collect_instances(node[key]))
        return found

    if isinstance(node, (list, tuple)):
        found: list[dict[str, Any]] = []
        for item in node:
            found.extend(collect_instances(item))
        return found

    return []


def extract_keypoints(instance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    scores_raw = instance.get("keypoint_scores", instance.get("keypoints_scores", []))
    scores = np.asarray(scores_raw, dtype=float).reshape(-1)
    return keypoints[:, :2], scores


def run_inferencer(inferencer: MMPoseInferencer, frame: np.ndarray, bbox_thr: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_vis": False,
        "show": False,
        "bbox_thr": bbox_thr,
    }
    try:
        return next(inferencer(frame, **kwargs))
    except TypeError:
        kwargs.pop("bbox_thr", None)
        return next(inferencer(frame, **kwargs))


def side_guess_from_bbox(instance: dict[str, Any], frame_w: int) -> str:
    bbox = np.asarray(instance.get("bbox", []), dtype=float).reshape(-1)
    if bbox.size >= 4:
        cx = (bbox[0] + bbox[2]) * 0.5
        return "left_side" if cx < frame_w * 0.5 else "right_side"
    return "unknown"


def main() -> None:
    args = parse_args()

    input_path = Path(path_arg(args.input))
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_csv_raw = path_arg(args.output_csv)
    output_video_raw = path_arg(args.output_video)
    pose2d_weights = path_arg(args.pose2d_weights)

    output_csv = (
        Path(output_csv_raw)
        if output_csv_raw
        else input_path.with_name(f"{input_path.stem}_rtmpose_fingertips.csv")
    )

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

    inferencer_kwargs: dict[str, Any] = {
        "pose2d": args.pose2d,
        "device": args.device,
        "show_progress": False,
    }
    if pose2d_weights:
        inferencer_kwargs["pose2d_weights"] = pose2d_weights

    try:
        inferencer = MMPoseInferencer(**inferencer_kwargs)
    except Exception as exc:
        if str(args.device).lower() != "cpu":
            print(f"[WARN] Failed on device {args.device}: {exc}")
            print("[INFO] Falling back to CPU...")
            inferencer_kwargs["device"] = "cpu"
            inferencer = MMPoseInferencer(**inferencer_kwargs)
        else:
            cap.release()
            if writer is not None:
                writer.release()
            raise

    processed_frames = 0
    detected_frames = 0
    fingertip_rows = 0
    total_infer_seconds = 0.0
    frame_index = 0
    warned_keypoint_layout = False
    t_start = time.perf_counter()

    with output_csv.open("w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            [
                "frame_index",
                "timestamp_ms",
                "hand_index",
                "hand_side_guess",
                "tip_name",
                "x_px",
                "y_px",
                "score",
                "x_norm",
                "y_norm",
            ]
        )

        while ok:
            draw_frame = frame.copy()
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

                infer_start = time.perf_counter()
                prediction = run_inferencer(inferencer, proc_frame, bbox_thr=args.bbox_thr)
                total_infer_seconds += time.perf_counter() - infer_start
                processed_frames += 1

                instances = collect_instances(prediction.get("predictions"))
                instances.sort(
                    key=lambda inst: as_float(
                        inst.get("bbox_score", inst.get("score", 0.0)),
                        default=0.0,
                    ),
                    reverse=True,
                )
                instances = instances[: max(0, args.max_hands)]

                if instances:
                    detected_frames += 1

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                for hand_i, inst in enumerate(instances):
                    keypoints, scores = extract_keypoints(inst)
                    if keypoints.shape[0] <= max(FINGERTIP_IDS.values()):
                        if not warned_keypoint_layout:
                            print(
                                "[WARN] Model returned fewer than 21 keypoints for hand instance; "
                                "default fingertip IDs (4,8,12,16,20) may not apply."
                            )
                            warned_keypoint_layout = True
                        continue

                    side_guess = side_guess_from_bbox(inst, frame_w=frame_w)

                    for tip_name, tip_idx in FINGERTIP_IDS.items():
                        x = int(keypoints[tip_idx, 0] * scale_x)
                        y = int(keypoints[tip_idx, 1] * scale_y)
                        x, y = safe_point(x, y, frame_w, frame_h)

                        score = float(scores[tip_idx]) if tip_idx < scores.size else float("nan")
                        if not np.isnan(score) and score < args.kpt_thr:
                            continue

                        x_norm = x / max(1, frame_w)
                        y_norm = y / max(1, frame_h)
                        csv_writer.writerow(
                            [
                                frame_index,
                                round(timestamp_ms, 3),
                                hand_i,
                                side_guess,
                                tip_name,
                                x,
                                y,
                                score,
                                round(x_norm, 6),
                                round(y_norm, 6),
                            ]
                        )
                        fingertip_rows += 1

                        if writer is not None or args.display:
                            color = TIP_COLORS[tip_name]
                            cv2.circle(draw_frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
                            if args.draw_labels:
                                label = f"h{hand_i}_{tip_name}"
                                cv2.putText(
                                    draw_frame,
                                    label,
                                    (x + 6, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )

            if writer is not None:
                writer.write(draw_frame)

            if args.display:
                cv2.imshow("RTMPose Hand Fingertip Detection", draw_frame)
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
