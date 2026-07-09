from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strings import StringGeometry, closest_point_and_distance_px, load_strings_json


REQUIRED_TOUCH_COLUMNS = [
    "timestamp_sec",
    "hand_side",
    "finger_type",
    "touched_string_id",
    "touch_conf",
    "contact_x",
    "contact_y",
    "finger_x",
    "finger_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay strings and selected contact events onto video frames."
    )
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "video.mp4")
    parser.add_argument("--strings", type=Path, default=REPO_ROOT / "strings.json")
    parser.add_argument("--touch-events", type=Path, default=REPO_ROOT / "touch_events.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "overlay_frames")
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--min-gap-sec", type=float, default=0.05)
    return parser.parse_args()


def load_touch_events_csv(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        raise FileNotFoundError(f"touch_events.csv not found: {path}")

    out: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = [col for col in REQUIRED_TOUCH_COLUMNS if col not in headers]
        if missing:
            raise ValueError(f"touch_events.csv missing columns: {missing}")

        for row_idx, row in enumerate(reader, start=2):
            try:
                out.append(
                    {
                        "timestamp_sec": float(row["timestamp_sec"]),
                        "hand_side": (row["hand_side"] or "").strip().lower(),
                        "finger_type": (row["finger_type"] or "").strip().lower(),
                        "touched_string_id": str(row["touched_string_id"]),
                        "touch_conf": float(row["touch_conf"]),
                        "contact_x": float(row["contact_x"]),
                        "contact_y": float(row["contact_y"]),
                        "finger_x": float(row["finger_x"]),
                        "finger_y": float(row["finger_y"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid row {row_idx} in touch_events.csv: {exc}") from exc
    return out


def _string_color(string_id: int | str) -> tuple[int, int, int]:
    key = str(string_id)
    seed = sum(ord(c) for c in key) % 255
    b = 60 + (37 * seed) % 180
    g = 60 + (73 * seed) % 180
    r = 60 + (109 * seed) % 180
    return int(b), int(g), int(r)


def draw_all_strings(frame, strings: list[StringGeometry]) -> None:
    for geom in strings:
        pts = geom.points
        color = _string_color(geom.string_id)
        for i in range(1, len(pts)):
            p1 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
            p2 = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        label_pt = (int(round(pts[0][0])) + 4, int(round(pts[0][1])) - 4)
        cv2.putText(
            frame,
            f"s{geom.string_id}",
            label_pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def select_events(events: list[dict[str, float | str]], max_events: int, min_gap_sec: float) -> list[dict[str, float | str]]:
    if max_events <= 0:
        return []
    sorted_events = sorted(events, key=lambda e: float(e["timestamp_sec"]))
    selected: list[dict[str, float | str]] = []
    last_t = -1e9
    for event in sorted_events:
        t = float(event["timestamp_sec"])
        if t - last_t < min_gap_sec:
            continue
        selected.append(event)
        last_t = t
        if len(selected) >= max_events:
            break
    return selected


def draw_event_overlay(
    frame,
    event: dict[str, float | str],
    string_by_id: dict[str, StringGeometry],
) -> None:
    contact = (int(round(float(event["contact_x"]))), int(round(float(event["contact_y"]))))
    finger = (int(round(float(event["finger_x"]))), int(round(float(event["finger_y"]))))
    string_id = str(event["touched_string_id"])
    geom = string_by_id.get(string_id)

    cv2.circle(frame, contact, 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, finger, 4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, finger, contact, (0, 255, 255), 1, cv2.LINE_AA)

    note = f"t={float(event['timestamp_sec']):.3f}s str={string_id} {event['finger_type']}"
    cv2.putText(frame, note, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if geom is not None:
        closest_pt, dist_px = closest_point_and_distance_px(
            geom,
            float(event["contact_x"]),
            float(event["contact_y"]),
        )
        cp = (int(round(closest_pt[0])), int(round(closest_pt[1])))
        cv2.circle(frame, cp, 4, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, contact, cp, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"dist={dist_px:.2f}px",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    strings = load_strings_json(args.strings)
    touch_events = load_touch_events_csv(args.touch_events)
    selected_events = select_events(touch_events, args.max_events, args.min_gap_sec)
    string_by_id = {str(geom.string_id): geom for geom in strings}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 60.0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not selected_events:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame from video.")
        draw_all_strings(frame, strings)
        out_path = args.output_dir / "frame_000000_strings_only.jpg"
        cv2.imwrite(str(out_path), frame)
        cap.release()
        print(f"No events selected. Wrote strings-only frame: {out_path}")
        return

    saved = 0
    for i, event in enumerate(selected_events, start=1):
        frame_idx = max(0, int(round(float(event["timestamp_sec"]) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Could not read frame {frame_idx} for event {i}.")
            continue

        draw_all_strings(frame, strings)
        draw_event_overlay(frame, event, string_by_id)
        out_path = args.output_dir / f"event_{i:02d}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print(f"Loaded strings: {len(strings)}")
    print(f"Loaded touch events: {len(touch_events)}")
    print(f"Selected events: {len(selected_events)}")
    print(f"Saved overlay frames: {saved} -> {args.output_dir}")


if __name__ == "__main__":
    main()
