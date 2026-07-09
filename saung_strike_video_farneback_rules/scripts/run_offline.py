from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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

ALLOWED_HAND_SIDE = {"right"}
ALLOWED_FINGER_TYPES = {"thumb", "index"}

RESULT_COLUMNS = [
    "timestamp_sec",
    "string_id",
    "hand_side",
    "finger_type",
    "touch_conf",
    "rule_label",
    "rule_score",
    "notes",
]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def load_touch_events(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"touch_events.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = [col for col in REQUIRED_TOUCH_COLUMNS if col not in headers]
        if missing:
            raise ValueError(f"touch_events.csv missing columns: {missing}")
        rows = list(reader)

    for i, row in enumerate(rows, start=2):
        hand_side = (row.get("hand_side") or "").strip().lower()
        finger_type = (row.get("finger_type") or "").strip().lower()
        if hand_side not in ALLOWED_HAND_SIDE:
            raise ValueError(f"Row {i}: hand_side must be 'right', got '{row.get('hand_side')}'")
        if finger_type not in ALLOWED_FINGER_TYPES:
            raise ValueError(f"Row {i}: finger_type must be one of {sorted(ALLOWED_FINGER_TYPES)}")
    return rows


def load_strings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"strings.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        strings = data.get("strings")
    else:
        strings = data

    if not isinstance(strings, list):
        raise ValueError("strings.json must be a list or an object with key 'strings'.")

    for idx, item in enumerate(strings):
        if not isinstance(item, dict):
            raise ValueError(f"strings[{idx}] must be an object.")
        if "string_id" not in item:
            raise ValueError(f"strings[{idx}] missing 'string_id'.")
        has_endpoints = "endpoints" in item
        has_polyline = "polyline_points" in item
        if not has_endpoints and not has_polyline:
            raise ValueError(
                f"strings[{idx}] must have either 'endpoints' or 'polyline_points'."
            )
    return strings


def ensure_video_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"video.mp4 not found: {path}")


def write_results(path: Path, touch_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in touch_rows:
            writer.writerow(
                {
                    "timestamp_sec": row["timestamp_sec"],
                    "string_id": row["touched_string_id"],
                    "hand_side": row["hand_side"],
                    "finger_type": row["finger_type"],
                    "touch_conf": row["touch_conf"],
                    "rule_label": "pending",
                    "rule_score": "",
                    "notes": "Skeleton output: Farneback rule engine not implemented yet.",
                }
            )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Offline skeleton for Saung strike detection using Farneback rules."
    )
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "config.yaml")
    parser.add_argument("--video", type=Path, default=repo_root / "video.mp4")
    parser.add_argument("--touch-events", type=Path, default=repo_root / "touch_events.csv")
    parser.add_argument("--strings", type=Path, default=repo_root / "strings.json")
    parser.add_argument("--output", type=Path, default=repo_root / "outputs" / "results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    ensure_video_exists(args.video)
    touch_rows = load_touch_events(args.touch_events)
    strings = load_strings(args.strings)
    write_results(args.output, touch_rows)
    print(f"Loaded config keys: {sorted(config.keys())}")
    print(f"Loaded touch events: {len(touch_rows)}")
    print(f"Loaded strings: {len(strings)}")
    print(f"Wrote results: {args.output}")


if __name__ == "__main__":
    main()
