from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_json(path: Path) -> Any | None:
    try:
        return read_json(path)
    except Exception:
        return None


def derive_output_stem_from_touch_json(touch_events_json_path: str | Path) -> tuple[Path, str]:
    src_path = Path(touch_events_json_path)
    out_dir = src_path.parent
    name = src_path.name
    suffix = "_touch_events.json"
    base_stem = name[: -len(suffix)] if name.endswith(suffix) else src_path.stem
    return out_dir, base_stem


def build_meta(
    *,
    source_video: str | None = None,
    touch_events_json_path: str | None = None,
    fps: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if source_video is not None:
        meta["source_video"] = str(source_video)
    if touch_events_json_path is not None:
        meta["touch_events_json_path"] = str(touch_events_json_path)
    if fps is not None:
        meta["fps"] = float(fps)
    if extra:
        meta.update(extra)
    return meta


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(Counter(str(v) for v in values))


def update_touch_events_bundle_paths(
    touch_events_json_path: str | Path,
    path_updates: dict[str, Any],
    count_updates: dict[str, Any] | None = None,
) -> None:
    src_path = Path(touch_events_json_path)
    if not src_path.exists():
        return
    try:
        payload = read_json(src_path)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    for key, value in path_updates.items():
        payload[key] = value
    for key, value in (count_updates or {}).items():
        payload[key] = value
    write_json(src_path, payload)

