from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class StringGeometry:
    string_id: int | str
    mode: str
    points: tuple[Point, ...]


def _to_point(value: Any, field_name: str) -> Point:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Invalid point for {field_name}: {value!r}")


def _parse_endpoints(value: Any, *, string_id: int | str) -> tuple[Point, Point]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = value
        return (float(x1), float(y1)), (float(x2), float(y2))

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        p1 = _to_point(value[0], f"strings[{string_id}].endpoints[0]")
        p2 = _to_point(value[1], f"strings[{string_id}].endpoints[1]")
        return p1, p2

    raise ValueError(
        f"strings[{string_id}].endpoints must be [x1,y1,x2,y2] or [[x1,y1],[x2,y2]]."
    )


def _parse_polyline_points(value: Any, *, string_id: int | str) -> tuple[Point, ...]:
    if not isinstance(value, list):
        raise ValueError(f"strings[{string_id}].polyline_points must be a list of points.")
    points = tuple(_to_point(p, f"strings[{string_id}].polyline_points") for p in value)
    if len(points) < 2:
        raise ValueError(f"strings[{string_id}].polyline_points needs at least 2 points.")
    return points


def _coerce_string_geom(value: StringGeometry | dict[str, Any]) -> StringGeometry:
    if isinstance(value, StringGeometry):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Invalid string geometry: {value!r}")

    if "string_id" not in value:
        raise ValueError("String geometry is missing 'string_id'.")
    string_id = value["string_id"]

    if "points" in value:
        points = tuple(_to_point(p, f"strings[{string_id}].points") for p in value["points"])
        if len(points) < 2:
            raise ValueError(f"strings[{string_id}].points needs at least 2 points.")
        mode = str(value.get("mode") or "polyline")
        return StringGeometry(string_id=string_id, mode=mode, points=points)

    if "polyline_points" in value:
        points = _parse_polyline_points(value["polyline_points"], string_id=string_id)
        return StringGeometry(string_id=string_id, mode="polyline", points=points)

    if "endpoints" in value:
        p1, p2 = _parse_endpoints(value["endpoints"], string_id=string_id)
        return StringGeometry(string_id=string_id, mode="endpoints", points=(p1, p2))

    raise ValueError("String geometry needs 'points', 'polyline_points', or 'endpoints'.")


def load_strings_json(path: Path) -> list[StringGeometry]:
    if not path.exists():
        raise FileNotFoundError(f"strings.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_strings = data.get("strings") if isinstance(data, dict) else data
    if not isinstance(raw_strings, list):
        raise ValueError("strings.json must be a list or an object with key 'strings'.")

    out: list[StringGeometry] = []
    for idx, item in enumerate(raw_strings):
        if not isinstance(item, dict):
            raise ValueError(f"strings[{idx}] must be an object.")
        if "string_id" not in item:
            raise ValueError(f"strings[{idx}] missing 'string_id'.")
        string_id = item["string_id"]

        if "polyline_points" in item:
            points = _parse_polyline_points(item["polyline_points"], string_id=string_id)
            out.append(StringGeometry(string_id=string_id, mode="polyline", points=points))
            continue

        if "endpoints" in item:
            p1, p2 = _parse_endpoints(item["endpoints"], string_id=string_id)
            out.append(StringGeometry(string_id=string_id, mode="endpoints", points=(p1, p2)))
            continue

        raise ValueError(
            f"strings[{idx}] must include either 'endpoints' or 'polyline_points'."
        )

    return out


def _segment_length(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _polyline_lengths(points: tuple[Point, ...]) -> list[float]:
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + _segment_length(points[i - 1], points[i]))
    return lengths


def _point_at_distance(points: tuple[Point, ...], distance: float) -> Point:
    cumulative = _polyline_lengths(points)
    total = cumulative[-1]
    if total <= 1e-6:
        return points[0]

    d = max(0.0, min(distance, total))
    for i in range(1, len(points)):
        seg_start_d = cumulative[i - 1]
        seg_end_d = cumulative[i]
        if d <= seg_end_d:
            seg_len = seg_end_d - seg_start_d
            if seg_len <= 1e-6:
                return points[i]
            t = (d - seg_start_d) / seg_len
            x = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
            y = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
            return (x, y)
    return points[-1]


def _project_point_to_segment(p1: Point, p2: Point, point: Point) -> tuple[Point, float]:
    x1, y1 = p1
    x2, y2 = p2
    px, py = point
    dx = x2 - x1
    dy = y2 - y1
    den = dx * dx + dy * dy

    if den <= 1e-12:
        cp = p1
        return cp, math.hypot(px - cp[0], py - cp[1])

    t = ((px - x1) * dx + (py - y1) * dy) / den
    t = max(0.0, min(1.0, t))
    cp = (x1 + t * dx, y1 + t * dy)
    dist = math.hypot(px - cp[0], py - cp[1])
    return cp, dist


def closest_point_and_distance_px(
    string_geom: StringGeometry | dict[str, Any], x: float, y: float
) -> tuple[Point, float]:
    geom = _coerce_string_geom(string_geom)
    if len(geom.points) < 2:
        raise ValueError(f"String {geom.string_id} must have at least 2 points.")

    best_cp: Point = geom.points[0]
    best_dist = float("inf")
    point = (float(x), float(y))

    for i in range(1, len(geom.points)):
        cp, dist = _project_point_to_segment(geom.points[i - 1], geom.points[i], point)
        if dist < best_dist:
            best_cp = cp
            best_dist = dist

    return best_cp, best_dist


def sample_mid_segment(
    string_geom: StringGeometry | dict[str, Any], trim_ends_ratio: float
) -> tuple[Point, Point]:
    geom = _coerce_string_geom(string_geom)
    if len(geom.points) < 2:
        raise ValueError(f"String {geom.string_id} must have at least 2 points.")

    trim = max(0.0, min(float(trim_ends_ratio), 0.45))
    total_len = _polyline_lengths(geom.points)[-1]
    if total_len <= 1e-6:
        return geom.points[0], geom.points[1]

    start_d = total_len * trim
    end_d = total_len * (1.0 - trim)
    if end_d - start_d <= 1e-6:
        start_d = total_len * 0.25
        end_d = total_len * 0.75

    mid_d = 0.5 * (start_d + end_d)
    half_window = max(2.0, 0.1 * (end_d - start_d))
    p1 = _point_at_distance(geom.points, max(start_d, mid_d - half_window))
    p2 = _point_at_distance(geom.points, min(end_d, mid_d + half_window))

    if _segment_length(p1, p2) <= 1e-6:
        p1 = _point_at_distance(geom.points, start_d)
        p2 = _point_at_distance(geom.points, end_d)

    return p1, p2


def sample_points_along_mid_segment(
    string_geom: StringGeometry | dict[str, Any],
    *,
    sample_count: int = 5,
    trim_ends_ratio: float = 0.15,
) -> list[Point]:
    geom = _coerce_string_geom(string_geom)
    if len(geom.points) < 2:
        raise ValueError(f"String {geom.string_id} must have at least 2 points.")

    trim = max(0.0, min(float(trim_ends_ratio), 0.45))
    total_len = _polyline_lengths(geom.points)[-1]
    if total_len <= 1e-6:
        return [geom.points[0]]

    start_d = total_len * trim
    end_d = total_len * (1.0 - trim)
    if end_d - start_d <= 1e-6:
        start_d = total_len * 0.25
        end_d = total_len * 0.75

    n = max(1, int(sample_count))
    if n == 1:
        return [_point_at_distance(geom.points, 0.5 * (start_d + end_d))]

    distances = np.linspace(start_d, end_d, num=n, dtype=np.float64)
    return [_point_at_distance(geom.points, float(d)) for d in distances]


def estimate_neighbor_spacing_px(
    target_geom: StringGeometry | dict[str, Any],
    all_strings: list[StringGeometry | dict[str, Any]],
    sample_count: int = 5,
    trim_ends_ratio: float = 0.15,
) -> float | None:
    target = _coerce_string_geom(target_geom)
    samples = sample_points_along_mid_segment(
        target,
        sample_count=sample_count,
        trim_ends_ratio=trim_ends_ratio,
    )
    if not samples:
        return None

    best_dist = float("inf")
    for other_raw in all_strings:
        other = _coerce_string_geom(other_raw)
        if str(other.string_id) == str(target.string_id):
            continue
        for x, y in samples:
            _, dist = closest_point_and_distance_px(other, x=x, y=y)
            if dist < best_dist:
                best_dist = dist

    if not math.isfinite(best_dist):
        return None
    return float(best_dist)


def string_direction_and_normal(string_geom: StringGeometry | dict[str, Any]) -> tuple[Point, Point]:
    geom = _coerce_string_geom(string_geom)
    p1, p2 = sample_mid_segment(geom, trim_ends_ratio=0.15)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    mag = math.hypot(dx, dy)

    if mag <= 1e-6:
        for i in range(1, len(geom.points)):
            dx = geom.points[i][0] - geom.points[i - 1][0]
            dy = geom.points[i][1] - geom.points[i - 1][1]
            mag = math.hypot(dx, dy)
            if mag > 1e-6:
                break

    if mag <= 1e-6:
        raise ValueError(f"Cannot compute direction for degenerate string: {geom.string_id}")

    direction = (dx / mag, dy / mag)
    normal = (-direction[1], direction[0])
    return direction, normal
