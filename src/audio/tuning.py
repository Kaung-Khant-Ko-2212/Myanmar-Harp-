from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TuningEntry:
    string_id: int
    frequency_hz: float
    note_name: str | None = None


@dataclass(frozen=True)
class PitchMatchResult:
    matched_string_id: int | None
    cents_error: float | None
    candidate_strings: list[int]
    status: str
    debug: dict[str, Any]


def load_tuning_table(path: str | Path) -> dict[int, TuningEntry]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("strings"), list):
        items = data["strings"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Invalid tuning table format: {p}")
    out: dict[int, TuningEntry] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = int(item["string_id"])
        hz = float(item["frequency_hz"])
        out[sid] = TuningEntry(string_id=sid, frequency_hz=hz, note_name=item.get("note_name"))
    if not out:
        raise ValueError(f"Empty tuning table: {p}")
    return out


def cents_error(f0_hz: float, target_hz: float) -> float:
    if f0_hz <= 0 or target_hz <= 0:
        raise ValueError("frequencies must be positive")
    return float(1200.0 * math.log2(float(f0_hz) / float(target_hz)))


def select_candidate_string_ids(
    *,
    touched_string_id: int | None,
    available_string_ids: list[int],
    distance_px: float | None,
    candidate_radius_default: int,
    candidate_radius_close_contact: int,
    contact_dist_px_thr: float,
) -> list[int]:
    if not available_string_ids:
        return []
    ids_sorted = sorted(set(int(s) for s in available_string_ids))
    if touched_string_id is None:
        return ids_sorted

    radius = int(candidate_radius_default)
    if distance_px is not None and float(distance_px) <= float(contact_dist_px_thr):
        radius = int(candidate_radius_close_contact)
    radius = max(0, radius)
    sid = int(touched_string_id)
    out = [s for s in ids_sorted if abs(int(s) - sid) <= radius]
    return out or ids_sorted


def match_f0_to_tuning(
    *,
    f0_hz: float,
    tuning_by_string: dict[int, TuningEntry],
    candidate_string_ids: list[int],
    max_cents_error: float,
) -> PitchMatchResult:
    debug: dict[str, Any] = {"f0_hz": float(f0_hz)}
    candidates = [int(s) for s in candidate_string_ids if int(s) in tuning_by_string]
    if not candidates:
        return PitchMatchResult(None, None, [], "no_match", {**debug, "reason": "no_candidates"})

    best_sid: int | None = None
    best_cents: float | None = None
    for sid in candidates:
        c = cents_error(float(f0_hz), float(tuning_by_string[sid].frequency_hz))
        if best_cents is None or abs(c) < abs(best_cents):
            best_sid = int(sid)
            best_cents = float(c)

    debug["candidate_evals"] = [
        {
            "string_id": int(sid),
            "frequency_hz": float(tuning_by_string[sid].frequency_hz),
            "cents_error": float(cents_error(float(f0_hz), float(tuning_by_string[sid].frequency_hz))),
        }
        for sid in candidates
    ]
    if best_sid is None or best_cents is None:
        return PitchMatchResult(None, None, candidates, "no_match", {**debug, "reason": "no_best"})

    if abs(best_cents) > float(max_cents_error):
        return PitchMatchResult(
            None,
            float(best_cents),
            candidates,
            "no_match",
            {**debug, "reason": "max_cents_error_exceeded", "max_cents_error": float(max_cents_error), "best_string_id": int(best_sid)},
        )
    return PitchMatchResult(int(best_sid), float(best_cents), candidates, "ok", debug)

