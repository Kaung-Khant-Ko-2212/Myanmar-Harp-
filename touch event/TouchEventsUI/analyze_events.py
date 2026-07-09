import argparse
import json
import os
from pathlib import Path


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")

    if isinstance(data.get("touch_events"), list):
        return {
            "source_type": "touch_events",
            "video_name": data.get("video_name", "N/A"),
            "frames_processed": data.get("frames_processed", 0),
            "fps": data.get("fps", 0),
            "events": [ev for ev in data.get("touch_events", []) if isinstance(ev, dict)],
            "filter_pinky_default": True,
        }

    if isinstance(data.get("events"), list):
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        normalized_events = []
        for ev in data.get("events", []):
            if not isinstance(ev, dict):
                continue
            finger_type = str(ev.get("finger_type") or "").strip().lower()
            fingertip = str(ev.get("fingertip") or (f"{finger_type}_tip" if finger_type else "unknown"))
            timestamp_sec = ev.get("timestamp_sec", ev.get("time_sec", 0.0))
            struck_id = ev.get("struck_string_id", ev.get("string_id"))
            normalized_events.append(
                {
                    "time_sec": timestamp_sec,
                    "timestamp_sec": timestamp_sec,
                    "frame_index": ev.get("frame_index"),
                    "hand": str(ev.get("hand") or ev.get("hand_side") or "right"),
                    "hand_side": str(ev.get("hand_side") or ev.get("hand") or "right"),
                    "fingertip": fingertip,
                    "finger_type": finger_type,
                    "string_id": struck_id,
                    "struck_string_id": ev.get("struck_string_id"),
                    "touched_string_id": ev.get("touched_string_id"),
                    "beat_label": ev.get("beat_label"),
                    "confidence": ev.get("confidence"),
                    "confidence_label": ev.get("confidence_label"),
                    "strategy": ev.get("strategy"),
                }
            )
        source_video = meta.get("source_video") or meta.get("video_name") or "right_av_strike_events"
        return {
            "source_type": "av_strike_events",
            "video_name": str(Path(str(source_video)).name),
            "frames_processed": meta.get("frames_processed", 0),
            "fps": meta.get("fps", 0),
            "events": normalized_events,
            "filter_pinky_default": False,
        }

    raise ValueError("Unsupported JSON schema: expected `touch_events` or `events` list")


def _find_latest_av_strike_json() -> str | None:
    base = Path("backend/touch_events")
    if not base.exists():
        return None
    matches = sorted(
        base.glob("**/*_right_av_strike_events.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(matches[0]) if matches else None


def main():
    parser = argparse.ArgumentParser(description="Analyze touch events JSON or AV strike events JSON.")
    parser.add_argument("filepath", nargs="?", default=None, help="Path to the JSON file.")
    parser.add_argument("--bpm", type=float, default=120.0, help="Beats Per Minute for analysis (default: 120).")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Tolerance in seconds for ON beat (default: 0.1).")

    args = parser.parse_args()
    filepath = args.filepath or _find_latest_av_strike_json()
    bpm = args.bpm
    tolerance = args.tolerance

    if not filepath:
        print("Error: No filepath provided and no latest *_right_av_strike_events.json found under backend/touch_events")
        return

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    try:
        normalized = _normalize_payload(data)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 75)
    print(" EVENTS ANALYSIS REPORT ".center(75, "="))
    print(f" BPM: {bpm} | Tolerance: {tolerance}s ".center(75, "="))
    print("=" * 75 + "\n")

    print(" [1] OVERVIEW ".ljust(75, "-"))
    print(f"  Source Type      : {normalized.get('source_type', 'unknown')}")
    print(f"  File             : {filepath}")
    print(f"  Video Name       : {normalized.get('video_name', 'N/A')}")
    print(f"  Frames Processed : {normalized.get('frames_processed', 0)}")
    print(f"  Video FPS        : {normalized.get('fps', 0)}")

    events = normalized.get("events", [])
    print(f"  Total Raw Events : {len(events)}")

    hands = {}
    fingertips = {}
    strings = {}
    beat_stats = {"ON beat": 0, "OFF beat": 0}
    beat_source_stats = {"json": 0, "computed": 0}

    beat_interval = 60.0 / bpm if bpm > 0 else 1.0

    filtered_events = []
    filter_pinky = bool(normalized.get("filter_pinky_default", True))
    for ev in events:
        if filter_pinky and ev.get("fingertip") == "pinky_tip":
            continue

        filtered_events.append(ev)

        h = str(ev.get("hand", "unknown"))
        hands[h] = hands.get(h, 0) + 1

        f = str(ev.get("fingertip", "unknown"))
        fingertips[f] = fingertips.get(f, 0) + 1

        s = ev.get("string_id", "unknown")
        strings[s] = strings.get(s, 0) + 1

        beat_label = str(ev.get("beat_label") or "").strip().lower()
        if beat_label in {"on_beat", "off_beat"}:
            is_on_beat = beat_label == "on_beat"
            ev["beat_source"] = "json"
        else:
            time_sec = _to_float(ev.get("time_sec", 0.0), 0.0)
            remainder = time_sec % beat_interval
            is_on_beat = (remainder < tolerance) or (beat_interval - remainder < tolerance)
            ev["beat_source"] = "computed"
        beat_source_stats[ev["beat_source"]] += 1

        if is_on_beat:
            beat_stats["ON beat"] += 1
            ev["is_on_beat"] = True
        else:
            beat_stats["OFF beat"] += 1
            ev["is_on_beat"] = False

    total_filtered = len(filtered_events)
    filter_note = "Pinky removed" if filter_pinky else "No pinky filter"
    print(f"  Filtered Events  : {total_filtered} ({filter_note})\n")

    if not filtered_events:
        print("  No events found matching criteria.")
        return

    print(" [2] CLASSIFICATIONS ".ljust(75, "-"))

    print("\n  > Beat Accuracy:")
    on_count = beat_stats["ON beat"]
    pct = (on_count / total_filtered * 100) if total_filtered > 0 else 0
    print(f"    ON Beat      : {on_count:>5} events ({pct:.1f}%)")
    print(f"    OFF Beat     : {beat_stats['OFF beat']:>5} events ({100-pct:.1f}%)")
    print(f"    Beat Source  : json={beat_source_stats['json']} | computed={beat_source_stats['computed']}")

    print("\n  > By Hand (Left/Right):")
    for k, v in sorted(hands.items(), key=lambda item: item[1], reverse=True):
        print(f"    {k.capitalize():<12} : {v:>5} events ({(v / total_filtered * 100):.1f}%)")

    print("\n  > By Fingertip:")
    for k, v in sorted(fingertips.items(), key=lambda item: item[1], reverse=True):
        print(f"    {k.replace('_', ' ').capitalize():<12} : {v:>5} events ({(v / total_filtered * 100):.1f}%)")

    print("\n  > By String ID (Top 8):")
    sorted_strings = sorted(strings.items(), key=lambda item: item[1], reverse=True)[:8]
    for k, v in sorted_strings:
        print(f"    String {str(k):<5} : {v:>5} events ({(v / total_filtered * 100):.1f}%)")
    print()

    print(" [3] DETAILED LOG (Preview of first 30 events) ".ljust(75, "-"))
    print(f"  {'Time(s)':<8} | {'Frame':<6} | {'Hand':<6} | {'Fingertip':<12} | {'String':<8} | {'Beat':<8}")
    print("  " + "-" * 69)

    for ev in filtered_events[:30]:
        time_sec = _to_float(ev.get("time_sec", 0.0), 0.0)
        time_str = f"{time_sec:.3f}"
        frame = str(ev.get("frame_index", ""))
        hand = str(ev.get("hand", "")).capitalize()
        fingertip = str(ev.get("fingertip", "")).replace("_", " ").capitalize()
        string_id = f"s{ev.get('string_id', '?')}"
        beat_tag = "[ON]" if ev.get("is_on_beat") else "OFF"
        print(f"  {time_str:<8} | {frame:<6} | {hand:<6} | {fingertip:<12} | {string_id:<8} | {beat_tag:<8}")

    print("  " + "-" * 69)
    if total_filtered > 30:
        print(f"  ... and {total_filtered - 30:,} more events truncated ...")

    print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    main()
