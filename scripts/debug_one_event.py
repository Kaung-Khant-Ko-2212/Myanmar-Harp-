from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.audio.decision import build_canonical_right_events
from src.audio.extract import extract_audio_from_video
from src.audio.load import load_audio_mono
from src.audio.onset import prepare_onset_strength
from src.io.json_utils import derive_output_stem_from_touch_json, read_json
from src.pipeline.config import load_pipeline_config


def _float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(round(float(v)))
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def _derive_related_paths(touch_json: Path) -> dict[str, Path]:
    out_dir, base = derive_output_stem_from_touch_json(touch_json)
    return {
        "audio_decision": out_dir / f"{base}_right_audio_decision_events.json",
        "video_decision": out_dir / f"{base}_right_decision_events.json",
        "debug_dir": Path("outputs") / "debug",
    }


def _pick_event(
    *,
    audio_decision_payload: dict[str, Any] | None,
    canonical_events: list[dict[str, Any]],
    event_id: str | None,
    event_index: int | None,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    audio_events = []
    if isinstance(audio_decision_payload, dict) and isinstance(audio_decision_payload.get("events"), list):
        audio_events = [ev for ev in audio_decision_payload["events"] if isinstance(ev, dict)]

    if event_id:
        for i, ev in enumerate(audio_events):
            if str(ev.get("event_id")) == event_id:
                touch = ev.get("touch") if isinstance(ev.get("touch"), dict) else None
                return i, touch, ev
        for i, ev in enumerate(canonical_events):
            if str(ev.get("event_id")) == event_id:
                return i, ev, None
        raise ValueError(f"Event id not found: {event_id}")

    idx = int(event_index or 0)
    if audio_events and 0 <= idx < len(audio_events):
        ev = audio_events[idx]
        touch = ev.get("touch") if isinstance(ev.get("touch"), dict) else None
        return idx, touch, ev
    if 0 <= idx < len(canonical_events):
        return idx, canonical_events[idx], None
    raise IndexError(f"Event index out of range: {idx}")


def _align_video_event(
    video_decision_payload: dict[str, Any] | None,
    event_idx: int,
) -> dict[str, Any] | None:
    if not isinstance(video_decision_payload, dict):
        return None
    rows = video_decision_payload.get("right_decision_events")
    if not isinstance(rows, list):
        return None
    rows = [r for r in rows if isinstance(r, dict)]
    if 0 <= event_idx < len(rows):
        return rows[event_idx]
    return None


def _ensure_audio_path(
    *,
    args_audio: str | None,
    args_video: str | None,
    audio_decision_payload: dict[str, Any] | None,
    touch_json: Path,
    cfg: dict[str, Any],
    debug_dir: Path,
) -> Path | None:
    if args_audio:
        p = Path(args_audio)
        return p if p.exists() else None
    if isinstance(audio_decision_payload, dict):
        meta = audio_decision_payload.get("meta") if isinstance(audio_decision_payload.get("meta"), dict) else {}
        p = meta.get("audio_source_path")
        if p and Path(str(p)).exists():
            return Path(str(p))
    if args_video:
        video_path = Path(args_video)
        if not video_path.exists():
            return None
        sample_rate = int((cfg.get("audio") or {}).get("sample_rate", 16000))
        debug_dir.mkdir(parents=True, exist_ok=True)
        wav_path = debug_dir / f"{video_path.stem}_debug_audio.wav"
        res = extract_audio_from_video(video_path, wav_path, sample_rate=sample_rate)
        if res.ok and res.wav_path is not None:
            return res.wav_path
    return None


def _plot_debug(
    *,
    out_path: Path,
    audio: np.ndarray,
    sr: int,
    touch_time_sec: float,
    onset_time_sec: float | None,
    onset_cache,
    window_before: float = 0.35,
    window_after: float = 0.45,
) -> str:
    t0 = max(0.0, float(touch_time_sec) - float(window_before))
    t1 = min(float(audio.shape[0]) / max(float(sr), 1e-6), float(touch_time_sec) + float(window_after))
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))
    y = np.asarray(audio[i0:i1], dtype=np.float32)
    t = np.arange(i0, i1, dtype=np.float64) / float(sr)

    env = onset_cache.envelope
    env_t = onset_cache.times_sec
    env_mask = (env_t >= t0) & (env_t <= t1)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        out_path.with_suffix(".txt").write_text(
            f"matplotlib_unavailable: {exc}\nwindow: [{t0:.3f}, {t1:.3f}]\n",
            encoding="utf-8",
        )
        return str(out_path.with_suffix(".txt"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, y, color="#2C7FB8", linewidth=1.0)
    axes[0].axvline(float(touch_time_sec), color="#D95F0E", linestyle="--", label="touch")
    if onset_time_sec is not None:
        axes[0].axvline(float(onset_time_sec), color="#238B45", linestyle="-.", label="onset")
    axes[0].set_ylabel("Waveform")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(env_t[env_mask], env[env_mask], color="#756BB1", linewidth=1.2)
    axes[1].axvline(float(touch_time_sec), color="#D95F0E", linestyle="--")
    if onset_time_sec is not None:
        axes[1].axvline(float(onset_time_sec), color="#238B45", linestyle="-.")
    axes[1].set_ylabel("Onset Strength")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug a single touch/strike event with audio onset plot and decision summaries")
    parser.add_argument("--touch-json", type=str, required=True, help="Path to *_touch_events.json")
    parser.add_argument("--audio-json", type=str, default=None, help="Path to *_right_audio_decision_events.json")
    parser.add_argument("--video-json", type=str, default=None, help="Path to *_right_decision_events.json")
    parser.add_argument("--video", type=str, default=None, help="Source video path (used for audio extraction if needed)")
    parser.add_argument("--audio", type=str, default=None, help="Audio wav path (optional)")
    parser.add_argument("--config", type=str, default=str(Path("configs") / "config.yaml"))
    parser.add_argument("--event-id", type=str, default=None)
    parser.add_argument("--event-index", type=int, default=0, help="0-based index if event-id not provided")
    args = parser.parse_args()

    touch_json = Path(args.touch_json)
    touch_payload = _load_json(touch_json)
    touch_events = [ev for ev in touch_payload.get("touch_events", []) if isinstance(ev, dict)]
    cfg = load_pipeline_config(Path(args.config))
    fps = float(touch_payload.get("fps") or (cfg.get("general") or {}).get("fps") or 30.0)

    related = _derive_related_paths(touch_json)
    audio_json_path = Path(args.audio_json) if args.audio_json else related["audio_decision"]
    video_json_path = Path(args.video_json) if args.video_json else related["video_decision"]
    audio_decision_payload = _load_json(audio_json_path) if audio_json_path.exists() else None
    video_decision_payload = _load_json(video_json_path) if video_json_path.exists() else None

    video_events = []
    if isinstance(video_decision_payload, dict) and isinstance(video_decision_payload.get("right_decision_events"), list):
        video_events = [ev for ev in video_decision_payload["right_decision_events"] if isinstance(ev, dict)]
    canonical_events = build_canonical_right_events(
        touch_events=touch_events,
        fps=fps,
        video_decision_events=video_events if video_events else None,
    )

    event_idx, touch_event, audio_event = _pick_event(
        audio_decision_payload=audio_decision_payload,
        canonical_events=canonical_events,
        event_id=args.event_id,
        event_index=args.event_index,
    )
    if touch_event is None:
        raise ValueError("Could not resolve touch event for selected index/event_id")

    debug_dir = related["debug_dir"]
    audio_path = _ensure_audio_path(
        args_audio=args.audio,
        args_video=args.video,
        audio_decision_payload=audio_decision_payload,
        touch_json=touch_json,
        cfg=cfg,
        debug_dir=debug_dir,
    )
    if audio_path is None:
        print("No audio source available. Provide --audio or --video.")
        return 1

    sample_rate = int((cfg.get("audio") or {}).get("sample_rate", 16000))
    load_res = load_audio_mono(audio_path, sample_rate=sample_rate)
    if not load_res.ok or load_res.audio is None or load_res.sr is None:
        print(f"Failed to load audio: {load_res.error}")
        return 1
    audio = np.asarray(load_res.audio, dtype=np.float32)
    sr = int(load_res.sr)
    onset_cache = prepare_onset_strength(audio, sr, hop_length=int((cfg.get("audio") or {}).get("onset_strength_hop", 256)))

    touch_time = _float_or_none((touch_event or {}).get("timestamp_sec", (touch_event or {}).get("time_sec")))
    if touch_time is None:
        print("Selected event has no timestamp")
        return 1
    selected_event_id = str((audio_event or {}).get("event_id") or (touch_event or {}).get("event_id") or f"event_{event_idx}")

    audio_part = (audio_event or {}).get("audio") if isinstance((audio_event or {}).get("audio"), dict) else {}
    decision_part = (audio_event or {}).get("decision") if isinstance((audio_event or {}).get("decision"), dict) else {}
    onset_time = _float_or_none(audio_part.get("onset_time_sec")) if isinstance(audio_part, dict) else None
    out_plot = debug_dir / f"{selected_event_id}_audio_onset.png"
    saved_plot_path = _plot_debug(
        out_path=out_plot,
        audio=audio,
        sr=sr,
        touch_time_sec=float(touch_time),
        onset_time_sec=onset_time,
        onset_cache=onset_cache,
    )

    video_event = _align_video_event(video_decision_payload, event_idx)

    print(f"Event index:        {event_idx}")
    print(f"Event id:           {selected_event_id}")
    print(f"Touch time (sec):   {touch_time}")
    print(f"Touch frame index:  {_int_or_none((touch_event or {}).get('frame_index'))}")
    print(f"Finger type:        {(touch_event or {}).get('finger_type')}")
    print(f"Touched string id:  {_int_or_none((touch_event or {}).get('touched_string_id'))}")
    print(f"Audio source:       {audio_path}")
    print(f"Debug plot:         {saved_plot_path}")

    if isinstance(audio_part, dict):
        print("Audio decision:")
        print(f"  status:           {audio_part.get('status')}")
        print(f"  onset_time_sec:   {audio_part.get('onset_time_sec')}")
        print(f"  onset_score:      {audio_part.get('onset_score')}")
        print(f"  pitch_backend:    {audio_part.get('pitch_backend')}")
        print(f"  f0_hz:            {audio_part.get('f0_hz')}")
        print(f"  pitch_conf:       {audio_part.get('pitch_conf')}")
        print(f"  matched_string:   {audio_part.get('matched_string_id')}")
        print(f"  cents_error:      {audio_part.get('cents_error')}")
        print(f"  candidates:       {audio_part.get('candidate_strings')}")
    if isinstance(decision_part, dict):
        print("Audio final:")
        print(f"  struck_string_id: {decision_part.get('struck_string_id')}")
        print(f"  confidence:       {decision_part.get('confidence')}")
        print(f"  confidence_label: {decision_part.get('confidence_label')}")
        print(f"  reject_reason:    {decision_part.get('reject_reason')}")

    if isinstance(video_event, dict):
        print("Video decision:")
        print(f"  label:            {video_event.get('label')}")
        print(f"  decision_reason:  {video_event.get('decision_reason')}")
        print(f"  struck_id:        {video_event.get('struck_id')}")
        print(f"  candidate_score:  {video_event.get('candidate_score')}")
        print(f"  peak_z:           {video_event.get('peak_z')}")
        print(f"  duration:         {video_event.get('duration')}")
        print(f"  impulse:          {video_event.get('impulse')}")
        decision_debug = video_event.get("decision_debug") if isinstance(video_event.get("decision_debug"), dict) else {}
        if decision_debug:
            print(f"  peak_frame:       {decision_debug.get('peak_frame')}")
            print(f"  event_frame:      {decision_debug.get('event_frame_index')}")
            if "candidate_z_scores" in decision_debug:
                print(f"  candidate_z_scores:{decision_debug.get('candidate_z_scores')}")
            else:
                print("  candidate_z_scores: not available in current video decision schema")
    else:
        print("Video decision:     not available")

    # Save a small debug bundle for later inspection.
    debug_bundle_path = debug_dir / f"{selected_event_id}_debug_bundle.json"
    debug_bundle = {
        "event_index": event_idx,
        "event_id": selected_event_id,
        "touch_event": touch_event,
        "audio_event": audio_event,
        "video_event": video_event,
        "audio_source_path": str(audio_path),
        "plot_path": str(saved_plot_path),
    }
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_bundle_path.write_text(json.dumps(debug_bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Debug bundle:       {debug_bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
