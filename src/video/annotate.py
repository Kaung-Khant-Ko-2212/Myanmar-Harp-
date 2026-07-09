from __future__ import annotations

from pathlib import Path
import time
from typing import Any


def _int_or_none(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(round(float(v)))
    except Exception:
        return None


def _build_event_overlays(fusion_decision_payload: dict[str, Any], fps: float, hold_frames: int = 10) -> dict[int, list[str]]:
    overlays: dict[int, list[str]] = {}
    events = fusion_decision_payload.get("events") if isinstance(fusion_decision_payload, dict) else None
    if not isinstance(events, list):
        return overlays
    for item in events:
        if not isinstance(item, dict):
            continue
        touch = item.get("touch") if isinstance(item.get("touch"), dict) else {}
        video = item.get("video") if isinstance(item.get("video"), dict) else {}
        audio = item.get("audio") if isinstance(item.get("audio"), dict) else {}
        fusion = item.get("fusion") if isinstance(item.get("fusion"), dict) else {}
        timing = fusion.get("timing") if isinstance(fusion.get("timing"), dict) else {}

        frame_candidates = [
            _int_or_none(timing.get("onset_frame")),
            _int_or_none(touch.get("frame_index")),
            _int_or_none(video.get("frame_index")),
        ]
        frame_idx = next((v for v in frame_candidates if v is not None), None)
        if frame_idx is None:
            onset_time = timing.get("onset_time_sec")
            try:
                frame_idx = int(round(float(onset_time) * max(float(fps), 1e-6)))
            except Exception:
                continue

        touch_sid = _int_or_none(touch.get("touched_string_id"))
        video_sid = _int_or_none(video.get("struck_string_id"))
        audio_sid = _int_or_none(audio.get("struck_string_id"))
        fused_sid = _int_or_none(fusion.get("struck_string_id"))
        conf_label = str(fusion.get("confidence_label") or "")
        strategy = str(fusion.get("strategy") or "")
        status = str(fusion.get("status") or "")
        if status != "strike":
            continue
        beat_label = str(fusion.get("beat_label") or "")
        finger = str(touch.get("finger_type") or "")
        beat_suffix = f" {beat_label}" if beat_label else ""
        line = f"{finger} T:{touch_sid if touch_sid is not None else '-'} V:{video_sid if video_sid is not None else '-'} A:{audio_sid if audio_sid is not None else '-'} F:{fused_sid if fused_sid is not None else '-'} {status}{beat_suffix} {conf_label} {strategy}".strip()
        for f in range(int(frame_idx), int(frame_idx) + max(1, int(hold_frames))):
            overlays.setdefault(int(f), []).append(line)
    return overlays


def _float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _parse_point(v: Any) -> tuple[int, int] | None:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    x = _float_or_none(v[0])
    y = _float_or_none(v[1])
    if x is None or y is None:
        return None
    return int(round(x)), int(round(y))


def _build_string_endpoint_map(string_geometries: Any) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    out: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    if not isinstance(string_geometries, list):
        return out
    for item in string_geometries:
        if not isinstance(item, dict):
            continue
        sid = _int_or_none(item.get("string_id"))
        endpoints = item.get("endpoints")
        if sid is None or not isinstance(endpoints, list) or len(endpoints) < 2:
            continue
        p1 = _parse_point(endpoints[0])
        p2 = _parse_point(endpoints[1])
        if p1 is None or p2 is None:
            continue
        out[int(sid)] = (p1, p2)
    return out


def _build_strike_highlights(
    *,
    fusion_strike_payload: dict[str, Any] | None,
    fusion_decision_payload: dict[str, Any],
    fps: float,
    hold_frames: int,
) -> dict[int, list[int]]:
    highlights: dict[int, list[int]] = {}
    events = fusion_strike_payload.get("events") if isinstance(fusion_strike_payload, dict) else None

    if not isinstance(events, list):
        # Fallback: derive strike events from fusion decision events.
        decisions = fusion_decision_payload.get("events") if isinstance(fusion_decision_payload, dict) else None
        events = []
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                fusion = item.get("fusion") if isinstance(item.get("fusion"), dict) else {}
                if str(fusion.get("status") or "") != "strike":
                    continue
                events.append(
                    {
                        "frame_index": (fusion.get("timing") or {}).get("onset_frame")
                        if isinstance(fusion.get("timing"), dict)
                        else None,
                        "timestamp_sec": (fusion.get("timing") or {}).get("onset_time_sec")
                        if isinstance(fusion.get("timing"), dict)
                        else None,
                        "struck_string_id": fusion.get("struck_string_id"),
                    }
                )

    if not isinstance(events, list):
        return highlights

    for item in events:
        if not isinstance(item, dict):
            continue
        sid = _int_or_none(item.get("struck_string_id"))
        if sid is None:
            continue

        frame_idx = _int_or_none(item.get("frame_index"))
        if frame_idx is None:
            t_sec = _float_or_none(item.get("timestamp_sec"))
            if t_sec is None:
                t_sec = _float_or_none(item.get("time_sec"))
            if t_sec is not None:
                frame_idx = int(round(float(t_sec) * max(float(fps), 1e-6)))
        if frame_idx is None:
            continue

        for f in range(int(frame_idx), int(frame_idx) + max(1, int(hold_frames))):
            highlights.setdefault(int(f), []).append(int(sid))

    return highlights


def overlay_av_decisions_on_video(
    *,
    input_video_path: str | Path,
    output_video_path: str | Path,
    fusion_decision_payload: dict[str, Any],
    fusion_strike_payload: dict[str, Any] | None = None,
    string_geometries: list[dict[str, Any]] | None = None,
    source_with_audio: str | Path | None = None,
    hold_frames: int = 10,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"opencv_missing:{exc}"}

    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    if not in_path.exists():
        return {"ok": False, "error": f"input_video_missing:{in_path}"}

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        return {"ok": False, "error": f"cannot_open_video:{in_path}"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return {"ok": False, "error": "invalid_video_dimensions"}

    overlays = _build_event_overlays(fusion_decision_payload, fps=fps, hold_frames=hold_frames)
    string_endpoint_map = _build_string_endpoint_map(string_geometries)
    strike_highlights = _build_strike_highlights(
        fusion_strike_payload=fusion_strike_payload,
        fusion_decision_payload=fusion_decision_payload,
        fps=fps,
        hold_frames=hold_frames,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"ok": False, "error": f"cannot_open_writer:{out_path}"}

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0
    strike_frames_drawn = 0
    draw_started_at = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            struck_sids = strike_highlights.get(frame_idx, [])
            if struck_sids and string_endpoint_map:
                highlight_overlay = frame.copy()
                drawn_any = False
                # Draw only unique string ids in this frame to avoid overdrawing duplicates.
                for sid in sorted(set(int(s) for s in struck_sids)):
                    endpoints = string_endpoint_map.get(int(sid))
                    if endpoints is None:
                        continue
                    p1, p2 = endpoints
                    cv2.line(highlight_overlay, p1, p2, (0, 120, 255), 8, cv2.LINE_AA)
                    cv2.line(highlight_overlay, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.circle(highlight_overlay, p1, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(highlight_overlay, p2, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    drawn_any = True
                if drawn_any:
                    cv2.addWeighted(highlight_overlay, 0.75, frame, 0.25, 0.0, frame)
                    strike_frames_drawn += 1

            lines = overlays.get(frame_idx, [])
            # Header strip
            if lines:
                overlay = frame.copy()
                header_h = min(height, 26 + 22 * min(len(lines), 4))
                cv2.rectangle(overlay, (8, 8), (width - 8, 8 + header_h), (0, 0, 0), thickness=-1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
                y = 28
                cv2.putText(frame, "AV Strike Overlay", (16, y), font, 0.65, (60, 230, 255), 2, cv2.LINE_AA)
                y += 20
                for line in lines[:4]:
                    cv2.putText(frame, line[:140], (16, y), font, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                    y += 20

            cv2.putText(frame, f"frame {frame_idx}", (12, height - 14), font, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    draw_elapsed_sec = round(float(time.perf_counter() - draw_started_at), 3)

    # Best-effort H.264 transcode with preserved audio by reusing existing backend helper.
    transcoded = False
    audio_muxed = False
    final_path = out_path
    transcode_elapsed_sec = 0.0
    try:
        try:
            from backend.post_processing import transcode_to_h264  # type: ignore
        except Exception:
            from post_processing import transcode_to_h264  # type: ignore
        transcode_started_at = time.perf_counter()
        final_path, transcoded, audio_muxed = transcode_to_h264(
            final_path,
            source_with_audio=Path(source_with_audio) if source_with_audio else None,
        )
        transcode_elapsed_sec = round(float(time.perf_counter() - transcode_started_at), 3)
    except Exception:
        pass

    return {
        "ok": True,
        "output_video_path": str(final_path),
        "fps": fps,
        "frames_annotated": int(frame_idx),
        "strike_highlight_frames": int(strike_frames_drawn),
        "strike_highlight_strings_available": int(len(string_endpoint_map)),
        "transcoded": bool(transcoded),
        "audio_muxed": bool(audio_muxed),
        "draw_elapsed_sec": float(draw_elapsed_sec),
        "transcode_elapsed_sec": float(transcode_elapsed_sec),
        "elapsed_sec": round(float(time.perf_counter() - started_at), 3),
    }
