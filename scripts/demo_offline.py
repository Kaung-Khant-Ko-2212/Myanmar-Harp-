from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline demo for audio-first + AV fusion pipeline")
    parser.add_argument("--video", type=str, help="Path to source video")
    parser.add_argument("--touch-json", type=str, default=None, help="Existing *_touch_events.json (skip detection stage)")
    parser.add_argument("--right-video-json", type=str, default=None, help="Existing *_right_decision_events.json")
    parser.add_argument("--annotated-video", type=str, default=None, help="Existing annotated video to overlay AV labels on")
    parser.add_argument("--model", type=str, default=str(Path("harp_pose_v11m_prepped") / "weights" / "best.pt"))
    parser.add_argument("--config", type=str, default=str(Path("configs") / "config.yaml"))
    parser.add_argument("--fusion-mode", type=str, default=None, help="audio_only | video_only | av_fuse")
    parser.add_argument("--audio-enabled", dest="audio_enabled", action="store_true")
    parser.add_argument("--no-audio", dest="audio_enabled", action="store_false")
    parser.set_defaults(audio_enabled=None)
    args = parser.parse_args()

    if args.touch_json:
        from src.pipeline.run import run_audio_fusion_postprocess

        video_path = Path(args.video) if args.video else None
        out = run_audio_fusion_postprocess(
            video_path=video_path,
            touch_events_json_path=Path(args.touch_json),
            right_video_decision_events_json_path=Path(args.right_video_json) if args.right_video_json else None,
            annotated_video_path=Path(args.annotated_video) if args.annotated_video else None,
            config_path=Path(args.config),
            fusion_mode=args.fusion_mode,
            audio_enabled=args.audio_enabled,
        )
        print("Audio/Fusion postprocess complete")
        print(f"  right_audio_decision_events_count: {out.get('right_audio_decision_events_count')}")
        print(f"  right_audio_strike_events_count:   {out.get('right_audio_strike_events_count')}")
        print(f"  right_av_decision_events_count:    {out.get('right_av_decision_events_count')}")
        print(f"  right_av_strike_events_count:      {out.get('right_av_strike_events_count')}")
        print(f"  right_audio_decision_events_json:  {out.get('right_audio_decision_events_json_path')}")
        print(f"  right_audio_strike_events_json:    {out.get('right_audio_strike_events_json_path')}")
        print(f"  right_av_decision_events_json:     {out.get('right_av_decision_events_json_path')}")
        print(f"  right_av_strike_events_json:       {out.get('right_av_strike_events_json_path')}")
        print(f"  annotated_video_path:              {out.get('annotated_video_path')}")
        if out.get("audio_error"):
            print(f"  audio_error:                       {out.get('audio_error')}")
        return 0

    if not args.video:
        parser.error("--video is required when --touch-json is not provided")

    from src.pipeline.run import run_pipeline

    result = run_pipeline(
        video_path=Path(args.video),
        model_path=Path(args.model),
        config_path=Path(args.config),
        fusion_mode=args.fusion_mode,
        audio_enabled=args.audio_enabled,
        run_video_stage=True,
        run_vibration_stage=False,  # existing vibration stage is currently embedded in backend/app.py
    )
    av = result.get("av_pipeline", {})
    print("Pipeline run complete")
    print(f"  frames_processed:                  {result.get('frames_processed')}")
    print(f"  touch_events_json_path:            {result.get('touch_events_json_path')}")
    print(f"  out_video_path:                    {result.get('out_video_path')}")
    print(f"  out_video_path_av:                 {result.get('out_video_path_av')}")
    print(f"  right_audio_decision_events_count: {av.get('right_audio_decision_events_count')}")
    print(f"  right_audio_strike_events_count:   {av.get('right_audio_strike_events_count')}")
    print(f"  right_av_decision_events_count:    {av.get('right_av_decision_events_count')}")
    print(f"  right_av_strike_events_count:      {av.get('right_av_strike_events_count')}")
    if av.get("audio_error"):
        print(f"  audio_error:                       {av.get('audio_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
