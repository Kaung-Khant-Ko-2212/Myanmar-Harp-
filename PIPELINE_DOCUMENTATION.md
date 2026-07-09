# Myanmar Harp Analysis Pipeline

## 1. System Overview

This repository implements a staged Myanmar harp analysis pipeline with the following high-level flow:

1. Video upload and backend entry point.
2. Video processing and string/touch detection.
3. Legacy vibration-based strike inference.
4. Audio extraction and audio-first strike decision.
5. AV fusion to combine video and audio evidence.
6. Optional beat alignment and slot summary generation.
7. Final annotated video and JSON artifact delivery.

The pipeline is hybrid: it combines deep learning detection, classical optical-flow vibration analysis, signal processing, and rule-based fusion.

## 2. Main Components

### Backend

- `backend/app.py`
  - FastAPI server and `/api/predict-video` endpoint.
  - Upload handling, caching, orchestration of video, strike, audio, and fusion stages.
  - Builds final response payload consumed by the frontend.

- `backend/post_processing.py`
  - Video-stage implementation.
  - YOLO-based harp string detection.
  - MediaPipe hand tracking and fingertip-to-string touch event generation.
  - Annotated video creation and touch event JSON output.
  - Optional strike highlight re-overlay after video-stage inference.

### Shared Pipeline

- `src/pipeline/run.py`
  - Shared orchestration for audio processing and AV fusion.
  - Exposes `run_audio_fusion_postprocess()` and `run_pipeline()`.

- `src/pipeline/config.py`
  - Default pipeline configuration.
  - Loads YAML overrides from `configs/config.yaml`.
  - Maps video vibration settings into the legacy strike config schema.

### Audio

- `src/audio/extract.py`
  - Audio extraction from video using `ffmpeg` or `imageio_ffmpeg`.

- `src/audio/load.py`
  - Loads mono audio data for processing.

- `src/audio/onset.py`
  - Onset detection logic for strike timing.

- `src/audio/pitch.py`
  - Pitch estimation and fallback handling.

- `src/audio/tuning.py`
  - Tuning table loading and matching pitches to string IDs.

- `src/audio/decision.py`
  - Builds canonical right-hand events.
  - Computes audio confidence and strike decisions.

### Fusion

- `src/fusion/fuse.py`
  - Combines video and audio decisions into fused AV judgments.
  - Computes fused confidence, timing choice, and final event labels.

### Video Annotation

- `src/video/annotate.py`
  - Overlays fused AV decisions onto annotated video.

### Legacy Strike Logic

- `saung_strike_video_farneback_rules/`
  - Legacy vibration-based strike inference package.
  - Uses Farneback optical flow, probe windows, domination checks, and shake rejection.
  - Important modules:
    - `src/decision.py`
    - `src/windows.py`
    - `src/rules.py`
    - `src/roi.py`
    - `src/strings.py`

## 3. End-to-End Request Flow

### 3.1 Video Upload

- Client uploads a video to `backend/app.py` via `POST /api/predict-video`.
- The upload is saved under `backend/uploads/`.
- The backend may also accept multipart uploads via `/predict`.

### 3.2 Video Processing Stage

`backend/app.py` calls `backend.post_processing.run_video_predict()`.

This stage does the following:

- Loads the YOLO harp model from `harp_pose_v11m_prepped/weights/best.pt`.
- Opens the uploaded video.
- Runs string detection every frame or on a cadence.
- Detects hands with MediaPipe and optionally uses a worker-thread pipeline.
- Maps right-hand fingertips to detected strings to create touch events.
- Records:
  - annotated video at `backend/predict_postprocessed/<tag>/<video>_annotated.mp4`
  - touch events JSON at `backend/touch_events/<tag>/<video>_touch_events.json`
  - left/right touch JSON split files
  - per-frame string geometry JSONL `*_strings_by_frame.jsonl`
- Returns `touch_events_json_path`, `out_video_path`, and other metadata.

Outputs from this stage are consumed by the later audio and fusion stages.

### 3.3 Legacy Vibration-Based Strike Inference

`backend/app.py` uses the legacy strike package when enabled.

Key steps:

- Imports `saung_strike_video_farneback_rules.src.decision.decide_touch_events`.
- Converts the new pipeline touch events into legacy `TouchEvent` objects.
- Uses string endpoint geometries to build `StringGeometry` instances.
- Applies Farneback optical-flow vibration analysis to string candidate ROIs.
- Uses thresholds, domination checks, finger gate heuristics, and shake rejection.
- Writes:
  - `*_right_decision_events.json`
  - `*_right_strike_events.json`

These outputs become the video decision payload used during fusion.

### 3.4 Audio Extraction and Decision

`src.pipeline.run.run_audio_fusion_postprocess()` handles audio and audio-only decision logic.

Audio stage steps:

- Loads pipeline config from `configs/config.yaml`.
- Determines FPS from touch payload or config.
- Extracts audio from the uploaded video with `ffmpeg` if needed.
- Loads mono audio at configured sample rate.
- Builds canonical right-hand touch events.
- Runs audio decision logic from `src.audio.decision.run_audio_decision_for_right_events()`.
- Optionally loads tuning table from `configs/saung_tuning.json`.
- Writes:
  - `*_right_audio_decision_events.json`
  - `*_right_audio_strike_events.json`

This stage produces confidence and string match information using onset strength, optional pitch matching, and touch context.

### 3.5 AV Fusion

After audio decisions, the system fuses audio and video using `src.fusion.fuse.fuse_audio_video_decisions()`.

Fusion behavior:

- Aligns canonical touch events with video decisions and audio decisions.
- Computes fused status based on:
  - `fusion.mode` (`av_fuse`, `audio_only`, `video_only`)
  - audio/video confidence values
  - thresholds from config
  - audio presence and missing-data rules
- Selects timing source according to `fusion.timing_source`.
- Builds final AV decision and strike payloads.
- Writes:
  - `*_right_av_decision_events.json`
  - `*_right_av_strike_events.json`

### 3.6 Beat Alignment and Slot Summary

`run_audio_fusion_postprocess()` optionally performs beat alignment:

- Uses `librosa` to detect beats from audio.
- Attaches beat metadata to fused decision and strike payloads.
- Labels fused strikes as `on_beat` or `off_beat`.
- Builds an alternating on/off slot summary:
  - `*_alternating_on_off_slots.json`
  - Infers string candidates per beat slot.
  - Detects left-hand involvement from nearby left touch events.

### 3.7 Overlay and Response

If `enable_overlay` is true and an annotated video exists:

- Calls `src.video.annotate.overlay_av_decisions_on_video()`.
- Produces an AV-overlay video variant.
- Final annotated video path is updated in the response.

The backend assembles a response containing:

- `predicted_video_url`
- `annotated_video_path`
- `touch_events_json_path`
- `right_decision_events_json_path`
- `right_strike_events_json_path`
- `right_audio_decision_events_json_path`
- `right_audio_strike_events_json_path`
- `right_av_decision_events_json_path`
- `right_av_strike_events_json_path`
- `right_av_alternating_on_off_slots_json_path`
- counts for touch, decision, and strike events
- timing metadata and pipeline stats
- any audio extraction or fusion errors

## 4. Configuration

The pipeline uses defaults from `src/pipeline/config.py` and loads overrides from `configs/config.yaml`.

Important configuration sections:

- `general`
  - `fps`
  - `timezone`

- `video_vibration`
  - `baseline_sec`, `action_sec`, `action_start_frame_offset`
  - ROI geometry, Farneback optical flow params, and rule thresholds
  - `domination.ratio`
  - `global_shake` thresholds

- `audio`
  - `enabled`
  - `decision_mode`: `onset_only` or `onset_pitch_match`
  - sample rate and onset thresholds
  - `pitch_backend`
  - pitch confidence and cents error thresholds
  - `tuning_table_path`
  - `confidence_weights`

- `fusion`
  - `mode`: `av_fuse`
  - `prefer_audio_when_conf_ge`
  - `prefer_video_when_audio_missing`
  - `timing_source`: `hybrid`, `audio`, `video`
  - `confidence_thresholds`

- `paths`
  - `legacy_strike_config_path`

`src/pipeline/config.py` also includes:

- `_deep_merge_dict()` for YAML overrides.
- `confidence_label()` to assign `high`/`medium`/`low`.
- `apply_video_vibration_overrides_to_legacy_strike_config()` 
  - maps video vibration config into the older legacy strike schema.

## 5. Offline and Script Usage

The repository includes offline scripts that exercise the shared pipeline:

- `scripts/demo_offline.py`
  - can call `src.pipeline.run.run_pipeline()`.

- `scripts/debug_one_event.py`
  - loads config and runs audio or pipeline debugging.

`src.pipeline.run.run_pipeline()` supports:

- full video stage via `backend.post_processing.run_video_predict`
- optional vibration stage callback
- audio/AV fusion
- final AV annotated video output

## 6. Key Artifact Paths

- `backend/uploads/`: uploaded video files
- `backend/predict_postprocessed/<tag>/...`: annotated videos and outputs
- `backend/touch_events/<tag>/...`: touch and string geometry JSON and JSONL
- `backend/cache/`: request cache manifests
- `configs/config.yaml`: pipeline runtime overrides
- `configs/saung_tuning.json`: tuning table for pitch matching

## 7. Important Notes

- The backend must preserve repository import precedence so that `src/` from the repo is preferred over the legacy package's `src/` namespace.
- The current pipeline is hybrid: new shared `src/` code is integrated with legacy `saung_strike_video_farneback_rules/` logic.
- `backend/post_processing.py` is the video stage origin, while `src/pipeline/run.py` is the shared AV pipeline orchestrator.
- Beat alignment and slot summary are optional pipeline enhancements, not core strike inference.
