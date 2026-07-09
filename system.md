# System Overview

This repository implements an end-to-end Myanmar harp analysis system with three main layers:

1. A FastAPI backend that accepts uploaded video, runs computer vision, infers touch and strike events, runs audio and AV post-processing, and serves the final annotated media.
2. A React frontend that uploads a video, polls for the result, and visualizes the returned annotations and event summaries.
3. A shared Python pipeline under `src/` plus a legacy strike-decision package under `saung_strike_video_farneback_rules/`.

The system is not a single monolithic model. It is a staged pipeline that combines:

- YOLO pose/keypoint detection for harp string geometry
- MediaPipe hand tracking for fingertip positions
- touch-event generation from fingertip-to-string proximity
- video-based strike inference from optical-flow vibration
- audio-first onset and optional pitch-based strike inference
- AV fusion to combine video and audio evidence
- final video annotation and JSON artifact generation

## Top-Level Layout

- `backend/`
  Main backend application and core video-processing implementation.
- `src/`
  Shared pipeline code for audio, fusion, config mapping, JSON helpers, and video overlays.
- `saung_strike_video_farneback_rules/`
  Legacy but active strike-decision package used for video-based vibration analysis.
- `configs/`
  Primary pipeline config for audio, fusion, and shared runtime settings.
- `myanma-saung-main/`
  Frontend React/Vite application.
- `harp_pose_v11m_prepped/`
  YOLO weights used by the harp detector.
- `backend/predict_postprocessed/`
  Generated annotated videos and related outputs.
- `backend/touch_events/`
  Generated touch and decision JSON files.

## Main Entry Points

### Backend API

Primary backend entrypoint:

- `backend/app.py`

Important responsibilities:

- creates the FastAPI app
- exposes `/api/predict-video`
- serves generated videos from `/predictions`
- runs the video stage via `backend.post_processing.run_video_predict`
- runs video strike inference
- runs audio and AV post-processing via `src.pipeline.run.run_audio_fusion_postprocess`
- assembles the final response payload used by the frontend

### Video Processing Stage

Primary implementation:

- `backend/post_processing.py`

This file handles:

- loading YOLO weights
- reading the uploaded video
- running string detection on frames
- drawing the string geometry on the output video
- running MediaPipe hand tracking
- mapping fingertips to strings to produce touch events
- saving per-frame string geometry samples
- writing annotated video output
- writing touch-event JSON
- writing per-frame string geometry JSONL
- running a second-pass strike highlight overlay that reuses the exact frame-level strings already drawn in the video stage

### Shared Pipeline Stage

Primary orchestrator:

- `src/pipeline/run.py`

This file handles:

- loading touch-event JSON and optional prior decision JSON
- extracting or loading audio
- running audio-first strike inference
- fusing audio and video decisions
- optional beat alignment
- writing audio, AV, and summary JSON files
- producing a final overlay video for AV decisions

### Frontend

Primary page:

- `myanma-saung-main/src/pages/Index.tsx`

This page:

- uploads a video to the backend
- calls `/api/predict-video`
- displays the returned video URL
- shows touch, strike, audio, and AV event summaries
- reads paths and counts for generated JSON artifacts

## End-to-End Request Flow

### 1. Upload

The frontend posts a video file directly to:

- `/api/predict-video`

The backend stores the upload under:

- `backend/uploads/`

### 2. Video Stage

`backend/app.py` calls:

- `backend.post_processing.run_video_predict`

This stage:

- loads the YOLO harp model from `harp_pose_v11m_prepped/weights/best.pt`
- reads each frame from the uploaded video
- detects harp geometry and extracts corrected string endpoints
- draws the strings on the frame
- optionally runs hand tracking
- maps fingertips to nearest strings
- emits touch events

Outputs from this stage include:

- annotated video
- touch events list
- touch events JSON
- median string geometry summary
- per-frame string geometry JSONL

### 3. Video Strike Inference

Still inside `backend/app.py`, the backend converts touch events and string geometry into the format expected by the legacy strike package and calls:

- `saung_strike_video_farneback_rules.src.decision.decide_touch_events`

That package:

- builds candidate strings near the touched string
- extracts small ROIs around each candidate string
- measures vibration using Farneback optical flow
- compares pre-touch baseline frames to post-touch action frames
- computes per-candidate vibration metrics
- selects a struck string if the best candidate passes thresholds and domination checks
- rejects shake-heavy events with global-shake heuristics

Key code:

- `saung_strike_video_farneback_rules/src/windows.py`
- `saung_strike_video_farneback_rules/src/rules.py`
- `saung_strike_video_farneback_rules/src/decision.py`

Generated JSON artifacts:

- `*_right_decision_events.json`
- `*_right_strike_events.json`

### 4. Second-Pass Strike Highlight

After video strike inference, the backend now performs a second pass on the annotated video using:

- `backend.post_processing.highlight_strikes_on_video`

Purpose:

- reuse the exact per-frame strings that were already drawn during `run_video_predict`
- avoid the older approximation where highlight overlays were based only on median string geometry

This stage reads:

- the annotated video from the video stage
- the per-frame string geometry JSONL
- the strike decision list

It outputs a strike-highlighted video variant before AV post-processing starts.

### 5. Audio Stage

`backend/app.py` then calls:

- `src.pipeline.run.run_audio_fusion_postprocess`

The audio portion:

- extracts audio from the uploaded video if needed
- computes onset strength near right-hand touch events
- optionally estimates pitch and matches it to a tuning table
- emits audio decision JSON and audio strike JSON

Important modules:

- `src/audio/extract.py`
- `src/audio/load.py`
- `src/audio/onset.py`
- `src/audio/pitch.py`
- `src/audio/tuning.py`
- `src/audio/decision.py`

Generated JSON artifacts:

- `*_right_audio_decision_events.json`
- `*_right_audio_strike_events.json`

### 6. AV Fusion

The shared pipeline then combines video and audio evidence using:

- `src.fusion.fuse.fuse_audio_video_decisions`

Fusion decides:

- whether the event is a strike
- which string identity to trust
- confidence level
- decision strategy metadata

Generated JSON artifacts:

- `*_right_av_decision_events.json`
- `*_right_av_strike_events.json`

### 7. Beat Alignment and Slot Summary

If enabled, the pipeline:

- detects beats from audio
- labels fused strike events as on-beat or off-beat
- creates an alternating on/off slot summary

Generated JSON artifact:

- `*_alternating_on_off_slots.json`

### 8. Final Response

The backend returns a single JSON response containing:

- final video URL
- local output paths
- touch events
- strike results
- audio events
- AV events
- slot summaries
- config and timing metadata

The frontend uses this response directly.

## How Touch Events Are Built

Touch events are generated in `backend/post_processing.py`.

Flow:

1. detect fingertip landmarks with MediaPipe
2. keep only the configured touch tips
3. find the nearest string line to each fingertip
4. accept a touch if fingertip-to-string distance is under threshold
5. emit an event only when the fingertip changes into a new touched string state

Current right-hand touch tips:

- thumb
- index

Current left-hand touch tip:

- thumb

This is enforced directly in `backend/post_processing.py`.

## How Video Strike Identity Is Chosen

The video strike module does not simply reuse the touched string.

For each touch:

1. choose nearby candidate strings
2. measure optical-flow vibration for each candidate
3. compute z-score based metrics against a baseline
4. mark candidates as vibrating if they pass thresholds
5. pick the best vibrating candidate
6. reject if dominance is weak or if many strings appear to vibrate globally

Important consequence:

- `touched_string_id` and `struck_id` can differ

## How Audio Strike Identity Is Chosen

Audio identity depends on `configs/config.yaml`.

Current mode:

- `audio.decision_mode: onset_only`

That means:

- audio confirms that an onset happened near the touch
- it does not independently identify the string from pitch
- it typically reuses the touched string as the audio strike identity

If switched to `onset_pitch_match`, audio can try to infer the string from pitch and tuning instead.

## Configuration Layers

There are two main configuration files.

### Shared pipeline config

- `configs/config.yaml`

Controls:

- global FPS defaults
- audio analysis and pitch-matching settings
- fusion mode and confidence thresholds
- beat alignment
- mapping into the legacy strike config schema

### Legacy strike config

- `saung_strike_video_farneback_rules/configs/config.yaml`

Controls:

- baseline and action windows
- candidate radius near the touched string
- optical-flow ROI sizing
- vibration thresholds
- shake rejection thresholds
- finger gating
- stabilization
- allowed right-hand finger types for strike decisions

## Important Generated Artifacts

### Video outputs

- `backend/predict_postprocessed/<tag>/<video>_annotated.mp4`
- `backend/predict_postprocessed/<tag>/<video>_annotated_strike.mp4`
- later AV overlay variants produced by `src/video/annotate.py`

### Touch and decision JSON

- `backend/touch_events/<tag>/<video>_touch_events.json`
- `backend/touch_events/<tag>/<video>_left_touch_events.json`
- `backend/touch_events/<tag>/<video>_right_decision_events.json`
- `backend/touch_events/<tag>/<video>_right_strike_events.json`
- `backend/touch_events/<tag>/<video>_right_audio_decision_events.json`
- `backend/touch_events/<tag>/<video>_right_audio_strike_events.json`
- `backend/touch_events/<tag>/<video>_right_av_decision_events.json`
- `backend/touch_events/<tag>/<video>_right_av_strike_events.json`
- `backend/touch_events/<tag>/<video>_alternating_on_off_slots.json`
- `backend/touch_events/<tag>/<video>_strings_by_frame.jsonl`

## Frontend/Backend Contract

The frontend expects the backend response to include:

- `predicted_video_url`
- `touch_events`
- `strike_results`
- audio and AV event arrays
- JSON artifact paths
- counts and metadata

Primary consumer:

- `myanma-saung-main/src/pages/Index.tsx`

The frontend defaults to:

- `VITE_API_BASE_URL=http://127.0.0.1:8000`

## Run Modes

### Full web flow

1. start backend from `backend/app.py`
2. start frontend from `myanma-saung-main`
3. upload a video through the UI

### Direct backend CLI flow

You can also run the video stage directly through:

- `backend/post_processing.py`

This is useful for local debugging when you want to inspect the raw CV stage without the full web app.

## Known Architectural Characteristics

- The system is a hybrid pipeline, not an end-to-end learned strike classifier.
- Video strike inference still relies on a legacy package with its own config schema.
- Audio and video can disagree on string identity; AV fusion resolves the final output.
- The final visual highlight is now closer to the video-stage geometry because the backend saves per-frame strings and reuses them in a second pass.
- The codebase currently mixes newer shared pipeline code under `src/` with older specialized strike logic under `saung_strike_video_farneback_rules/`.

## Files To Read First

If you need to understand or modify the system quickly, start with these files in order:

1. `backend/app.py`
2. `backend/post_processing.py`
3. `src/pipeline/run.py`
4. `src/audio/decision.py`
5. `src/fusion/fuse.py`
6. `saung_strike_video_farneback_rules/src/decision.py`
7. `myanma-saung-main/src/pages/Index.tsx`
