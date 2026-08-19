from __future__ import annotations

import numpy as np

from src.audio import decision
from src.audio.onset import OnsetDetectionResult
from src.audio.pitch import PitchEstimateResult
from src.audio.tuning import TuningEntry
from src.pipeline.config import load_pipeline_config


def _config() -> dict:
    return {
        "audio": {
            "enabled": True,
            "decision_mode": "onset_pitch_match",
            "pitch_backend": "test",
            "pitch_fallback_to_onset": True,
            "require_cv_vibration_for_audio_strike": True,
            "onset_window_sec": 0.18,
            "baseline_window_sec": 0.30,
            "onset_threshold": 1.40,
            "pitch_window_sec": 0.12,
            "min_f0_hz": 60.0,
            "max_f0_hz": 1000.0,
            "min_pitch_conf": 0.50,
            "max_cents_error": 50.0,
            "confidence_weights": {
                "onset_z": 1.2,
                "pitch_conf": 1.5,
                "cents_penalty": 1.0,
                "bias": -0.4,
            },
        },
        "fusion": {
            "confidence_thresholds": {
                "high": 0.80,
                "medium": 0.55,
            }
        },
    }


def _touch_event() -> dict:
    return {
        "timestamp_sec": 0.5,
        "frame_index": 15,
        "hand_side": "right",
        "finger_type": "index",
        "touched_string_id": 1,
        "touch_conf": 0.9,
    }


def _patch_audio_detection(monkeypatch, *, f0_hz: float) -> None:
    monkeypatch.setattr(decision, "prepare_onset_strength", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        decision,
        "detect_onset_in_window",
        lambda *args, **kwargs: OnsetDetectionResult(
            onset_time_sec=0.52,
            onset_score=2.2,
            status="onset",
            debug={},
        ),
    )
    monkeypatch.setattr(
        decision,
        "estimate_pitch_with_fallbacks",
        lambda *args, **kwargs: PitchEstimateResult(
            f0_hz=f0_hz,
            pitch_conf=0.9,
            backend="test",
            status="ok",
            debug={},
        ),
    )


def test_high_pitch_is_removed_and_note_uses_cv_vibrated_string(monkeypatch) -> None:
    _patch_audio_detection(monkeypatch, f0_hz=1500.0)
    tuning = {
        1: TuningEntry(string_id=1, frequency_hz=110.0, note_name="A2"),
        2: TuningEntry(string_id=2, frequency_hz=123.47, note_name="B2"),
    }
    video_decision = {
        "time_sec": 0.5,
        "frame_index": 15,
        "finger_type": "index",
        "touched_id": 1,
        "struck_id": 2,
        "label": "strike",
    }

    artifacts = decision.run_audio_decision_for_right_events(
        touch_events=[_touch_event()],
        fps=30.0,
        audio=np.zeros(16000, dtype=np.float32),
        sr=16000,
        config=_config(),
        video_decision_events=[video_decision],
        tuning_by_string=tuning,
    )

    event = artifacts.decision_payload["events"][0]
    assert event["touch"]["cv_vibrated_string_id"] == 2
    assert event["audio"]["status"] == "strike"
    assert event["audio"]["matched_string_id"] == 2
    assert event["audio"]["note_name"] == "B2"
    assert event["audio"]["f0_hz"] is None
    assert event["audio"]["recognition_source"] == "cv_vibration_fallback"
    assert artifacts.strike_payload["events"][0]["struck_string_id"] == 2


def test_audio_onset_without_cv_vibration_is_not_a_strike(monkeypatch) -> None:
    _patch_audio_detection(monkeypatch, f0_hz=220.0)
    video_decision = {
        "time_sec": 0.5,
        "frame_index": 15,
        "finger_type": "index",
        "touched_id": 1,
        "struck_id": None,
        "label": "touch_only",
    }

    artifacts = decision.run_audio_decision_for_right_events(
        touch_events=[_touch_event()],
        fps=30.0,
        audio=np.zeros(16000, dtype=np.float32),
        sr=16000,
        config=_config(),
        video_decision_events=[video_decision],
        tuning_by_string={
            1: TuningEntry(string_id=1, frequency_hz=110.0, note_name="A2"),
        },
    )

    event = artifacts.decision_payload["events"][0]
    assert event["audio"]["status"] == "no_cv_vibration"
    assert event["decision"]["struck_string_id"] is None
    assert artifacts.strike_payload["events"] == []


def test_default_audio_ceiling_is_1000_hz() -> None:
    assert load_pipeline_config(None)["audio"]["max_f0_hz"] == 1000.0
