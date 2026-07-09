from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from saung_strike_video_farneback_rules.src.roi import extract_rotated_roi
from saung_strike_video_farneback_rules.src.rules import (
    evaluate_candidate_scores,
    evaluate_candidate_series,
)
from saung_strike_video_farneback_rules.src.strings import StringGeometry
from saung_strike_video_farneback_rules.src.windows import (
    TouchEvent,
    _compute_candidate_scores,
    build_dynamic_action_windows,
    build_frame_windows,
    select_event_candidates,
)


def _geom(string_id: int, p1: tuple[float, float], p2: tuple[float, float]) -> StringGeometry:
    return StringGeometry(string_id=string_id, mode="endpoints", points=(p1, p2))


def test_build_frame_windows_with_baseline_gap() -> None:
    baseline, action = build_frame_windows(
        f0=100,
        baseline_len=4,
        action_len=4,
        action_start_frame_offset=1,
        baseline_gap_frames=2,
    )
    assert baseline == [94, 95, 96, 97]
    assert action == [101, 102, 103, 104]


def test_build_dynamic_action_windows() -> None:
    baseline, actions = build_dynamic_action_windows(
        f0=100,
        baseline_len=3,
        action_len=4,
        candidate_offsets=[-1, 0, 3],
        baseline_gap=2,
    )
    assert baseline == [95, 96, 97]
    assert actions[-1] == [99, 100, 101, 102]
    assert actions[0] == [100, 101, 102, 103]
    assert actions[3] == [103, 104, 105, 106]


def test_geometry_candidate_selection_and_missing_touched_fallback() -> None:
    strings = {
        10: _geom(10, (10.0, 10.0), (10.0, 40.0)),
        20: _geom(20, (20.0, 10.0), (20.0, 40.0)),
        40: _geom(40, (40.0, 10.0), (40.0, 40.0)),
    }
    event = TouchEvent(
        timestamp_sec=0.0,
        hand_side="right",
        finger_type="thumb",
        touched_string_id=99,
        touch_conf=1.0,
        contact_x=21.0,
        contact_y=25.0,
        finger_x=21.0,
        finger_y=25.0,
        row_index=1,
    )
    candidates, debug = select_event_candidates(
        event=event,
        strings_by_id=strings,
        candidate_radius_default=1,
        candidate_radius_close_contact=0,
        contact_dist_px_thr=8.0,
        geometry_enabled=False,
        missing_touched_id_fallback="nearest_geometry",
        fallback_top_k=2,
        fallback_max_distance_px=15.0,
    )
    assert candidates == [10, 20]
    assert debug["reason"] == "touched_id_missing_geometry_fallback_used"


def test_dynamic_candidate_series_selects_best_offset() -> None:
    series = {
        "candidate_string_id": 8,
        "baseline_seq": [0.01, 0.01, 0.01],
        "action_seq": [0.02, 0.02, 0.02],
        "score_by_frame": {
            100: 0.02,
            101: 0.03,
            102: 0.04,
            103: 0.20,
            104: 0.25,
            105: 0.30,
        },
        "action_windows": {
            0: [100, 101, 102],
            3: [103, 104, 105],
        },
    }
    decision = evaluate_candidate_series(
        series,
        z_thr=1.0,
        thr_peak=1.0,
        thr_duration_frames=1,
        thr_impulse=1.0,
        dynamic_select_metric="max_impulse",
    )
    assert decision.selected_offset == 3
    assert decision.selected_action_frames == [103, 104, 105]


def test_robust_normalization_and_absolute_motion_gate() -> None:
    decision = evaluate_candidate_scores(
        candidate_string_id=1,
        baseline_scores=[0.0, 0.0, 0.0, 0.0],
        action_scores=[0.01, 0.01, 0.01],
        z_thr=1.0,
        thr_peak=1.0,
        thr_duration_frames=1,
        thr_impulse=1.0,
        normalization_mode="robust_mad",
        require_absolute_motion=True,
        min_action_mean=0.02,
        min_action_max=0.05,
    )
    assert decision.normalization_mode == "robust_mad"
    assert decision.baseline_scale >= 0.05
    assert decision.absolute_motion_pass is False
    assert "low_absolute_motion" in decision.reject_reasons


@pytest.mark.skipif(cv2 is None, reason="OpenCV not available")
def test_roi_border_rejection() -> None:
    frame = np.zeros((20, 20), dtype=np.uint8)
    roi, _, debug = extract_rotated_roi(
        frame=frame,
        string_geom=_geom(1, (0.0, 0.0), (8.0, 0.0)),
        roi_w=20,
        roi_h=10,
        border_mode="reject",
        min_inside_fraction=0.95,
        return_debug=True,
    )
    assert roi is None
    assert debug["roi_rejected"] is True


@pytest.mark.skipif(cv2 is None, reason="OpenCV not available")
def test_non_consecutive_transition_scoring() -> None:
    frame0 = np.zeros((64, 64), dtype=np.uint8)
    frame2 = np.zeros((64, 64), dtype=np.uint8)
    frame0[30:33, 10:54] = 255
    frame2[32:35, 10:54] = 255

    event = TouchEvent(
        timestamp_sec=0.0,
        hand_side="right",
        finger_type="thumb",
        touched_string_id=1,
        touch_conf=1.0,
        contact_x=32.0,
        contact_y=32.0,
        finger_x=32.0,
        finger_y=32.0,
        row_index=1,
    )
    geom = _geom(1, (10.0, 32.0), (54.0, 32.0))
    result = _compute_candidate_scores(
        event=event,
        candidate_id=1,
        candidate_geom=geom,
        strings=[geom],
        frames_for_event={0: frame0, 2: frame2},
        frame_indices=[0, 2],
        baseline_frames=[0],
        action_frames=[2],
        action_windows=None,
        roi_w=32,
        roi_h=12,
        trim_ends_ratio=0.0,
        center_band_h=6,
        enable_hand_mask=False,
        hand_mask_mode="finger_point",
        contact_band_exclusion_px=10.0,
        mask_contact_region=True,
        hand_mask_expand_px=0.0,
        farneback_params=None,
        dy_thr=0.1,
        adaptive_roi_enabled=False,
        adaptive_height_ratio=0.45,
        min_roi_h=5,
        max_roi_h=18,
        min_neighbor_distance_px=4.0,
        border_mode="replicate",
        constant_border_value=0,
        reject_if_out_of_frame=False,
        min_inside_fraction=0.95,
        allow_small_gaps=True,
        max_gap_frames=2,
        normalize_by_gap=True,
    )
    assert result.debug["non_consecutive_transitions_scored"] == 1
    assert result.debug["transitions_zeroed_due_to_gap"] == 0
