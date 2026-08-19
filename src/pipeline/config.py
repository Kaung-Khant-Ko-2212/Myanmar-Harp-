from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "general": {
        "fps": 60,
        "timezone": "Asia/Yangon",
    },
    "video_vibration": {
        "baseline_sec": 0.25,
        "action_sec": 0.25,
        "action_start_frame_offset": 1,
        "baseline_gap_frames": 0,
        "roi_h": 32,
        "roi_w": 160,
        "roi_thickness_px": 2,
        "center_band_h": 10,
        "trim_ends_ratio": 0.15,
        "roi": {
            "adaptive_enabled": False,
            "adaptive_height_ratio": 0.45,
            "min_roi_h": 5,
            "max_roi_h": 18,
            "min_neighbor_distance_px": 4.0,
            "reject_if_neighbor_overlap": False,
            "reject_if_out_of_frame": False,
            "border_mode": "replicate",
            "constant_border_value": 0,
            "min_inside_fraction": 0.95,
        },
        "windows": {
            "baseline_len": 8,
            "action_len": 8,
            "action_start_frame_offset": 1,
            "baseline_gap_frames": 0,
            "dynamic_action_enabled": False,
            "dynamic_offsets": [-2, -1, 0, 1, 2, 3, 4, 5, 6],
            "dynamic_select_metric": "max_impulse",
        },
        "candidates": {
            "candidate_radius_default": 2,
            "candidate_radius_close_contact": 1,
            "contact_dist_px_thr": 8.0,
            "geometry_enabled": False,
            "geometry_top_k": 5,
            "geometry_max_distance_px": 35.0,
            "always_include_touched_id": True,
            "include_id_radius_fallback": True,
            "missing_touched_id_fallback": "none",
            "fallback_top_k": 3,
            "fallback_max_distance_px": 40.0,
            "log_string_id_inconsistency": True,
        },
        "hand_mask": {
            "enabled": True,
            "mode": "finger_point",
            "expand_px": 8.0,
            "contact_band_exclusion_px": 10.0,
            "mask_contact_region": True,
        },
        "frame_transitions": {
            "allow_small_gaps": False,
            "max_gap_frames": 2,
            "normalize_by_gap": True,
        },
        "farneback": {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        },
        "rules": {
            "z_thr": 2.35,
            "thr_peak": 4.0,
            "thr_duration_frames": 2,
            "thr_impulse": 7.0,
            "normalization_mode": "zscore",
            "min_scale": 0.05,
            "mad_scale_factor": 1.4826,
            "percentile_low": 25,
            "percentile_high": 75,
            "require_absolute_motion": False,
            "min_action_mean": 0.02,
            "min_action_max": 0.05,
            "max_baseline_mean": None,
        },
        "domination": {
            "ratio": 1.2,
        },
        "global_shake": {
            "thr_global_median_peak": 5.0,
            "thr_many_strings_vibrating": 6,
            "shake_probe_count": 6,
        },
    },
    "audio": {
        "enabled": True,
        "decision_mode": "onset_only",  # onset_only | onset_pitch_match
        "sample_rate": 16000,
        "extract_audio": True,
        "onset_window_sec": 0.18,
        "baseline_window_sec": 0.30,
        "onset_strength_hop": 256,
        "onset_threshold": 1.40,
        "pitch_backend": "torchcrepe",
        "pitch_window_sec": 0.12,
        "min_f0_hz": 60.0,
        "max_f0_hz": 1000.0,
        "min_pitch_conf": 0.50,
        "max_cents_error": 50.0,
        "require_cv_vibration_for_audio_strike": True,
        "candidate_radius_default": 2,
        "candidate_radius_close_contact": 1,
        "contact_dist_px_thr": 8.0,
        "tuning_table_path": "configs/saung_tuning.json",
        "confidence_weights": {
            "onset_z": 1.2,
            "pitch_conf": 1.5,
            "cents_penalty": 1.0,
            "bias": -0.4,
        },
    },
    "fusion": {
        "mode": "av_fuse",
        "prefer_audio_when_conf_ge": 0.75,
        "prefer_video_when_audio_missing": True,
        "timing_source": "hybrid",
        "confidence_thresholds": {
            "high": 0.80,
            "medium": 0.55,
        },
    },
    "paths": {
        "legacy_strike_config_path": "saung_strike_video_farneback_rules/configs/config.yaml",
    },
}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_repo_relative_paths(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    repo_dir = Path(__file__).resolve().parents[2]
    audio = dict(out.get("audio") or {})
    tuning_value = audio.get("tuning_table_path")
    if tuning_value:
        tuning_path = Path(str(tuning_value))
        if not tuning_path.is_absolute():
            tuning_path = repo_dir / tuning_path
        audio["tuning_table_path"] = str(tuning_path.resolve())
    out["audio"] = audio
    return out


def load_pipeline_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = DEFAULT_PIPELINE_CONFIG
    if config_path is None:
        return _resolve_repo_relative_paths(cfg)
    path = Path(config_path)
    if not path.exists():
        return _resolve_repo_relative_paths(cfg)
    try:
        import yaml  # type: ignore
    except Exception:
        return _resolve_repo_relative_paths(cfg)
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        return _resolve_repo_relative_paths(cfg)
    if not isinstance(loaded, dict):
        return _resolve_repo_relative_paths(cfg)
    return _resolve_repo_relative_paths(_deep_merge_dict(cfg, loaded))


def confidence_label(score: float, thresholds: dict[str, Any]) -> str:
    high = float(thresholds.get("high", 0.80))
    medium = float(thresholds.get("medium", 0.55))
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def apply_video_vibration_overrides_to_legacy_strike_config(
    legacy_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Map `video_vibration` section into the legacy strike config schema used today."""
    out = dict(legacy_cfg or {})
    vv = pipeline_cfg.get("video_vibration", {})
    if not isinstance(vv, dict):
        return out

    windows = dict(out.get("windows") or {})
    vv_windows = vv.get("windows", {})
    if not isinstance(vv_windows, dict):
        vv_windows = {}
    for k in ("baseline_sec", "action_sec", "action_start_frame_offset", "baseline_gap_frames"):
        if k in vv:
            windows[k] = vv[k]
    for k in (
        "baseline_len",
        "action_len",
        "action_start_frame_offset",
        "baseline_gap_frames",
        "dynamic_action_enabled",
        "dynamic_offsets",
        "dynamic_select_metric",
    ):
        if k in vv_windows:
            windows[k] = vv_windows[k]
    if windows:
        out["windows"] = windows

    roi = dict(out.get("ROI") or {})
    for k in ("roi_h", "roi_w", "center_band_h", "trim_ends_ratio"):
        if k in vv:
            roi[k] = vv[k]
    vv_roi = vv.get("roi", {})
    if isinstance(vv_roi, dict):
        for k in (
            "adaptive_enabled",
            "adaptive_height_ratio",
            "min_roi_h",
            "max_roi_h",
            "min_neighbor_distance_px",
            "reject_if_neighbor_overlap",
            "reject_if_out_of_frame",
            "border_mode",
            "constant_border_value",
            "min_inside_fraction",
        ):
            if k in vv_roi:
                roi[k] = vv_roi[k]
    if roi:
        out["ROI"] = roi

    fb = dict(out.get("farneback_params") or {})
    vv_fb = vv.get("farneback", {})
    if isinstance(vv_fb, dict):
        fb.update(vv_fb)
    if fb:
        out["farneback_params"] = fb

    rules = dict(out.get("rules") or {})
    vv_rules = vv.get("rules", {})
    if isinstance(vv_rules, dict):
        rules.update(vv_rules)
    if rules:
        out["rules"] = rules

    candidates = dict(out.get("candidates") or {})
    for k in ("candidate_radius_default", "candidate_radius_close_contact", "contact_dist_px_thr"):
        if k in vv:
            candidates[k] = vv[k]
    vv_candidates = vv.get("candidates", {})
    if isinstance(vv_candidates, dict):
        candidates.update(vv_candidates)
    if candidates:
        out["candidates"] = candidates

    masking = dict(out.get("masking") or {})
    vv_hand_mask = vv.get("hand_mask", {})
    if isinstance(vv_hand_mask, dict):
        if "enabled" in vv_hand_mask:
            masking["enable_hand_mask"] = vv_hand_mask["enabled"]
        if "expand_px" in vv_hand_mask:
            masking["hand_mask_expand_px"] = vv_hand_mask["expand_px"]
        for k in ("mode", "contact_band_exclusion_px", "mask_contact_region"):
            if k in vv_hand_mask:
                masking[k] = vv_hand_mask[k]
    if masking:
        out["masking"] = masking

    frame_transitions = dict(out.get("frame_transitions") or {})
    vv_frame_transitions = vv.get("frame_transitions", {})
    if isinstance(vv_frame_transitions, dict):
        frame_transitions.update(vv_frame_transitions)
    if frame_transitions:
        out["frame_transitions"] = frame_transitions

    dom = vv.get("domination", {})
    if isinstance(dom, dict) and "ratio" in dom:
        rules = dict(out.get("rules") or {})
        rules["thr_domination_ratio"] = dom["ratio"]
        out["rules"] = rules

    gs = dict(out.get("global_shake") or {})
    vv_gs = vv.get("global_shake", {})
    if isinstance(vv_gs, dict):
        gs.update(vv_gs)
    if gs:
        out["global_shake"] = gs

    return out
