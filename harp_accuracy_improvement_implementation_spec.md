# Myanmar Harp Strike Detection Accuracy Improvements

## Purpose

This document is an implementation brief for improving the accuracy of the Myanmar harp/Saung Gauk strike-detection pipeline.

The current pipeline detects strings and hand touches, then uses Farneback optical flow to infer string vibration, audio onset/pitch analysis to infer acoustic strikes, and fusion logic to produce final AV strike decisions.

The 10 tasks below target the most likely causes of poor accuracy in the vibration stage and related decision logic:

1. ROI design is too wide and not string-specific.
2. Action window alignment is fixed and may miss the vibration peak.
3. Candidate selection uses integer string ID radius instead of geometry.
4. Z-score normalization is unstable.
5. Baseline/action windows are rigid and can be contaminated.
6. `BORDER_REPLICATE` may fabricate motion at ROI edges.
7. Hand/finger masking is weak.
8. Score computation ignores non-consecutive frame transitions.
9. Decision rules rely only on relative impulse, not absolute motion.
10. Candidate selection assumes string IDs are reliable and present.

The implementation should be minimally invasive, configurable, and backward-compatible. Existing behavior should remain available behind config flags.

---

## Relevant Files

Primary files:

- `saung_strike_video_farneback_rules/src/roi.py`
- `saung_strike_video_farneback_rules/src/windows.py`
- `saung_strike_video_farneback_rules/src/rules.py`

Likely supporting files:

- `saung_strike_video_farneback_rules/src/mask.py`
- `saung_strike_video_farneback_rules/src/strings.py`
- `saung_strike_video_farneback_rules/src/decision.py`
- `src/pipeline/config.py`
- `configs/config.yaml`

Existing key functions:

- `roi_box_corners()`
- `extract_rotated_roi()`
- `build_frame_windows()`
- `select_event_candidates()`
- `_compute_candidate_scores()`
- `process_single_event_windows()`
- `evaluate_candidate_scores()`
- `evaluate_candidate_series()`
- `evaluate_all_candidates()`

---

## Global Implementation Requirements

### Backward Compatibility

All new behavior must be controlled by config flags. The current behavior should remain the default unless the calling config enables the new behavior.

Example:

```yaml
video_vibration:
  roi:
    adaptive_enabled: false
  windows:
    dynamic_action_enabled: false
  candidates:
    geometry_enabled: false
  rules:
    robust_norm_enabled: false
```

### Debuggability

Every task must add structured debug output. The debug output should make it possible to compare old vs new behavior per event.

At minimum, each event debug payload should include:

```json
{
  "event_row_index": 123,
  "f0": 456,
  "touched_string_id": 8,
  "candidate_ids": [7, 8, 9],
  "winner_candidate_id": 8,
  "baseline_frames": [440, 441],
  "action_frames": [457, 458],
  "peak_frame": 461,
  "peak_relative_to_f0": 5,
  "reject_reasons": [],
  "roi_debug": {},
  "candidate_debug": {},
  "score_debug": {}
}
```

### Measurement Requirements

After each implementation, evaluate on the same fixed validation set.

Required metrics:

- Event-level precision
- Event-level recall
- Event-level F1
- False positives
- False negatives
- Candidate coverage: whether the true string was included in the candidate set
- Peak alignment: detected vibration peak frame minus touch frame `f0`
- Score stability: distribution of baseline mean/std and action mean/max
- ROI validity rate: percent of ROIs fully inside frame
- Missing-frame rate
- Neighbor-string confusion rate

If no ground-truth dataset exists, create a small manually labeled set first.

Minimum useful validation set:

- 20 clear true strikes
- 20 non-strike touches/rests
- 10 weak strikes
- 10 fast repeated strikes
- 10 edge/occlusion cases
- 10 cases with neighboring strings close together

---

# 1. Adaptive, String-Specific ROI Design

## Problem

The current ROI is a fixed-width rotated rectangle around the string mid-segment. If strings are close together, the ROI may include neighboring strings, finger motion, or background texture. This contaminates the Farneback optical-flow score.

Current behavior:

- `roi_box_corners()` creates a fixed rectangle using `roi_w` and `roi_h`.
- `extract_rotated_roi()` warps that rectangle from the full frame into a grayscale ROI.
- The ROI does not account for nearest-neighbor string spacing.

## Goal

Make ROI height adaptive based on neighboring string distance, so each candidate ROI is narrow enough to isolate one string while still covering visible vibration.

## Files to Modify

- `roi.py`
- `windows.py`
- possibly `strings.py`

## New Config

```yaml
video_vibration:
  roi:
    adaptive_enabled: true
    adaptive_height_ratio: 0.45
    min_roi_h: 5
    max_roi_h: 18
    min_neighbor_distance_px: 4.0
    reject_if_neighbor_overlap: false
    reject_if_out_of_frame: false
```

## Implementation Details

### 1. Add helper to compute nearest-neighbor string distance

Create a function such as:

```python
def estimate_neighbor_spacing_px(
    target_geom: StringGeometry,
    all_strings: list[StringGeometry],
    sample_count: int = 5,
    trim_ends_ratio: float = 0.15,
) -> float | None:
    ...
```

Recommended logic:

1. Sample several points along the target string mid-segment.
2. For every other string, compute the minimum perpendicular/nearest distance to those sample points.
3. Return the minimum distance to the nearest neighboring string.
4. If no valid neighbor exists, return `None`.

### 2. Compute adaptive ROI height

```python
adaptive_h = int(neighbor_spacing_px * adaptive_height_ratio)
adaptive_h = max(min_roi_h, min(adaptive_h, max_roi_h))
```

If `neighbor_spacing_px` is missing or too small, fallback to existing `roi_h`.

### 3. Preserve existing `roi_w`

Initially keep `roi_w` unchanged. Optimize height first.

### 4. Add ROI debug data

For each candidate, record:

```json
{
  "roi_w": 96,
  "roi_h_original": 20,
  "roi_h_used": 9,
  "adaptive_roi_enabled": true,
  "neighbor_spacing_px": 22.4,
  "adaptive_height_ratio": 0.45
}
```

## Acceptance Criteria

- Existing fixed ROI behavior still works when `adaptive_enabled: false`.
- Adaptive ROI reduces neighboring-string overlap in debug overlays.
- Strike precision improves or stays equal without major recall loss.
- Candidate score for neighboring strings decreases in known confusion cases.

---

# 2. Dynamic Action Window Alignment

## Problem

The current action window starts at a fixed offset from the rounded touch frame:

```python
action_start = f0 + action_start_frame_offset
```

But real harp vibration often starts after finger release, not at initial contact. The true vibration peak may occur several frames before or after the fixed action window.

## Goal

Search several possible action windows around `f0` and choose the window with the strongest valid vibration evidence.

## Files to Modify

- `windows.py`
- `rules.py` if decision output needs peak frame or selected window metadata

## New Config

```yaml
video_vibration:
  windows:
    dynamic_action_enabled: true
    dynamic_offsets: [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    dynamic_select_metric: "max_impulse"  # options: max_impulse, max_peak, max_action_mean
    action_len: 8
    baseline_len: 8
```

## Implementation Details

### 1. Keep existing `build_frame_windows()`

Do not break current behavior. Add a new function:

```python
def build_dynamic_action_windows(
    *,
    f0: int,
    baseline_len: int,
    action_len: int,
    candidate_offsets: list[int],
    baseline_gap: int = 0,
    max_frame_index: int | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
    ...
```

Return:

- one baseline frame list
- a dictionary mapping offset to action frame list

Example:

```python
{
  -1: [99, 100, 101, 102],
   0: [100, 101, 102, 103],
   3: [103, 104, 105, 106]
}
```

### 2. Score each candidate action window

For each string candidate:

1. Compute scores over the union of baseline frames and all possible action frames.
2. Evaluate each action window separately.
3. Select the best window using configured metric.

### 3. Record selected offset and peak frame

Debug payload:

```json
{
  "dynamic_action_enabled": true,
  "tested_offsets": [-2, -1, 0, 1, 2, 3, 4, 5, 6],
  "selected_offset": 4,
  "selected_action_frames": [104, 105, 106, 107],
  "peak_frame": 106,
  "peak_relative_to_f0": 6
}
```

## Acceptance Criteria

- Existing fixed-window behavior works when dynamic mode is disabled.
- The system logs peak location relative to `f0`.
- Recall improves on late-release strikes.
- False positives do not significantly increase.

---

# 3. Geometry-Based Candidate Selection

## Problem

The current candidate selection uses integer string ID radius:

```python
for sid in range(event.touched_string_id - radius, event.touched_string_id + radius + 1):
```

This fails when:

- the touched string ID is wrong
- string IDs are non-contiguous
- detected string ordering is inconsistent
- a nearby physical string has a non-neighboring ID
- the true vibrating string is outside the fixed radius

## Goal

Select candidate strings by geometric distance from the touch/contact point, not only by integer ID.

## Files to Modify

- `windows.py`
- possibly `strings.py`

## New Config

```yaml
video_vibration:
  candidates:
    geometry_enabled: true
    geometry_top_k: 5
    geometry_max_distance_px: 35.0
    always_include_touched_id: true
    include_id_radius_fallback: true
```

## Implementation Details

### 1. Add nearest-geometry candidate function

```python
def select_event_candidates_by_geometry(
    *,
    event: TouchEvent,
    strings_by_id: dict[int, StringGeometry],
    top_k: int,
    max_distance_px: float,
    always_include_touched_id: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    ...
```

For each string:

1. Compute closest point from `event.contact_x/contact_y` to string geometry.
2. Store distance.
3. Sort strings by distance.
4. Keep strings with distance <= `max_distance_px`.
5. Limit to `top_k`.
6. Always include touched ID if configured and present.

### 2. Combine with existing ID-radius candidates

If `include_id_radius_fallback` is true:

```python
candidate_ids = sorted(set(geometry_candidates + id_radius_candidates))
```

### 3. Add debug data

```json
{
  "candidate_method": "geometry_plus_id_radius",
  "geometry_candidates": [
    {"string_id": 8, "distance_px": 2.1},
    {"string_id": 9, "distance_px": 7.4}
  ],
  "id_radius_candidates": [7, 8, 9],
  "final_candidates": [7, 8, 9]
}
```

## Acceptance Criteria

- Candidate coverage improves.
- Events with wrong touched ID still include the correct nearby string.
- False negatives caused by missing candidates decrease.
- No crash if touched string ID is missing.

---

# 4. Robust Z-Score Normalization

## Problem

The current rule uses mean/std normalization:

```python
denom_std = max(float(baseline_std), float(MIN_BASELINE_STD_FOR_Z))
z_action = (action - baseline_mean) / denom_std
```

This can be unstable when baseline variance is tiny. The arbitrary floor `0.05` may still amplify tiny motion and produce false positives.

## Goal

Add robust normalization using median and MAD or percentile-based scaling, while keeping existing z-score behavior available.

## Files to Modify

- `rules.py`

## New Config

```yaml
video_vibration:
  rules:
    normalization_mode: "robust_mad"  # options: zscore, robust_mad, percentile
    min_scale: 0.05
    mad_scale_factor: 1.4826
    percentile_low: 25
    percentile_high: 75
```

## Implementation Details

### 1. Add robust scale helper

```python
def robust_center_scale(
    values: np.ndarray,
    *,
    mode: str,
    min_scale: float,
    mad_scale_factor: float = 1.4826,
    percentile_low: float = 25,
    percentile_high: float = 75,
) -> tuple[float, float, dict[str, float]]:
    ...
```

For `robust_mad`:

```python
center = median(baseline)
mad = median(abs(baseline - center))
scale = max(mad * 1.4826, min_scale)
```

For `percentile`:

```python
center = median(baseline)
q_low = percentile(baseline, percentile_low)
q_high = percentile(baseline, percentile_high)
scale = max(q_high - q_low, min_scale)
```

For `zscore`, preserve existing behavior.

### 2. Update `CandidateDecision`

Add fields:

```python
normalization_mode: str
baseline_center: float
baseline_scale: float
baseline_median: float
baseline_mad: float
```

### 3. Keep old fields

Do not remove:

- `baseline_mean`
- `baseline_std`
- `z_action`

Existing downstream code may depend on them.

## Acceptance Criteria

- Old behavior preserved with `normalization_mode: zscore`.
- False positives decrease in low-baseline-variance cases.
- Debug output shows normalization mode and scale.
- No division-by-zero or extreme score explosions.

---

# 5. Baseline Gap and Less Contaminated Baseline Window

## Problem

The current baseline window is immediately before the touch frame:

```python
baseline_frames = [f for f in range(f0 - baseline_len, f0)]
```

This may include pre-strike finger movement, string pulling, or hand motion. That contaminates the baseline and weakens the action-vs-baseline contrast.

## Goal

Allow a configurable gap between the baseline window and touch frame.

## Files to Modify

- `windows.py`
- config mapping code if necessary

## New Config

```yaml
video_vibration:
  windows:
    baseline_gap_frames: 2
```

## Implementation Details

### 1. Extend `build_frame_windows()`

Add parameter:

```python
baseline_gap_frames: int = 0
```

Update baseline construction:

```python
baseline_end = f0 - baseline_gap_frames
baseline_start = baseline_end - baseline_len
baseline_frames = list(range(baseline_start, baseline_end))
```

Current behavior is preserved when `baseline_gap_frames = 0`.

### 2. Update callers

Update:

- `process_single_event_windows()`
- `process_touch_events_windows()`

Pass `baseline_gap_frames`.

### 3. Debug output

```json
{
  "baseline_gap_frames": 2,
  "baseline_frames": [90, 91, 92, 93],
  "baseline_end_before_f0": 2
}
```

## Acceptance Criteria

- Baseline mean is less affected by finger-preparation motion.
- Weak strike recall improves or stays equal.
- Config value `0` exactly reproduces old behavior.

---

# 6. ROI Border Validity and Border Mode Handling

## Problem

Current ROI extraction uses:

```python
borderMode=cv2.BORDER_REPLICATE
```

If an ROI extends outside image bounds, OpenCV replicates edge pixels. This can fabricate artificial motion and corrupt optical-flow scores.

## Goal

Detect out-of-frame ROIs and allow either rejection or safer border handling.

## Files to Modify

- `roi.py`
- `windows.py`

## New Config

```yaml
video_vibration:
  roi:
    border_mode: "replicate"  # options: replicate, constant, reject
    constant_border_value: 0
    min_inside_fraction: 0.95
```

## Implementation Details

### 1. Add ROI bounds check

Create:

```python
def roi_inside_fraction(
    corners: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> float:
    ...
```

Simple first implementation:

- Count how many corners are inside frame.
- Or compute polygon intersection if practical.
- Return approximate fraction.

### 2. Update `extract_rotated_roi()`

Add parameters:

```python
border_mode: str = "replicate"
constant_border_value: int = 0
reject_if_out_of_frame: bool = False
min_inside_fraction: float = 0.95
```

If `border_mode == "reject"` and inside fraction is too low:

```python
return None, mat_img_to_roi, debug
```

If `border_mode == "constant"`:

```python
borderMode=cv2.BORDER_CONSTANT
borderValue=constant_border_value
```

If `border_mode == "replicate"`:

keep old behavior.

### 3. Update callers

`_compute_candidate_scores()` must handle rejected ROIs:

```python
if roi is None:
    score_by_frame[frame_idx] = 0.0
    rejected_roi_frames.append(frame_idx)
    continue
```

### 4. Debug output

```json
{
  "border_mode": "reject",
  "inside_fraction": 0.87,
  "roi_rejected": true,
  "rejected_reason": "roi_out_of_frame"
}
```

## Acceptance Criteria

- Edge ROIs are logged.
- Rejected ROI frames do not crash scoring.
- False positives near frame edge decrease.
- Old behavior preserved with `border_mode: replicate`.

---

# 7. Stronger Hand/Finger Masking

## Problem

Current hand masking appears to use only the event finger point in `_compute_candidate_scores()`:

```python
finger_point_img=_event_finger_point(event)
```

This may not mask enough hand/finger motion. Finger motion inside the ROI can be interpreted as string vibration.

## Goal

Allow masking of larger hand/finger regions, including optional hand bounding boxes or polygons if available.

## Files to Modify

- `windows.py`
- `mask.py`
- touch event schema conversion code if hand bbox exists elsewhere

## New Config

```yaml
video_vibration:
  hand_mask:
    enabled: true
    mode: "finger_point_plus_contact_band"  # options: finger_point, finger_point_plus_contact_band, hand_bbox
    expand_px: 12.0
    contact_band_exclusion_px: 10.0
    mask_contact_region: true
```

## Implementation Details

### 1. Extend `TouchEvent`

Add optional fields if available:

```python
hand_bbox_x1: float | None = None
hand_bbox_y1: float | None = None
hand_bbox_x2: float | None = None
hand_bbox_y2: float | None = None
```

If CSV/JSON does not include them, keep `None`.

### 2. Update mask creation

Enhance `create_roi_mask()` to support:

- finger point mask
- contact point mask
- hand bbox mask
- optional exclusion band around the contact point

The mask should exclude high-motion hand/finger pixels while preserving the string center band where possible.

### 3. Add debug visualization support

For each ROI, optionally save:

- raw ROI
- mask image
- masked ROI
- optical flow magnitude image

Suggested filenames:

```text
debug/event_0123_string_08_frame_0456_roi.png
debug/event_0123_string_08_frame_0456_mask.png
debug/event_0123_string_08_frame_0456_flow.png
```

### 4. Debug output

```json
{
  "hand_mask_enabled": true,
  "hand_mask_mode": "finger_point_plus_contact_band",
  "hand_mask_expand_px": 12.0,
  "valid_frac_mean": 0.72
}
```

## Acceptance Criteria

- False positives from finger/hand movement decrease.
- `valid_frac_mean` remains high enough for reliable scoring.
- Debug overlays confirm hand motion is excluded.
- No major recall loss from over-masking the actual string.

---

# 8. Non-Consecutive Frame Transition Handling

## Problem

Current optical flow is only computed for consecutive frames:

```python
if prev_roi is None or prev_frame_idx is None or (frame_idx - prev_frame_idx) != 1:
    score_by_frame[frame_idx] = 0.0
```

If frames are missing or the pipeline samples non-consecutive frames, valid motion can be lost.

## Goal

Allow optical flow over small frame gaps or explicitly log/reject gaps.

## Files to Modify

- `windows.py`

## New Config

```yaml
video_vibration:
  frame_transitions:
    allow_small_gaps: true
    max_gap_frames: 2
    normalize_by_gap: true
```

## Implementation Details

### 1. Modify transition check

Current behavior:

```python
(frame_idx - prev_frame_idx) != 1
```

New behavior:

```python
gap = frame_idx - prev_frame_idx

if gap == 1:
    compute flow normally
elif allow_small_gaps and gap <= max_gap_frames:
    compute flow and optionally divide score by gap
else:
    score = 0.0
```

### 2. Normalize score by gap

If computing flow over a 2-frame gap:

```python
score = raw_score / gap
```

This avoids overestimating motion because more time passed.

### 3. Debug output

```json
{
  "transitions_scored": 8,
  "non_consecutive_transitions_scored": 2,
  "transitions_zeroed_due_to_gap": 1,
  "max_gap_seen": 3
}
```

## Acceptance Criteria

- Videos with occasional dropped frames lose fewer scores.
- Missing-frame diagnostics become visible.
- Accuracy improves or remains stable on normal videos.
- Old behavior preserved when `allow_small_gaps: false`.

---

# 9. Add Absolute Motion/Energy Requirements

## Problem

Current decision logic is based on relative z-score impulse:

```python
vibrates = peak >= thr_peak and duration >= thr_duration_frames and impulse >= thr_impulse
candidate_score = impulse
```

A tiny absolute motion can produce a large relative z-score if the baseline is extremely quiet.

## Goal

Require minimum absolute action motion in addition to relative z-score evidence.

## Files to Modify

- `rules.py`
- `windows.py` if extra action statistics need to be passed

## New Config

```yaml
video_vibration:
  rules:
    require_absolute_motion: true
    min_action_mean: 0.02
    min_action_max: 0.05
    max_baseline_mean: null
```

## Implementation Details

### 1. Compute raw action statistics

Inside `evaluate_candidate_scores()`:

```python
action_mean = float(np.mean(action)) if action.size else 0.0
action_max = float(np.max(action)) if action.size else 0.0
baseline_max = float(np.max(baseline)) if baseline.size else 0.0
```

### 2. Extend `CandidateDecision`

Add:

```python
action_mean: float
action_max: float
baseline_max: float
absolute_motion_pass: bool
```

### 3. Update decision rule

```python
relative_pass = (
    peak >= thr_peak
    and duration >= thr_duration_frames
    and impulse >= thr_impulse
)

absolute_pass = True
if require_absolute_motion:
    absolute_pass = (
        action_mean >= min_action_mean
        or action_max >= min_action_max
    )

baseline_pass = True
if max_baseline_mean is not None:
    baseline_pass = baseline_mean <= max_baseline_mean

vibrates = relative_pass and absolute_pass and baseline_pass
```

Use `or` between mean and max initially to avoid over-rejecting brief real strikes.

### 4. Debug output

```json
{
  "relative_pass": true,
  "absolute_motion_pass": false,
  "baseline_pass": true,
  "action_mean": 0.008,
  "action_max": 0.019,
  "reject_reason": "low_absolute_motion"
}
```

## Acceptance Criteria

- Spike-only low-energy false positives decrease.
- Real visible string vibrations still pass.
- Debug output clearly explains absolute-threshold rejections.

---

# 10. Candidate Selection Fallback for Missing/Unreliable String IDs

## Problem

The current candidate logic assumes:

- `event.touched_string_id` exists
- string IDs are sequential
- string IDs are stable and correct

If the touched string is missing, current behavior returns no candidates and adds reason `"touched_string_missing"`.

## Goal

When the touched string is missing or unreliable, fallback to nearest strings by geometry.

## Files to Modify

- `windows.py`

## New Config

```yaml
video_vibration:
  candidates:
    missing_touched_id_fallback: "nearest_geometry"  # options: none, nearest_geometry
    fallback_top_k: 3
    fallback_max_distance_px: 40.0
    log_string_id_inconsistency: true
```

## Implementation Details

### 1. Detect missing touched ID

Existing code:

```python
touched_geom = strings_by_id.get(event.touched_string_id)
if touched_geom is None:
    return [], {"reason": "touched_string_missing"}
```

Replace or extend:

```python
if touched_geom is None and missing_touched_id_fallback == "nearest_geometry":
    return nearest_geometry_candidates, debug
```

### 2. Detect suspicious ID mapping

Add debug warnings when:

- IDs are non-contiguous
- touched ID is outside min/max
- nearest geometric string differs from touched ID by more than configured threshold
- candidate set is empty

Example debug:

```json
{
  "string_id_warning": true,
  "reason": "touched_id_missing_geometry_fallback_used",
  "touched_string_id": 12,
  "nearest_geometry_candidates": [10, 11, 13]
}
```

### 3. Reuse implementation from Task 3

Do not duplicate candidate-by-geometry code. Use the same helper function.

## Acceptance Criteria

- Events with missing touched IDs still produce candidate strings.
- Candidate coverage improves in incomplete string-detection cases.
- Debug logs make ID reliability issues obvious.

---

# Recommended Implementation Order

Implement in this order:

1. Add debug instrumentation and validation metrics first.
2. Geometry-based candidate selection.
3. Dynamic action-window search.
4. Baseline gap.
5. Absolute motion threshold.
6. Robust normalization.
7. Adaptive ROI height.
8. ROI border validity handling.
9. Stronger hand/finger masking.
10. Non-consecutive frame transition handling.

Reason:

- Candidate selection and action timing are likely highest impact.
- Baseline gap and absolute threshold are simple and reduce false positives.
- ROI/mask changes are powerful but require visual debugging to avoid overfitting.
- Non-consecutive frames matter only if videos actually have frame gaps.

---

# Suggested Config Block

Add this to `configs/config.yaml` or equivalent:

```yaml
video_vibration:
  roi:
    adaptive_enabled: true
    adaptive_height_ratio: 0.45
    min_roi_h: 5
    max_roi_h: 18
    min_neighbor_distance_px: 4.0
    reject_if_neighbor_overlap: false
    border_mode: "replicate"
    constant_border_value: 0
    min_inside_fraction: 0.95

  windows:
    baseline_len: 8
    action_len: 8
    action_start_frame_offset: 0
    baseline_gap_frames: 2
    dynamic_action_enabled: true
    dynamic_offsets: [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    dynamic_select_metric: "max_impulse"

  candidates:
    geometry_enabled: true
    geometry_top_k: 5
    geometry_max_distance_px: 35.0
    always_include_touched_id: true
    include_id_radius_fallback: true
    missing_touched_id_fallback: "nearest_geometry"
    fallback_top_k: 3
    fallback_max_distance_px: 40.0
    log_string_id_inconsistency: true

  hand_mask:
    enabled: true
    mode: "finger_point_plus_contact_band"
    expand_px: 12.0
    contact_band_exclusion_px: 10.0
    mask_contact_region: true

  frame_transitions:
    allow_small_gaps: true
    max_gap_frames: 2
    normalize_by_gap: true

  rules:
    normalization_mode: "robust_mad"
    min_scale: 0.05
    mad_scale_factor: 1.4826
    percentile_low: 25
    percentile_high: 75

    require_absolute_motion: true
    min_action_mean: 0.02
    min_action_max: 0.05
    max_baseline_mean: null
```

---

# Required Debug Artifacts

For each processed event, optionally save:

```text
debug/
  event_000123/
    summary.json
    frame_000456_full_overlay.png
    string_08_roi.png
    string_08_mask.png
    string_08_flow_mag.png
    string_08_score_plot.json
```

`summary.json` should include:

```json
{
  "event": {
    "row_index": 123,
    "timestamp_sec": 3.72,
    "f0": 112,
    "touched_string_id": 8,
    "contact_x": 431.2,
    "contact_y": 289.8
  },
  "candidates": {
    "method": "geometry_plus_id_radius",
    "final_ids": [7, 8, 9],
    "distances_px": {
      "7": 12.5,
      "8": 2.1,
      "9": 9.7
    }
  },
  "windows": {
    "baseline_gap_frames": 2,
    "baseline_frames": [102, 103, 104, 105],
    "dynamic_action_enabled": true,
    "tested_offsets": [-2, -1, 0, 1, 2, 3, 4, 5, 6],
    "selected_offset": 4,
    "selected_action_frames": [116, 117, 118, 119]
  },
  "winner": {
    "candidate_string_id": 8,
    "vibrates": true,
    "peak": 5.4,
    "duration": 3,
    "impulse": 7.9,
    "action_mean": 0.08,
    "action_max": 0.14
  },
  "reject_reasons": []
}
```

---

# Test Plan

## Unit Tests

Add tests for:

1. `build_frame_windows()` with and without baseline gap.
2. `build_dynamic_action_windows()`.
3. Geometry candidate selection with:
   - contiguous IDs
   - non-contiguous IDs
   - missing touched ID
   - wrong touched ID
4. Robust normalization on:
   - empty baseline
   - zero-variance baseline
   - noisy baseline
5. Absolute motion threshold logic.
6. ROI border rejection logic.
7. Non-consecutive frame transition scoring.

## Integration Tests

Run the full pipeline on a fixed set of videos and compare:

```text
before_metrics.json
after_metrics.json
```

Metrics:

```json
{
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "false_positives": 0,
  "false_negatives": 0,
  "candidate_coverage": 0.0,
  "avg_peak_relative_to_f0": 0.0,
  "roi_out_of_frame_rate": 0.0,
  "missing_transition_rate": 0.0
}
```

## Regression Requirement

Do not accept a change unless at least one of these is true:

- F1 improves.
- Precision improves with recall decrease below 3%.
- Recall improves with precision decrease below 3%.
- Debug output reveals a previously invisible failure mode.

---

# Final Agent Instruction

Implement these 10 improvements as configurable, backward-compatible changes. Do not rewrite the pipeline architecture. Prefer small, testable patches.

For each completed task, provide:

1. Files changed.
2. Config options added.
3. Before/after behavior.
4. Debug fields added.
5. Unit tests added.
6. Expected accuracy impact.
7. Known risks or tradeoffs.

The main objective is to make the vibration stage more robust to timing error, ROI contamination, string-ID errors, and unstable heuristic scoring.
