# Detection Code Guide

This document describes the current UAV detection runtime for another AI or engineer to understand and modify the code safely.

Current version: `frame-diff-20260614`

Main files:

- `main.py`: board-side multi-camera detection runtime.
- `yololib.py`: RKNN YOLOv5 wrapper and post-processing.
- `comms.py`: UDP video/data sender.
- `README.md`: short project summary.
- `CHANGELOG.md`: version-level change notes.

## Design Goal

The runtime detects small UAV targets from 5 fixed cameras on an RK3588 board. The current production path is:

```text
camera frame
-> grayscale resize to 1920x1080
-> frame difference
-> static background mask
-> morphology
-> motion contour filtering
-> 640x640 ROI crop/fusion
-> RKNN YOLO inference
-> trajectory filtering/confirmation
-> display and UDP target output
```

The older sky-gray absolute gray-threshold candidate path was removed from `main.py`. Current target confirmation depends on YOLO detections plus trajectory rules.

## Runtime Entry

The program starts from `if __name__ == "__main__"` in `main.py`.

Startup sequence:

1. Create OpenCV display windows if `SHOW_INDIVIDUAL_WINDOWS = True`.
2. Print `ALGORITHM_VERSION` and `ALGORITHM_NOTE`.
3. Start one global `inference_worker()` thread.
4. Start one `capture_job(cam_idx)` thread per camera.
5. Start a timer thread that enables video sending after `INIT_TIME`.
6. Main thread only refreshes OpenCV display windows.

The detection work happens inside each `capture_job`, while RKNN inference is centralized in `inference_worker`.

## Global Configuration

Important top-level constants in `main.py`:

```text
N_CAM = 5
CAPTURE_W, CAPTURE_H = 2560, 1440
PROCESS_EVERY_N_FRAMES = 3
DIFF_W, DIFF_H = 1920, 1080
CROP_SIZE = 640
MODEL_PATH = '../model/yolov5s.rknn'
CONF_THRESH = 0.45
```

Board/layer settings:

```text
BOARD_ROW_IDX = 1
LOW_LAYER_ROW_IDX = 0
HIGH_LAYER_MODE = BOARD_ROW_IDX != LOW_LAYER_ROW_IDX
```

With the current defaults, `HIGH_LAYER_MODE = True`.

Frame-difference path:

```text
ENABLE_FRAME_DIFF_ROIS = True
ENABLE_STATIC_BG_MASK = True
ENABLE_LOW_LAYER_STATIC_BG_MASK = True
DIFF_THRESH = 8
MIN_DIFF_AREA = 50
MIN_LOCAL_DIFF_MEAN = 16.0
ENABLE_LCM_FILTER = True
```

ROI limits:

```text
MAX_ROIS_PER_FRAME = 6 if HIGH_LAYER_MODE else 4
MAX_TRACK_ROIS_PER_FRAME = 3
INF_QUEUE_SIZE = MAX_ROIS_PER_FRAME + MAX_TRACK_ROIS_PER_FRAME + 2
```

Display/send:

```text
SHOW_INDIVIDUAL_WINDOWS = True
ENABLE_FUSION_DISPLAY = True
VIDEO_SEND_EVERY_N_FRAMES = 3
```

## Thread And Queue Model

Queues:

```text
inf_queues[cam_idx]  # ROI tasks to RKNN worker
res_queues[cam_idx]  # RKNN results back to camera thread
display_queues[cam_idx]  # latest 640x360 display frame
```

`put_latest(q, item)` is non-blocking. If a queue is full, it drops the oldest item and inserts the newest one. This prevents camera threads from blocking indefinitely when RKNN or display is slower than capture.

There is one global RKNN thread:

```text
inference_worker()
```

It loops over all camera queues, takes the latest ROI task, runs `YoloRKNN.infer(roi)`, and puts mapped results into `res_queues[cam_idx]`.

## Camera Processing Flow

Each `capture_job(cam_idx)` does this:

1. Resolve the V4L2 camera node with `get_camera_node`.
2. Open camera using MJPG at `2560x1440`, `15 FPS`.
3. Create one `TrajectoryFilter` instance for that camera.
4. Initialize static background samples.
5. Loop until `stop_event`.

Per frame:

```text
read frame
ensure BGR frame
increment f_idx
if f_idx is not scheduled for detection:
    only drain results/display previous boxes
else:
    convert to grayscale
    resize grayscale to 1920x1080
    if static background not ready:
        collect background sample and continue
    build frame-difference mask
    generate motion ROIs
    add track-seeded ROIs if allowed
    submit up to MAX_ROIS_PER_FRAME ROI crops to RKNN queue
drain RKNN results
map ROI detections back to full-frame coordinates
update tracker
draw display boxes
send confirmed target data
```

## Static Background Model

During startup, each camera collects `LOW_BG_SAMPLE_FRAMES = 60` processed grayscale frames.

`build_static_bg_model(samples)` computes:

- `bg`: per-pixel median background.
- `tol`: per-pixel tolerance based on standard deviation.

Formula:

```text
tol = LOW_BG_ABS_DELTA + LOW_BG_STD_MULT * std
LOW_BG_ABS_DELTA = 14
LOW_BG_STD_MULT = 3.0
```

`apply_static_bg_mask(mask, gray, bg, tol)` keeps only pixels that differ from the learned background:

```text
bg_diff = abs(gray - bg)
changed = bg_diff > tol
mask = mask AND changed
```

Purpose:

- Suppress stable buildings/trees/texture.
- Allow newly changed pixels to pass into frame-difference ROI filtering.

Limitation:

- If the camera shakes, static edges can still enter the changed mask.
- Moving leaves, flashing lights, clouds, or real illumination changes are still dynamic.

## Frame Difference ROI Generation

The main motion path begins here:

```python
aligned_t1 = align_previous_gray(frame_t1, gray_small)
diff_img = cv2.absdiff(aligned_t1, gray_small)
_, mask = cv2.threshold(diff_img, DIFF_THRESH, 255, cv2.THRESH_BINARY)
mask = apply_static_bg_mask(mask, gray_small, bg_gray, bg_tol)
mask = cleanup_motion_mask(mask)
diff_rois = motion_rois_from_mask(...)
```

Current global motion compensation is disabled:

```text
ENABLE_GLOBAL_MOTION_COMP = False
```

So `align_previous_gray` returns the previous frame unchanged.

`cleanup_motion_mask` applies:

```text
erode 1 iteration, 2x2 rect
dilate 1 iteration, 3x3 rect
close 1 iteration, 3x3 ellipse
```

`motion_rois_from_mask` then:

1. Finds contours.
2. Rejects vertical strip-like boxes.
3. Counts active pixels in contour mask.
4. Rejects small contours below `MIN_DIFF_AREA`.
5. Measures local diff mean.
6. Optionally computes local contrast measure.
7. Optionally suppresses gray noise in low-layer mode.
8. Applies far/near compactness and size rules.
9. Converts small-frame coordinates back to full-frame coordinates.
10. Creates normal 640-centered ROI or tight padded ROI depending on mode.

### Far/near object filters

Far tiny target:

```text
area <= MAX_DIFF_AREA
w <= MAX_DIFF_BOX_W
h <= MAX_DIFF_BOX_H
compactness >= FAR_MIN_COMPACTNESS
texture <= EDGE_DENSITY_THRESH
```

Near compact target:

```text
area <= NEAR_MAX_DIFF_AREA
w <= NEAR_MAX_DIFF_BOX_W
h <= NEAR_MAX_DIFF_BOX_H
compactness >= NEAR_MIN_COMPACTNESS
texture <= NEAR_EDGE_DENSITY_THRESH
```

Current values:

```text
MAX_DIFF_AREA = 600
MAX_DIFF_BOX_W = 120
MAX_DIFF_BOX_H = 120
NEAR_MAX_DIFF_AREA = 6000
NEAR_MAX_DIFF_BOX_W = 320
NEAR_MAX_DIFF_BOX_H = 320
FAR_MIN_COMPACTNESS = 0.18
NEAR_MIN_COMPACTNESS = 0.06
```

## ROI Fusion

YOLO receives `640x640` ROI images.

If `ENABLE_FUSION_INFERENCE = True`, `make_fused_roi` creates a 3-channel input:

```text
channel 0 = grayscale ROI
channel 1 = grayscale ROI
channel 2 = resized motion/fusion mask for the ROI
```

So the model sees both raw grayscale intensity and motion evidence.

Display uses `make_fused_display_preview`, which builds a lightweight `640x360` preview:

```text
channel 0 = grayscale preview
channel 1 = grayscale preview
channel 2 = motion/fusion mask preview
```

This display image is only for visualization, not for detection.

## RKNN YOLO Inference

`yololib.py` wraps RKNNLite:

```python
yolo = YoloRKNN(MODEL_PATH, (640, 640), CONF_THRESH, 0.45)
res = yolo.infer(roi)
```

Input:

- BGR or fused BGR-like ROI.
- Resized to 640x640 if needed.

Output per detection:

```text
[x1, y1, x2, y2, score, class_id]
```

Coordinates are in ROI space. The camera thread maps them back to full-frame coordinates using the ROI origin and ROI scaling.

Debug environment variables supported by `yololib.py`:

```text
YOLO_RKNN_DEBUG=1
YOLO_RKNN_KEEP_BGR=1
YOLO_RKNN_FLOAT_INPUT=1
```

## Tracking And Confirmation

Each camera has one `TrajectoryFilter`.

Track state fields:

```text
id
box
cx, cy
w, h
vx, vy
hits
misses
last_frame
score
yolo_hits
history
frame_history
hit_history
template
```

Matching score uses:

```text
distance score
YOLO confidence score
template similarity score
motion consistency score
```

Weighted score:

```text
0.42 * distance_score
+ 0.23 * yolo_score
+ 0.17 * template_score
+ 0.18 * motion_score
```

A detection matches a track only if:

```text
distance <= dynamic_gate
match_score >= TRACKER_MATCH_SCORE
```

Current `TRACKER_MATCH_SCORE = 0.25`.

### Dynamic gate

`_dynamic_gate` grows with:

- current track velocity,
- allowed physical target speed,
- number of misses.

It is clamped by:

```text
TRACKER_MAX_DIST = 100
TRACKER_MAX_GATE = 320
TRACKER_MAX_SPEED_MPS = 10.0
```

### Confirmed green-box logic

Basic confirmation has two paths.

Fast YOLO confirmation:

```text
yolo_hits >= 3
recent_hits >= 3
misses <= 0
score >= 0.40
```

Trajectory confirmation:

```text
hits >= 4
recent_hits >= 3
misses <= 2
score >= 0.50
trajectory_score >= 0.55
```

Because `ENABLE_REFERENCE_TRAJ_FILTER = HIGH_LAYER_MODE`, high-layer mode also requires a reference trajectory rule:

```text
label == target_like
traj_rule_score >= 0.58
```

Reference trajectory features include:

- duration,
- valid hit ratio,
- missed ratio,
- path length,
- net displacement,
- straightness,
- speed coefficient of variation,
- heading standard deviation,
- line-fit residual RMSE,
- lateral jitter ratio,
- stationary rejection.

Stationary rejection:

```text
duration >= 15
net displacement < 4 px
=> stationary_like, not confirmed
```

This is intended to prevent stable building/vegetation false positives from staying green forever.

## Track-Seeded Search ROIs

Confirmed tracks can seed future search ROIs via:

```python
tracker.get_yolo_seeded_search_rois(...)
```

Conditions:

```text
track age <= TRACK_SEARCH_MAX_AGE_FRAMES
misses <= YOLO_TRACK_MAX_SEARCH_MISSES
track is confirmed
net motion >= TRACK_SEARCH_MIN_NET_MOTION_PX
yolo_hits >= TRACK_SEARCH_MIN_YOLO_HITS
recent_hits >= TRACK_SEARCH_MIN_RECENT_HITS
score >= TRACK_SEARCH_MIN_SCORE
```

If a confirmed track is temporarily missed, the predicted box can be used for search when:

```text
0 < misses <= TRACK_SEARCH_PREDICT_MAX_MISSES
```

The seeded ROI is a normal `640x640` crop centered around the current or predicted track center.

In current high-layer mode:

```text
TRACK_SEARCH_MIN_YOLO_HITS = 2
TRACK_SEARCH_MIN_RECENT_HITS = 2
TRACK_SEARCH_MIN_SCORE = 0.32
TRACK_SEARCH_MIN_NET_MOTION_PX = 6.0
```

## Output

Confirmed tracks are sent to the gimbal endpoint:

```python
gimbal_data = [
    [x1, y1, x2, y2, rough_range_m],
    ...
]
data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")
```

Range is a rough monocular estimate:

```text
distance = target_real_width * fx_px / target_pixel_width
ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
range clamped to 20..2000 m
rounded to nearest 10 m
```

Video preview is sent by UDP through `VideoSender` when `VIDEO_STREAM_ALLOWED = True`.

Preview boxes:

```text
white ROI     = frame-difference ROI sent to YOLO
yellow TRKROI = track-seeded search ROI
red Raw       = raw YOLO detection
green TARGET  = confirmed target
```

## Important Removed/Disabled Paths

The following path is intentionally not present in `main.py` anymore:

```text
absolute gray threshold
-> isolated sky candidate
-> sky-gray weak boxes
-> tracker confirmation
```

Do not reintroduce it into `main.py` unless the runtime budget and false-positive strategy are redesigned.

Reference/debug scripts may still exist in the repository, for example:

- `make_gru_gray_threshold_debug_video.py`
- `make_gru_gray_threshold_two_target_video.py`
- `rk3588_gru_detect_lite.py`

These are analysis/debug tools, not part of the current board runtime.

## Common Modification Points

### Reduce lag

Try, in order:

1. Increase `PROCESS_EVERY_N_FRAMES` from `3` to `4` or `5`.
2. Reduce `MAX_ROIS_PER_FRAME`.
3. Disable `SHOW_INDIVIDUAL_WINDOWS`.
4. Disable `ENABLE_FUSION_DISPLAY`.
5. Consider reducing `DIFF_W, DIFF_H`.

### Reduce false positives from static structures

Tune:

```text
LOW_BG_SAMPLE_FRAMES
LOW_BG_ABS_DELTA
LOW_BG_STD_MULT
ENABLE_GLOBAL_MOTION_COMP
REF_TRAJ_STATIONARY_MAX_DISP
REF_TRAJ_MIN_SCORE
```

If camera shake is the cause, enabling and validating `ENABLE_GLOBAL_MOTION_COMP` is more important than only raising thresholds.

### Detect smaller/farther targets

Tune carefully:

```text
DIFF_THRESH
MIN_DIFF_AREA
MIN_LOCAL_DIFF_MEAN
FAR_MIN_COMPACTNESS
MAX_DIFF_AREA
MAX_DIFF_BOX_W/H
LCM_MIN_SCORE
LCM_MIN_RATIO
```

Lower thresholds improve sensitivity but increase noise ROIs and RKNN load.

### Make green box faster

Tune:

```text
YOLO_DIRECT_CONFIRM_HITS
YOLO_DIRECT_CONFIRM_RECENT_HITS
TRACKER_MIN_HITS
TRACKER_MIN_RECENT_HITS
TRACKER_CONFIRM_SCORE
REF_TRAJ_MIN_SCORE
```

Lowering confirmation thresholds improves response time but increases false positives.

## Minimal Pseudocode

```python
start inference_worker
for each camera:
    start capture_job

def capture_job(cam_idx):
    open camera
    create tracker
    while running:
        frame = cap.read()
        if detection_frame:
            gray = resize(to_gray(frame), 1920x1080)
            if background not ready:
                collect background
                continue

            diff = abs(prev_gray - gray)
            mask = threshold(diff, DIFF_THRESH)
            mask = mask AND static_background_changed(gray)
            mask = morphology(mask)
            rois = motion_rois_from_mask(mask)

            track_rois = tracker.get_yolo_seeded_search_rois()
            rois = merge/prioritize(rois + track_rois)

            for roi in rois[:MAX_ROIS_PER_FRAME]:
                roi_img = fused_gray_mask_roi(frame, mask, roi)
                inf_queue[cam].put_latest(roi_img, origin)

            prev_gray = gray

        detections = drain_rknn_results()
        full_boxes = map_roi_detections_to_full_frame(detections)
        tracker.update(full_boxes)
        confirmed = tracker.get_confirmed_tracks()
        send confirmed targets
        draw display frame
```

## Expected Invariants

These should remain true after modifications:

- `main.py` should compile with `python -m py_compile main.py`.
- `main.py` should not contain `SKY_GRAY`, `sky_gray`, `gray_hits`, or `skygray` in the current frame-diff version.
- Every ROI sent to YOLO should include its full-frame origin and source frame index.
- Old RKNN results should be discarded using `MAX_INFERENCE_RESULT_AGE_FRAMES`.
- Confirmed output should only use live confirmed tracks:

```python
live_confirmed_tracks = [t for t in confirmed_tracks if t.get("live", True)]
```

- The static background model must be built per camera, not shared globally.

## Current Known Limitations

- Global motion compensation is disabled, so camera shake can still create motion ROIs on building/tree edges.
- One global RKNN worker services all camera streams; too many ROIs can create inference lag.
- Static background masking suppresses stable structures but not dynamic vegetation, lights, clouds, or strong illumination changes.
- Rough range is approximate and should not be treated as final ranging data.
