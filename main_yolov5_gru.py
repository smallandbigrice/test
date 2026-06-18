#!/usr/bin/env python3
import cv2
import threading
import multiprocessing as mp
import queue
import time
import numpy as np
import subprocess
import math
from collections import deque
from yololib import YoloRKNN 
from comms import DataSender, VideoSender
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ==========================================
# 1
# ==========================================
ALGORITHM_VERSION = "rknn-yolov5-gru-cascade-20260618"
ALGORITHM_NOTE = "YOLOv5 + GRU cascade tracking and trajectory prediction with 1080p camera input."
N_CAM = 5                        
INIT_TIME = 5                    

TRACKER_MIN_HITS = 4            
TRACKER_MAX_DIST = 100           
TRACKER_MAX_GATE = 320           
TRACKER_RECENT_WINDOW = 18       
TRACKER_MIN_RECENT_HITS = 3      
TRACKER_CONFIRM_SCORE = 0.50     
TRACKER_MIN_TRAJ_SCORE = 0.55
TRACKER_MATCH_SCORE = 0.25
TRACKER_TEMPLATE_SIZE = 31       
TRACKER_MAX_SPEED_MPS = 10.0     
CONF_THRESH = 0.45               
YOLO_DIRECT_CONFIRM_HITS = 3
YOLO_DIRECT_CONFIRM_RECENT_HITS = 3
YOLO_DIRECT_CONFIRM_SCORE = 0.40
YOLO_DIRECT_CONFIRM_MAX_MISSES = 0
YOLO_TRACK_MAX_CONFIRMED_MISSES = 2
YOLO_TRACK_MAX_SEARCH_MISSES = 8
TRACK_SEARCH_PREDICT_MAX_MISSES = 2
MISS_VELOCITY_DECAY = 0.55
MAX_TRACK_ROIS_PER_FRAME = 3
SHOW_INDIVIDUAL_WINDOWS = True
VIDEO_SEND_EVERY_N_FRAMES = 3
MAX_DRAW_BOXES = 80
CAPTURE_W, CAPTURE_H = 1920, 1080

BOARD_ID = "BOARD_2" 
BOARD_ROW_IDX = 1 

DATA_TARGETS = {
    "gimbal": ("192.168.0.100", 8888)
}
VIDEO_TARGET_IP = "192.168.0.200"       
VIDEO_BASE_PORT = 9999                  

CROP_SIZE = 640      
LOW_LAYER_ROW_IDX = 0
HIGH_LAYER_MODE = BOARD_ROW_IDX != LOW_LAYER_ROW_IDX
ENABLE_TRAJECTORY_TRACKING = HIGH_LAYER_MODE
TRACK_SEARCH_MIN_YOLO_HITS = 2
TRACK_SEARCH_MIN_RECENT_HITS = 2
TRACK_SEARCH_MIN_SCORE = 0.32 if HIGH_LAYER_MODE else 0.40
TRACK_SEARCH_CONFIRMED_ONLY = True
TRACK_SEARCH_MIN_NET_MOTION_PX = 6.0 if HIGH_LAYER_MODE else 0.0
ENABLE_REFERENCE_TRAJ_FILTER = HIGH_LAYER_MODE
REF_TRAJ_MIN_SCORE = 0.58
REF_TRAJ_MIN_DURATION = 8
REF_TRAJ_MIN_VALID_RATIO = 0.55
REF_TRAJ_MAX_MISSED_RATIO = 0.45
REF_TRAJ_EARLY_STRICT_DURATION = 10
REF_TRAJ_EARLY_MIN_VALID_RATIO = 0.65
REF_TRAJ_MIN_STRAIGHTNESS = 0.30 if HIGH_LAYER_MODE else 0.45
REF_TRAJ_MAX_CV_SPEED = 1.25
REF_TRAJ_MAX_HEADING_STD = 95.0
REF_TRAJ_MAX_LATERAL_JITTER_RATIO = 0.90
REF_TRAJ_MAX_CV_RESIDUAL_RMSE = 80.0
REF_TRAJ_REJECT_STATIONARY = True
REF_TRAJ_STATIONARY_MIN_DURATION = 15
REF_TRAJ_STATIONARY_MAX_DISP = 4.0
ENABLE_STATIC_BG_MASK = True
ENABLE_LOW_LAYER_STATIC_BG_MASK = ENABLE_STATIC_BG_MASK
ENABLE_DYNAMIC_BG_MODEL = True
DYNAMIC_BG_SECONDS = 60.0
DYNAMIC_BG_SAMPLE_INTERVAL = 0.5
DYNAMIC_BG_START_DELAY_SECONDS = 0.0
DYNAMIC_BG_SERIAL_GAP_SECONDS = 0.0
DYNAMIC_BG_ABS_DELTA = 20
DYNAMIC_BG_STD_MULT = 5.0
ENABLE_TIGHT_MOTION_ROI = not HIGH_LAYER_MODE
TIGHT_MOTION_ROI_PAD = 32
TIGHT_MOTION_ROI_FILL = 114
ENABLE_FLICKER_SUPPRESSOR = not HIGH_LAYER_MODE
FLICKER_CELL_PX = 48
FLICKER_WINDOW_FRAMES = 120
FLICKER_COOLDOWN_FRAMES = 240
FLICKER_MIN_HITS = 4
FLICKER_STATIONARY_PX = 10.0
FLICKER_MAX_AREA = 1400
FLICKER_MAX_BOX = 180
FLICKER_MIN_BRIGHT_RATIO = 0.35
FLICKER_MIN_BRIGHT_DELTA = 10.0
PROCESS_EVERY_N_FRAMES = 3
MAX_INFERENCE_RESULT_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 2
TRACK_LIVE_MAX_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 2
TRACK_SEARCH_MAX_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 3
LOW_BG_SAMPLE_FRAMES = 60
LOW_BG_ABS_DELTA = 14
LOW_BG_STD_MULT = 3.0
ENABLE_FRAME_DIFF_ROIS = True
DIFF_THRESH = 8
MIN_LOCAL_DIFF_MEAN = 10.0
ENABLE_GRAY_NOISE_SUPPRESSOR = False
ENABLE_FUSION_INFERENCE = True
ENABLE_FUSION_DISPLAY = True
FUSION_REQUIRE_TRACK_MOTION = True
FUSION_TRACK_MIN_PIXELS = 3
FUSION_TRACK_CENTER_SIZE = 120
ENABLE_VERTICAL_STRIP_FILTER = True
VERTICAL_STRIP_ASPECT_RATIO = 1.6
VERTICAL_STRIP_MIN_HEIGHT = 8
GRAY_TEXTURE_PAD = 28
MAX_LOCAL_GRAY_STD = 30.0
BRIGHT_SPOT_ABS_THRESH = 210
BRIGHT_SPOT_REL_THRESH = 32.0
BRIGHT_SPOT_MAX_AREA = 140
BRIGHT_SPOT_MIN_BG_STD = 10.0
ENABLE_LCM_FILTER = ENABLE_FRAME_DIFF_ROIS
LCM_BG_PAD = 18
LCM_MIN_SCORE = 1.8
LCM_MIN_RATIO = 1.25
LCM_REQUIRE_BOTH = False
LCM_SCORE_WEIGHT = 2.0
MOTION_ERODE_ITER = 0
MOTION_DILATE_ITER = 1
MOTION_CLOSE_ITER = 1
ENABLE_MOTION_OPENING = True
MOTION_OPEN_ITER = 0
MOTION_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MOTION_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
MOTION_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MOTION_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MAX_ROIS_PER_FRAME = 6 if HIGH_LAYER_MODE else 4
ENABLE_ROI_GRID_QUOTA = True
ROI_GRID_COLS = 4
ROI_GRID_ROWS = 3
ROI_GRID_MAX_PER_CELL = 1
INF_QUEUE_SIZE = MAX_ROIS_PER_FRAME + MAX_TRACK_ROIS_PER_FRAME + 2
RES_QUEUE_SIZE = MAX_ROIS_PER_FRAME + 4
DIFF_W, DIFF_H = 1920, 1080
MODEL_PATH = '../model/yolov5s.rknn'
ENABLE_GLOBAL_MOTION_COMP = False
ENABLE_EDGE_DENSITY_FILTER = False
STAB_W, STAB_H = 480, 270
STAB_MIN_RESPONSE = 0.08
STAB_MAX_SHIFT = 80
MAX_DIFF_AREA = 600
MAX_DIFF_BOX_W = 120
MAX_DIFF_BOX_H = 120
NEAR_MAX_DIFF_AREA = 6000
NEAR_MAX_DIFF_BOX_W = 320
NEAR_MAX_DIFF_BOX_H = 320
NEAR_MIN_COMPACTNESS = 0.06
NEAR_EDGE_DENSITY_THRESH = 0.30
EDGE_GRAD_THRESH = 45
EDGE_DENSITY_THRESH = 0.35 if HIGH_LAYER_MODE else 0.18
EDGE_DENSITY_PAD = 24
EDGE_DENSITY_SOFT_LIMIT = 0.50 if HIGH_LAYER_MODE else 0.38
EDGE_DENSITY_SCORE_PENALTY = 45.0 if HIGH_LAYER_MODE else 65.0
ENABLE_LOW_LAYER_EDGE_HARD_FILTER = False
ENABLE_SPATIAL_CHAOS_FILTER = False
CHAOS_CELL_PX = 220
CHAOS_LOCAL_ROI_LIMIT = 5
CHAOS_KEEP_PER_CELL = 1
CHAOS_SCORE_PENALTY = 40.0
ENABLE_ADAPTIVE_TRACK_CONFIRM = False
TRACK_CONFIRM_MIN_NET_MOTION_PX = 6.0 if HIGH_LAYER_MODE else 10.0
TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX = 10.0 if HIGH_LAYER_MODE else 16.0
TRACK_CONFIRM_MIN_STRAIGHTNESS = 0.24 if HIGH_LAYER_MODE else 0.35
TRACK_CONFIRM_RISK_MIN_STRAIGHTNESS = 0.45
TRACK_CONFIRM_RISK_THRESHOLD = 0.45
TRACK_CONFIRM_SMALL_BOX_MAX = 56
TRACK_CONFIRM_SMALL_MIN_HITS = 6
TRACK_CONFIRM_RISK_MIN_HITS = 7

ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
IMG_W, IMG_H = CAPTURE_W, CAPTURE_H
LOW_LAYER_MIN_DIFF_AREA = 50
MIN_DIFF_AREA = LOW_LAYER_MIN_DIFF_AREA if ENABLE_FRAME_DIFF_ROIS else 0
FAR_MIN_COMPACTNESS = 0.18

ROUGH_RANGE_MIN_M = 20
ROUGH_RANGE_MAX_M = 2000
ROUGH_RANGE_ROUND_M = 10
CAM_MAP = {i: f"00000000{i+1}" for i in range(5)}

# ==========================================
# 2
# ==========================================
data_sender = DataSender(DATA_TARGETS, BOARD_ID)
video_senders =[VideoSender(VIDEO_TARGET_IP, VIDEO_BASE_PORT) for i in range(N_CAM)]

inf_queues = [queue.Queue(maxsize=INF_QUEUE_SIZE) for _ in range(N_CAM)]
res_queues =[queue.Queue(maxsize=RES_QUEUE_SIZE) for _ in range(N_CAM)]
display_queues =[queue.Queue(maxsize=2) for _ in range(N_CAM)]

stop_event = threading.Event()
video_allowed_event = None
SYSTEM_START_TIME = time.time()
INIT_SIGNAL_SENT = False
VIDEO_STREAM_ALLOWED = False

def init_runtime_ipc():
    global inf_queues, res_queues, display_queues, stop_event, video_allowed_event
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("--> multiprocessing fork is unavailable; fallback to camera threads.", flush=True)
        video_allowed_event = None
        return None
    inf_queues = [ctx.Queue(maxsize=INF_QUEUE_SIZE) for _ in range(N_CAM)]
    res_queues = [ctx.Queue(maxsize=RES_QUEUE_SIZE) for _ in range(N_CAM)]
    display_queues = [ctx.Queue(maxsize=2) for _ in range(N_CAM)]
    stop_event = ctx.Event()
    video_allowed_event = ctx.Event()
    return ctx

def is_video_stream_allowed():
    if video_allowed_event is not None:
        try:
            return video_allowed_event.is_set()
        except Exception:
            pass
    return VIDEO_STREAM_ALLOWED

def put_latest(q, item):
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            return False

# ==========================================
# 3.
# ==========================================
def estimate_rough_range_m(box, frame_width=IMG_W):
    pixel_width = max(1.0, float(box[2] - box[0]))
    frame_width = max(1.0, float(frame_width))
    fx_px = (frame_width / 2.0) / math.tan(math.radians(CAM_H_FOV / 2.0))
    distance = (ROUGH_TARGET_WIDTH_M * fx_px) / pixel_width
    distance = max(ROUGH_RANGE_MIN_M, min(ROUGH_RANGE_MAX_M, distance))
    return int(round(distance / ROUGH_RANGE_ROUND_M) * ROUGH_RANGE_ROUND_M)

def merge_nearby_boxes(boxes, dist_thresh=120):
    if not boxes: return []
    merged = []
    for box in boxes:
        is_dup = False
        bcx, bcy = (box[0]+box[2])/2, (box[1]+box[3])/2
        for idx, m_box in enumerate(merged):
            mcx, mcy = (m_box[0]+m_box[2])/2, (m_box[1]+m_box[3])/2
            dist = math.sqrt((bcx-mcx)**2 + (bcy-mcy)**2)
            if dist < dist_thresh:
                box_score = float(box[4]) if len(box) > 4 else 1.0
                merged_score = float(m_box[4]) if len(m_box) > 4 else 1.0
                if box_score > merged_score:
                    merged[idx] = box
                is_dup = True; break
        if not is_dup: merged.append(box)
    return merged

def build_static_bg_model(samples):
    if not samples:
        return None, None
    stack = np.stack(samples, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    std = np.std(stack, axis=0)
    tol = np.clip(LOW_BG_ABS_DELTA + LOW_BG_STD_MULT * std, LOW_BG_ABS_DELTA, 255).astype(np.uint8)
    return bg, tol

def apply_static_bg_mask(mask, gray, bg, tol):
    if not ENABLE_LOW_LAYER_STATIC_BG_MASK or bg is None or tol is None:
        return mask
    bg_diff = cv2.absdiff(gray, bg)
    changed = (bg_diff > tol).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, changed)

def build_dynamic_bg_model(samples):
    if not samples:
        return None, None
    stack = np.stack(samples, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    std = np.std(stack, axis=0)
    tol = np.clip(DYNAMIC_BG_ABS_DELTA + DYNAMIC_BG_STD_MULT * std, DYNAMIC_BG_ABS_DELTA, 255).astype(np.uint8)
    return bg, tol

def cleanup_motion_mask(mask):
    if ENABLE_MOTION_OPENING and MOTION_OPEN_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MOTION_OPEN_KERNEL, iterations=MOTION_OPEN_ITER)
    if MOTION_ERODE_ITER > 0:
        mask = cv2.erode(mask, MOTION_ERODE_KERNEL, iterations=MOTION_ERODE_ITER)
    if MOTION_DILATE_ITER > 0:
        mask = cv2.dilate(mask, MOTION_DILATE_KERNEL, iterations=MOTION_DILATE_ITER)
    if MOTION_CLOSE_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MOTION_CLOSE_KERNEL, iterations=MOTION_CLOSE_ITER)
    return mask

def make_fused_roi(frame, mask_small, roi_box, full_w, full_h):
    x1, y1, x2, y2 = [int(v) for v in roi_box[:4]]
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return roi
    gray_roi = frame_to_gray(roi)
    if mask_small is None:
        mask_roi = np.zeros_like(gray_roi)
    else:
        sx = mask_small.shape[1] / float(max(1, full_w))
        sy = mask_small.shape[0] / float(max(1, full_h))
        mx1 = max(0, min(mask_small.shape[1] - 1, int(round(x1 * sx))))
        my1 = max(0, min(mask_small.shape[0] - 1, int(round(y1 * sy))))
        mx2 = max(0, min(mask_small.shape[1], int(round(x2 * sx))))
        my2 = max(0, min(mask_small.shape[0], int(round(y2 * sy))))
        mask_patch = mask_small[my1:my2, mx1:mx2]
        if mask_patch.size == 0:
            mask_roi = np.zeros_like(gray_roi)
        else:
            mask_roi = cv2.resize(mask_patch, (gray_roi.shape[1], gray_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray_roi, gray_roi, mask_roi])

def make_padded_roi_canvas(frame, mask_small, roi_info, full_w, full_h, fusion=True, crop_size=CROP_SIZE, fill_value=114):
    x1, y1, x2, y2 = [int(v) for v in roi_info[:4]]
    pad_x = int(roi_info[5]) if len(roi_info) > 5 else 0
    pad_y = int(roi_info[6]) if len(roi_info) > 6 else 0
    src_w = max(0, x2 - x1)
    src_h = max(0, y2 - y1)
    if src_w <= 0 or src_h <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    if fusion:
        patch = make_fused_roi(frame, mask_small, (x1, y1, x2, y2), full_w, full_h)
        canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        canvas[:, :, 0] = np.uint8(fill_value)
        canvas[:, :, 1] = np.uint8(fill_value)
    else:
        patch = frame[y1:y2, x1:x2]
        canvas = np.full((crop_size, crop_size, 3), np.uint8(fill_value), dtype=np.uint8)

    if patch.size == 0:
        return patch

    dst_x1 = max(0, min(crop_size, pad_x))
    dst_y1 = max(0, min(crop_size, pad_y))
    dst_x2 = min(crop_size, dst_x1 + patch.shape[1])
    dst_y2 = min(crop_size, dst_y1 + patch.shape[0])
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch[:dst_y2 - dst_y1, :dst_x2 - dst_x1]
    return canvas

def make_fused_display_frame(frame, mask_small, full_w, full_h):
    gray = frame_to_gray(frame)
    if mask_small is None:
        mask_full = np.zeros_like(gray)
    elif mask_small.shape[1] == gray.shape[1] and mask_small.shape[0] == gray.shape[0]:
        mask_full = mask_small
    else:
        mask_full = cv2.resize(mask_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray, gray, mask_full])

def make_fused_display_preview(frame, mask_small, out_w=640, out_h=360):
    gray = frame_to_gray(frame)
    gray_preview = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if mask_small is None:
        mask_preview = np.zeros((out_h, out_w), dtype=np.uint8)
    else:
        mask_preview = cv2.resize(mask_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray_preview, gray_preview, mask_preview])

def track_roi_has_motion(track_roi, mask_small, full_w, full_h, min_pixels=FUSION_TRACK_MIN_PIXELS, center_size=FUSION_TRACK_CENTER_SIZE):
    if mask_small is None:
        return True
    x1, y1, x2, y2 = [float(v) for v in track_roi[:4]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    sx = mask_small.shape[1] / float(max(1, full_w))
    sy = mask_small.shape[0] / float(max(1, full_h))
    half = float(center_size) * 0.5
    mx1 = int(max(0, round((cx - half) * sx)))
    my1 = int(max(0, round((cy - half) * sy)))
    mx2 = int(min(mask_small.shape[1], round((cx + half) * sx)))
    my2 = int(min(mask_small.shape[0], round((cy + half) * sy)))
    if mx2 <= mx1 or my2 <= my1:
        return False
    return cv2.countNonZero(mask_small[my1:my2, mx1:mx2]) >= int(min_pixels)

def limit_mask_to_rois(mask_small, rois, full_w, full_h):
    if mask_small is None or not rois:
        return None
    limited = np.zeros_like(mask_small)
    sx = mask_small.shape[1] / float(max(1, full_w))
    sy = mask_small.shape[0] / float(max(1, full_h))
    for roi in rois:
        x1, y1, x2, y2 = [float(v) for v in roi[:4]]
        mx1 = int(max(0, min(mask_small.shape[1], math.floor(x1 * sx))))
        my1 = int(max(0, min(mask_small.shape[0], math.floor(y1 * sy))))
        mx2 = int(max(0, min(mask_small.shape[1], math.ceil(x2 * sx))))
        my2 = int(max(0, min(mask_small.shape[0], math.ceil(y2 * sy))))
        if mx2 <= mx1 or my2 <= my1:
            continue
        limited[my1:my2, mx1:mx2] = mask_small[my1:my2, mx1:mx2]
    return limited

def boost_roi_score(roi, boost):
    values = list(roi)
    if len(values) < 5:
        values.append(0.0)
    values[4] = float(values[4]) + float(boost)
    return tuple(values)

def roi_is_padded_canvas(roi):
    return len(roi) >= 8

def get_roi_risk(roi):
    if len(roi) >= 8:
        return float(roi[7])
    if len(roi) >= 6:
        return float(roi[5])
    return 0.0

def set_roi_score_and_risk(roi, score, risk):
    values = list(roi)
    if len(values) < 5:
        values.append(0.0)
    values[4] = float(score)
    risk = max(0.0, min(1.0, float(risk)))
    if len(values) >= 8:
        values[7] = risk
    elif len(values) >= 6:
        values[5] = risk
    else:
        values.append(risk)
    return tuple(values)

def edge_texture_risk(texture):
    if not ENABLE_EDGE_DENSITY_FILTER:
        return 0.0
    texture = float(texture)
    if texture <= EDGE_DENSITY_THRESH:
        return 0.0
    span = max(0.01, float(EDGE_DENSITY_SOFT_LIMIT) - float(EDGE_DENSITY_THRESH))
    return max(0.0, min(1.0, (texture - float(EDGE_DENSITY_THRESH)) / span))

def apply_spatial_chaos_filter(rois, full_w, full_h):
    if not ENABLE_SPATIAL_CHAOS_FILTER or len(rois) < CHAOS_LOCAL_ROI_LIMIT:
        return rois
    cell_px = max(32, int(CHAOS_CELL_PX))
    groups = {}
    keys = []
    for roi in rois:
        cx = (float(roi[0]) + float(roi[2])) * 0.5
        cy = (float(roi[1]) + float(roi[3])) * 0.5
        key = (
            max(0, min(int(full_w // cell_px), int(cx // cell_px))),
            max(0, min(int(full_h // cell_px), int(cy // cell_px))),
        )
        groups.setdefault(key, []).append(roi)
        keys.append(key)

    local_counts = {}
    for key in groups:
        kx, ky = key
        total = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                total += len(groups.get((kx + dx, ky + dy), []))
        local_counts[key] = total

    filtered = []
    for key, items in groups.items():
        items = sorted(items, key=lambda r: float(r[4]) if len(r) > 4 else 0.0, reverse=True)
        is_chaotic = local_counts.get(key, 0) >= CHAOS_LOCAL_ROI_LIMIT
        if not is_chaotic:
            filtered.extend(items)
            continue
        for roi in items[:max(1, int(CHAOS_KEEP_PER_CELL))]:
            score = float(roi[4]) - float(CHAOS_SCORE_PENALTY)
            risk = max(get_roi_risk(roi), 0.70)
            filtered.append(set_roi_score_and_risk(roi, score, risk))
    return filtered

def crop_roi_from_center(cx, cy, full_w, full_h, crop_size=CROP_SIZE):
    half = crop_size // 2
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(full_w, x1 + crop_size)
    y2 = min(full_h, y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)
    return [x1, y1, x2, y2]

def frame_to_gray(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def align_previous_gray(prev_gray, curr_gray):
    if not ENABLE_GLOBAL_MOTION_COMP or prev_gray is None:
        return prev_gray
    try:
        prev_small = cv2.resize(prev_gray, (STAB_W, STAB_H))
        curr_small = cv2.resize(curr_gray, (STAB_W, STAB_H))
        shift, response = cv2.phaseCorrelate(np.float32(prev_small), np.float32(curr_small))
        dx = float(shift[0]) * (prev_gray.shape[1] / float(STAB_W))
        dy = float(shift[1]) * (prev_gray.shape[0] / float(STAB_H))
        if response < STAB_MIN_RESPONSE or abs(dx) > STAB_MAX_SHIFT or abs(dy) > STAB_MAX_SHIFT:
            return prev_gray
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(prev_gray, mat, (prev_gray.shape[1], prev_gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return prev_gray

def build_edge_mask(gray):
    if not ENABLE_EDGE_DENSITY_FILTER:
        return None
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    edge_mask = (grad >= EDGE_GRAD_THRESH).astype(np.uint8) * 255
    return cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

def edge_density(edge_mask, x, y, w, h):
    if edge_mask is None:
        return 0.0
    x1 = max(0, x - EDGE_DENSITY_PAD)
    y1 = max(0, y - EDGE_DENSITY_PAD)
    x2 = min(edge_mask.shape[1], x + w + EDGE_DENSITY_PAD)
    y2 = min(edge_mask.shape[0], y + h + EDGE_DENSITY_PAD)
    patch = edge_mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(cv2.countNonZero(patch)) / float(patch.size)

def local_contrast_measure(diff_img, mask_patch, x, y, w, h):
    if not ENABLE_LCM_FILTER or diff_img is None or mask_patch is None:
        return 999.0, 999.0
    pad = max(int(LCM_BG_PAD), int(max(w, h) * 2))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(diff_img.shape[1], x + w + pad)
    y2 = min(diff_img.shape[0], y + h + pad)
    local_patch = diff_img[y1:y2, x1:x2].astype(np.float32)
    if local_patch.size == 0:
        return 999.0, 999.0

    active = mask_patch > 0
    target_vals = diff_img[y:y+h, x:x+w][active].astype(np.float32) if np.any(active) else diff_img[y:y+h, x:x+w].astype(np.float32).reshape(-1)
    if target_vals.size == 0:
        return 0.0, 0.0

    ring_mask = np.ones(local_patch.shape, dtype=np.uint8)
    rx1 = max(0, x - x1)
    ry1 = max(0, y - y1)
    rx2 = min(ring_mask.shape[1], rx1 + w)
    ry2 = min(ring_mask.shape[0], ry1 + h)
    ring_mask[ry1:ry2, rx1:rx2] = 0
    ring_vals = local_patch[ring_mask > 0]
    if ring_vals.size == 0:
        return 999.0, 999.0

    target_mean = float(np.mean(target_vals))
    bg_mean = float(np.mean(ring_vals))
    bg_std = float(np.std(ring_vals))
    lcm_score = (target_mean - bg_mean) / max(bg_std, 1.0)
    lcm_ratio = target_mean / max(bg_mean, 1.0)
    return lcm_score, lcm_ratio

def is_vertical_strip_box(box_or_wh):
    if not ENABLE_VERTICAL_STRIP_FILTER:
        return False
    if len(box_or_wh) >= 4:
        bw = float(box_or_wh[2] - box_or_wh[0])
        bh = float(box_or_wh[3] - box_or_wh[1])
    else:
        bw, bh = float(box_or_wh[0]), float(box_or_wh[1])
    if bw <= 0 or bh <= 0:
        return False
    return bh >= VERTICAL_STRIP_MIN_HEIGHT and bh >= bw * VERTICAL_STRIP_ASPECT_RATIO

def prioritize_diverse_rois(rois, full_w, full_h, max_rois=MAX_ROIS_PER_FRAME):
    if not ENABLE_ROI_GRID_QUOTA or not rois:
        return rois
    sorted_rois = sorted(rois, key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    selected, overflow = [], []
    cell_counts = {}
    cols = max(1, int(ROI_GRID_COLS))
    rows = max(1, int(ROI_GRID_ROWS))
    max_per_cell = max(1, int(ROI_GRID_MAX_PER_CELL))
    for roi in sorted_rois:
        cx = (float(roi[0]) + float(roi[2])) * 0.5
        cy = (float(roi[1]) + float(roi[3])) * 0.5
        cell_x = max(0, min(cols - 1, int(cx / max(1.0, float(full_w)) * cols)))
        cell_y = max(0, min(rows - 1, int(cy / max(1.0, float(full_h)) * rows)))
        cell = (cell_x, cell_y)
        if cell_counts.get(cell, 0) < max_per_cell:
            selected.append(roi)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        else:
            overflow.append(roi)
    return (selected[:max_rois] + overflow) if len(selected) >= max_rois else (selected + overflow)

def should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
    if not ENABLE_GRAY_NOISE_SUPPRESSOR or gray is None:
        return False
    x1 = max(0, x - GRAY_TEXTURE_PAD)
    y1 = max(0, y - GRAY_TEXTURE_PAD)
    x2 = min(gray.shape[1], x + w + GRAY_TEXTURE_PAD)
    y2 = min(gray.shape[0], y + h + GRAY_TEXTURE_PAD)
    local_patch = gray[y1:y2, x1:x2]
    if local_patch.size == 0:
        return False

    local_std = float(np.std(local_patch))
    if local_std > MAX_LOCAL_GRAY_STD:
        return True

    active = mask_patch > 0
    if not np.any(active):
        return False
    active_vals = gray[y:y+h, x:x+w][active]
    if active_vals.size == 0:
        return False

    local_median = float(np.median(local_patch))
    bright_thresh = max(float(BRIGHT_SPOT_ABS_THRESH), local_median + BRIGHT_SPOT_REL_THRESH)
    active_mean = float(np.mean(active_vals))
    bright_ratio = float(np.count_nonzero(active_vals >= bright_thresh)) / float(active_vals.size)
    return (
        area <= BRIGHT_SPOT_MAX_AREA
        and bright_ratio >= 0.55
        and active_mean >= local_median + BRIGHT_SPOT_REL_THRESH
        and (local_std >= BRIGHT_SPOT_MIN_BG_STD or bright_ratio >= 0.80)
    )

class FixedFlickerSuppressor:
    def __init__(
        self,
        cell_px=FLICKER_CELL_PX,
        window_frames=FLICKER_WINDOW_FRAMES,
        cooldown_frames=FLICKER_COOLDOWN_FRAMES,
        min_hits=FLICKER_MIN_HITS,
        stationary_px=FLICKER_STATIONARY_PX,
        max_area=FLICKER_MAX_AREA,
        max_box=FLICKER_MAX_BOX,
        min_bright_ratio=FLICKER_MIN_BRIGHT_RATIO,
        min_bright_delta=FLICKER_MIN_BRIGHT_DELTA,
    ):
        self.cell_px = max(4, int(cell_px))
        self.window_frames = max(1, int(window_frames))
        self.cooldown_frames = max(1, int(cooldown_frames))
        self.min_hits = max(2, int(min_hits))
        self.stationary_px = float(stationary_px)
        self.max_area = int(max_area)
        self.max_box = int(max_box)
        self.min_bright_ratio = float(min_bright_ratio)
        self.min_bright_delta = float(min_bright_delta)
        self.history = {}
        self.blocked_until = {}
        self.suppressed_count = 0
        self.blocked_count = 0

    def _key(self, cx, cy):
        return int(round(float(cx) / self.cell_px)), int(round(float(cy) / self.cell_px))

    def _near_keys(self, key):
        kx, ky = key
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yield kx + dx, ky + dy

    def _purge(self, frame_idx):
        min_frame = int(frame_idx) - self.window_frames
        for key in list(self.history.keys()):
            vals = [v for v in self.history[key] if v[0] >= min_frame]
            if vals:
                self.history[key] = vals
            else:
                self.history.pop(key, None)
        for key in list(self.blocked_until.keys()):
            if self.blocked_until[key] < frame_idx:
                self.blocked_until.pop(key, None)

    def _is_static_flicker_candidate(self, gray, diff_img, mask_patch, x, y, w, h, area):
        if area > self.max_area or max(w, h) > self.max_box:
            return False
        active = mask_patch > 0
        if not np.any(active):
            return False
        y1 = max(0, y - 12)
        x1 = max(0, x - 12)
        y2 = min(gray.shape[0], y + h + 12)
        x2 = min(gray.shape[1], x + w + 12)
        local = gray[y1:y2, x1:x2]
        if local.size == 0:
            return False
        active_vals = gray[y:y+h, x:x+w][active]
        if active_vals.size == 0:
            return False
        diff_vals = diff_img[y:y+h, x:x+w][active] if diff_img is not None else active_vals
        diff_mean = float(np.mean(diff_vals)) if diff_vals.size else 0.0
        local_median = float(np.median(local))
        active_mean = float(np.mean(active_vals))
        bright_thresh = max(float(BRIGHT_SPOT_ABS_THRESH), local_median + self.min_bright_delta)
        bright_ratio = float(np.count_nonzero(active_vals >= bright_thresh)) / float(active_vals.size)
        is_bright_flicker = bright_ratio >= self.min_bright_ratio and active_mean >= local_median + self.min_bright_delta
        is_diff_flicker = diff_mean >= max(float(DIFF_THRESH), self.min_bright_delta)
        return is_bright_flicker or is_diff_flicker

    def observe_and_should_suppress(self, frame_idx, gray, diff_img, mask_patch, x, y, w, h, area):
        cx = float(x) + float(w) * 0.5
        cy = float(y) + float(h) * 0.5
        key = self._key(cx, cy)
        self._purge(frame_idx)
        if any(self.blocked_until.get(k, -1) >= frame_idx for k in self._near_keys(key)):
            self.suppressed_count += 1
            return True

        if not self._is_static_flicker_candidate(gray, diff_img, mask_patch, x, y, w, h, area):
            return False

        records = self.history.setdefault(key, [])
        records.append((int(frame_idx), cx, cy, int(area)))
        min_frame = int(frame_idx) - self.window_frames
        self.history[key] = [v for v in records if v[0] >= min_frame]
        records = self.history[key]
        if len(records) < self.min_hits:
            return False

        pts = np.array([[r[1], r[2]] for r in records], dtype=np.float32)
        span = float(np.max(np.linalg.norm(pts - np.mean(pts, axis=0, keepdims=True), axis=1)))
        net = float(np.linalg.norm(pts[-1] - pts[0]))
        if span <= self.stationary_px and net <= self.stationary_px:
            self.blocked_until[key] = int(frame_idx) + self.cooldown_frames
            self.blocked_count += 1
            self.suppressed_count += 1
            return True
        return False

    def is_blocked_full_box(self, frame_idx, box, scale_x, scale_y):
        self._purge(frame_idx)
        if not self.blocked_until:
            return False
        cx_full = (float(box[0]) + float(box[2])) * 0.5
        cy_full = (float(box[1]) + float(box[3])) * 0.5
        cx = cx_full / max(float(scale_x), 1e-6)
        cy = cy_full / max(float(scale_y), 1e-6)
        key = self._key(cx, cy)
        return any(self.blocked_until.get(k, -1) >= frame_idx for k in self._near_keys(key))

def motion_rois_from_mask(mask, diff_img, edge_mask, gray, scale_x, scale_y, full_w, full_h, flicker_suppressor=None, frame_idx=0):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        if is_vertical_strip_box((w, h)):
            continue
        mask_patch = mask[y:y+h, x:x+w]
        area = int(cv2.countNonZero(mask_patch))
        if area < MIN_DIFF_AREA:
            continue
        texture = edge_density(edge_mask, x, y, w, h)
        roi_risk = edge_texture_risk(texture)
        active = mask_patch > 0
        local_diff = float(diff_img[y:y+h, x:x+w][active].mean()) if np.any(active) else 0.0
        if local_diff < MIN_LOCAL_DIFF_MEAN:
            continue
        lcm_score, lcm_ratio = local_contrast_measure(diff_img, mask_patch, x, y, w, h)
        if ENABLE_LCM_FILTER:
            if LCM_REQUIRE_BOTH:
                if lcm_score < LCM_MIN_SCORE or lcm_ratio < LCM_MIN_RATIO:
                    continue
            elif lcm_score < LCM_MIN_SCORE and lcm_ratio < LCM_MIN_RATIO:
                continue
        if should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
            continue
        compactness = area / max(1.0, float(w * h))

        is_far_tiny = (
            area <= MAX_DIFF_AREA
            and w <= MAX_DIFF_BOX_W
            and h <= MAX_DIFF_BOX_H
            and compactness >= FAR_MIN_COMPACTNESS
            and (not ENABLE_LOW_LAYER_EDGE_HARD_FILTER or texture <= EDGE_DENSITY_THRESH)
        )
        is_near_compact = (
            area <= NEAR_MAX_DIFF_AREA
            and w <= NEAR_MAX_DIFF_BOX_W
            and h <= NEAR_MAX_DIFF_BOX_H
            and compactness >= NEAR_MIN_COMPACTNESS
            and (not ENABLE_LOW_LAYER_EDGE_HARD_FILTER or texture <= NEAR_EDGE_DENSITY_THRESH)
        )
        if not (is_far_tiny or is_near_compact):
            continue

        if flicker_suppressor is not None and flicker_suppressor.observe_and_should_suppress(
            frame_idx,
            gray,
            diff_img,
            mask_patch,
            x,
            y,
            w,
            h,
            area,
        ):
            continue

        if is_near_compact and not is_far_tiny:
            score = local_diff + 18.0 * compactness - 0.003 * area - 18.0 * texture
        else:
            score = local_diff + 10.0 * compactness - 0.015 * area - 20.0 * texture
        score += LCM_SCORE_WEIGHT * max(0.0, min(float(lcm_score), 8.0))
        score -= EDGE_DENSITY_SCORE_PENALTY * roi_risk

        fx, fy = int(math.floor(x * scale_x)), int(math.floor(y * scale_y))
        fx2 = int(math.ceil((x + w) * scale_x))
        fy2 = int(math.ceil((y + h) * scale_y))
        fw, fh = max(1, fx2 - fx), max(1, fy2 - fy)
        cx, cy = fx + fw // 2, fy + fh // 2
        if ENABLE_TIGHT_MOTION_ROI:
            pad = max(0, int(TIGHT_MOTION_ROI_PAD))
            sx1 = max(0, fx - pad)
            sy1 = max(0, fy - pad)
            sx2 = min(full_w, fx2 + pad)
            sy2 = min(full_h, fy2 + pad)
            sw, sh = sx2 - sx1, sy2 - sy1
            if sw <= CROP_SIZE and sh <= CROP_SIZE:
                px = (CROP_SIZE - sw) // 2
                py = (CROP_SIZE - sh) // 2
                temp_rois.append((
                    sx1,
                    sy1,
                    sx2,
                    sy2,
                    score,
                    px,
                    py,
                    roi_risk,
                ))
                continue
        rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
        temp_rois.append((
            rx1,
            ry1,
            rx2,
            ry2,
            score,
            roi_risk,
        ))
    temp_rois = apply_spatial_chaos_filter(temp_rois, full_w, full_h)
    temp_rois.sort(key=lambda r: r[4], reverse=True)
    rois = merge_nearby_boxes(temp_rois, dist_thresh=300)
    rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    return prioritize_diverse_rois(rois, full_w, full_h)


class UAVTrajectoryNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 1,
        future_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, future_steps * 2),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f"Input must be 3D tensor [B,T,F], received {x.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input feature dim must be {self.input_dim}, received {x.shape[-1]}")
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)
        pred = self.predictor(last_hidden)
        future_offsets = pred.view(-1, self.future_steps, 2)
        return logits, future_offsets
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.gru.num_layers,
            "future_steps": self.future_steps,
            "dropout": self.classifier[2].p if len(self.classifier) > 2 else 0,
        }
    
    @staticmethod
    def load_from_checkpoint(path: str, device: str = "cpu") -> "UAVTrajectoryNet":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint.get("model_config", {})
        model = UAVTrajectoryNet(
            input_dim=config.get("input_dim", 8),
            hidden_dim=config.get("hidden_dim", 32),
            num_layers=config.get("num_layers", 1),
            future_steps=config.get("future_steps", 5),
            dropout=config.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

class FeatureTracker:
    def __init__(
        self,
        seq_len: int = 20,
        input_dim: int = 8,
        smooth: bool = True,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.smooth = smooth
        self.normalize = normalize
        
        self.history_buffer = []
        self.invalid_dist_frames = 0
        self.locked_positions = []
        self.max_drone_move = 130.0 # max pixel movement in 2K space (130px)

    def update(self, frame: np.ndarray, detections: np.ndarray):
        candidates = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                w = float(x2 - x1)
                h = float(y2 - y1)
                candidates.append((cX, cY, w, h, conf))
                
        drone_pos = None
        drone_w, drone_h = 50.0, 50.0
        drone_conf = 1.0
        
        last_pos = None
        if len(self.history_buffer) > 0:
            last_pos = (int(self.history_buffer[-1]['px']), int(self.history_buffer[-1]['py']))
            
        if last_pos is not None:
            best_candidate = None
            min_dist = float('inf')
            for c in candidates:
                dist = np.hypot(c[0] - last_pos[0], c[1] - last_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = c
                    
            if min_dist < self.max_drone_move and best_candidate is not None:
                drone_pos = (best_candidate[0], best_candidate[1])
                drone_w, drone_h = best_candidate[2], best_candidate[3]
                drone_conf = float(best_candidate[4])
                self.invalid_dist_frames = 0
            else:
                drone_pos = last_pos
                drone_w = self.history_buffer[-1]['w']
                drone_h = self.history_buffer[-1]['h']
                drone_conf = self.history_buffer[-1]['conf']
                self.invalid_dist_frames += 1
        else:
            if candidates:
                best = max(candidates, key=lambda x: x[4])
                drone_pos = (best[0], best[1])
                drone_w, drone_h = best[2], best[3]
                drone_conf = float(best[4])
                
        is_deadlocked = False
        if drone_pos is not None:
            temp_locked = list(self.locked_positions) + [drone_pos]
            
            if len(self.history_buffer) >= 20:
                xs = [pt['x'] for pt in self.history_buffer]
                ys = [pt['y'] for pt in self.history_buffer]
                if np.std(xs) < 1.2 and np.std(ys) < 1.2:
                    is_deadlocked = True
                    print(f"--> [FeatureTracker] Deadlock standard deviation limit reached (std_x={np.std(xs):.2f}, std_y={np.std(ys):.2f})!")

            if self.invalid_dist_frames >= 12:
                is_deadlocked = True
                print(f"--> [FeatureTracker] Target lost for {self.invalid_dist_frames} frames. Triggering fast capture reset.")

            if not is_deadlocked and len(temp_locked) >= 30:
                xs = [p[0] for p in temp_locked if p is not None]
                ys = [p[1] for p in temp_locked if p is not None]
                if xs and ys:
                    span_x = max(xs) - min(xs)
                    span_y = max(ys) - min(ys)
                    if span_x < 5 and span_y < 5:
                        is_deadlocked = True
                    elif len(temp_locked) >= 50 and span_x < 8 and span_y < 8 and self.invalid_dist_frames >= 20:
                        is_deadlocked = True

        if is_deadlocked:
            print(f"--> [FeatureTracker] Static deadlock detected! Resetting tracker. locked_positions len: {len(self.locked_positions)}")
            self.history_buffer.clear()
            self.locked_positions.clear()
            self.invalid_dist_frames = 0
            drone_pos = None

        if drone_pos is not None:
            self.locked_positions.append(drone_pos)
            if len(self.locked_positions) > 50:
                self.locked_positions.pop(0)
                
            self.history_buffer.append({
                'x': float(drone_pos[0]), 'y': float(drone_pos[1]),
                'px': float(drone_pos[0]), 'py': float(drone_pos[1]),
                'w': drone_w, 'h': drone_h, 'conf': drone_conf
            })
            
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)
            
        is_valid = len(self.history_buffer) == self.seq_len
        return drone_pos, (0.0, 0.0), is_valid

    def get_features(self) -> np.ndarray:
        x = np.array([pt['x'] for pt in self.history_buffer], dtype=np.float64)
        y = np.array([pt['y'] for pt in self.history_buffer], dtype=np.float64)
        w = np.array([pt['w'] for pt in self.history_buffer], dtype=np.float64)
        h = np.array([pt['h'] for pt in self.history_buffer], dtype=np.float64)
        conf = np.array([pt['conf'] for pt in self.history_buffer], dtype=np.float64)
        
        if self.smooth and len(x) >= 5:
            x = savgol_filter(x, 5, 2)
            y = savgol_filter(y, 5, 2)
            
        if self.normalize:
            x_norm = x / 2560.0
            y_norm = y / 1440.0
            w_norm = w / 2560.0
            h_norm = h / 1440.0
        else:
            x_norm = x
            y_norm = y
            w_norm = w
            h_norm = h
            
        relative_x = x_norm - x_norm[0]
        relative_y = y_norm - y_norm[0]
        
        velocity_x = np.diff(x_norm, prepend=x_norm[0])
        velocity_y = np.diff(y_norm, prepend=y_norm[0])
        
        acceleration_x = np.diff(velocity_x, prepend=velocity_x[0])
        acceleration_y = np.diff(velocity_y, prepend=velocity_y[0])
        
        feature_list = [
            x_norm, y_norm, relative_x, relative_y,
            velocity_x, velocity_y, acceleration_x, acceleration_y
        ]
        
        if self.input_dim == 12:
            aspect_ratio = w_norm / (h_norm + 1e-8)
            feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
            
        features = np.stack(feature_list, axis=-1)
        return features.astype(np.float32)

class MockFallbackImpl:
    def __init__(self, conf_thres: float = 0.3, nms_thres: float = 0.45):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = cv2.blur(gray, (15, 15))
        diff = cv2.absdiff(bg, gray)
        
        _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        border = 8
        centroids = []
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=True)[:150]
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if 4 <= area < 400:
                cX = int(x + w / 2.0)
                cY = int(y + h / 2.0)
                if border <= cX < width - border and border <= cY < height - border:
                    max_contrast = int(np.max(diff[y:y+h, x:x+w]))
                    centroids.append((cX, cY, area, max_contrast, x, y, w, h))
                    
        if not centroids:
            return np.empty((0, 6), dtype=np.float32)
            
        coords = np.array([[p[0], p[1]] for p in centroids])
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        counts = tree.query_ball_point(coords, r=150, return_length=True)
        neighbors = counts - 1
        
        isolated_dets = []
        for i, p in enumerate(centroids):
            if neighbors[i] < 4:
                score = p[2] * p[3]
                isolated_dets.append((p, score))
                
        if not isolated_dets:
            return np.empty((0, 6), dtype=np.float32)
            
        isolated_dets.sort(key=lambda x: x[1], reverse=True)
        
        res = []
        for item, score_val in isolated_dets:
            cX, cY, area, max_contrast, rx, ry, rw, rh = item
            pad_w = max(15, rw)
            pad_h = max(15, rh)
            x1 = max(0, cX - pad_w)
            y1 = max(0, cY - pad_h)
            x2 = min(width - 1, cX + pad_w)
            y2 = min(height - 1, cY + pad_h)
            confidence = min(0.99, max(0.40, 0.5 + 0.05 * max_contrast))
            res.append([x1, y1, x2, y2, confidence, 0])
            
        return np.array(res, dtype=np.float32)

class TrajectoryFilter:
    def __init__(
        self,
        max_dist=100,
        min_hits=3,
        recent_window=15,
        min_recent_hits=2,
        confirm_score=0.45,
        template_size=31,
        max_gate=320,
        max_speed_mps=10.0,
        fps=15.0,
    ):
        self.trackers = []
        self.next_id = 1
        self.max_dist = float(max_dist)
        self.min_hits = int(min_hits)
        self.recent_window = int(recent_window)
        self.min_recent_hits = int(min_recent_hits)
        self.confirm_score = float(confirm_score)
        self.template_size = int(template_size if template_size % 2 == 1 else template_size + 1)
        self.max_gate = float(max_gate)
        self.max_speed_mps = float(max_speed_mps)
        self.fps = max(1.0, float(fps))

    def _center(self, box):
        return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)

    def _box_wh(self, box):
        return (max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1])))

    def _det_score(self, box):
        return float(box[4]) if len(box) > 4 else CONF_THRESH

    def _det_risk(self, box):
        if len(box) > 5:
            return self._clamp01(float(box[5]))
        return 0.0

    def _clamp01(self, value):
        return max(0.0, min(1.0, float(value)))

    def _crop_template(self, frame, box):
        if frame is None: return None
        h, w = frame.shape[:2]
        cx, cy = self._center(box)
        r = self.template_size // 2
        x1 = max(0, int(round(cx)) - r)
        y1 = max(0, int(round(cy)) - r)
        x2 = min(w, int(round(cx)) + r + 1)
        y2 = min(h, int(round(cy)) + r + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return None
        gray = frame_to_gray(patch)
        gray = cv2.resize(gray, (self.template_size, self.template_size))
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _template_score(self, trk, frame, det):
        if trk.get('template') is None:
            return 0.65
        patch = self._crop_template(frame, det)
        if patch is None:
            return 0.55
        if float(np.std(patch)) < 1.0 or float(np.std(trk['template'])) < 1.0:
            return 0.55
        raw = cv2.matchTemplate(patch, trk['template'], cv2.TM_CCOEFF_NORMED)[0][0]
        if not np.isfinite(raw):
            return 0.55
        return self._clamp01((float(raw) + 1.0) * 0.5)

    def _predict_center(self, trk, frame_idx):
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        return trk['cx'] + trk['vx'] * dt, trk['cy'] + trk['vy'] * dt, dt

    def _predicted_box(self, trk, frame_idx):
        cx, cy, _ = self._predict_center(trk, frame_idx)
        w, h = trk['w'], trk['h']
        return [int(round(cx - w * 0.5)), int(round(cy - h * 0.5)), int(round(cx + w * 0.5)), int(round(cy + h * 0.5))]

    def _dynamic_gate(self, trk, frame_idx):
        _, _, dt = self._predict_center(trk, frame_idx)
        speed_px_frame = math.sqrt(trk['vx'] * trk['vx'] + trk['vy'] * trk['vy'])
        width_px = max(1.0, trk['w'])
        physical_motion_px = self.max_speed_mps * (dt / self.fps) * width_px / max(ROUGH_TARGET_WIDTH_M, 0.01)
        gate = self.max_dist + 0.35 * speed_px_frame * dt + 0.75 * physical_motion_px + 12.0 * trk['misses']
        return max(self.max_dist, min(self.max_gate, gate))

    def _motion_score(self, trk, det, frame_idx):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        nvx = (cx - trk['cx']) / dt
        nvy = (cy - trk['cy']) / dt
        old_speed = math.sqrt(trk['vx'] * trk['vx'] + trk['vy'] * trk['vy'])
        new_speed = math.sqrt(nvx * nvx + nvy * nvy)
        if old_speed < 2.0 or new_speed < 2.0:
            return 0.75
        speed_change = abs(new_speed - old_speed) / max(old_speed, 8.0)
        dot = trk['vx'] * nvx + trk['vy'] * nvy
        cos_v = dot / max(old_speed * new_speed, 1e-6)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_v))))
        return self._clamp01(1.0 - 0.35 * min(1.0, speed_change) - 0.45 * min(1.0, angle / 150.0))

    def _trajectory_score(self, trk):
        pts = list(trk['history'])
        if len(pts) < 4:
            return 0.7
        steps = [math.sqrt((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2) for i in range(1, len(pts))]
        if not steps:
            return 0.7
        median_step = float(np.median(steps))
        max_step = max(steps)
        score = 1.0
        if median_step > 1.0 and max_step > median_step * 3.0 + 35.0:
            score -= 0.35
        headings = []
        for i in range(1, len(pts)):
            dx, dy = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]
            if abs(dx) + abs(dy) > 1.0:
                headings.append(math.degrees(math.atan2(dy, dx)))
        if len(headings) >= 4:
            turns = [abs((headings[i] - headings[i-1] + 180.0) % 360.0 - 180.0) for i in range(1, len(headings))]
            sharp_turns = sum(1 for t in turns if t > 130.0)
            if sharp_turns >= 2:
                score -= 0.25
        recent_hits = sum(trk['hit_history'])
        if len(trk['hit_history']) >= self.recent_window and recent_hits < self.min_recent_hits:
            score -= 0.35
        return self._clamp01(score)

    def _net_motion_px(self, trk):
        pts = list(trk.get('history', []))
        if len(pts) < 3:
            return 0.0
        return math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)

    def _adaptive_required_hits(self, trk):
        if not ENABLE_ADAPTIVE_TRACK_CONFIRM:
            return self.min_hits
        required = int(self.min_hits)
        risk = float(trk.get('bg_risk', 0.0))
        max_side = max(float(trk.get('w', 1.0)), float(trk.get('h', 1.0)))
        if risk >= TRACK_CONFIRM_RISK_THRESHOLD:
            required = max(required, int(TRACK_CONFIRM_RISK_MIN_HITS))
        elif (not HIGH_LAYER_MODE) and max_side <= TRACK_CONFIRM_SMALL_BOX_MAX:
            required = max(required, int(TRACK_CONFIRM_SMALL_MIN_HITS))
        return required

    def _passes_motion_confirmation(self, trk):
        if not ENABLE_ADAPTIVE_TRACK_CONFIRM:
            return True
        f = self._reference_features(trk)
        duration = int(f['duration_frames'])
        if duration < REF_TRAJ_MIN_DURATION:
            return True
        risk = float(trk.get('bg_risk', 0.0))
        min_net = TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX if risk >= TRACK_CONFIRM_RISK_THRESHOLD else TRACK_CONFIRM_MIN_NET_MOTION_PX
        min_straight = TRACK_CONFIRM_RISK_MIN_STRAIGHTNESS if risk >= TRACK_CONFIRM_RISK_THRESHOLD else TRACK_CONFIRM_MIN_STRAIGHTNESS
        if float(f['net_displacement_px']) < float(min_net):
            return False
        if float(f['straightness']) < float(min_straight):
            return False
        return True

    def _reference_features(self, trk):
        pts = np.array(list(trk.get('history', [])), dtype=np.float32)
        frames = np.array(list(trk.get('frame_history', [])), dtype=np.float32)
        hit_history = list(trk.get('hit_history', []))
        duration = len(hit_history)
        valid_ratio = float(sum(hit_history)) / float(max(1, duration))
        missed_ratio = 1.0 - valid_ratio
        features = {
            'duration_frames': duration,
            'valid_ratio': valid_ratio,
            'missed_ratio': missed_ratio,
            'path_length_px': 0.0,
            'net_displacement_px': 0.0,
            'straightness': 0.0,
            'cv_speed': 0.0,
            'heading_std_deg': 0.0,
            'cv_residual_rmse': 0.0,
            'lateral_jitter_ratio': 0.0,
            'mean_speed_px_s': 0.0,
            'std_speed_px_s': 0.0,
        }
        if len(pts) < 2:
            return features

        if len(frames) != len(pts):
            frames = np.arange(len(pts), dtype=np.float32)
        dxy = np.diff(pts, axis=0)
        steps = np.linalg.norm(dxy, axis=1)
        path_length = float(np.sum(steps))
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        features['path_length_px'] = path_length
        features['net_displacement_px'] = net_disp
        features['straightness'] = float(net_disp / max(path_length, 1e-6))

        dt_frames = np.diff(frames)
        dt = np.maximum(dt_frames / max(self.fps, 1e-6), 1.0 / max(self.fps, 1e-6))
        speeds = steps / dt
        if len(speeds):
            mean_speed = float(np.mean(speeds))
            std_speed = float(np.std(speeds))
            features['mean_speed_px_s'] = mean_speed
            features['std_speed_px_s'] = std_speed
            features['cv_speed'] = float(std_speed / max(abs(mean_speed), 1e-6))

        if len(dxy) >= 2:
            headings = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
            features['heading_std_deg'] = float(np.std(np.degrees(np.unwrap(np.radians(headings)))))

        if len(pts) >= 3:
            times = (frames - frames[0]) / max(self.fps, 1e-6)
            design = np.vstack([np.ones_like(times), times]).T
            try:
                coef_x, *_ = np.linalg.lstsq(design, pts[:, 0], rcond=None)
                coef_y, *_ = np.linalg.lstsq(design, pts[:, 1], rcond=None)
                fit_pts = np.column_stack([design @ coef_x, design @ coef_y])
                residuals = np.linalg.norm(pts - fit_pts, axis=1)
                features['cv_residual_rmse'] = float(np.sqrt(np.mean(residuals ** 2)))
            except np.linalg.LinAlgError:
                pass

            centered = pts - np.mean(pts, axis=0, keepdims=True)
            if float(np.linalg.norm(centered)) > 1e-6:
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    main_dir = vh[0]
                    normal = np.array([-main_dir[1], main_dir[0]], dtype=np.float32)
                    lateral = centered @ normal
                    features['lateral_jitter_ratio'] = float(np.std(lateral) / max(net_disp, 1e-6))
                except np.linalg.LinAlgError:
                    pass
        return features

    def _reference_label_score(self, trk):
        f = self._reference_features(trk)
        duration = int(f['duration_frames'])
        valid_ratio = float(f['valid_ratio'])
        missed_ratio = float(f['missed_ratio'])
        net_disp = float(f['net_displacement_px'])

        if duration < REF_TRAJ_MIN_DURATION:
            return 'uncertain', 0.15, f
        if duration < REF_TRAJ_EARLY_STRICT_DURATION and valid_ratio < REF_TRAJ_EARLY_MIN_VALID_RATIO:
            return 'uncertain', 0.20, f
        if valid_ratio < REF_TRAJ_MIN_VALID_RATIO:
            return 'noise_like', 0.10, f
        if missed_ratio > REF_TRAJ_MAX_MISSED_RATIO:
            return 'noise_like', 0.10, f
        if (
            REF_TRAJ_REJECT_STATIONARY
            and duration >= REF_TRAJ_STATIONARY_MIN_DURATION
            and net_disp < REF_TRAJ_STATIONARY_MAX_DISP
        ):
            return 'stationary_like', 0.20, f

        score = 0.25
        if duration >= REF_TRAJ_MIN_DURATION:
            score += 0.12
        if valid_ratio >= REF_TRAJ_MIN_VALID_RATIO:
            score += 0.16
        if missed_ratio <= REF_TRAJ_MAX_MISSED_RATIO:
            score += 0.08
        if float(f['straightness']) >= REF_TRAJ_MIN_STRAIGHTNESS:
            score += 0.10
        if float(f['cv_speed']) <= REF_TRAJ_MAX_CV_SPEED:
            score += 0.10
        if float(f['heading_std_deg']) <= REF_TRAJ_MAX_HEADING_STD:
            score += 0.08
        if float(f['lateral_jitter_ratio']) <= REF_TRAJ_MAX_LATERAL_JITTER_RATIO:
            score += 0.08
        if float(f['cv_residual_rmse']) <= REF_TRAJ_MAX_CV_RESIDUAL_RMSE:
            score += 0.08

        bird_votes = 0
        if float(f['heading_std_deg']) > REF_TRAJ_MAX_HEADING_STD:
            bird_votes += 1
        if float(f['cv_speed']) > REF_TRAJ_MAX_CV_SPEED:
            bird_votes += 1
        if float(f['lateral_jitter_ratio']) > REF_TRAJ_MAX_LATERAL_JITTER_RATIO:
            bird_votes += 1
        if float(f['cv_residual_rmse']) > REF_TRAJ_MAX_CV_RESIDUAL_RMSE:
            bird_votes += 1
        if bird_votes >= 3:
            return 'bird_like', min(0.45, score), f

        label = 'target_like' if score >= REF_TRAJ_MIN_SCORE else 'uncertain'
        return label, self._clamp01(score), f

    def _match_score(self, trk, det, frame, frame_idx):
        pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
        cx, cy = self._center(det)
        dist = math.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
        gate = self._dynamic_gate(trk, frame_idx)
        if dist > gate:
            return None
        distance_score = self._clamp01(1.0 - dist / max(gate, 1.0))
        yolo_score = self._clamp01(self._det_score(det))
        tmpl_score = self._template_score(trk, frame, det)
        motion_score = self._motion_score(trk, det, frame_idx)
        score = 0.42 * distance_score + 0.23 * yolo_score + 0.17 * tmpl_score + 0.18 * motion_score
        return score, dist, tmpl_score

    def _start_track(self, det, frame, frame_idx):
        cx, cy = self._center(det)
        w, h = self._box_wh(det)
        trk = {
            'id': self.next_id,
            'box': det[:4],
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'vx': 0.0,
            'vy': 0.0,
            'hits': 1,
            'misses': 0,
            'last_frame': int(frame_idx),
            'score': 0.35 + 0.35 * self._clamp01(self._det_score(det)),
            'bg_risk': self._det_risk(det),
            'yolo_hits': 1,
            'history': deque([(cx, cy)], maxlen=self.recent_window),
            'frame_history': deque([int(frame_idx)], maxlen=self.recent_window),
            'hit_history': deque([1], maxlen=self.recent_window),
            'template': self._crop_template(frame, det),
        }
        self.next_id += 1
        self.trackers.append(trk)

    def _update_track(self, trk, det, frame, frame_idx, match_score, tmpl_score):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        nvx = (cx - trk['cx']) / dt
        nvy = (cy - trk['cy']) / dt
        alpha = 0.55
        trk['vx'] = alpha * nvx + (1.0 - alpha) * trk['vx']
        trk['vy'] = alpha * nvy + (1.0 - alpha) * trk['vy']
        trk['cx'], trk['cy'] = cx, cy
        trk['w'], trk['h'] = self._box_wh(det)
        trk['box'] = det[:4]
        trk['last_frame'] = int(frame_idx)
        trk['hits'] += 1
        trk['yolo_hits'] = trk.get('yolo_hits', 0) + 1
        trk['misses'] = 0
        trk['score'] = self._clamp01(0.75 * trk['score'] + 0.25 * match_score)
        trk['bg_risk'] = self._clamp01(0.70 * float(trk.get('bg_risk', 0.0)) + 0.30 * self._det_risk(det))
        trk['history'].append((cx, cy))
        trk['frame_history'].append(int(frame_idx))
        trk['hit_history'].append(1)
        new_template = self._crop_template(frame, det)
        if new_template is not None:
            if trk.get('template') is None:
                trk['template'] = new_template
            elif tmpl_score >= 0.45 and match_score >= 0.45:
                trk['template'] = cv2.addWeighted(trk['template'], 0.85, new_template, 0.15, 0)

    def _miss_track(self, trk):
        trk['misses'] += 1
        trk['vx'] *= MISS_VELOCITY_DECAY
        trk['vy'] *= MISS_VELOCITY_DECAY
        trk['score'] = self._clamp01(trk['score'] * 0.92)
        trk['hit_history'].append(0)

    def _basic_is_confirmed(self, trk):
        recent_hits = sum(trk['hit_history'])
        traj_score = self._trajectory_score(trk)
        required_hits = self._adaptive_required_hits(trk)
        required_direct_hits = max(YOLO_DIRECT_CONFIRM_HITS, required_hits if float(trk.get('bg_risk', 0.0)) >= TRACK_CONFIRM_RISK_THRESHOLD else YOLO_DIRECT_CONFIRM_HITS)
        required_direct_recent = max(YOLO_DIRECT_CONFIRM_RECENT_HITS, min(required_direct_hits, self.recent_window))
        if trk.get('yolo_hits', 0) >= required_direct_hits:
            return (
                recent_hits >= required_direct_recent
                and trk['misses'] <= YOLO_DIRECT_CONFIRM_MAX_MISSES
                and trk['score'] >= YOLO_DIRECT_CONFIRM_SCORE
            )
        return (
            trk['hits'] >= required_hits
            and recent_hits >= self.min_recent_hits
            and trk['misses'] <= YOLO_TRACK_MAX_CONFIRMED_MISSES
            and trk['score'] >= self.confirm_score
            and traj_score >= TRACKER_MIN_TRAJ_SCORE
        )

    def _is_confirmed(self, trk):
        if not self._basic_is_confirmed(trk):
            return False
        if not self._passes_motion_confirmation(trk):
            return False
        if not ENABLE_REFERENCE_TRAJ_FILTER:
            return True
        label, score, _ = self._reference_label_score(trk)
        return label == 'target_like' and score >= REF_TRAJ_MIN_SCORE

    def update(self, detections, frame=None, frame_idx=0):
        detections = [list(d) for d in detections]
        pairs = []
        for ti, trk in enumerate(self.trackers):
            for di, det in enumerate(detections):
                result = self._match_score(trk, det, frame, frame_idx)
                if result is None:
                    continue
                score, dist, tmpl_score = result
                if score >= TRACKER_MATCH_SCORE:
                    pairs.append((score, dist, tmpl_score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_tracks, matched_dets = set(), set()
        for score, _, tmpl_score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            self._update_track(self.trackers[ti], detections[di], frame, frame_idx, score, tmpl_score)
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, trk in enumerate(self.trackers):
            if ti not in matched_tracks:
                self._miss_track(trk)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self._start_track(det, frame, frame_idx)

        kept = []
        for trk in self.trackers:
            if trk['misses'] <= YOLO_TRACK_MAX_SEARCH_MISSES and trk['score'] >= 0.12:
                kept.append(trk)
        self.trackers = kept
        return [t['box'] for t in self.get_confirmed_tracks(frame_idx)]

    def get_yolo_seeded_search_rois(self, frame_idx, full_w, full_h, max_rois=MAX_TRACK_ROIS_PER_FRAME):
        rois = []
        for trk in self.trackers:
            track_age = max(0, int(frame_idx) - int(trk.get('last_frame', frame_idx)))
            if track_age > TRACK_SEARCH_MAX_AGE_FRAMES:
                continue
            if trk['misses'] > YOLO_TRACK_MAX_SEARCH_MISSES:
                continue
            if TRACK_SEARCH_CONFIRMED_ONLY and not self._is_confirmed(trk):
                continue
            if TRACK_SEARCH_MIN_NET_MOTION_PX > 0.0 and self._net_motion_px(trk) < TRACK_SEARCH_MIN_NET_MOTION_PX:
                continue
            recent_hits = sum(trk.get('hit_history', []))
            if (
                trk.get('yolo_hits', 0) < TRACK_SEARCH_MIN_YOLO_HITS
                or recent_hits < TRACK_SEARCH_MIN_RECENT_HITS
                or float(trk.get('score', 0.0)) < TRACK_SEARCH_MIN_SCORE
            ):
                continue
            box = self._predicted_box(trk, frame_idx) if 0 < trk['misses'] <= TRACK_SEARCH_PREDICT_MAX_MISSES else trk['box']
            cx = (float(box[0]) + float(box[2])) * 0.5
            cy = (float(box[1]) + float(box[3])) * 0.5
            rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
            priority = 1000.0 + float(trk.get('score', 0.0)) - 20.0 * float(trk['misses'])
            rois.append((rx1, ry1, rx2, ry2, priority))
        rois = merge_nearby_boxes(rois, dist_thresh=160)
        rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
        return rois[:max_rois]

    def get_confirmed_tracks(self, frame_idx=0):
        confirmed = []
        for trk in self.trackers:
            track_age = max(0, int(frame_idx) - int(trk.get('last_frame', frame_idx)))
            if track_age > TRACK_SEARCH_MAX_AGE_FRAMES:
                continue
            if not self._is_confirmed(trk):
                continue
            box = trk['box']
            history = list(trk['history'])
            traj_label, traj_rule_score, _ = self._reference_label_score(trk)
            confirmed.append({
                'id': trk['id'],
                'box': box,
                'history': history,
                'score': trk['score'],
                'traj_rule_label': traj_label,
                'traj_rule_score': traj_rule_score,
                'live': trk['misses'] == 0 and track_age <= TRACK_LIVE_MAX_AGE_FRAMES,
                'misses': trk['misses'],
                'age': track_age,
            })
        return confirmed

def get_camera_node(bus_info_keyword):
    try:
        out = subprocess.check_output("v4l2-ctl --list-devices", shell=True).decode("utf-8").split('\n')
        is_target = False
        for line in out:
            line = line.strip()
            if not line: continue
            if not line.startswith('/dev/video'): is_target = (bus_info_keyword in line)
            elif is_target and line.startswith('/dev/video'): return line
    except: pass
    return None

# ==========================================
# 4
# ==========================================
def inference_worker():
    try: 
        yolo = YoloRKNN(MODEL_PATH, (640, 640), CONF_THRESH, 0.45)
        print("--> RKNN Loaded.", flush=True)
    except Exception as e:
        print(f"[Fatal] RKNN Init Failed: {e}"); return
    
    while not stop_event.is_set():
        did_work = False
        for i in range(N_CAM):
            try:
                item = inf_queues[i].get_nowait()
                if len(item) >= 4:
                    roi, x, y, src_frame_idx = item[:4]
                else:
                    roi, x, y = item
                    src_frame_idx = -1
                roi_risk = float(item[4]) if len(item) >= 5 else 0.0
                did_work = True
                res = yolo.infer(roi)
                rh, rw = roi.shape[:2]
                if res is None:
                    res = []
                put_latest(res_queues[i], (res, x, y, rw, rh, src_frame_idx, roi_risk))
            except queue.Empty: pass
        if not did_work: time.sleep(0.001)
    if yolo: yolo.release()

# ==========================================
# 5
# ==========================================
def capture_job(cam_idx):
    if cam_idx not in CAM_MAP: return
    target_hw_id = CAM_MAP[cam_idx]
    dev_path = get_camera_node(target_hw_id) or f"/dev/video{cam_idx * 2}"

    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_t1 = None
    tracker = FeatureTracker(seq_len=20, input_dim=8, smooth=True, normalize=True)
    try:
        gru_model = UAVTrajectoryNet.load_from_checkpoint("model/gru_baseline.pth", device="cpu")
        gru_model.eval()
    except Exception as e:
        print(f"--> Cam {cam_idx} failed to load GRU model: {e}", flush=True)
        gru_model = None
    fallback_detector = MockFallbackImpl(conf_thres=0.4, nms_thres=0.45)
    pending_fallback_rois = []
    flicker_suppressor = FixedFlickerSuppressor() if ENABLE_FLICKER_SUPPRESSOR else None
    
    f_idx = 0
    local_data_sender = DataSender(DATA_TARGETS, BOARD_ID)
    v_sender = VideoSender(VIDEO_TARGET_IP, VIDEO_BASE_PORT)
    learning_mode = True
    current_draw_boxes =[] 
    bg_samples = []
    bg_gray, bg_tol = None, None
    dynamic_bg_samples = []
    dynamic_bg_ready = not ENABLE_DYNAMIC_BG_MODEL
    dynamic_bg_start_ts = None
    dynamic_bg_next_sample_ts = 0.0
    dynamic_bg_ready_deadline_ts = None
    latest_fusion_mask = None
    last_tracker_update_frame = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        gray_frame = None
        if len(frame.shape) == 2:
            gray_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray_frame = frame[:, :, 0]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]
        f_idx += 1
        stream_allowed = is_video_stream_allowed()
        visual_enabled = stream_allowed or SHOW_INDIVIDUAL_WINDOWS
        send_video_this_frame = stream_allowed and ((f_idx + cam_idx) % VIDEO_SEND_EVERY_N_FRAMES == 0)
        draw_this_frame = send_video_this_frame or SHOW_INDIVIDUAL_WINDOWS
        processed_for_detection = False
        sent_inference_this_frame = False
        if learning_mode and (time.time() - SYSTEM_START_TIME > INIT_TIME):
            learning_mode = False

        if (f_idx + cam_idx) % PROCESS_EVERY_N_FRAMES == 0:
            gray = gray_frame if gray_frame is not None else frame_to_gray(frame)
            if gray.shape[1] == DIFF_W and gray.shape[0] == DIFF_H:
                gray_small = gray
            else:
                gray_small = cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = W / float(DIFF_W), H / float(DIFF_H)

            if ENABLE_DYNAMIC_BG_MODEL and not dynamic_bg_ready:
                now_ts = time.time()
                if dynamic_bg_start_ts is None:
                    per_cam_learning_span = DYNAMIC_BG_SECONDS + DYNAMIC_BG_SERIAL_GAP_SECONDS
                    start_delay = cam_idx * per_cam_learning_span + DYNAMIC_BG_START_DELAY_SECONDS
                    dynamic_bg_start_ts = SYSTEM_START_TIME + start_delay
                    dynamic_bg_next_sample_ts = dynamic_bg_start_ts
                    dynamic_bg_ready_deadline_ts = dynamic_bg_start_ts + DYNAMIC_BG_SECONDS
                    print(
                        f"--> Cam {cam_idx} dynamic background scheduled. "
                        f"start_delay={start_delay:.1f}s "
                        f"window={DYNAMIC_BG_SECONDS:.1f}s",
                        flush=True,
                    )
                if now_ts >= dynamic_bg_start_ts:
                    if now_ts >= dynamic_bg_next_sample_ts:
                        dynamic_bg_samples.append(gray_small.copy())
                        dynamic_bg_next_sample_ts = now_ts + DYNAMIC_BG_SAMPLE_INTERVAL
                    if (
                        dynamic_bg_ready_deadline_ts is not None
                        and now_ts >= dynamic_bg_ready_deadline_ts
                    ):
                        dynamic_bg_gray, dynamic_bg_tol = build_dynamic_bg_model(dynamic_bg_samples)
                        if dynamic_bg_gray is not None and dynamic_bg_tol is not None:
                            bg_gray = dynamic_bg_gray
                            if bg_tol is None:
                                bg_tol = dynamic_bg_tol
                            else:
                                bg_tol = np.maximum(bg_tol, dynamic_bg_tol)
                        dynamic_bg_ready = True
                        print(
                            f"--> Cam {cam_idx} dynamic background ready. "
                            f"start_delay={start_delay:.1f}s "
                            f"window={DYNAMIC_BG_SECONDS:.1f}s samples={len(dynamic_bg_samples)}",
                            flush=True,
                        )
                        dynamic_bg_samples = []

            if ENABLE_LOW_LAYER_STATIC_BG_MASK and bg_gray is None:
                bg_samples.append(gray_small.copy())
                if len(bg_samples) >= LOW_BG_SAMPLE_FRAMES:
                    bg_gray, bg_tol = build_static_bg_model(bg_samples)
                    bg_samples = []
                    print(f"--> Cam {cam_idx} static background ready.", flush=True)
                frame_t1 = gray_small
                continue
            
            processed_for_detection = True
            
            # A. YOLO Cascade Detection on 1080p (W, H are 1920, 1080)
            last_pos_1080 = None
            if len(tracker.history_buffer) > 0:
                last_pos_2k = (tracker.history_buffer[-1]['px'], tracker.history_buffer[-1]['py'])
                last_pos_1080 = (
                    int(last_pos_2k[0] * (W / 2560.0)),
                    int(last_pos_2k[1] * (H / 1440.0))
                )

            sent_inference_this_frame = False
            
            if last_pos_1080 is not None:
                # 1. ROI search mode: Crop 640x640 around last_pos_1080
                cX, cY = last_pos_1080
                x1 = max(0, cX - 320)
                y1 = max(0, cY - 320)
                if x1 + 640 > W:
                    x1 = W - 640
                if y1 + 640 > H:
                    y1 = H - 640
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x1 + 640)
                y2 = min(H, y1 + 640)
                
                roi_img = frame[y1:y2, x1:x2].copy()
                if roi_img.size > 0:
                    if put_latest(inf_queues[cam_idx], (roi_img, x1, y1, f_idx, 0.0)):
                        sent_inference_this_frame = True
                        if visual_enabled:
                            current_draw_boxes.append([x1, y1, x2, y2, (0, 255, 255), "TRKROI", 2])
            else:
                # 2. Fallback or Global mode
                if pending_fallback_rois:
                    # Push up to 3 cropped 256x256 ROIs (resized to 640x640)
                    for f_det in pending_fallback_rois[:3]:
                        fx1, fy1, fx2, fy2 = f_det[:4]
                        fcX = int((fx1 + fx2) / 2.0)
                        fcY = int((fy1 + fy2) / 2.0)
                        
                        rx1 = max(0, fcX - 128)
                        ry1 = max(0, fcY - 128)
                        if rx1 + 256 > W:
                            rx1 = W - 256
                        if ry1 + 256 > H:
                            ry1 = H - 256
                        rx1 = max(0, rx1)
                        ry1 = max(0, ry1)
                        rx2 = min(W, rx1 + 256)
                        ry2 = min(H, ry1 + 256)
                        
                        roi_img_256 = frame[ry1:ry2, rx1:rx2].copy()
                        if roi_img_256.size > 0:
                            roi_img = cv2.resize(roi_img_256, (640, 640))
                            if put_latest(inf_queues[cam_idx], (roi_img, rx1, ry1, f_idx, 2.0)):
                                sent_inference_this_frame = True
                                if visual_enabled:
                                    current_draw_boxes.append([rx1, ry1, rx2, ry2, (255, 255, 255), "FALLBACK", 2])
                    pending_fallback_rois = []
                else:
                    # Push full 1080p frame resized to 640x640
                    roi_img = cv2.resize(frame, (640, 640))
                    if put_latest(inf_queues[cam_idx], (roi_img, 0, 0, f_idx, 1.0)):
                        sent_inference_this_frame = True
                        if visual_enabled:
                            current_draw_boxes.append([0, 0, W, H, (255, 255, 255), "GLOBAL", 2])

        # B. Read NPU Inference Results from queue
        raw_boxes_in_this_frame = []
        got_inference_result = False
        is_global_capture_miss = False
        
        while True:
            try:
                result = res_queues[cam_idx].get_nowait()
                if len(result) >= 7:
                    dets, xo, yo, roi_w, roi_h, src_frame_idx, roi_risk = result[:7]
                elif len(result) >= 6:
                    dets, xo, yo, roi_w, roi_h, src_frame_idx = result[:6]
                    roi_risk = 0.0
                else:
                    dets, xo, yo = result
                    roi_w, roi_h = 640, 640
                    src_frame_idx = f_idx
                    roi_risk = 0.0
                    
                if src_frame_idx >= 0 and f_idx - int(src_frame_idx) > MAX_INFERENCE_RESULT_AGE_FRAMES:
                    continue
                    
                got_inference_result = True
                
                is_global = abs(float(roi_risk) - 1.0) < 0.01
                if is_global and len(dets) == 0:
                    is_global_capture_miss = True
                
                sx = float(roi_w) / 640.0
                sy = float(roi_h) / 640.0
                
                for d in dets:
                    score = float(d[4]) if len(d) > 4 else CONF_THRESH
                    mapped_box = [
                        int(max(0, min(W, d[0] * sx + xo))),
                        int(max(0, min(H, d[1] * sy + yo))),
                        int(max(0, min(W, d[2] * sx + xo))),
                        int(max(0, min(H, d[3] * sy + yo))),
                        score,
                        float(roi_risk)
                    ]
                    if (
                        mapped_box[2] > mapped_box[0]
                        and mapped_box[3] > mapped_box[1]
                        and not is_vertical_strip_box(mapped_box)
                    ):
                        raw_boxes_in_this_frame.append(mapped_box)
            except queue.Empty:
                break

        unique_raw_boxes = merge_nearby_boxes(raw_boxes_in_this_frame, dist_thresh=100)
        
        if is_global_capture_miss:
            fallback_dets = fallback_detector.detect(frame)
            pending_fallback_rois = fallback_dets[:3]

        for rb in unique_raw_boxes:
            if visual_enabled:
                current_draw_boxes.append([rb[0], rb[1], rb[2], rb[3], (0, 0, 255), "Raw", 2])

        # Scale coordinates to 2K space for tracker
        dets_2k = []
        for rb in unique_raw_boxes:
            x1_2k = rb[0] * (2560.0 / W)
            y1_2k = rb[1] * (1440.0 / H)
            x2_2k = rb[2] * (2560.0 / W)
            y2_2k = rb[3] * (1440.0 / H)
            dets_2k.append([x1_2k, y1_2k, x2_2k, y2_2k, rb[4], 0.0])
        dets_2k = np.array(dets_2k, dtype=np.float32) if dets_2k else np.empty((0, 6), dtype=np.float32)

        # C. Update tracker in 2K space
        should_empty_update = (
            not got_inference_result
            and processed_for_detection
            and not sent_inference_this_frame
            and f_idx - last_tracker_update_frame >= PROCESS_EVERY_N_FRAMES
        )
        
        drone_pos_2k = None
        is_valid = False
        
        if got_inference_result or should_empty_update:
            dummy_frame_2k = np.empty((1440, 2560, 3), dtype=np.uint8)
            drone_pos_2k, _, is_valid = tracker.update(dummy_frame_2k, dets_2k if got_inference_result else np.empty((0, 6), dtype=np.float32))
            last_tracker_update_frame = f_idx

        # D. GRU Inference and Trajectory Prediction
        classification_prob = None
        pred_coords_1080 = []
        
        if is_valid and gru_model is not None:
            feats_np = tracker.get_features()
            feats_t = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits, pred_offsets = gru_model(feats_t)
                pred_offsets = pred_offsets.squeeze(0).numpy()
                classification_prob = torch.sigmoid(logits).item()
                
            if drone_pos_2k is not None:
                current_x_2k, current_y_2k = drone_pos_2k
                scale_x = 2560.0 / 2.0
                scale_y = 1440.0 / 2.0
                
                for off in pred_offsets:
                    px_2k = current_x_2k + off[0] * scale_x
                    py_2k = current_y_2k + off[1] * scale_y
                    px_1080 = int(px_2k * (W / 2560.0))
                    py_1080 = int(py_2k * (H / 1440.0))
                    pred_coords_1080.append((px_1080, py_1080))

        # E. Rendering Bounding Box and Trajectories on 1080p frame
        # Draw history trajectory
        if len(tracker.history_buffer) > 1:
            for i in range(1, len(tracker.history_buffer)):
                p0 = tracker.history_buffer[i - 1]
                p1 = tracker.history_buffer[i]
                pt0_1080 = (
                    int(p0["px"] * (W / 2560.0)),
                    int(p0["py"] * (H / 1440.0))
                )
                pt1_1080 = (
                    int(p1["px"] * (W / 2560.0)),
                    int(p1["py"] * (H / 1440.0))
                )
                if visual_enabled:
                    current_draw_boxes.append([pt0_1080[0], pt0_1080[1], pt1_1080[0], pt1_1080[1], (0, 255, 0), "LINE", 1])

        # Draw predicted future trajectory
        if is_valid and drone_pos_2k is not None and pred_coords_1080:
            for i in range(1, len(pred_coords_1080)):
                if visual_enabled:
                    current_draw_boxes.append([pred_coords_1080[i-1][0], pred_coords_1080[i-1][1], pred_coords_1080[i][0], pred_coords_1080[i][1], (255, 191, 0), "PREDLINE", 1])
            for coord in pred_coords_1080:
                if visual_enabled:
                    current_draw_boxes.append([coord[0], coord[1], 0, 0, (255, 191, 0), "PREDPT", 1])

        # Draw yellow drone bounding box
        drone_pos_1080 = None
        if drone_pos_2k is not None:
            latest = tracker.history_buffer[-1]
            drone_pos_1080 = (
                int(drone_pos_2k[0] * (W / 2560.0)),
                int(drone_pos_2k[1] * (H / 1440.0))
            )
            w_box = max(20, int(latest.get("w", 50) * (W / 2560.0)))
            h_box = max(20, int(latest.get("h", 50) * (H / 1440.0)))
            
            x1_b = drone_pos_1080[0] - w_box // 2
            y1_b = drone_pos_1080[1] - h_box // 2
            x2_b = drone_pos_1080[0] + w_box // 2
            y2_b = drone_pos_1080[1] + h_box // 2
            
            label_box = "TARGET"
            if classification_prob is not None:
                is_noise = classification_prob < 0.5
                label_box = f"NOISE:{classification_prob:.2f}" if is_noise else f"TARGET:{classification_prob:.2f}"
                color_box = (0, 0, 255) if is_noise else (0, 255, 0)
            else:
                color_box = (0, 255, 255)
                
            if visual_enabled:
                current_draw_boxes.append([x1_b, y1_b, x2_b, y2_b, color_box, label_box, 1])

        # F. Send telemetry to Gimbal via UDP
        if not learning_mode and drone_pos_1080 is not None:
            if classification_prob is None or classification_prob >= 0.5:
                w_box = max(20, int(latest.get("w", 50) * (W / 2560.0)))
                h_box = max(20, int(latest.get("h", 50) * (H / 1440.0)))
                x1_b = drone_pos_1080[0] - w_box // 2
                y1_b = drone_pos_1080[1] - h_box // 2
                x2_b = drone_pos_1080[0] + w_box // 2
                y2_b = drone_pos_1080[1] + h_box // 2
                gimbal_data = [[x1_b, y1_b, x2_b, y2_b, estimate_rough_range_m([x1_b, y1_b, x2_b, y2_b], W)]]
                local_data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")

        if visual_enabled and len(current_draw_boxes) > MAX_DRAW_BOXES:
            current_draw_boxes = current_draw_boxes[-MAX_DRAW_BOXES:]

        if draw_this_frame:
            try:
                if ENABLE_FUSION_DISPLAY:
                    show_frame = make_fused_display_preview(frame, latest_fusion_mask, 640, 360)
                else:
                    show_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                dsx, dsy = 640.0/W, 360.0/H
                for i in range(len(current_draw_boxes)-1, -1, -1):
                    x1, y1, x2, y2, color, text, life = current_draw_boxes[i]
                    if text in ("LINE", "PREDLINE"):
                        cv2.line(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 2, cv2.LINE_AA)
                    elif text == "PREDPT":
                        cv2.circle(show_frame, (int(x1*dsx), int(y1*dsy)), 3, color, -1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 1 if text in ("ROI", "TRKROI", "FALLBACK") else 2)
                        if text not in ("ROI", "TRKROI", "FALLBACK", "Raw"):
                            cv2.putText(show_frame, text, (int(x1*dsx), int(y1*dsy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    current_draw_boxes[i][6] -= 1
                    if current_draw_boxes[i][6] <= 0: current_draw_boxes.pop(i)
                
                if send_video_this_frame: v_sender.send(BOARD_ID, cam_idx, show_frame)
                if SHOW_INDIVIDUAL_WINDOWS:
                    put_latest(display_queues[cam_idx], show_frame)
            except: pass
    cap.release()

if __name__ == '__main__':
    runtime_ctx = init_runtime_ipc()
    use_camera_processes = runtime_ctx is not None
    latest_display_frames = [None for _ in range(N_CAM)]
    
    print(f"--> Detection enabled. version={ALGORITHM_VERSION}", flush=True)
    print(f"--> {ALGORITHM_NOTE}", flush=True)
    print(f"--> Camera workers: {'processes' if use_camera_processes else 'threads'}", flush=True)
    t_inf = threading.Thread(target=inference_worker); t_inf.daemon = True; t_inf.start()
    camera_workers = []
    for i in range(N_CAM):
        if use_camera_processes:
            worker = runtime_ctx.Process(target=capture_job, args=(i,), daemon=True)
        else:
            worker = threading.Thread(target=capture_job, args=(i,), daemon=True)
        worker.start()
        camera_workers.append(worker)
        time.sleep(0.5)

    if SHOW_INDIVIDUAL_WINDOWS:
        for i in range(N_CAM):
            cv2.namedWindow(f"Cam {i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Cam {i}", 640, 360)
    
    def timer_job():
        global INIT_SIGNAL_SENT, VIDEO_STREAM_ALLOWED
        while not stop_event.is_set():
            if not INIT_SIGNAL_SENT and (time.time() - SYSTEM_START_TIME > INIT_TIME):

                VIDEO_STREAM_ALLOWED = True
                if video_allowed_event is not None:
                    video_allowed_event.set()
                INIT_SIGNAL_SENT = True
            time.sleep(1)
    
    threading.Thread(target=timer_job, daemon=True).start()

    try:
        while True:
            if SHOW_INDIVIDUAL_WINDOWS:
                for i in range(N_CAM):
                    while True:
                        try:
                            latest_display_frames[i] = display_queues[i].get_nowait()
                        except queue.Empty:
                            break
                    if latest_display_frames[i] is not None:
                        cv2.imshow(f"Cam {i}", latest_display_frames[i])
                if cv2.waitKey(10) & 0xFF == ord('q'): break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt: pass
    stop_event.set()
    for worker in camera_workers:
        try:
            worker.join(timeout=1.0)
        except Exception:
            pass
        if use_camera_processes:
            try:
                if worker.is_alive():
                    worker.terminate()
            except Exception:
                pass
    time.sleep(0.5)
    cv2.destroyAllWindows()
