#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shutil
import sys
import time
import types
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


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
TRACK_SEARCH_MIN_YOLO_HITS = 2
TRACK_SEARCH_MIN_RECENT_HITS = 2
TRACK_SEARCH_MIN_SCORE = 0.32
ENABLE_REFERENCE_TRAJ_FILTER = False
REF_TRAJ_MIN_SCORE = 0.58
REF_TRAJ_MIN_DURATION = 6
REF_TRAJ_MIN_VALID_RATIO = 0.45
REF_TRAJ_MAX_MISSED_RATIO = 0.60
REF_TRAJ_MIN_STRAIGHTNESS = 0.45
REF_TRAJ_MAX_CV_SPEED = 1.25
REF_TRAJ_MAX_HEADING_STD = 95.0
REF_TRAJ_MAX_LATERAL_JITTER_RATIO = 0.90
REF_TRAJ_MAX_CV_RESIDUAL_RMSE = 80.0
REF_TRAJ_REJECT_STATIONARY = True
REF_TRAJ_STATIONARY_MIN_DURATION = 15
REF_TRAJ_STATIONARY_MAX_DISP = 4.0
FUSION_REQUIRE_TRACK_MOTION = True
FUSION_TRACK_MIN_PIXELS = 3
FUSION_TRACK_CENTER_SIZE = 120
ENABLE_VERTICAL_STRIP_FILTER = True
VERTICAL_STRIP_ASPECT_RATIO = 1.6
VERTICAL_STRIP_MIN_HEIGHT = 8
ENABLE_STATIC_BG_MASK = True
STATIC_BG_SECONDS = 10.0
STATIC_BG_ABS_DELTA = 12.0
STATIC_BG_STD_MULT = 3.0
STATIC_BG_MAX_DELTA = 60.0
STATIC_BG_MAX_SAMPLES = 40
ENABLE_LOWER_DYNAMIC_MASK = False
LOWER_DYNAMIC_MASK_SECONDS = 60.0
LOWER_DYNAMIC_MASK_Y_RATIO = 0.5
LOWER_DYNAMIC_MASK_DIFF_THRESH = 8
LOWER_DYNAMIC_MASK_HIT_RATIO = 0.12
ENABLE_FLICKER_SUPPRESSOR = False
FLICKER_CELL_PX = 48
FLICKER_WINDOW_FRAMES = 120
FLICKER_COOLDOWN_FRAMES = 240
FLICKER_MIN_HITS = 4
FLICKER_STATIONARY_PX = 10.0
FLICKER_MAX_AREA = 1400
FLICKER_MAX_BOX = 180
FLICKER_MIN_BRIGHT_RATIO = 0.35
FLICKER_MIN_BRIGHT_DELTA = 10.0

CROP_SIZE = 640
TIGHT_MOTION_ROI = False
TIGHT_MOTION_ROI_PAD = 32
TIGHT_MOTION_ROI_FILL = 114
PROCESS_EVERY_N_FRAMES = 3
TRACK_LIVE_MAX_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 2
TRACK_SEARCH_MAX_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 3
DIFF_THRESH = 6
MOTION_MODE = "frame-diff"
MEDIAN_BG_INIT_FRAMES = 40
MEDIAN_BG_WINDOW = 41
MEDIAN_BG_Z_THRESH = 2.5
MEDIAN_BG_MIN_THRESH = DIFF_THRESH
MEDIAN_BG_UPDATE = False
MIN_LOCAL_DIFF_MEAN = 10.0
ENABLE_GRAY_NOISE_SUPPRESSOR = True
GRAY_TEXTURE_PAD = 28
MAX_LOCAL_GRAY_STD = 38.0
BRIGHT_SPOT_ABS_THRESH = 210
BRIGHT_SPOT_REL_THRESH = 32.0
BRIGHT_SPOT_MAX_AREA = 220
BRIGHT_SPOT_MIN_BG_STD = 10.0
ENABLE_LCM_FILTER = True
LCM_BG_PAD = 18
LCM_MIN_SCORE = 1.8
LCM_MIN_RATIO = 1.25
LCM_SCORE_WEIGHT = 3.0
MIN_OBJ_SIZE = 20
FAR_MIN_COMPACTNESS = 0.18
MOTION_ERODE_ITER = 0
MOTION_DILATE_ITER = 1
MOTION_CLOSE_ITER = 1
ENABLE_MOTION_OPENING = True
MOTION_OPEN_ITER = 0
MOTION_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MOTION_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
MOTION_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MOTION_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
BG_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
BG_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MAX_ROIS_PER_FRAME = 6
ENABLE_ROI_GRID_QUOTA = True
ROI_GRID_COLS = 4
ROI_GRID_ROWS = 3
ROI_GRID_MAX_PER_CELL = 1
DIFF_W, DIFF_H = 1920, 1080
ENABLE_GLOBAL_MOTION_COMP = False
ENABLE_EDGE_DENSITY_FILTER = False
STAB_W, STAB_H = 480, 270
STAB_MIN_RESPONSE = 0.08
STAB_MAX_SHIFT = 80
MAX_DIFF_AREA = 1000
MAX_DIFF_BOX_W = 180
MAX_DIFF_BOX_H = 180
NEAR_MAX_DIFF_AREA = 10000
NEAR_MAX_DIFF_BOX_W = 450
NEAR_MAX_DIFF_BOX_H = 450
NEAR_MIN_COMPACTNESS = 0.04
NEAR_EDGE_DENSITY_THRESH = 0.45
EDGE_GRAD_THRESH = 45
EDGE_DENSITY_THRESH = 0.35
EDGE_DENSITY_PAD = 24
EDGE_DENSITY_SOFT_LIMIT = 0.38
EDGE_DENSITY_SCORE_PENALTY = 65.0
ENABLE_LOW_LAYER_EDGE_HARD_FILTER = False
ENABLE_SPATIAL_CHAOS_FILTER = False
CHAOS_CELL_PX = 220
CHAOS_LOCAL_ROI_LIMIT = 5
CHAOS_KEEP_PER_CELL = 1
CHAOS_SCORE_PENALTY = 40.0
ENABLE_LOWER_CHAOS_SUPPRESSOR = False
LOWER_CHAOS_Y_RATIO = 0.58
LOWER_CHAOS_LOCAL_ROI_LIMIT = 3
LOWER_CHAOS_KEEP_PER_CLUSTER = 1
LOWER_CHAOS_SCORE_PENALTY = 55.0
LOWER_CHAOS_MIN_RISK = 0.92
ENABLE_ADAPTIVE_TRACK_CONFIRM = False
TRACK_CONFIRM_MIN_NET_MOTION_PX = 10.0
TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX = 16.0
TRACK_CONFIRM_MIN_STRAIGHTNESS = 0.35
TRACK_CONFIRM_RISK_MIN_STRAIGHTNESS = 0.45
TRACK_CONFIRM_RISK_THRESHOLD = 0.45
TRACK_CONFIRM_SMALL_BOX_MAX = 56
TRACK_CONFIRM_SMALL_MIN_HITS = 6
TRACK_CONFIRM_RISK_MIN_HITS = 7

ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
IMG_W, IMG_H = 2560, 1440
DIFF_SCALE_AREA = (DIFF_W * DIFF_H) / float(IMG_W * IMG_H)
MIN_DIFF_AREA = max(2, int(MIN_OBJ_SIZE * DIFF_SCALE_AREA))
ROUGH_RANGE_MIN_M = 20
ROUGH_RANGE_MAX_M = 2000
ROUGH_RANGE_ROUND_M = 10


def estimate_rough_range_m(box, frame_width=IMG_W):
    pixel_width = max(1.0, float(box[2] - box[0]))
    frame_width = max(1.0, float(frame_width))
    fx_px = (frame_width / 2.0) / math.tan(math.radians(CAM_H_FOV / 2.0))
    distance = (ROUGH_TARGET_WIDTH_M * fx_px) / pixel_width
    distance = max(ROUGH_RANGE_MIN_M, min(ROUGH_RANGE_MAX_M, distance))
    return int(round(distance / ROUGH_RANGE_ROUND_M) * ROUGH_RANGE_ROUND_M)


def merge_nearby_boxes(boxes, dist_thresh=120):
    if not boxes:
        return []
    merged = []
    for box in boxes:
        is_dup = False
        bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for idx, m_box in enumerate(merged):
            mcx, mcy = (m_box[0] + m_box[2]) / 2, (m_box[1] + m_box[3]) / 2
            dist = math.sqrt((bcx - mcx) ** 2 + (bcy - mcy) ** 2)
            if dist < dist_thresh:
                box_score = float(box[4]) if len(box) > 4 else 1.0
                merged_score = float(m_box[4]) if len(m_box) > 4 else 1.0
                if box_score > merged_score:
                    merged[idx] = box
                is_dup = True
                break
        if not is_dup:
            merged.append(box)
    return merged


def cleanup_motion_mask(mask):
    if MOTION_ERODE_ITER > 0:
        mask = cv2.erode(mask, MOTION_ERODE_KERNEL, iterations=MOTION_ERODE_ITER)
    if MOTION_DILATE_ITER > 0:
        mask = cv2.dilate(mask, MOTION_DILATE_KERNEL, iterations=MOTION_DILATE_ITER)
    if MOTION_CLOSE_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MOTION_CLOSE_KERNEL, iterations=MOTION_CLOSE_ITER)
    return mask


def cleanup_motion_mask_with_iters(mask, erode_iter, dilate_iter, close_iter):
    if ENABLE_MOTION_OPENING and MOTION_OPEN_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MOTION_OPEN_KERNEL, iterations=int(MOTION_OPEN_ITER))
    if erode_iter > 0:
        mask = cv2.erode(mask, MOTION_ERODE_KERNEL, iterations=int(erode_iter))
    if dilate_iter > 0:
        mask = cv2.dilate(mask, MOTION_DILATE_KERNEL, iterations=int(dilate_iter))
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MOTION_CLOSE_KERNEL, iterations=int(close_iter))
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


def map_padded_detection_to_frame(det, roi_info, full_w, full_h):
    x1, y1, x2, y2 = [int(v) for v in roi_info[:4]]
    pad_x = int(roi_info[5]) if len(roi_info) > 5 else 0
    pad_y = int(roi_info[6]) if len(roi_info) > 6 else 0
    src_w = max(0, x2 - x1)
    src_h = max(0, y2 - y1)
    dx1 = max(0, min(src_w, int(round(det[0] - pad_x))))
    dy1 = max(0, min(src_h, int(round(det[1] - pad_y))))
    dx2 = max(0, min(src_w, int(round(det[2] - pad_x))))
    dy2 = max(0, min(src_h, int(round(det[3] - pad_y))))
    if dx2 <= dx1 or dy2 <= dy1:
        return None
    return [
        max(0, min(full_w, x1 + dx1)),
        max(0, min(full_h, y1 + dy1)),
        max(0, min(full_w, x1 + dx2)),
        max(0, min(full_h, y1 + dy2)),
        float(det[4]),
    ]


def make_fused_display_frame(frame, mask_small, full_w, full_h):
    gray = frame_to_gray(frame)
    if mask_small is None:
        mask_full = np.zeros_like(gray)
    elif mask_small.shape[1] == gray.shape[1] and mask_small.shape[0] == gray.shape[0]:
        mask_full = mask_small
    else:
        mask_full = cv2.resize(mask_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.merge([gray, gray, mask_full])


def track_roi_has_motion(track_roi, mask_small, full_w, full_h, min_pixels=FUSION_TRACK_MIN_PIXELS, center_size=FUSION_TRACK_CENTER_SIZE):
    if mask_small is None:
        return False
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
    for roi in rois:
        cx = (float(roi[0]) + float(roi[2])) * 0.5
        cy = (float(roi[1]) + float(roi[3])) * 0.5
        key = (
            max(0, min(int(full_w // cell_px), int(cx // cell_px))),
            max(0, min(int(full_h // cell_px), int(cy // cell_px))),
        )
        groups.setdefault(key, []).append(roi)

    local_counts = {}
    for key in groups:
        kx, ky = key
        total = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                total += len(groups.get((kx + dx, ky + dy), []))
        local_counts[key] = total

    filtered = []
    def cell_is_lower(key):
        _, ky = key
        cell_center_y = (float(ky) + 0.5) * float(cell_px)
        return ENABLE_LOWER_CHAOS_SUPPRESSOR and cell_center_y >= float(full_h) * float(LOWER_CHAOS_Y_RATIO)

    chaotic_cells = set()
    for key, count in local_counts.items():
        limit = LOWER_CHAOS_LOCAL_ROI_LIMIT if cell_is_lower(key) else CHAOS_LOCAL_ROI_LIMIT
        if count >= int(limit):
            chaotic_cells.add(key)
    processed_cells = set()

    for key, items in groups.items():
        items = sorted(items, key=lambda r: float(r[4]) if len(r) > 4 else 0.0, reverse=True)
        if key not in chaotic_cells:
            filtered.extend(items)
            continue
        if key in processed_cells:
            continue

        cluster_cells = []
        stack = [key]
        while stack:
            cur = stack.pop()
            if cur in processed_cells or cur not in chaotic_cells:
                continue
            processed_cells.add(cur)
            cluster_cells.append(cur)
            cx, cy = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nxt = (cx + dx, cy + dy)
                    if nxt not in processed_cells and nxt in chaotic_cells:
                        stack.append(nxt)

        cluster_items = []
        for cell in cluster_cells:
            cluster_items.extend(groups.get(cell, []))
        cluster_items.sort(key=lambda r: float(r[4]) if len(r) > 4 else 0.0, reverse=True)

        cluster_is_lower = any(cell_is_lower(cell) for cell in cluster_cells)
        keep_n = LOWER_CHAOS_KEEP_PER_CLUSTER if cluster_is_lower else CHAOS_KEEP_PER_CELL
        score_penalty = LOWER_CHAOS_SCORE_PENALTY if cluster_is_lower else CHAOS_SCORE_PENALTY
        min_risk = LOWER_CHAOS_MIN_RISK if cluster_is_lower else 0.85

        for roi in cluster_items[:max(1, int(keep_n))]:
            score = float(roi[4]) - float(score_penalty)
            risk = max(get_roi_risk(roi), float(min_risk))
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


def build_initial_bg_model(video_path, seconds, diff_size, abs_delta, std_mult, max_delta, max_samples):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for static background: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    sample_span = max(1, int(round(float(seconds) * fps)))
    sample_stride = max(1, int(np.ceil(sample_span / float(max(1, int(max_samples))))))

    samples = []
    frame_idx = 0
    while frame_idx < sample_span:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % sample_stride != 0:
            continue
        gray = frame_to_gray(frame)
        if gray.shape[1] != diff_size[0] or gray.shape[0] != diff_size[1]:
            gray = cv2.resize(gray, diff_size, interpolation=cv2.INTER_AREA)
        samples.append(gray)
    cap.release()

    if not samples:
        raise RuntimeError("no frames sampled for static background")
    stack = np.stack(samples, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    std = np.std(stack, axis=0)
    tol = np.clip(float(abs_delta) + float(std_mult) * std, float(abs_delta), float(max_delta)).astype(np.uint8)
    return bg, tol, len(samples), sample_span


def build_median_background(samples):
    if not samples:
        return None
    stack = np.stack(samples, axis=0)
    kth = len(samples) // 2
    return np.partition(stack, kth, axis=0)[kth].astype(np.uint8)


def robust_residual_threshold(diff_img, z_thresh, min_thresh):
    values = diff_img.astype(np.float32).reshape(-1)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = max(1.4826 * mad, 1.0)
    return max(float(min_thresh), med + float(z_thresh) * sigma), med, sigma


def build_static_bg_change_mask(gray, bg, tol):
    if bg is None or tol is None:
        return None
    bg_diff = cv2.absdiff(gray, bg)
    changed = (bg_diff.astype(np.uint16) > tol.astype(np.uint16)).astype(np.uint8) * 255
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, BG_OPEN_KERNEL, iterations=1)
    changed = cv2.dilate(changed, BG_DILATE_KERNEL, iterations=1)
    return changed


def build_lower_dynamic_mask(video_path, seconds, diff_size, lower_y_ratio, diff_thresh, hit_ratio):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for lower dynamic mask: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    sample_span = max(2, int(round(float(seconds) * fps)))
    lower_y = max(0, min(diff_size[1] - 1, int(round(float(diff_size[1]) * float(lower_y_ratio)))))

    hit_counts = np.zeros((diff_size[1], diff_size[0]), dtype=np.uint16)
    valid_frames = 0
    prev_gray = None

    frame_idx = 0
    while frame_idx < sample_span:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = frame_to_gray(frame)
        if gray.shape[1] != diff_size[0] or gray.shape[0] != diff_size[1]:
            gray = cv2.resize(gray, diff_size, interpolation=cv2.INTER_AREA)
        if prev_gray is not None:
            diff_img = cv2.absdiff(prev_gray, gray)
            _, mask = cv2.threshold(diff_img, int(diff_thresh), 255, cv2.THRESH_BINARY)
            mask = cleanup_motion_mask_with_iters(mask, 0, 1, 1)
            lower_mask = np.zeros_like(mask)
            lower_mask[lower_y:, :] = mask[lower_y:, :]
            hit_counts += (lower_mask > 0).astype(np.uint16)
            valid_frames += 1
        prev_gray = gray
    cap.release()

    if valid_frames <= 0:
        return None, 0, lower_y

    freq = hit_counts.astype(np.float32) / float(valid_frames)
    dyn_mask = np.zeros((diff_size[1], diff_size[0]), dtype=np.uint8)
    dyn_mask[(freq >= float(hit_ratio))] = 255
    dyn_mask[:lower_y, :] = 0
    dyn_mask = cv2.morphologyEx(dyn_mask, cv2.MORPH_CLOSE, BG_DILATE_KERNEL, iterations=1)
    dyn_mask = cv2.dilate(dyn_mask, BG_DILATE_KERNEL, iterations=1)
    return dyn_mask, valid_frames, lower_y


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
        return cv2.warpAffine(
            prev_gray,
            mat,
            (prev_gray.shape[1], prev_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
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


def local_contrast_measure(diff_img, mask_patch, x, y, w, h, bg_pad=LCM_BG_PAD):
    if diff_img is None or mask_patch is None:
        return 999.0, 999.0
    pad = max(int(bg_pad), int(max(w, h) * 2))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(diff_img.shape[1], x + w + pad)
    y2 = min(diff_img.shape[0], y + h + pad)
    local_patch = diff_img[y1:y2, x1:x2].astype(np.float32)
    if local_patch.size == 0:
        return 999.0, 999.0

    active = mask_patch > 0
    target_vals = diff_img[y:y + h, x:x + w][active].astype(np.float32) if np.any(active) else diff_img[y:y + h, x:x + w].astype(np.float32).reshape(-1)
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
    active_vals = gray[y:y + h, x:x + w][active]
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

    def _is_bright_static_candidate(self, gray, diff_img, mask_patch, x, y, w, h, area):
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
        active_vals = gray[y:y + h, x:x + w][active]
        if active_vals.size == 0:
            return False
        diff_vals = diff_img[y:y + h, x:x + w][active] if diff_img is not None else active_vals
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

        if not self._is_bright_static_candidate(gray, diff_img, mask_patch, x, y, w, h, area):
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


def motion_rois_from_mask(
    mask,
    diff_img,
    edge_mask,
    gray,
    scale_x,
    scale_y,
    full_w,
    full_h,
    tight_roi=False,
    tight_pad=64,
    min_diff_area=MIN_DIFF_AREA,
    far_min_compactness=FAR_MIN_COMPACTNESS,
    lcm_filter=ENABLE_LCM_FILTER,
    lcm_bg_pad=LCM_BG_PAD,
    lcm_min_score=LCM_MIN_SCORE,
    lcm_min_ratio=LCM_MIN_RATIO,
    lcm_score_weight=LCM_SCORE_WEIGHT,
    lcm_require_both=False,
    max_rois=MAX_ROIS_PER_FRAME,
    flicker_suppressor=None,
    frame_idx=0,
):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        if is_vertical_strip_box((w, h)):
            continue
        mask_patch = mask[y:y + h, x:x + w]
        area = int(cv2.countNonZero(mask_patch))
        if area < int(min_diff_area):
            continue
        texture = edge_density(edge_mask, x, y, w, h)
        roi_risk = edge_texture_risk(texture)
        active = mask_patch > 0
        local_diff = float(diff_img[y:y + h, x:x + w][active].mean()) if np.any(active) else 0.0
        if local_diff < MIN_LOCAL_DIFF_MEAN:
            continue
        lcm_score, lcm_ratio = local_contrast_measure(diff_img, mask_patch, x, y, w, h, lcm_bg_pad)
        if lcm_filter:
            if lcm_require_both:
                if lcm_score < float(lcm_min_score) or lcm_ratio < float(lcm_min_ratio):
                    continue
            elif lcm_score < float(lcm_min_score) and lcm_ratio < float(lcm_min_ratio):
                continue
        if should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
            continue
        compactness = area / max(1.0, float(w * h))

        is_far_tiny = (
            area <= MAX_DIFF_AREA
            and w <= MAX_DIFF_BOX_W
            and h <= MAX_DIFF_BOX_H
            and compactness >= float(far_min_compactness)
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
        score += float(lcm_score_weight) * max(0.0, min(float(lcm_score), 8.0))
        score -= EDGE_DENSITY_SCORE_PENALTY * roi_risk

        fx, fy = int(math.floor(x * scale_x)), int(math.floor(y * scale_y))
        fx2 = int(math.ceil((x + w) * scale_x))
        fy2 = int(math.ceil((y + h) * scale_y))
        fw, fh = max(1, fx2 - fx), max(1, fy2 - fy)
        cx, cy = fx + fw // 2, fy + fh // 2
        if tight_roi:
            pad = max(0, int(tight_pad))
            sx1 = max(0, fx - pad)
            sy1 = max(0, fy - pad)
            sx2 = min(full_w, fx2 + pad)
            sy2 = min(full_h, fy2 + pad)
            sw, sh = sx2 - sx1, sy2 - sy1
            if sw <= CROP_SIZE and sh <= CROP_SIZE:
                px = (CROP_SIZE - sw) // 2
                py = (CROP_SIZE - sh) // 2
                temp_rois.append((sx1, sy1, sx2, sy2, score, px, py, roi_risk))
                continue

        rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
        temp_rois.append((rx1, ry1, rx2, ry2, score, roi_risk))
    temp_rois = apply_spatial_chaos_filter(temp_rois, full_w, full_h)
    temp_rois.sort(key=lambda r: r[4], reverse=True)
    rois = merge_nearby_boxes(temp_rois, dist_thresh=300)
    rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    return prioritize_diverse_rois(rois, full_w, full_h, max_rois=max_rois)


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
        if frame is None:
            return None
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
        if trk.get("template") is None:
            return 0.65
        patch = self._crop_template(frame, det)
        if patch is None:
            return 0.55
        if float(np.std(patch)) < 1.0 or float(np.std(trk["template"])) < 1.0:
            return 0.55
        raw = cv2.matchTemplate(patch, trk["template"], cv2.TM_CCOEFF_NORMED)[0][0]
        if not np.isfinite(raw):
            return 0.55
        return self._clamp01((float(raw) + 1.0) * 0.5)

    def _predict_center(self, trk, frame_idx):
        dt = max(1, int(frame_idx) - int(trk["last_frame"]))
        return trk["cx"] + trk["vx"] * dt, trk["cy"] + trk["vy"] * dt, dt

    def _predicted_box(self, trk, frame_idx):
        cx, cy, _ = self._predict_center(trk, frame_idx)
        w, h = trk["w"], trk["h"]
        return [
            int(round(cx - w * 0.5)),
            int(round(cy - h * 0.5)),
            int(round(cx + w * 0.5)),
            int(round(cy + h * 0.5)),
        ]

    def _dynamic_gate(self, trk, frame_idx):
        _, _, dt = self._predict_center(trk, frame_idx)
        speed_px_frame = math.sqrt(trk["vx"] * trk["vx"] + trk["vy"] * trk["vy"])
        width_px = max(1.0, trk["w"])
        physical_motion_px = self.max_speed_mps * (dt / self.fps) * width_px / max(ROUGH_TARGET_WIDTH_M, 0.01)
        gate = self.max_dist + 0.35 * speed_px_frame * dt + 0.75 * physical_motion_px + 12.0 * trk["misses"]
        return max(self.max_dist, min(self.max_gate, gate))

    def _motion_score(self, trk, det, frame_idx):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk["last_frame"]))
        nvx = (cx - trk["cx"]) / dt
        nvy = (cy - trk["cy"]) / dt
        old_speed = math.sqrt(trk["vx"] * trk["vx"] + trk["vy"] * trk["vy"])
        new_speed = math.sqrt(nvx * nvx + nvy * nvy)
        if old_speed < 2.0 or new_speed < 2.0:
            return 0.75
        speed_change = abs(new_speed - old_speed) / max(old_speed, 8.0)
        dot = trk["vx"] * nvx + trk["vy"] * nvy
        cos_v = dot / max(old_speed * new_speed, 1e-6)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_v))))
        return self._clamp01(1.0 - 0.35 * min(1.0, speed_change) - 0.45 * min(1.0, angle / 150.0))

    def _trajectory_score(self, trk):
        pts = list(trk["history"])
        if len(pts) < 4:
            return 0.7
        steps = [math.sqrt((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) for i in range(1, len(pts))]
        if not steps:
            return 0.7
        median_step = float(np.median(steps))
        max_step = max(steps)
        score = 1.0
        if median_step > 1.0 and max_step > median_step * 3.0 + 35.0:
            score -= 0.35
        headings = []
        for i in range(1, len(pts)):
            dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
            if abs(dx) + abs(dy) > 1.0:
                headings.append(math.degrees(math.atan2(dy, dx)))
        if len(headings) >= 4:
            turns = [abs((headings[i] - headings[i - 1] + 180.0) % 360.0 - 180.0) for i in range(1, len(headings))]
            sharp_turns = sum(1 for t in turns if t > 130.0)
            if sharp_turns >= 2:
                score -= 0.25
        recent_hits = sum(trk["hit_history"])
        if len(trk["hit_history"]) >= self.recent_window and recent_hits < self.min_recent_hits:
            score -= 0.35
        return self._clamp01(score)

    def _adaptive_required_hits(self, trk):
        if not ENABLE_ADAPTIVE_TRACK_CONFIRM:
            return self.min_hits
        required = int(self.min_hits)
        risk = float(trk.get("bg_risk", 0.0))
        max_side = max(float(trk.get("w", 1.0)), float(trk.get("h", 1.0)))
        if risk >= TRACK_CONFIRM_RISK_THRESHOLD:
            required = max(required, int(TRACK_CONFIRM_RISK_MIN_HITS))
        elif max_side <= TRACK_CONFIRM_SMALL_BOX_MAX:
            required = max(required, int(TRACK_CONFIRM_SMALL_MIN_HITS))
        return required

    def _passes_motion_confirmation(self, trk):
        if not ENABLE_ADAPTIVE_TRACK_CONFIRM:
            return True
        f = self._reference_features(trk)
        duration = int(f["duration_frames"])
        if duration < REF_TRAJ_MIN_DURATION:
            return True
        risk = float(trk.get("bg_risk", 0.0))
        min_net = TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX if risk >= TRACK_CONFIRM_RISK_THRESHOLD else TRACK_CONFIRM_MIN_NET_MOTION_PX
        min_straight = TRACK_CONFIRM_RISK_MIN_STRAIGHTNESS if risk >= TRACK_CONFIRM_RISK_THRESHOLD else TRACK_CONFIRM_MIN_STRAIGHTNESS
        if float(f["net_displacement_px"]) < float(min_net):
            return False
        if float(f["straightness"]) < float(min_straight):
            return False
        return True

    def _reference_features(self, trk):
        pts = np.array(list(trk.get("history", [])), dtype=np.float32)
        frames = np.array(list(trk.get("frame_history", [])), dtype=np.float32)
        hit_history = list(trk.get("hit_history", []))
        duration = len(hit_history)
        valid_ratio = float(sum(hit_history)) / float(max(1, duration))
        missed_ratio = 1.0 - valid_ratio
        features = {
            "duration_frames": duration,
            "valid_ratio": valid_ratio,
            "missed_ratio": missed_ratio,
            "path_length_px": 0.0,
            "net_displacement_px": 0.0,
            "straightness": 0.0,
            "cv_speed": 0.0,
            "heading_std_deg": 0.0,
            "cv_residual_rmse": 0.0,
            "lateral_jitter_ratio": 0.0,
            "mean_speed_px_s": 0.0,
            "std_speed_px_s": 0.0,
        }
        if len(pts) < 2:
            return features

        if len(frames) != len(pts):
            frames = np.arange(len(pts), dtype=np.float32)
        dxy = np.diff(pts, axis=0)
        steps = np.linalg.norm(dxy, axis=1)
        path_length = float(np.sum(steps))
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        features["path_length_px"] = path_length
        features["net_displacement_px"] = net_disp
        features["straightness"] = float(net_disp / max(path_length, 1e-6))

        dt_frames = np.diff(frames)
        dt = np.maximum(dt_frames / max(self.fps, 1e-6), 1.0 / max(self.fps, 1e-6))
        speeds = steps / dt
        if len(speeds):
            mean_speed = float(np.mean(speeds))
            std_speed = float(np.std(speeds))
            features["mean_speed_px_s"] = mean_speed
            features["std_speed_px_s"] = std_speed
            features["cv_speed"] = float(std_speed / max(abs(mean_speed), 1e-6))

        if len(dxy) >= 2:
            headings = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
            features["heading_std_deg"] = float(np.std(np.degrees(np.unwrap(np.radians(headings)))))

        if len(pts) >= 3:
            times = (frames - frames[0]) / max(self.fps, 1e-6)
            design = np.vstack([np.ones_like(times), times]).T
            try:
                coef_x, *_ = np.linalg.lstsq(design, pts[:, 0], rcond=None)
                coef_y, *_ = np.linalg.lstsq(design, pts[:, 1], rcond=None)
                fit_pts = np.column_stack([design @ coef_x, design @ coef_y])
                residuals = np.linalg.norm(pts - fit_pts, axis=1)
                features["cv_residual_rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
            except np.linalg.LinAlgError:
                pass

            centered = pts - np.mean(pts, axis=0, keepdims=True)
            if float(np.linalg.norm(centered)) > 1e-6:
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    main_dir = vh[0]
                    normal = np.array([-main_dir[1], main_dir[0]], dtype=np.float32)
                    lateral = centered @ normal
                    features["lateral_jitter_ratio"] = float(np.std(lateral) / max(net_disp, 1e-6))
                except np.linalg.LinAlgError:
                    pass
        return features

    def _reference_label_score(self, trk):
        f = self._reference_features(trk)
        reasons = []
        duration = int(f["duration_frames"])
        valid_ratio = float(f["valid_ratio"])
        missed_ratio = float(f["missed_ratio"])
        net_disp = float(f["net_displacement_px"])

        if duration < REF_TRAJ_MIN_DURATION:
            return "uncertain", 0.15, f, [f"duration {duration} < {REF_TRAJ_MIN_DURATION}"]
        if valid_ratio < REF_TRAJ_MIN_VALID_RATIO:
            return "noise_like", 0.10, f, [f"valid_ratio {valid_ratio:.2f} < {REF_TRAJ_MIN_VALID_RATIO:.2f}"]
        if missed_ratio > REF_TRAJ_MAX_MISSED_RATIO:
            return "noise_like", 0.10, f, [f"missed_ratio {missed_ratio:.2f} > {REF_TRAJ_MAX_MISSED_RATIO:.2f}"]
        if (
            REF_TRAJ_REJECT_STATIONARY
            and duration >= REF_TRAJ_STATIONARY_MIN_DURATION
            and net_disp < REF_TRAJ_STATIONARY_MAX_DISP
        ):
            return "stationary_like", 0.20, f, [f"net_disp {net_disp:.1f} < {REF_TRAJ_STATIONARY_MAX_DISP:.1f}"]

        score = 0.25
        if duration >= REF_TRAJ_MIN_DURATION:
            score += 0.12
            reasons.append("duration")
        if valid_ratio >= REF_TRAJ_MIN_VALID_RATIO:
            score += 0.16
            reasons.append("valid")
        if missed_ratio <= REF_TRAJ_MAX_MISSED_RATIO:
            score += 0.08
            reasons.append("missed")
        if float(f["straightness"]) >= REF_TRAJ_MIN_STRAIGHTNESS:
            score += 0.10
            reasons.append("straight")
        if float(f["cv_speed"]) <= REF_TRAJ_MAX_CV_SPEED:
            score += 0.10
            reasons.append("speed")
        if float(f["heading_std_deg"]) <= REF_TRAJ_MAX_HEADING_STD:
            score += 0.08
            reasons.append("heading")
        if float(f["lateral_jitter_ratio"]) <= REF_TRAJ_MAX_LATERAL_JITTER_RATIO:
            score += 0.08
            reasons.append("jitter")
        if float(f["cv_residual_rmse"]) <= REF_TRAJ_MAX_CV_RESIDUAL_RMSE:
            score += 0.08
            reasons.append("cv_residual")

        bird_votes = 0
        if float(f["heading_std_deg"]) > REF_TRAJ_MAX_HEADING_STD:
            bird_votes += 1
        if float(f["cv_speed"]) > REF_TRAJ_MAX_CV_SPEED:
            bird_votes += 1
        if float(f["lateral_jitter_ratio"]) > REF_TRAJ_MAX_LATERAL_JITTER_RATIO:
            bird_votes += 1
        if float(f["cv_residual_rmse"]) > REF_TRAJ_MAX_CV_RESIDUAL_RMSE:
            bird_votes += 1
        if bird_votes >= 3:
            return "bird_like", min(0.45, score), f, ["unstable trajectory"]

        label = "target_like" if score >= REF_TRAJ_MIN_SCORE else "uncertain"
        return label, self._clamp01(score), f, reasons

    def _match_score(self, trk, det, frame, frame_idx):
        pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
        cx, cy = self._center(det)
        dist = math.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2)
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
            "id": self.next_id,
            "box": det[:4],
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "vx": 0.0,
            "vy": 0.0,
            "hits": 1,
            "misses": 0,
            "last_frame": int(frame_idx),
            "score": 0.35 + 0.35 * self._clamp01(self._det_score(det)),
            "bg_risk": self._det_risk(det),
            "yolo_hits": 1,
            "history": deque([(cx, cy)], maxlen=self.recent_window),
            "frame_history": deque([int(frame_idx)], maxlen=self.recent_window),
            "hit_history": deque([1], maxlen=self.recent_window),
            "template": self._crop_template(frame, det),
        }
        self.next_id += 1
        self.trackers.append(trk)

    def _update_track(self, trk, det, frame, frame_idx, match_score, tmpl_score):
        cx, cy = self._center(det)
        dt = max(1, int(frame_idx) - int(trk["last_frame"]))
        nvx = (cx - trk["cx"]) / dt
        nvy = (cy - trk["cy"]) / dt
        alpha = 0.55
        trk["vx"] = alpha * nvx + (1.0 - alpha) * trk["vx"]
        trk["vy"] = alpha * nvy + (1.0 - alpha) * trk["vy"]
        trk["cx"], trk["cy"] = cx, cy
        trk["w"], trk["h"] = self._box_wh(det)
        trk["box"] = det[:4]
        trk["last_frame"] = int(frame_idx)
        trk["hits"] += 1
        trk["yolo_hits"] = trk.get("yolo_hits", 0) + 1
        trk["misses"] = 0
        trk["score"] = self._clamp01(0.75 * trk["score"] + 0.25 * match_score)
        trk["bg_risk"] = self._clamp01(0.70 * float(trk.get("bg_risk", 0.0)) + 0.30 * self._det_risk(det))
        trk["history"].append((cx, cy))
        trk["frame_history"].append(int(frame_idx))
        trk["hit_history"].append(1)
        new_template = self._crop_template(frame, det)
        if new_template is not None:
            if trk.get("template") is None:
                trk["template"] = new_template
            elif tmpl_score >= 0.45 and match_score >= 0.45:
                trk["template"] = cv2.addWeighted(trk["template"], 0.85, new_template, 0.15, 0)

    def _miss_track(self, trk):
        trk["misses"] += 1
        trk["vx"] *= MISS_VELOCITY_DECAY
        trk["vy"] *= MISS_VELOCITY_DECAY
        trk["score"] = self._clamp01(trk["score"] * 0.92)
        trk["hit_history"].append(0)

    def _basic_is_confirmed(self, trk):
        recent_hits = sum(trk["hit_history"])
        traj_score = self._trajectory_score(trk)
        required_hits = self._adaptive_required_hits(trk)
        required_direct_hits = max(
            YOLO_DIRECT_CONFIRM_HITS,
            required_hits if float(trk.get("bg_risk", 0.0)) >= TRACK_CONFIRM_RISK_THRESHOLD else YOLO_DIRECT_CONFIRM_HITS,
        )
        required_direct_recent = max(YOLO_DIRECT_CONFIRM_RECENT_HITS, min(required_direct_hits, self.recent_window))
        if trk.get("yolo_hits", 0) >= required_direct_hits:
            return (
                recent_hits >= required_direct_recent
                and trk["misses"] <= YOLO_DIRECT_CONFIRM_MAX_MISSES
                and trk["score"] >= YOLO_DIRECT_CONFIRM_SCORE
            )
        return (
            trk["hits"] >= required_hits
            and recent_hits >= self.min_recent_hits
            and trk["misses"] <= YOLO_TRACK_MAX_CONFIRMED_MISSES
            and trk["score"] >= self.confirm_score
            and traj_score >= TRACKER_MIN_TRAJ_SCORE
        )

    def _is_confirmed(self, trk):
        if not self._basic_is_confirmed(trk):
            return False
        if not self._passes_motion_confirmation(trk):
            return False
        if not ENABLE_REFERENCE_TRAJ_FILTER:
            return True
        label, score, _, _ = self._reference_label_score(trk)
        return label == "target_like" and score >= REF_TRAJ_MIN_SCORE

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
            if trk["misses"] <= YOLO_TRACK_MAX_SEARCH_MISSES and trk["score"] >= 0.12:
                kept.append(trk)
        self.trackers = kept
        return [t["box"] for t in self.get_confirmed_tracks(frame_idx)]

    def get_yolo_seeded_search_rois(self, frame_idx, full_w, full_h, max_rois=MAX_TRACK_ROIS_PER_FRAME, confirmed_only=False):
        rois = []
        for trk in self.trackers:
            track_age = max(0, int(frame_idx) - int(trk.get("last_frame", frame_idx)))
            if track_age > TRACK_SEARCH_MAX_AGE_FRAMES:
                continue
            if trk["misses"] > YOLO_TRACK_MAX_SEARCH_MISSES:
                continue
            if confirmed_only and not self._is_confirmed(trk):
                continue
            recent_hits = sum(trk.get("hit_history", []))
            if (
                trk.get("yolo_hits", 0) < TRACK_SEARCH_MIN_YOLO_HITS
                or recent_hits < TRACK_SEARCH_MIN_RECENT_HITS
                or float(trk.get("score", 0.0)) < TRACK_SEARCH_MIN_SCORE
            ):
                continue
            box = self._predicted_box(trk, frame_idx) if 0 < trk["misses"] <= TRACK_SEARCH_PREDICT_MAX_MISSES else trk["box"]
            cx = (float(box[0]) + float(box[2])) * 0.5
            cy = (float(box[1]) + float(box[3])) * 0.5
            rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
            priority = 1000.0 + float(trk.get("score", 0.0)) - 20.0 * float(trk["misses"])
            rois.append((rx1, ry1, rx2, ry2, priority))
        rois = merge_nearby_boxes(rois, dist_thresh=160)
        rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
        return rois[:max_rois]

    def get_confirmed_tracks(self, frame_idx=0):
        confirmed = []
        for trk in self.trackers:
            track_age = max(0, int(frame_idx) - int(trk.get("last_frame", frame_idx)))
            if track_age > TRACK_SEARCH_MAX_AGE_FRAMES:
                continue
            if not self._is_confirmed(trk):
                continue
            box = trk["box"]
            history = list(trk["history"])
            label, rule_score, features, reasons = self._reference_label_score(trk)
            confirmed.append({
                "id": trk["id"],
                "box": box,
                "history": history,
                "score": trk["score"],
                "traj_rule_label": label,
                "traj_rule_score": rule_score,
                "traj_rule_features": features,
                "traj_rule_reasons": ";".join(reasons[:4]),
            })
            confirmed[-1]["live"] = trk["misses"] == 0 and track_age <= TRACK_LIVE_MAX_AGE_FRAMES
            confirmed[-1]["misses"] = trk["misses"]
            confirmed[-1]["age"] = track_age
        return confirmed


class PtDetector:
    def __init__(self, model_path, conf=CONF_THRESH, iou=0.45, imgsz=640, device=None):
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device_arg = device
        self.device = None if device in (None, "", "auto") else device
        self.backend = "yolov8"

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
        except Exception as exc:
            print(f"YOLOv8 loader failed, trying YOLOv5 loader: {exc}", flush=True)
            self.backend = "yolov5"
            self.model = self._load_yolov5(model_path)

    def _load_yolov5(self, model_path):
        import torch

        if self.device_arg in (None, "", "auto"):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._ensure_ipython_display_stub()
        model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=model_path,
            autoshape=True,
            trust_repo=True,
            force_reload=False,
            verbose=False,
            device=self.device,
        )
        model.conf = self.conf
        model.iou = self.iou
        model.max_det = 20
        return model

    @staticmethod
    def _ensure_ipython_display_stub():
        try:
            from IPython.display import display  # noqa: F401
            return
        except Exception:
            pass
        ipython_mod = types.ModuleType("IPython")
        display_mod = types.ModuleType("IPython.display")

        def display(*args, **kwargs):
            return None

        ipython_mod.get_ipython = lambda: None
        ipython_mod.version_info = (0, 0, 0)
        display_mod.display = display
        ipython_mod.display = display_mod
        sys.modules.setdefault("IPython", ipython_mod)
        sys.modules.setdefault("IPython.display", display_mod)

    def infer(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return []
        if self.backend == "yolov5":
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.model(img_rgb, size=self.imgsz)
            pred = results.xyxy[0].detach().cpu().numpy()
            if pred.size == 0:
                return []
            xyxy = pred[:, :4]
            confs = pred[:, 4]
        else:
            result = self.model.predict(
                img_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
        dets = []
        h, w = img_bgr.shape[:2]
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = box.tolist()
            dets.append([
                int(max(0, min(w - 1, round(x1)))),
                int(max(0, min(h - 1, round(y1)))),
                int(max(0, min(w, round(x2)))),
                int(max(0, min(h, round(y2)))),
                float(score),
            ])
        return dets


def draw_boxes(frame, draw_boxes, confirmed, frame_idx, fps_text=""):
    out = frame.copy()
    for x1, y1, x2, y2, color, text in draw_boxes:
        thickness = 1 if text in ("ROI", "TRKROI") else 2
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(out, text, (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    for box in confirmed:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(out, "TARGET", (int(x1), max(26, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"frame {frame_idx} {fps_text}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def open_mask_writer(path, fps, size):
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, False)
    if writer.isOpened():
        return writer
    writer.release()
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size, True)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create mask video: {path}")
    return writer


def write_mask_frame(writer, mask, size):
    if writer is None:
        return
    if mask is None:
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    elif mask.shape[1] != size[0] or mask.shape[0] != size[1]:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 2:
        writer.write(mask)
    else:
        writer.write(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))


def det_to_yolo_line(det, img_w, img_h):
    x1, y1, x2, y2 = [float(v) for v in det[:4]]
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    cls_id = int(det[5]) if len(det) > 5 else 0
    return f"{cls_id} {cx / img_w:.6f} {cy / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def prepare_infer_roi_dataset(root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for name in ("images", "labels", "preview"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("uav\n", encoding="utf-8")
    (root / "labels" / "classes.txt").write_text("uav\n", encoding="utf-8")
    return root


def save_infer_roi_sample(root, sample_idx, frame_idx, roi_idx, infer_roi, label_lines, dets, jpeg_quality=95):
    root = Path(root)
    stem = f"frame_{int(frame_idx):06d}_roi{int(roi_idx):02d}_{int(sample_idx):06d}"
    image_path = root / "images" / f"{stem}.jpg"
    label_path = root / "labels" / f"{stem}.txt"
    cv2.imwrite(str(image_path), infer_roi)
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    if sample_idx <= 32:
        preview = infer_roi.copy()
        for det in dets:
            x1, y1, x2, y2 = [int(round(float(v))) for v in det[:4]]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"f{frame_idx} r{roi_idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(root / "preview" / f"{stem}.jpg"), preview)


def make_image_contact_sheet(image_dir, output_path, cols=4, thumb_w=320):
    images = sorted(Path(image_dir).glob("*.jpg"))[:32]
    thumbs = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = max(1, int(round(h * thumb_w / max(1, w))))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % cols:
        thumbs.append(blank.copy())
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def run(args):
    global ENABLE_GRAY_NOISE_SUPPRESSOR, TRACK_SEARCH_MIN_YOLO_HITS
    global TRACK_SEARCH_MIN_RECENT_HITS, TRACK_SEARCH_MIN_SCORE
    global ENABLE_REFERENCE_TRAJ_FILTER, REF_TRAJ_MIN_SCORE, REF_TRAJ_MIN_DURATION
    global REF_TRAJ_MIN_VALID_RATIO, REF_TRAJ_MAX_MISSED_RATIO, REF_TRAJ_MIN_STRAIGHTNESS
    global REF_TRAJ_MAX_CV_SPEED, REF_TRAJ_MAX_HEADING_STD
    global REF_TRAJ_MAX_LATERAL_JITTER_RATIO, REF_TRAJ_MAX_CV_RESIDUAL_RMSE
    global REF_TRAJ_REJECT_STATIONARY, REF_TRAJ_STATIONARY_MIN_DURATION, REF_TRAJ_STATIONARY_MAX_DISP
    global DIFF_THRESH, MIN_LOCAL_DIFF_MEAN, MAX_LOCAL_GRAY_STD, BRIGHT_SPOT_MAX_AREA
    global MAX_DIFF_AREA, MAX_DIFF_BOX_W, MAX_DIFF_BOX_H
    global NEAR_MAX_DIFF_AREA, NEAR_MAX_DIFF_BOX_W, NEAR_MAX_DIFF_BOX_H
    global NEAR_MIN_COMPACTNESS, NEAR_EDGE_DENSITY_THRESH, EDGE_DENSITY_THRESH
    global ENABLE_SPATIAL_CHAOS_FILTER, CHAOS_CELL_PX, CHAOS_LOCAL_ROI_LIMIT
    global CHAOS_KEEP_PER_CELL, CHAOS_SCORE_PENALTY
    global ENABLE_LOWER_CHAOS_SUPPRESSOR, LOWER_CHAOS_Y_RATIO, LOWER_CHAOS_LOCAL_ROI_LIMIT
    global LOWER_CHAOS_KEEP_PER_CLUSTER, LOWER_CHAOS_SCORE_PENALTY, LOWER_CHAOS_MIN_RISK
    global ENABLE_ADAPTIVE_TRACK_CONFIRM
    ENABLE_GRAY_NOISE_SUPPRESSOR = bool(args.gray_noise_suppressor)
    TRACK_SEARCH_MIN_YOLO_HITS = int(args.track_search_min_yolo_hits)
    TRACK_SEARCH_MIN_RECENT_HITS = int(args.track_search_min_recent_hits)
    TRACK_SEARCH_MIN_SCORE = float(args.track_search_min_score)
    ENABLE_SPATIAL_CHAOS_FILTER = bool(args.spatial_chaos_filter)
    CHAOS_CELL_PX = int(args.chaos_cell_px)
    CHAOS_LOCAL_ROI_LIMIT = int(args.chaos_local_roi_limit)
    CHAOS_KEEP_PER_CELL = int(args.chaos_keep_per_cell)
    CHAOS_SCORE_PENALTY = float(args.chaos_score_penalty)
    ENABLE_LOWER_CHAOS_SUPPRESSOR = bool(args.lower_chaos_suppressor)
    LOWER_CHAOS_Y_RATIO = float(args.lower_chaos_y_ratio)
    LOWER_CHAOS_LOCAL_ROI_LIMIT = int(args.lower_chaos_local_roi_limit)
    LOWER_CHAOS_KEEP_PER_CLUSTER = int(args.lower_chaos_keep_per_cluster)
    LOWER_CHAOS_SCORE_PENALTY = float(args.lower_chaos_score_penalty)
    LOWER_CHAOS_MIN_RISK = float(args.lower_chaos_min_risk)
    ENABLE_ADAPTIVE_TRACK_CONFIRM = bool(args.adaptive_track_confirm)
    ENABLE_REFERENCE_TRAJ_FILTER = bool(args.trajectory_rule_filter)
    REF_TRAJ_MIN_SCORE = float(args.traj_rule_min_score)
    REF_TRAJ_MIN_DURATION = int(args.traj_rule_min_duration)
    REF_TRAJ_MIN_VALID_RATIO = float(args.traj_rule_min_valid_ratio)
    REF_TRAJ_MAX_MISSED_RATIO = float(args.traj_rule_max_missed_ratio)
    REF_TRAJ_MIN_STRAIGHTNESS = float(args.traj_rule_min_straightness)
    REF_TRAJ_MAX_CV_SPEED = float(args.traj_rule_max_cv_speed)
    REF_TRAJ_MAX_HEADING_STD = float(args.traj_rule_max_heading_std)
    REF_TRAJ_MAX_LATERAL_JITTER_RATIO = float(args.traj_rule_max_lateral_jitter_ratio)
    REF_TRAJ_MAX_CV_RESIDUAL_RMSE = float(args.traj_rule_max_cv_residual_rmse)
    REF_TRAJ_REJECT_STATIONARY = bool(args.traj_rule_reject_stationary)
    REF_TRAJ_STATIONARY_MIN_DURATION = int(args.traj_rule_stationary_min_duration)
    REF_TRAJ_STATIONARY_MAX_DISP = float(args.traj_rule_stationary_max_disp)
    if args.layer_mode == "low":
        DIFF_THRESH = 8
        MIN_LOCAL_DIFF_MEAN = 10.0
        MAX_LOCAL_GRAY_STD = 30.0
        BRIGHT_SPOT_MAX_AREA = 140
        MAX_DIFF_AREA = 600
        MAX_DIFF_BOX_W = 120
        MAX_DIFF_BOX_H = 120
        NEAR_MAX_DIFF_AREA = 6000
        NEAR_MAX_DIFF_BOX_W = 320
        NEAR_MAX_DIFF_BOX_H = 320
        NEAR_MIN_COMPACTNESS = 0.06
        NEAR_EDGE_DENSITY_THRESH = 0.30
        EDGE_DENSITY_THRESH = 0.18
        ENABLE_GRAY_NOISE_SUPPRESSOR = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_gray, bg_tol = None, None
    if args.static_bg_mask:
        bg_gray, bg_tol, bg_sample_count, bg_frame_span = build_initial_bg_model(
            args.video,
            args.bg_seconds,
            (DIFF_W, DIFF_H),
            args.bg_abs_delta,
            args.bg_std_mult,
            args.bg_max_delta,
            args.bg_max_samples,
        )
        print(
            f"Static background ready: seconds={args.bg_seconds} "
            f"frames={bg_frame_span} samples={bg_sample_count} "
            f"abs_delta={args.bg_abs_delta} std_mult={args.bg_std_mult}",
            flush=True,
        )

    lower_dynamic_mask = None
    if args.lower_dynamic_mask:
        lower_dynamic_mask, lower_dyn_frames, lower_dyn_y = build_lower_dynamic_mask(
            args.video,
            args.lower_dynamic_mask_seconds,
            (DIFF_W, DIFF_H),
            args.lower_dynamic_mask_y_ratio,
            args.lower_dynamic_mask_diff_thresh,
            args.lower_dynamic_mask_hit_ratio,
        )
        masked_pixels = int(cv2.countNonZero(lower_dynamic_mask)) if lower_dynamic_mask is not None else 0
        print(
            f"Lower dynamic mask ready: seconds={args.lower_dynamic_mask_seconds} "
            f"frames={lower_dyn_frames} lower_y={lower_dyn_y} masked_pixels={masked_pixels} "
            f"hit_ratio={args.lower_dynamic_mask_hit_ratio}",
            flush=True,
        )

    output_size = (width, height)
    if args.output_scale != 1.0:
        output_size = (max(1, int(width * args.output_scale)), max(1, int(height * args.output_scale)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot create output video: {args.output}")
    mask_writer = open_mask_writer(args.mask_output, fps, (DIFF_W, DIFF_H))
    infer_roi_dataset_root = prepare_infer_roi_dataset(args.infer_roi_dataset) if args.save_infer_roi_dataset else None
    infer_roi_saved = 0

    csv_file = open(args.csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame", "type", "track_id", "x1", "y1", "x2", "y2", "score", "range_m",
            "traj_rule_label", "traj_rule_score", "valid_ratio", "missed_ratio",
            "straightness", "cv_speed", "heading_std_deg", "lateral_jitter_ratio",
            "cv_residual_rmse", "traj_rule_reasons",
        ],
    )
    csv_writer.writeheader()

    detector = PtDetector(args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device)
    tracker = TrajectoryFilter(
        max_dist=TRACKER_MAX_DIST,
        min_hits=TRACKER_MIN_HITS,
        recent_window=TRACKER_RECENT_WINDOW,
        min_recent_hits=TRACKER_MIN_RECENT_HITS,
        confirm_score=TRACKER_CONFIRM_SCORE,
        template_size=TRACKER_TEMPLATE_SIZE,
        max_gate=TRACKER_MAX_GATE,
        max_speed_mps=TRACKER_MAX_SPEED_MPS,
        fps=fps,
    )
    flicker_suppressor = (
        FixedFlickerSuppressor(
            cell_px=args.flicker_cell_px,
            window_frames=args.flicker_window_frames,
            cooldown_frames=args.flicker_cooldown_frames,
            min_hits=args.flicker_min_hits,
            stationary_px=args.flicker_stationary_px,
            max_area=args.flicker_max_area,
            max_box=args.flicker_max_box,
            min_bright_ratio=args.flicker_min_bright_ratio,
            min_bright_delta=args.flicker_min_bright_delta,
        )
        if args.flicker_suppressor
        else None
    )

    frame_t1 = None
    median_bg_samples = deque(maxlen=max(int(args.median_bg_window), int(args.median_bg_init_frames)))
    median_bg_gray = None
    frame_idx = 0
    t0 = time.time()
    processed = 0
    latest_fusion_mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        draw_list = []
        raw_boxes = []
        got_inference = False
        fusion_mask = None
        processed_for_detection = False

        if frame_idx % args.process_every == 0:
            processed_for_detection = True
            gray = frame_to_gray(frame)
            if gray.shape[1] == DIFF_W and gray.shape[0] == DIFF_H:
                gray_small = gray
            else:
                gray_small = cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = w / float(DIFF_W), h / float(DIFF_H)
            bg_change_mask = build_static_bg_change_mask(gray_small, bg_gray, bg_tol) if args.static_bg_mask else None

            rois = []
            track_rois = (
                tracker.get_yolo_seeded_search_rois(
                    frame_idx,
                    w,
                    h,
                    confirmed_only=args.track_search_confirmed_only,
                )
                if args.track_search
                else []
            )
            track_roi_keys = set()
            if args.motion_mode == "median-bg":
                if median_bg_gray is None:
                    median_bg_samples.append(gray_small.copy())
                    if len(median_bg_samples) >= int(args.median_bg_init_frames):
                        median_bg_gray = build_median_background(list(median_bg_samples))
                        print(
                            f"Median background ready: samples={len(median_bg_samples)} "
                            f"init_frames={args.median_bg_init_frames} update={args.median_bg_update}",
                            flush=True,
                        )
                else:
                    diff_img = cv2.absdiff(gray_small, median_bg_gray)
                    threshold_value, _, _ = robust_residual_threshold(
                        diff_img,
                        args.median_bg_z_thresh,
                        args.median_bg_min_thresh,
                    )
                    _, mask = cv2.threshold(diff_img, threshold_value, 255, cv2.THRESH_BINARY)
                    if bg_change_mask is not None:
                        mask = cv2.bitwise_and(mask, bg_change_mask)
                    if lower_dynamic_mask is not None:
                        mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_dynamic_mask))
                    mask = cleanup_motion_mask_with_iters(mask, args.erode_iter, args.dilate_iter, args.close_iter)
                    fusion_mask = mask
                    latest_fusion_mask = fusion_mask
                    edge_mask = build_edge_mask(gray_small)
                    rois = motion_rois_from_mask(
                        mask,
                        diff_img,
                        edge_mask,
                        gray_small,
                        scale_x,
                        scale_y,
                        w,
                        h,
                        tight_roi=args.tight_motion_roi,
                        tight_pad=args.tight_roi_pad,
                        min_diff_area=args.min_diff_area,
                        far_min_compactness=args.far_min_compactness,
                        lcm_filter=args.lcm_filter,
                        lcm_bg_pad=args.lcm_bg_pad,
                        lcm_min_score=args.lcm_min_score,
                        lcm_min_ratio=args.lcm_min_ratio,
                        lcm_score_weight=args.lcm_score_weight,
                        lcm_require_both=args.lcm_require_both,
                        max_rois=args.max_rois_per_frame,
                        flicker_suppressor=flicker_suppressor,
                        frame_idx=frame_idx,
                    )
                    if args.median_bg_update:
                        median_bg_samples.append(gray_small.copy())
                        median_bg_gray = build_median_background(list(median_bg_samples))
            elif frame_t1 is not None:
                aligned_t1 = align_previous_gray(frame_t1, gray_small)
                diff_img = cv2.absdiff(aligned_t1, gray_small)
                _, mask = cv2.threshold(diff_img, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                if bg_change_mask is not None:
                    mask = cv2.bitwise_and(mask, bg_change_mask)
                if lower_dynamic_mask is not None:
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(lower_dynamic_mask))
                mask = cleanup_motion_mask_with_iters(mask, args.erode_iter, args.dilate_iter, args.close_iter)
                fusion_mask = mask
                latest_fusion_mask = fusion_mask
                edge_mask = build_edge_mask(gray_small)
                rois = motion_rois_from_mask(
                    mask,
                    diff_img,
                    edge_mask,
                    gray_small,
                    scale_x,
                    scale_y,
                    w,
                    h,
                    tight_roi=args.tight_motion_roi,
                    tight_pad=args.tight_roi_pad,
                    min_diff_area=args.min_diff_area,
                    far_min_compactness=args.far_min_compactness,
                    lcm_filter=args.lcm_filter,
                    lcm_bg_pad=args.lcm_bg_pad,
                    lcm_min_score=args.lcm_min_score,
                    lcm_min_ratio=args.lcm_min_ratio,
                    lcm_score_weight=args.lcm_score_weight,
                    lcm_require_both=args.lcm_require_both,
                    max_rois=args.max_rois_per_frame,
                    flicker_suppressor=flicker_suppressor,
                    frame_idx=frame_idx,
                )
            if track_rois:
                if args.require_track_motion:
                    track_rois = [
                        r for r in track_rois
                        if track_roi_has_motion(r, fusion_mask, w, h, args.track_min_pixels, args.track_center_size)
                    ]
                if flicker_suppressor is not None:
                    track_rois = [
                        r for r in track_rois
                        if not flicker_suppressor.is_blocked_full_box(frame_idx, r, scale_x, scale_y)
                    ]
            if track_rois:
                if args.tight_motion_roi and fusion_mask is not None:
                    track_mask = limit_mask_to_rois(fusion_mask, track_rois, w, h)
                    track_motion_rois = motion_rois_from_mask(
                        track_mask,
                        diff_img,
                        edge_mask,
                        gray_small,
                        scale_x,
                        scale_y,
                        w,
                        h,
                        tight_roi=args.tight_motion_roi,
                        tight_pad=args.tight_roi_pad,
                        min_diff_area=args.min_diff_area,
                        far_min_compactness=args.far_min_compactness,
                        lcm_filter=args.lcm_filter,
                        lcm_bg_pad=args.lcm_bg_pad,
                        lcm_min_score=args.lcm_min_score,
                        lcm_min_ratio=args.lcm_min_ratio,
                        lcm_score_weight=args.lcm_score_weight,
                        lcm_require_both=args.lcm_require_both,
                        max_rois=args.max_rois_per_frame,
                        flicker_suppressor=flicker_suppressor,
                        frame_idx=frame_idx,
                    ) if track_mask is not None else []
                    track_motion_rois = [boost_roi_score(r, 1000.0) for r in track_motion_rois]
                    track_roi_keys = {tuple(int(v) for v in r[:4]) for r in track_motion_rois}
                    rois = merge_nearby_boxes(track_motion_rois + rois, dist_thresh=160)
                else:
                    track_roi_keys = {tuple(int(v) for v in r[:4]) for r in track_rois}
                    rois = merge_nearby_boxes(track_rois + rois, dist_thresh=160)
                rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                rois = prioritize_diverse_rois(rois, w, h, max_rois=args.max_rois_per_frame)

            frame_t1 = gray_small
            for roi_idx, roi in enumerate(rois[:args.max_rois_per_frame], 1):
                x1, y1, x2, y2 = [int(v) for v in roi[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                is_track_roi = tuple(int(v) for v in roi[:4]) in track_roi_keys
                label = "TRKROI" if is_track_roi else "ROI"
                color = (0, 255, 255) if is_track_roi else (255, 255, 255)
                draw_list.append((x1, y1, x2, y2, color, label))
                roi_risk = get_roi_risk(roi)
                use_padded_roi = args.tight_motion_roi and roi_is_padded_canvas(roi)
                if use_padded_roi:
                    infer_roi = make_padded_roi_canvas(
                        frame,
                        fusion_mask,
                        roi,
                        w,
                        h,
                        fusion=args.fusion,
                        fill_value=args.tight_roi_fill,
                    )
                else:
                    infer_roi = make_fused_roi(frame, fusion_mask, (x1, y1, x2, y2), w, h) if args.fusion else frame[y1:y2, x1:x2]
                dets = detector.infer(infer_roi)
                got_inference = True
                roi_label_lines = []
                roi_label_dets = []
                infer_h, infer_w = infer_roi.shape[:2]
                for d in dets:
                    if use_padded_roi:
                        mapped = map_padded_detection_to_frame(d, roi, w, h)
                        if mapped is None:
                            continue
                        mapped.append(float(roi_risk))
                    else:
                        mapped = [d[0] + x1, d[1] + y1, d[2] + x1, d[3] + y1, float(d[4]), float(roi_risk)]
                    if is_vertical_strip_box(mapped):
                        continue
                    if flicker_suppressor is not None and flicker_suppressor.is_blocked_full_box(frame_idx, mapped, scale_x, scale_y):
                        continue
                    roi_line = det_to_yolo_line(d, infer_w, infer_h)
                    if roi_line:
                        roi_label_lines.append(roi_line)
                        roi_label_dets.append(d)
                    raw_boxes.append(mapped)
                if infer_roi_dataset_root is not None and roi_label_lines:
                    infer_roi_saved += 1
                    save_infer_roi_sample(
                        infer_roi_dataset_root,
                        infer_roi_saved,
                        frame_idx,
                        roi_idx,
                        infer_roi,
                        roi_label_lines,
                        roi_label_dets,
                        jpeg_quality=args.roi_dataset_jpeg_quality,
                    )

        unique_raw_boxes = merge_nearby_boxes(raw_boxes, dist_thresh=100)
        for rb in unique_raw_boxes:
            draw_list.append((rb[0], rb[1], rb[2], rb[3], (0, 0, 255), "Raw"))
            csv_writer.writerow({
                "frame": frame_idx,
                "type": "raw",
                "track_id": "",
                "x1": int(rb[0]),
                "y1": int(rb[1]),
                "x2": int(rb[2]),
                "y2": int(rb[3]),
                "score": f"{float(rb[4]):.4f}" if len(rb) > 4 else "",
                "range_m": "",
            })

        if got_inference or processed_for_detection:
            tracker.update(unique_raw_boxes if got_inference else [], frame=frame, frame_idx=frame_idx)
        confirmed_tracks = tracker.get_confirmed_tracks(frame_idx)
        live_confirmed_tracks = [t for t in confirmed_tracks if t.get("live", True)]
        confirmed = [t["box"] for t in live_confirmed_tracks]
        for t in live_confirmed_tracks:
            box = t["box"]
            rng = estimate_rough_range_m(box, w)
            features = t.get("traj_rule_features", {})
            csv_writer.writerow({
                "frame": frame_idx,
                "type": "target",
                "track_id": t.get("id", ""),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "score": "",
                "range_m": rng,
                "traj_rule_label": t.get("traj_rule_label", ""),
                "traj_rule_score": f"{float(t.get('traj_rule_score', 0.0)):.4f}",
                "valid_ratio": f"{float(features.get('valid_ratio', 0.0)):.4f}",
                "missed_ratio": f"{float(features.get('missed_ratio', 0.0)):.4f}",
                "straightness": f"{float(features.get('straightness', 0.0)):.4f}",
                "cv_speed": f"{float(features.get('cv_speed', 0.0)):.4f}",
                "heading_std_deg": f"{float(features.get('heading_std_deg', 0.0)):.4f}",
                "lateral_jitter_ratio": f"{float(features.get('lateral_jitter_ratio', 0.0)):.4f}",
                "cv_residual_rmse": f"{float(features.get('cv_residual_rmse', 0.0)):.4f}",
                "traj_rule_reasons": t.get("traj_rule_reasons", ""),
            })

        processed += 1
        fps_text = ""
        if processed % 30 == 0:
            elapsed = max(1e-6, time.time() - t0)
            fps_text = f"proc_fps {processed / elapsed:.2f}"
            print(f"frame {frame_idx}/{frame_count} {fps_text}", flush=True)

        display_src = make_fused_display_frame(frame, latest_fusion_mask, w, h) if args.fusion_display else frame
        out_frame = draw_boxes(display_src, draw_list, confirmed, frame_idx, fps_text=fps_text)
        if output_size != (w, h):
            out_frame = cv2.resize(out_frame, output_size, interpolation=cv2.INTER_AREA)
        writer.write(out_frame)
        write_mask_frame(mask_writer, latest_fusion_mask, (DIFF_W, DIFF_H))

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    csv_file.close()
    print(f"Saved video: {args.output}")
    print(f"Saved csv: {args.csv}")
    if args.mask_output:
        print(f"Saved mask video: {args.mask_output}")
    if flicker_suppressor is not None:
        print(
            f"Flicker suppressor: blocked_cells={flicker_suppressor.blocked_count} "
            f"suppressed_rois={flicker_suppressor.suppressed_count}",
            flush=True,
        )
    if infer_roi_dataset_root is not None:
        make_image_contact_sheet(
            infer_roi_dataset_root / "preview",
            infer_roi_dataset_root / "preview_contact_sheet.jpg",
        )
        report = [
            "infer_roi_autolabel_dataset",
            f"video={args.video}",
            f"model={args.model}",
            f"source_csv={args.csv}",
            f"saved_positive_rois={infer_roi_saved}",
            f"images={infer_roi_dataset_root / 'images'}",
            f"labels={infer_roi_dataset_root / 'labels'}",
            f"preview={infer_roi_dataset_root / 'preview'}",
            "",
        ]
        (infer_roi_dataset_root / "roi_autolabel_report.txt").write_text("\n".join(report), encoding="utf-8")
        print(f"Saved infer ROI dataset: {infer_roi_dataset_root} samples={infer_roi_saved}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-video main-style UAV inference with a .pt YOLO model.")
    parser.add_argument("--model", default=r"E:\download\best.pt")
    parser.add_argument("--video", default=r"E:\detect uav\record_2k_pure\4.mp4")
    parser.add_argument("--output", default=r"E:\detect uav\record_2k_pure\4_best_main_infer.mp4")
    parser.add_argument("--csv", default=r"E:\detect uav\record_2k_pure\4_best_main_infer.csv")
    parser.add_argument("--mask-output", default="")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layer-mode", choices=["high", "low"], default="high")
    parser.add_argument("--process-every", type=int, default=PROCESS_EVERY_N_FRAMES)
    parser.add_argument("--motion-mode", choices=["frame-diff", "median-bg"], default=MOTION_MODE)
    parser.add_argument("--median-bg-init-frames", type=int, default=MEDIAN_BG_INIT_FRAMES)
    parser.add_argument("--median-bg-window", type=int, default=MEDIAN_BG_WINDOW)
    parser.add_argument("--median-bg-z-thresh", type=float, default=MEDIAN_BG_Z_THRESH)
    parser.add_argument("--median-bg-min-thresh", type=float, default=MEDIAN_BG_MIN_THRESH)
    parser.set_defaults(median_bg_update=MEDIAN_BG_UPDATE)
    parser.add_argument("--median-bg-update", dest="median_bg_update", action="store_true")
    parser.add_argument("--no-median-bg-update", dest="median_bg_update", action="store_false")
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--erode-iter", type=int, default=MOTION_ERODE_ITER)
    parser.add_argument("--dilate-iter", type=int, default=MOTION_DILATE_ITER)
    parser.add_argument("--close-iter", type=int, default=MOTION_CLOSE_ITER)
    parser.add_argument("--min-diff-area", type=int, default=MIN_DIFF_AREA)
    parser.add_argument("--far-min-compactness", type=float, default=FAR_MIN_COMPACTNESS)
    parser.add_argument("--max-rois-per-frame", type=int, default=MAX_ROIS_PER_FRAME)
    parser.set_defaults(spatial_chaos_filter=ENABLE_SPATIAL_CHAOS_FILTER)
    parser.add_argument("--spatial-chaos-filter", dest="spatial_chaos_filter", action="store_true")
    parser.add_argument("--no-spatial-chaos-filter", dest="spatial_chaos_filter", action="store_false")
    parser.add_argument("--chaos-cell-px", type=int, default=CHAOS_CELL_PX)
    parser.add_argument("--chaos-local-roi-limit", type=int, default=CHAOS_LOCAL_ROI_LIMIT)
    parser.add_argument("--chaos-keep-per-cell", type=int, default=CHAOS_KEEP_PER_CELL)
    parser.add_argument("--chaos-score-penalty", type=float, default=CHAOS_SCORE_PENALTY)
    parser.set_defaults(lower_chaos_suppressor=ENABLE_LOWER_CHAOS_SUPPRESSOR)
    parser.add_argument("--lower-chaos-suppressor", dest="lower_chaos_suppressor", action="store_true")
    parser.add_argument("--no-lower-chaos-suppressor", dest="lower_chaos_suppressor", action="store_false")
    parser.add_argument("--lower-chaos-y-ratio", type=float, default=LOWER_CHAOS_Y_RATIO)
    parser.add_argument("--lower-chaos-local-roi-limit", type=int, default=LOWER_CHAOS_LOCAL_ROI_LIMIT)
    parser.add_argument("--lower-chaos-keep-per-cluster", type=int, default=LOWER_CHAOS_KEEP_PER_CLUSTER)
    parser.add_argument("--lower-chaos-score-penalty", type=float, default=LOWER_CHAOS_SCORE_PENALTY)
    parser.add_argument("--lower-chaos-min-risk", type=float, default=LOWER_CHAOS_MIN_RISK)
    parser.set_defaults(adaptive_track_confirm=ENABLE_ADAPTIVE_TRACK_CONFIRM)
    parser.add_argument("--adaptive-track-confirm", dest="adaptive_track_confirm", action="store_true")
    parser.add_argument("--no-adaptive-track-confirm", dest="adaptive_track_confirm", action="store_false")
    parser.set_defaults(lcm_filter=ENABLE_LCM_FILTER)
    parser.add_argument("--lcm-filter", dest="lcm_filter", action="store_true")
    parser.add_argument("--no-lcm-filter", dest="lcm_filter", action="store_false")
    parser.add_argument("--lcm-bg-pad", type=int, default=LCM_BG_PAD)
    parser.add_argument("--lcm-min-score", type=float, default=LCM_MIN_SCORE)
    parser.add_argument("--lcm-min-ratio", type=float, default=LCM_MIN_RATIO)
    parser.add_argument("--lcm-score-weight", type=float, default=LCM_SCORE_WEIGHT)
    parser.set_defaults(lcm_require_both=False)
    parser.add_argument("--lcm-require-both", dest="lcm_require_both", action="store_true")
    parser.add_argument("--no-lcm-require-both", dest="lcm_require_both", action="store_false")
    parser.set_defaults(gray_noise_suppressor=ENABLE_GRAY_NOISE_SUPPRESSOR)
    parser.add_argument("--gray-noise-suppressor", dest="gray_noise_suppressor", action="store_true")
    parser.add_argument("--no-gray-noise-suppressor", dest="gray_noise_suppressor", action="store_false")
    parser.set_defaults(tight_motion_roi=TIGHT_MOTION_ROI)
    parser.add_argument("--tight-motion-roi", dest="tight_motion_roi", action="store_true")
    parser.add_argument("--no-tight-motion-roi", dest="tight_motion_roi", action="store_false")
    parser.add_argument("--tight-roi-pad", type=int, default=TIGHT_MOTION_ROI_PAD)
    parser.add_argument("--tight-roi-fill", type=int, default=TIGHT_MOTION_ROI_FILL)
    parser.set_defaults(fusion=True, fusion_display=False)
    parser.add_argument("--fusion", dest="fusion", action="store_true")
    parser.add_argument("--no-fusion", dest="fusion", action="store_false")
    parser.add_argument("--fusion-display", dest="fusion_display", action="store_true")
    parser.add_argument("--no-fusion-display", dest="fusion_display", action="store_false")
    parser.set_defaults(require_track_motion=FUSION_REQUIRE_TRACK_MOTION)
    parser.add_argument("--require-track-motion", dest="require_track_motion", action="store_true")
    parser.add_argument("--no-require-track-motion", dest="require_track_motion", action="store_false")
    parser.add_argument("--track-min-pixels", type=int, default=FUSION_TRACK_MIN_PIXELS)
    parser.add_argument("--track-center-size", type=int, default=FUSION_TRACK_CENTER_SIZE)
    parser.set_defaults(track_search=True)
    parser.add_argument("--track-search", dest="track_search", action="store_true")
    parser.add_argument("--no-track-search", dest="track_search", action="store_false")
    parser.set_defaults(track_search_confirmed_only=False)
    parser.add_argument("--track-search-confirmed-only", dest="track_search_confirmed_only", action="store_true")
    parser.add_argument("--no-track-search-confirmed-only", dest="track_search_confirmed_only", action="store_false")
    parser.add_argument("--track-search-min-yolo-hits", type=int, default=TRACK_SEARCH_MIN_YOLO_HITS)
    parser.add_argument("--track-search-min-recent-hits", type=int, default=TRACK_SEARCH_MIN_RECENT_HITS)
    parser.add_argument("--track-search-min-score", type=float, default=TRACK_SEARCH_MIN_SCORE)
    parser.set_defaults(trajectory_rule_filter=ENABLE_REFERENCE_TRAJ_FILTER)
    parser.add_argument("--trajectory-rule-filter", dest="trajectory_rule_filter", action="store_true")
    parser.add_argument("--no-trajectory-rule-filter", dest="trajectory_rule_filter", action="store_false")
    parser.add_argument("--traj-rule-min-score", type=float, default=REF_TRAJ_MIN_SCORE)
    parser.add_argument("--traj-rule-min-duration", type=int, default=REF_TRAJ_MIN_DURATION)
    parser.add_argument("--traj-rule-min-valid-ratio", type=float, default=REF_TRAJ_MIN_VALID_RATIO)
    parser.add_argument("--traj-rule-max-missed-ratio", type=float, default=REF_TRAJ_MAX_MISSED_RATIO)
    parser.add_argument("--traj-rule-min-straightness", type=float, default=REF_TRAJ_MIN_STRAIGHTNESS)
    parser.add_argument("--traj-rule-max-cv-speed", type=float, default=REF_TRAJ_MAX_CV_SPEED)
    parser.add_argument("--traj-rule-max-heading-std", type=float, default=REF_TRAJ_MAX_HEADING_STD)
    parser.add_argument("--traj-rule-max-lateral-jitter-ratio", type=float, default=REF_TRAJ_MAX_LATERAL_JITTER_RATIO)
    parser.add_argument("--traj-rule-max-cv-residual-rmse", type=float, default=REF_TRAJ_MAX_CV_RESIDUAL_RMSE)
    parser.set_defaults(traj_rule_reject_stationary=REF_TRAJ_REJECT_STATIONARY)
    parser.add_argument("--traj-rule-reject-stationary", dest="traj_rule_reject_stationary", action="store_true")
    parser.add_argument("--no-traj-rule-reject-stationary", dest="traj_rule_reject_stationary", action="store_false")
    parser.add_argument("--traj-rule-stationary-min-duration", type=int, default=REF_TRAJ_STATIONARY_MIN_DURATION)
    parser.add_argument("--traj-rule-stationary-max-disp", type=float, default=REF_TRAJ_STATIONARY_MAX_DISP)
    parser.set_defaults(static_bg_mask=ENABLE_STATIC_BG_MASK)
    parser.add_argument("--static-bg-mask", dest="static_bg_mask", action="store_true")
    parser.add_argument("--no-static-bg-mask", dest="static_bg_mask", action="store_false")
    parser.add_argument("--bg-seconds", type=float, default=STATIC_BG_SECONDS)
    parser.add_argument("--bg-abs-delta", type=float, default=STATIC_BG_ABS_DELTA)
    parser.add_argument("--bg-std-mult", type=float, default=STATIC_BG_STD_MULT)
    parser.add_argument("--bg-max-delta", type=float, default=STATIC_BG_MAX_DELTA)
    parser.add_argument("--bg-max-samples", type=int, default=STATIC_BG_MAX_SAMPLES)
    parser.set_defaults(lower_dynamic_mask=ENABLE_LOWER_DYNAMIC_MASK)
    parser.add_argument("--lower-dynamic-mask", dest="lower_dynamic_mask", action="store_true")
    parser.add_argument("--no-lower-dynamic-mask", dest="lower_dynamic_mask", action="store_false")
    parser.add_argument("--lower-dynamic-mask-seconds", type=float, default=LOWER_DYNAMIC_MASK_SECONDS)
    parser.add_argument("--lower-dynamic-mask-y-ratio", type=float, default=LOWER_DYNAMIC_MASK_Y_RATIO)
    parser.add_argument("--lower-dynamic-mask-diff-thresh", type=float, default=LOWER_DYNAMIC_MASK_DIFF_THRESH)
    parser.add_argument("--lower-dynamic-mask-hit-ratio", type=float, default=LOWER_DYNAMIC_MASK_HIT_RATIO)
    parser.set_defaults(flicker_suppressor=ENABLE_FLICKER_SUPPRESSOR)
    parser.add_argument("--flicker-suppressor", dest="flicker_suppressor", action="store_true")
    parser.add_argument("--no-flicker-suppressor", dest="flicker_suppressor", action="store_false")
    parser.add_argument("--flicker-cell-px", type=int, default=FLICKER_CELL_PX)
    parser.add_argument("--flicker-window-frames", type=int, default=FLICKER_WINDOW_FRAMES)
    parser.add_argument("--flicker-cooldown-frames", type=int, default=FLICKER_COOLDOWN_FRAMES)
    parser.add_argument("--flicker-min-hits", type=int, default=FLICKER_MIN_HITS)
    parser.add_argument("--flicker-stationary-px", type=float, default=FLICKER_STATIONARY_PX)
    parser.add_argument("--flicker-max-area", type=int, default=FLICKER_MAX_AREA)
    parser.add_argument("--flicker-max-box", type=int, default=FLICKER_MAX_BOX)
    parser.add_argument("--flicker-min-bright-ratio", type=float, default=FLICKER_MIN_BRIGHT_RATIO)
    parser.add_argument("--flicker-min-bright-delta", type=float, default=FLICKER_MIN_BRIGHT_DELTA)
    parser.set_defaults(save_infer_roi_dataset=False)
    parser.add_argument("--save-infer-roi-dataset", dest="save_infer_roi_dataset", action="store_true")
    parser.add_argument("--no-save-infer-roi-dataset", dest="save_infer_roi_dataset", action="store_false")
    parser.add_argument("--infer-roi-dataset", default=r"E:\detect uav\record_2k_pure\8_infer_roi_autolabel_only_big")
    parser.add_argument("--roi-dataset-jpeg-quality", type=int, default=95)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
