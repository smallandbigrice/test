#!/usr/bin/env python3
import os
os.environ.setdefault("NO_AT_BRIDGE", "1")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faulthandler
faulthandler.enable(all_threads=True)

import cv2
import threading
import queue
import time
import numpy as np
import subprocess
import math
import csv
import re
from collections import deque
from yololib import YoloRKNN
from comms import DataSender, VideoSender

try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ==========================================
# 1
# ==========================================
ALGORITHM_VERSION = "frame-diff-rknn-static-confirm-lowmotion-20260707"
ALGORITHM_NOTE = "Frame-diff RKNN detector with high-layer gray seed ROIs, visual lock, hover recheck, and low-motion static confirmation."
N_CAM = max(1, int(os.environ.get("UAV_VIDEO_CAM_COUNT", "5")))
INIT_TIME = 5
VIDEO_TEST_PATH = os.environ.get("UAV_VIDEO_PATH", "./4.mp4")
SIMULATE_BY_VIDEOS = os.environ.get("UAV_VIDEO_TEST", "0").strip().lower() not in ("0", "false", "no")
ENABLE_CAMERA_PREBUILD_BG = os.environ.get("UAV_CAMERA_PREBUILD_BG", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
VIDEO_TEST_MAX_SECONDS = max(0.0, float(os.environ.get("UAV_VIDEO_MAX_SECONDS", "0")))
VIDEO_TEST_REALTIME = os.environ.get("UAV_VIDEO_REALTIME", "1").strip().lower() not in ("0", "false", "no")
VIDEO_TEST_LOOP = os.environ.get("UAV_VIDEO_LOOP", "1").strip().lower() not in ("0", "false", "no")
VIDEO_TEST_OUTPUT = os.environ.get("UAV_VIDEO_OUTPUT", "").strip()
VIDEO_TEST_DETECTION_CSV = os.environ.get("UAV_VIDEO_DETECTION_CSV", "").strip()
VIDEO_TEST_SAVE_ROIS_DIR = os.environ.get("UAV_VIDEO_SAVE_ROIS_DIR", "").strip()
VIDEO_TEST_SAVE_ROIS_START = max(0.0, float(os.environ.get("UAV_VIDEO_SAVE_ROIS_START", "0")))
VIDEO_TEST_SAVE_ROIS_END = max(VIDEO_TEST_SAVE_ROIS_START, float(os.environ.get("UAV_VIDEO_SAVE_ROIS_END", "0")))
VIDEO_TEST_RESIZE_W = max(0, int(os.environ.get("UAV_VIDEO_RESIZE_W", "0")))
VIDEO_TEST_RESIZE_H = max(0, int(os.environ.get("UAV_VIDEO_RESIZE_H", "0")))
DEBUG_ASYNC_TRACE = os.environ.get("UAV_DEBUG_ASYNC_TRACE", "0").strip().lower() not in ("0", "false", "no")
DRAW_INTERMEDIATE_BOXES = os.environ.get("UAV_DRAW_INTERMEDIATE_BOXES", "0").strip().lower() not in ("0", "false", "no")


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off")


ENABLE_GRU_GRAY_SEED_ROIS = _env_bool("UAV_GRU_GRAY_SEED_ROIS", True)
GRU_GRAY_SEED_THRESH = max(1, int(os.environ.get("UAV_GRU_GRAY_SEED_THRESH", "120")))
GRU_GRAY_SEED_RELAXED_THRESH = max(GRU_GRAY_SEED_THRESH, int(os.environ.get("UAV_GRU_GRAY_SEED_RELAXED_THRESH", "130")))
GRU_GRAY_SEED_MIN_AREA = max(1, int(os.environ.get("UAV_GRU_GRAY_SEED_MIN_AREA", "1")))
GRU_GRAY_SEED_MAX_AREA = max(GRU_GRAY_SEED_MIN_AREA, int(os.environ.get("UAV_GRU_GRAY_SEED_MAX_AREA", "1200")))
GRU_GRAY_SEED_BORDER = max(0, int(os.environ.get("UAV_GRU_GRAY_SEED_BORDER", "32")))
GRU_GRAY_SEED_RADIUS = max(8.0, float(os.environ.get("UAV_GRU_GRAY_SEED_RADIUS", "130")))
GRU_GRAY_SEED_MAX_NEIGHBORS = max(0, int(os.environ.get("UAV_GRU_GRAY_SEED_MAX_NEIGHBORS", "3")))
GRU_GRAY_SEED_SKY_Y_RATIO = min(1.0, max(0.0, float(os.environ.get("UAV_GRU_GRAY_SEED_SKY_Y_RATIO", "0.88"))))
GRU_GRAY_SEED_MAX_ROIS = max(0, int(os.environ.get("UAV_GRU_GRAY_SEED_MAX_ROIS", "4")))
GRU_GRAY_SEED_SCORE_BONUS = float(os.environ.get("UAV_GRU_GRAY_SEED_SCORE_BONUS", "35.0"))

ENABLE_VISUAL_LOCK = _env_bool("UAV_VISUAL_LOCK", True)
VISUAL_LOCK_EVERY_N_FRAMES = max(1, int(os.environ.get("UAV_VISUAL_LOCK_EVERY_N_FRAMES", "3")))
VISUAL_LOCK_ENTER_YOLO_HITS = max(1, int(os.environ.get("UAV_VISUAL_LOCK_ENTER_YOLO_HITS", "5")))
VISUAL_LOCK_ENTER_RECENT_HITS = max(1, int(os.environ.get("UAV_VISUAL_LOCK_ENTER_RECENT_HITS", "5")))
VISUAL_LOCK_ENTER_GREEN_STREAK = max(1, int(os.environ.get("UAV_VISUAL_LOCK_ENTER_GREEN_STREAK", "4")))
VISUAL_LOCK_ENTER_TRACK_SCORE = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_TRACK_SCORE", "0.58"))))
VISUAL_LOCK_ENTER_TRAJ_SCORE = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_TRAJ_SCORE", "0.65"))))
VISUAL_LOCK_ENTER_MEAN_CONF = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_MEAN_CONF", "0.40"))))
VISUAL_LOCK_ENTER_MIN_NET_MOTION = max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_MIN_NET_MOTION", "0.0")))
VISUAL_LOCK_ENTER_MAX_CENTER_JITTER = max(1.0, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_MAX_CENTER_JITTER", "70.0")))
VISUAL_LOCK_ENTER_MAX_BOX_JITTER_RATIO = max(0.05, float(os.environ.get("UAV_VISUAL_LOCK_ENTER_MAX_BOX_JITTER_RATIO", "0.85")))
VISUAL_LOCK_WINDOW = max(64, int(os.environ.get("UAV_VISUAL_LOCK_WINDOW", "260")))
VISUAL_LOCK_RECHECK_INTERVAL = max(1, int(os.environ.get("UAV_VISUAL_LOCK_RECHECK_INTERVAL", "30")))
VISUAL_LOCK_RECHECK_GRACE_FRAMES = max(0, int(os.environ.get("UAV_VISUAL_LOCK_RECHECK_GRACE_FRAMES", "9")))
VISUAL_LOCK_RECHECK_DECAY = min(1.0, max(0.01, float(os.environ.get("UAV_VISUAL_LOCK_RECHECK_DECAY", "0.45"))))
VISUAL_LOCK_MAX_NO_YOLO_FRAMES = max(1, int(os.environ.get("UAV_VISUAL_LOCK_MAX_NO_YOLO_FRAMES", "45")))
VISUAL_LOCK_MAX_MISSES = max(1, int(os.environ.get("UAV_VISUAL_LOCK_MAX_MISSES", "8")))
VISUAL_LOCK_MIN_RESPONSE = max(1.0, float(os.environ.get("UAV_VISUAL_LOCK_MIN_RESPONSE", "5.0")))
VISUAL_LOCK_MIN_SCORE = max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_MIN_SCORE", "12.0")))
VISUAL_LOCK_MIN_TEMPLATE = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_MIN_TEMPLATE", "0.42"))))
VISUAL_LOCK_MAX_DIST_RATIO = min(1.0, max(0.05, float(os.environ.get("UAV_VISUAL_LOCK_MAX_DIST_RATIO", "0.36"))))
VISUAL_LOCK_MAX_AREA = max(1, int(os.environ.get("UAV_VISUAL_LOCK_MAX_AREA", "220")))
VISUAL_LOCK_MIN_BOX = max(6, int(os.environ.get("UAV_VISUAL_LOCK_MIN_BOX", "32")))
VISUAL_LOCK_MAX_BOX = max(VISUAL_LOCK_MIN_BOX, int(os.environ.get("UAV_VISUAL_LOCK_MAX_BOX", "90")))
VISUAL_LOCK_TEMPLATE_WEIGHT = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_TEMPLATE_WEIGHT", "0.18"))))
VISUAL_LOCK_SMOOTH_ALPHA = min(1.0, max(0.05, float(os.environ.get("UAV_VISUAL_LOCK_SMOOTH_ALPHA", "0.28"))))
VISUAL_LOCK_CONF_ENTER = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_CONF_ENTER", "0.55"))))
VISUAL_LOCK_CONF_EXIT = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_CONF_EXIT", "0.28"))))
VISUAL_LOCK_CONF_GAIN = min(1.0, max(0.01, float(os.environ.get("UAV_VISUAL_LOCK_CONF_GAIN", "0.32"))))
VISUAL_LOCK_CONF_DECAY = min(1.0, max(0.01, float(os.environ.get("UAV_VISUAL_LOCK_CONF_DECAY", "0.22"))))
VISUAL_LOCK_TEMPLATE_UPDATE_MIN_CONF = min(1.0, max(0.0, float(os.environ.get("UAV_VISUAL_LOCK_TEMPLATE_UPDATE_MIN_CONF", "0.55"))))

ENABLE_HOVER_HOLD = _env_bool("UAV_HOVER_HOLD", True)
HOVER_ENTER_YOLO_HITS = max(1, int(os.environ.get("UAV_HOVER_ENTER_YOLO_HITS", "3")))
HOVER_ENTER_RECENT_HITS = max(1, int(os.environ.get("UAV_HOVER_ENTER_RECENT_HITS", "2")))
HOVER_RECHECK_INTERVAL_FRAMES = max(1, int(os.environ.get("UAV_HOVER_RECHECK_INTERVAL_FRAMES", "20")))
HOVER_MAX_RECHECK_MISSES = max(1, int(os.environ.get("UAV_HOVER_MAX_RECHECK_MISSES", "6")))
HOVER_ROI_SIZE = max(160, int(os.environ.get("UAV_HOVER_ROI_SIZE", "640")))
HOVER_MAX_SPEED_PX_PER_FRAME = max(0.1, float(os.environ.get("UAV_HOVER_MAX_SPEED_PX_PER_FRAME", "2.5")))
HOVER_RECENT_WINDOW = max(3, int(os.environ.get("UAV_HOVER_RECENT_WINDOW", "8")))
HOVER_MAX_RECENT_NET_MOTION_PX = max(1.0, float(os.environ.get("UAV_HOVER_MAX_RECENT_NET_MOTION_PX", "28.0")))
HOVER_MIN_TRACK_SCORE = min(1.0, max(0.0, float(os.environ.get("UAV_HOVER_MIN_TRACK_SCORE", "0.45"))))
HOVER_MAX_YOLO_GAP_FRAMES = max(
    HOVER_RECHECK_INTERVAL_FRAMES,
    int(os.environ.get(
        "UAV_HOVER_MAX_YOLO_GAP_FRAMES",
        str(HOVER_RECHECK_INTERVAL_FRAMES * HOVER_MAX_RECHECK_MISSES),
    )),
)

ENABLE_STATIC_CONFIRM = _env_bool("UAV_STATIC_CONFIRM", True)
STATIC_CONFIRM_YOLO_HITS = max(1, int(os.environ.get("UAV_STATIC_CONFIRM_YOLO_HITS", "4")))
STATIC_CONFIRM_RECENT_HITS = max(1, int(os.environ.get("UAV_STATIC_CONFIRM_RECENT_HITS", "4")))
STATIC_CONFIRM_MAX_MISSES = max(0, int(os.environ.get("UAV_STATIC_CONFIRM_MAX_MISSES", "1")))
STATIC_CONFIRM_MIN_SCORE = min(1.0, max(0.0, float(os.environ.get("UAV_STATIC_CONFIRM_MIN_SCORE", "0.40"))))
STATIC_CONFIRM_MIN_MEAN_CONF = min(1.0, max(0.0, float(os.environ.get("UAV_STATIC_CONFIRM_MIN_MEAN_CONF", "0.38"))))
STATIC_CONFIRM_MAX_CENTER_JITTER = max(1.0, float(os.environ.get("UAV_STATIC_CONFIRM_MAX_CENTER_JITTER", "24.0")))
STATIC_CONFIRM_MAX_BOX_JITTER_RATIO = max(0.05, float(os.environ.get("UAV_STATIC_CONFIRM_MAX_BOX_JITTER_RATIO", "0.65")))
STATIC_CONFIRM_RECENT_WINDOW = max(3, int(os.environ.get("UAV_STATIC_CONFIRM_RECENT_WINDOW", str(HOVER_RECENT_WINDOW))))
STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME = max(0.1, float(os.environ.get("UAV_STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME", "2.2")))
STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX = max(1.0, float(os.environ.get("UAV_STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX", "24.0")))
STATIC_CONFIRM_MAX_BG_RISK = min(1.0, max(0.0, float(os.environ.get("UAV_STATIC_CONFIRM_MAX_BG_RISK", "0.70"))))
STATIC_CONFIRM_MIN_SEED_ALIGNMENT = min(1.0, max(0.0, float(os.environ.get("UAV_STATIC_CONFIRM_MIN_SEED_ALIGNMENT", "0.25"))))


def _env_choice(name, default, choices, aliases=None):
    raw = os.environ.get(name, default).strip().lower()
    if aliases:
        raw = aliases.get(raw, raw)
    if raw not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
    return raw


SCENE_MODE = _env_choice(
    "UAV_SCENE_MODE",
    os.environ.get("UAV_DAY_NIGHT_MODE", "day"),
    {"day", "night"},
    aliases={
        "d": "day",
        "daytime": "day",
        "white": "day",
        "n": "night",
        "nighttime": "night",
        "dark": "night",
    },
)
SCENE_IS_NIGHT = SCENE_MODE == "night"
DEFAULT_CAPTURE_W = 640 if SCENE_IS_NIGHT else 2560
DEFAULT_CAPTURE_H = 480 if SCENE_IS_NIGHT else 1440
DEFAULT_DIFF_W = 640 if SCENE_IS_NIGHT else 1920
DEFAULT_DIFF_H = 480 if SCENE_IS_NIGHT else 1080
DEFAULT_DIRECT_FULL_FRAME = SCENE_IS_NIGHT
DEFAULT_MOTION_ZOOM_CROP_SIZE = 160 if SCENE_IS_NIGHT else 0
DEFAULT_FULLFRAME_FALLBACK_ONLY = SCENE_IS_NIGHT


def parse_env_list(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [v.strip() for v in raw.replace(";", ",").split(",") if v.strip()]


def get_local_ipv4s():
    ips = []
    commands = (
        ["sh", "-c", "ip -o -4 addr show scope global | awk '{print $4}'"],
        ["sh", "-c", "hostname -I"],
    )
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=1.0).decode("utf-8", "ignore")
        except Exception:
            continue
        for token in re.split(r"\s+", out.strip()):
            if not token:
                continue
            ip = token.split("/")[0]
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip) and ip not in ips:
                ips.append(ip)
    return ips


def detect_board_id():
    forced = os.environ.get("UAV_BOARD_ID", os.environ.get("BOARD_ID", "")).strip()
    if forced:
        return forced
    ips = get_local_ipv4s()
    for prefix in ("192.168.0.", "192.168.1."):
        for ip in ips:
            if ip.startswith(prefix):
                suffix = ip.rsplit(".", 1)[-1]
                if suffix.isdigit():
                    return f"BOARD_{int(suffix)}"
    for ip in ips:
        suffix = ip.rsplit(".", 1)[-1]
        if suffix.isdigit():
            return f"BOARD_{int(suffix)}"
    return "BOARD_0"


VIDEO_TEST_PATHS = parse_env_list("UAV_VIDEO_PATHS")


def get_video_test_source(cam_idx):
    if VIDEO_TEST_PATHS:
        return VIDEO_TEST_PATHS[min(int(cam_idx), len(VIDEO_TEST_PATHS) - 1)]
    return VIDEO_TEST_PATH


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
YOLO_DIRECT_CONFIRM_HITS = 3
YOLO_DIRECT_CONFIRM_RECENT_HITS = 3
YOLO_DIRECT_CONFIRM_SCORE = 0.40
YOLO_DIRECT_CONFIRM_MAX_MISSES = 0
YOLO_TRACK_MAX_CONFIRMED_MISSES = 2
YOLO_TRACK_MAX_SEARCH_MISSES = 8
TRACK_SEARCH_PREDICT_MAX_MISSES = 2
MISS_VELOCITY_DECAY = 0.55
MAX_TRACK_ROIS_PER_FRAME = 3
SHOW_INDIVIDUAL_WINDOWS = _env_bool("UAV_SHOW_WINDOWS", False)
VIDEO_SEND_EVERY_N_FRAMES = max(1, int(os.environ.get("UAV_VIDEO_SEND_EVERY_N_FRAMES", "5")))
VIDEO_STREAM_W = max(1, int(os.environ.get("UAV_VIDEO_STREAM_W", "640")))
VIDEO_STREAM_H = max(1, int(os.environ.get("UAV_VIDEO_STREAM_H", "360")))
VIDEO_STREAM_QUALITY = max(5, min(95, int(os.environ.get("UAV_VIDEO_JPEG_QUALITY", "50"))))
VIDEO_STREAM_CAM_SPECS = parse_env_list("UAV_VIDEO_STREAM_CAMS")
MAX_DRAW_BOXES = 80
CAPTURE_W = max(320, int(os.environ.get("UAV_CAPTURE_W", str(DEFAULT_CAPTURE_W))))
CAPTURE_H = max(240, int(os.environ.get("UAV_CAPTURE_H", str(DEFAULT_CAPTURE_H))))

BOARD_ID = detect_board_id()
BOARD_ROW_IDX = int(os.environ.get("UAV_BOARD_ROW_IDX", "1"))

DATA_TARGETS = {
    "gimbal": ("192.168.0.100", 8888)
}
VIDEO_TARGET_IP = os.environ.get("UAV_VIDEO_TARGET_IP", "192.168.0.200")
VIDEO_BASE_PORT = int(os.environ.get("UAV_VIDEO_BASE_PORT", "9999"))

CROP_SIZE = 640      
LOW_LAYER_ROW_IDX = 0
LAYER_MODE = os.environ.get("UAV_LAYER_MODE", "auto").strip().lower()
if LAYER_MODE not in ("auto", "high", "low"):
    raise ValueError("UAV_LAYER_MODE must be auto, high, or low")
HIGH_LAYER_MODE = (BOARD_ROW_IDX != LOW_LAYER_ROW_IDX) if LAYER_MODE == "auto" else LAYER_MODE == "high"
CONF_THRESH = 0.40 if HIGH_LAYER_MODE else 0.30
YOLO_CONF_OVERRIDE = os.environ.get("UAV_YOLO_CONF", "").strip()
YOLO_CONF_OVERRIDE = float(YOLO_CONF_OVERRIDE) if YOLO_CONF_OVERRIDE else None
CAM_LAYER_MODE_SPECS = parse_env_list("UAV_CAM_LAYER_MODES")
ENABLE_AUTO_CAM_LAYER = os.environ.get("UAV_AUTO_CAM_LAYER", "1").strip().lower() not in ("0", "false", "no")
ENABLE_TRAJECTORY_TRACKING = True
ENABLE_TRACK_SEARCH_ROIS = _env_bool("UAV_TRACK_SEARCH_ROIS", SCENE_IS_NIGHT)
TRACK_SEARCH_MIN_YOLO_HITS = 2
TRACK_SEARCH_MIN_RECENT_HITS = 2
TRACK_SEARCH_MIN_SCORE = 0.32
TRACK_SEARCH_CONFIRMED_ONLY = True if HIGH_LAYER_MODE else False
TRACK_SEARCH_MIN_NET_MOTION_PX = 6.0 if HIGH_LAYER_MODE else 0.0
ENABLE_STATIC_BG_MASK = True
ENABLE_LOW_LAYER_STATIC_BG_MASK = ENABLE_STATIC_BG_MASK
PROCESS_EVERY_N_FRAMES = max(1, int(os.environ.get("UAV_PROCESS_EVERY_N_FRAMES", "3")))
FPS_ESTIMATE_WINDOW = max(6, int(os.environ.get("UAV_FPS_ESTIMATE_WINDOW", "12")))
MAX_INFERENCE_RESULT_AGE_FRAMES = max(
    PROCESS_EVERY_N_FRAMES * 2,
    int(os.environ.get("UAV_MAX_RESULT_AGE_FRAMES", "12")),
)
RESULT_GROUP_WAIT_SEC = max(0.05, float(os.environ.get("UAV_RESULT_GROUP_WAIT_SEC", "0.35")))
TRACK_LIVE_MAX_AGE_FRAMES = MAX_INFERENCE_RESULT_AGE_FRAMES
TRACK_SEARCH_MAX_AGE_FRAMES = PROCESS_EVERY_N_FRAMES * 3
HIGH_TRACK_SEARCH_MAX_AGE_FRAMES = max(
    TRACK_SEARCH_MAX_AGE_FRAMES,
    int(os.environ.get("UAV_HIGH_TRACK_SEARCH_MAX_AGE_FRAMES", str(PROCESS_EVERY_N_FRAMES * 12))),
)
HIGH_TRACK_SEARCH_PREDICT_MAX_MISSES = max(
    TRACK_SEARCH_PREDICT_MAX_MISSES,
    int(os.environ.get("UAV_HIGH_TRACK_SEARCH_PREDICT_MAX_MISSES", "8")),
)
HIGH_TRACK_SEARCH_MAX_ROIS = max(
    MAX_TRACK_ROIS_PER_FRAME,
    int(os.environ.get("UAV_HIGH_TRACK_SEARCH_MAX_ROIS", "4")),
)
HIGH_TRACK_SEARCH_MIN_RECENT_HITS = max(1, int(os.environ.get("UAV_HIGH_TRACK_SEARCH_MIN_RECENT_HITS", "1")))
HIGH_TRACK_SEARCH_MIN_SCORE = float(os.environ.get("UAV_HIGH_TRACK_SEARCH_MIN_SCORE", "0.25"))
TRACK_SEARCH_RECHECK_MIN_GAP_FRAMES = max(0, int(os.environ.get("UAV_TRACK_RECHECK_MIN_GAP_FRAMES", "0")))
TRACK_SEARCH_VISUAL_LOCK_ONLY = _env_bool("UAV_TRACK_RECHECK_VISUAL_LOCK_ONLY", False)
LOW_BG_SAMPLE_FRAMES = 40
LOW_BG_ABS_DELTA = 12
LOW_BG_STD_MULT = 3.0
ENABLE_BG_QRANGE_TOL = os.environ.get("UAV_BG_QRANGE_TOL", "1").strip().lower() not in ("0", "false", "no")
LOW_BG_QRANGE_BASE_DELTA = float(os.environ.get("UAV_BG_QRANGE_BASE_DELTA", "10.0"))
LOW_BG_QRANGE_MULT = float(os.environ.get("UAV_BG_QRANGE_MULT", "1.8"))
LOW_BG_QRANGE_MAX_DELTA = float(os.environ.get("UAV_BG_QRANGE_MAX_DELTA", "90.0"))
ENABLE_BG_UNSTABLE_EXTRA_TOL = os.environ.get("UAV_BG_UNSTABLE_EXTRA_TOL", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
LOW_BG_UNSTABLE_QRANGE = float(os.environ.get("UAV_BG_UNSTABLE_QRANGE", "18.0"))
LOW_BG_UNSTABLE_EXTRA_DELTA = float(os.environ.get("UAV_BG_UNSTABLE_EXTRA_DELTA", "18.0"))
DEFAULT_STATIC_BG_SECONDS = 10.0 if HIGH_LAYER_MODE else 30.0
STATIC_BG_SECONDS = max(
    1.0,
    float(os.environ.get("UAV_STATIC_BG_SECONDS", str(DEFAULT_STATIC_BG_SECONDS))),
)
ENABLE_FRAME_DIFF_ROIS = True
DIRECT_FULL_FRAME_INFERENCE = _env_bool("UAV_DIRECT_FULL_FRAME_INFERENCE", DEFAULT_DIRECT_FULL_FRAME)
MOTION_ZOOM_CROP_SIZE = max(0, int(os.environ.get("UAV_MOTION_ZOOM_CROP_SIZE", str(DEFAULT_MOTION_ZOOM_CROP_SIZE))))
MOTION_ZOOM_MAX_ROIS = max(0, int(os.environ.get("UAV_MOTION_ZOOM_MAX_ROIS", "1")))
TRACK_ZOOM_CROP_SIZE = max(0, int(os.environ.get("UAV_TRACK_ZOOM_CROP_SIZE", str(MOTION_ZOOM_CROP_SIZE))))
FULLFRAME_FALLBACK_ONLY = _env_bool("UAV_FULLFRAME_FALLBACK_ONLY", DEFAULT_FULLFRAME_FALLBACK_ONLY)
ENABLE_FULLFRAME_SEARCH_FALLBACK = _env_bool("UAV_FULLFRAME_SEARCH_FALLBACK", False)
FULLFRAME_SEARCH_INTERVAL_FRAMES = max(1, int(os.environ.get("UAV_FULLFRAME_SEARCH_INTERVAL_FRAMES", "90")))
FULLFRAME_SEARCH_NO_YOLO_GAP_FRAMES = max(0, int(os.environ.get("UAV_FULLFRAME_SEARCH_NO_YOLO_GAP_FRAMES", "45")))
FULLFRAME_SEARCH_SCORE = float(os.environ.get("UAV_FULLFRAME_SEARCH_SCORE", "0.5"))
DIFF_THRESH = 4 if HIGH_LAYER_MODE else 8
MIN_LOCAL_DIFF_MEAN = 10.0
HIGH_LAYER_MIN_LOCAL_DIFF_MEAN = float(os.environ.get("UAV_HIGH_MIN_LOCAL_DIFF_MEAN", "4.0"))
ENABLE_GRAY_NOISE_SUPPRESSOR = False
TRACK_REQUIRE_CURRENT_MOTION = True
TRACK_MOTION_MIN_PIXELS = 3
TRACK_MOTION_CENTER_SIZE = 120
ENABLE_VERTICAL_STRIP_FILTER = True
VERTICAL_STRIP_ASPECT_RATIO = 1.6
VERTICAL_STRIP_MIN_HEIGHT = 8
GRAY_TEXTURE_PAD = 28
MAX_LOCAL_GRAY_STD = 38.0 if HIGH_LAYER_MODE else 30.0
BRIGHT_SPOT_ABS_THRESH = 210
BRIGHT_SPOT_REL_THRESH = 32.0
BRIGHT_SPOT_MAX_AREA = 220 if HIGH_LAYER_MODE else 140
BRIGHT_SPOT_MIN_BG_STD = 10.0
ENABLE_LCM_FILTER = ENABLE_FRAME_DIFF_ROIS
LCM_BG_PAD = 18
LCM_MIN_SCORE = 1.8
LCM_MIN_RATIO = 1.25
LCM_REQUIRE_BOTH = False
LCM_SCORE_WEIGHT = 3.0
MOTION_ERODE_ITER = 0
MOTION_DILATE_ITER = 1
MOTION_CLOSE_ITER = 1
ENABLE_MOTION_OPENING = True
MOTION_OPEN_ITER = 0
MOTION_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MOTION_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
MOTION_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MOTION_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
STATIC_BG_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
STATIC_BG_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MAX_ROIS_PER_FRAME = 6
MAX_ENQUEUED_ROIS_PER_UPDATE = max(1, int(os.environ.get("UAV_MAX_ROIS_PER_UPDATE", "4")))
NPU_TASK_MAX_AGE_SEC = max(0.05, float(os.environ.get("UAV_NPU_TASK_MAX_AGE_SEC", "0.40")))
DEFAULT_NPU_WORKER_COUNT = 1 if N_CAM <= 2 else 3
NPU_WORKER_COUNT = min(
    3,
    max(1, int(os.environ.get("UAV_NPU_WORKERS", str(DEFAULT_NPU_WORKER_COUNT)))),
)
NPU_REPLACE_PENDING_ON_UPDATE = os.environ.get("UAV_REPLACE_PENDING", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
ENABLE_ROI_GRID_QUOTA = True
ROI_GRID_COLS = max(1, int(os.environ.get("UAV_ROI_GRID_COLS", "5")))
ROI_GRID_ROWS = max(1, int(os.environ.get("UAV_ROI_GRID_ROWS", "4")))
ROI_GRID_MAX_PER_CELL = max(1, int(os.environ.get("UAV_ROI_GRID_MAX_PER_CELL", "1")))
INF_QUEUE_SIZE = MAX_ENQUEUED_ROIS_PER_UPDATE
RES_QUEUE_SIZE = MAX_ROIS_PER_FRAME + 4
DIFF_W = max(320, int(os.environ.get("UAV_DIFF_W", str(DEFAULT_DIFF_W))))
DIFF_H = max(240, int(os.environ.get("UAV_DIFF_H", str(DEFAULT_DIFF_H))))
_local_model_path = PROJECT_ROOT / "model" / "yolov5s.rknn"
_board_model_path = PROJECT_ROOT.parent / "model" / "yolov5s.rknn"
MODEL_PATH = os.environ.get(
    "UAV_RKNN_MODEL",
    str(_local_model_path if _local_model_path.is_file() else _board_model_path),
)
MAX_DIFF_AREA = 1000 if HIGH_LAYER_MODE else 600
MAX_DIFF_BOX_W = 180 if HIGH_LAYER_MODE else 120
MAX_DIFF_BOX_H = 180 if HIGH_LAYER_MODE else 120
NEAR_MAX_DIFF_AREA = 10000 if HIGH_LAYER_MODE else 6000
NEAR_MAX_DIFF_BOX_W = 450 if HIGH_LAYER_MODE else 320
NEAR_MAX_DIFF_BOX_H = 450 if HIGH_LAYER_MODE else 320
NEAR_MIN_COMPACTNESS = 0.04 if HIGH_LAYER_MODE else 0.06
ENABLE_TIGHT_MOTION_ROI = os.environ.get("UAV_TIGHT_MOTION_ROI", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
TIGHT_MOTION_ROI_PAD = max(0, int(os.environ.get("UAV_TIGHT_MOTION_ROI_PAD", "32")))
TIGHT_MOTION_ROI_FILL = int(os.environ.get("UAV_TIGHT_MOTION_ROI_FILL", "114"))
TIGHT_MOTION_ROI_SCORE_PENALTY = float(os.environ.get("UAV_TIGHT_MOTION_ROI_SCORE_PENALTY", "4.0"))
HIGH_MAX_DET_BOX_W = float(os.environ.get("UAV_HIGH_MAX_DET_BOX_W", "220"))
HIGH_MAX_DET_BOX_H = float(os.environ.get("UAV_HIGH_MAX_DET_BOX_H", "180"))
HIGH_MAX_DET_BOX_AREA = float(os.environ.get("UAV_HIGH_MAX_DET_BOX_AREA", "32000"))
LOW_MAX_DET_BOX_W = float(os.environ.get("UAV_LOW_MAX_DET_BOX_W", "280"))
LOW_MAX_DET_BOX_H = float(os.environ.get("UAV_LOW_MAX_DET_BOX_H", "200"))
LOW_MAX_DET_BOX_AREA = float(os.environ.get("UAV_LOW_MAX_DET_BOX_AREA", "50000"))
TRACK_CONFIRM_MIN_NET_MOTION_PX = 6.0 if HIGH_LAYER_MODE else 10.0
TRACK_CONFIRM_MIN_STRAIGHTNESS = 0.24 if HIGH_LAYER_MODE else 0.35
TRACK_CONFIRM_SMALL_BOX_MAX = 56
TRACK_CONFIRM_SMALL_MIN_HITS = 6
ENABLE_BG_ANCHORED_TRACK_FILTER = os.environ.get("UAV_BG_ANCHORED_TRACK_FILTER", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
BG_ANCHORED_TRACK_MIN_BOX_W = float(os.environ.get("UAV_BG_ANCHOR_BOX_MIN_WIDTH", "140.0"))
BG_ANCHORED_TRACK_MIN_NET_RATIO = float(os.environ.get("UAV_BG_ANCHOR_NET_MOTION_RATIO", "0.30"))
BG_ANCHORED_TRACK_MIN_NET_PX = float(os.environ.get("UAV_BG_ANCHOR_MIN_NET_MOTION_PX", "12.0"))
ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
IMG_W, IMG_H = CAPTURE_W, CAPTURE_H
HIGH_LAYER_MIN_DIFF_AREA = 3
LOW_LAYER_MIN_DIFF_AREA = 11
MIN_DIFF_AREA = (HIGH_LAYER_MIN_DIFF_AREA if HIGH_LAYER_MODE else LOW_LAYER_MIN_DIFF_AREA) if ENABLE_FRAME_DIFF_ROIS else 0
FAR_MIN_COMPACTNESS = 0.0 if HIGH_LAYER_MODE else 0.18

ROUGH_RANGE_MIN_M = 20
ROUGH_RANGE_MAX_M = 2000
ROUGH_RANGE_ROUND_M = 10
CAM_MAP = {i: f"00000000{i+1}" for i in range(5)}

LAYER_CONTEXT = threading.local()
FRAME_DIFF_LOCK = threading.Lock()


def estimate_day_scene_from_bg(bg_gray):
    if SCENE_MODE == "day":
        return True, "scene=day:param"
    return False, "scene=night:param"


def make_layer_profile(high, reason="global", day_scene=None, layer_confidence=0.0):
    high = bool(high)
    return {
        "name": "high" if high else "low",
        "high": high,
        "reason": reason,
        "day_scene": day_scene,
        "layer_confidence": float(layer_confidence),
        "conf_thresh": YOLO_CONF_OVERRIDE if YOLO_CONF_OVERRIDE is not None else (0.40 if high else 0.30),
        "track_search_confirmed_only": True if high else False,
        "track_search_min_net_motion_px": 6.0 if high else 0.0,
        "track_search_max_age_frames": HIGH_TRACK_SEARCH_MAX_AGE_FRAMES if high else TRACK_SEARCH_MAX_AGE_FRAMES,
        "track_search_predict_max_misses": HIGH_TRACK_SEARCH_PREDICT_MAX_MISSES if high else TRACK_SEARCH_PREDICT_MAX_MISSES,
        "track_search_max_rois": HIGH_TRACK_SEARCH_MAX_ROIS if high else MAX_TRACK_ROIS_PER_FRAME,
        "track_search_min_yolo_hits": TRACK_SEARCH_MIN_YOLO_HITS,
        "track_search_min_recent_hits": HIGH_TRACK_SEARCH_MIN_RECENT_HITS if high else TRACK_SEARCH_MIN_RECENT_HITS,
        "track_search_min_score": HIGH_TRACK_SEARCH_MIN_SCORE if high else TRACK_SEARCH_MIN_SCORE,
        "track_search_recheck_min_gap_frames": TRACK_SEARCH_RECHECK_MIN_GAP_FRAMES,
        "track_search_visual_lock_only": TRACK_SEARCH_VISUAL_LOCK_ONLY,
        "track_search_require_current_motion": False if high else TRACK_REQUIRE_CURRENT_MOTION,
        "enable_tight_motion_roi": bool(ENABLE_TIGHT_MOTION_ROI and not high),
        "enable_static_bg_change_gate": False if high else True,
        "static_bg_seconds": 10.0 if high else 30.0,
        "diff_thresh": 4 if high else 8,
        "min_local_diff_mean": HIGH_LAYER_MIN_LOCAL_DIFF_MEAN if high else MIN_LOCAL_DIFF_MEAN,
        "enable_lcm_filter": False if high else ENABLE_LCM_FILTER,
        "enable_vertical_strip_filter": False if high else ENABLE_VERTICAL_STRIP_FILTER,
        "enable_gray_noise_suppressor": False,
        "roi_grid_max_per_cell": ROI_GRID_MAX_PER_CELL,
        "max_local_gray_std": 38.0 if high else 30.0,
        "bright_spot_max_area": 220 if high else 140,
        "max_diff_area": 1000 if high else 600,
        "max_diff_box_w": 180 if high else 120,
        "max_diff_box_h": 180 if high else 120,
        "near_max_diff_area": 10000 if high else 6000,
        "near_max_diff_box_w": 450 if high else 320,
        "near_max_diff_box_h": 450 if high else 320,
        "near_min_compactness": 0.04 if high else 0.06,
        "max_det_box_w": HIGH_MAX_DET_BOX_W if high else LOW_MAX_DET_BOX_W,
        "max_det_box_h": HIGH_MAX_DET_BOX_H if high else LOW_MAX_DET_BOX_H,
        "max_det_box_area": HIGH_MAX_DET_BOX_AREA if high else LOW_MAX_DET_BOX_AREA,
        "track_confirm_min_net_motion_px": 6.0 if high else 10.0,
        "track_confirm_min_straightness": 0.24 if high else 0.35,
        "min_diff_area": (HIGH_LAYER_MIN_DIFF_AREA if high else LOW_LAYER_MIN_DIFF_AREA) if ENABLE_FRAME_DIFF_ROIS else 0,
        "far_min_compactness": 0.0 if high else 0.18,
    }


def build_background_risk_model(bg_qrange):
    if bg_qrange is None or getattr(bg_qrange, "ndim", 0) != 2 or bg_qrange.size == 0:
        return None
    q = np.clip(bg_qrange, 0, 255).astype(np.uint8, copy=False)
    histogram = np.bincount(q.reshape(-1), minlength=256).astype(np.float64)
    cdf = np.cumsum(histogram)
    if cdf[-1] <= 0:
        return None
    cdf /= cdf[-1]
    baseline = int(np.searchsorted(cdf, 0.5, side="left"))
    baseline_cdf = float(cdf[baseline])
    return {
        "qrange": q,
        "cdf": cdf,
        "baseline": baseline,
        "baseline_cdf": baseline_cdf,
    }


def background_risk_for_box(box, risk_model, full_w, full_h):
    if risk_model is None or box is None or len(box) < 4:
        return 0.0
    q = risk_model["qrange"]
    qh, qw = q.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    qx1 = max(0, min(qw - 1, int(math.floor(x1 * qw / max(1.0, float(full_w))))))
    qy1 = max(0, min(qh - 1, int(math.floor(y1 * qh / max(1.0, float(full_h))))))
    qx2 = max(qx1 + 1, min(qw, int(math.ceil(x2 * qw / max(1.0, float(full_w))))))
    qy2 = max(qy1 + 1, min(qh, int(math.ceil(y2 * qh / max(1.0, float(full_h))))))
    patch = q[qy1:qy2, qx1:qx2]
    if patch.size == 0:
        return 0.0
    local_level = int(round(float(np.percentile(patch, 90))))
    baseline = int(risk_model["baseline"])
    if local_level <= baseline:
        return 0.0
    baseline_cdf = float(risk_model["baseline_cdf"])
    tail_mass = 1.0 - baseline_cdf
    if tail_mass <= 1e-9:
        return 0.0
    local_cdf = float(risk_model["cdf"][max(0, min(255, local_level))])
    return max(0.0, min(1.0, (local_cdf - baseline_cdf) / tail_mass))


def motion_seed_alignment(box, seed_cx, seed_cy):
    if box is None or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half_w = max(0.5, 0.5 * (x2 - x1))
    half_h = max(0.5, 0.5 * (y2 - y1))
    outside_x = max(0.0, abs(float(seed_cx) - cx) - half_w)
    outside_y = max(0.0, abs(float(seed_cy) - cy) - half_h)
    outside_dist = math.hypot(outside_x, outside_y)
    box_radius = max(1.0, math.hypot(half_w, half_h))
    return float(math.exp(-outside_dist / box_radius))


DEFAULT_LAYER_PROFILE = make_layer_profile(HIGH_LAYER_MODE, f"global:{LAYER_MODE}")
CAM_LAYER_PROFILES = [DEFAULT_LAYER_PROFILE for _ in range(N_CAM)]
ACTIVE_YOLO_CONF_THRESH = CONF_THRESH


def current_layer_profile():
    return getattr(LAYER_CONTEXT, "profile", DEFAULT_LAYER_PROFILE)


def set_current_layer_profile(profile):
    LAYER_CONTEXT.profile = profile or DEFAULT_LAYER_PROFILE


def get_cam_layer_profile(cam_idx):
    if 0 <= int(cam_idx) < len(CAM_LAYER_PROFILES):
        return CAM_LAYER_PROFILES[int(cam_idx)]
    return DEFAULT_LAYER_PROFILE


def estimate_low_layer_scene(bg_qrange):
    if bg_qrange is None or getattr(bg_qrange, "ndim", 0) != 2 or bg_qrange.size == 0:
        return False, 0.0, "no_qrange"
    q = bg_qrange.astype(np.float32)
    row_score = np.percentile(q, 75, axis=1).astype(np.float32)
    h = int(row_score.size)
    if h < 20:
        return False, 0.0, "short_qrange"
    smooth_rows = max(3, h // 40)
    if smooth_rows % 2 == 0:
        smooth_rows += 1
    smooth = np.convolve(
        row_score,
        np.ones(smooth_rows, dtype=np.float32) / float(smooth_rows),
        mode="same",
    )
    lo, hi = max(2, h // 6), min(h - 2, (5 * h) // 6)
    candidates = []
    for y in range(lo, hi):
        upper = float(np.median(smooth[:y]))
        lower = float(np.median(smooth[y:]))
        candidates.append((lower - upper, y, upper, lower))
    best_delta, split_y, upper, lower = max(candidates, key=lambda item: item[0])
    row_noise = float(np.median(np.abs(np.diff(smooth)))) * 1.4826
    global_spread = float(np.percentile(smooth, 75) - np.percentile(smooth, 25))
    sensor_scale = math.sqrt(max(0.0, float(np.median(smooth))) + 1.0)
    separation_scale = max(row_noise, 0.5 * global_spread, sensor_scale, 1e-6)
    separation = max(0.0, best_delta / separation_scale)
    is_low = best_delta > 0.0 and separation >= 1.0
    reason = (
        f"scene_low_score split={split_y / float(h):.3f} upper={upper:.2f} "
        f"lower={lower:.2f} separation={separation:.2f}"
    )
    return is_low, separation, reason


def estimate_layer_from_background(bg_gray, bg_qrange):
    day_scene, day_reason = estimate_day_scene_from_bg(bg_gray)
    if day_scene is False:
        return True, f"scene:night-high; {day_reason}", day_scene, 1.0
    is_low, layer_confidence, layer_reason = estimate_low_layer_scene(bg_qrange)
    if bg_qrange is None or getattr(bg_qrange, "ndim", 0) != 2 or bg_qrange.size == 0:
        return HIGH_LAYER_MODE, f"fallback:{layer_reason}; {day_reason}", day_scene, layer_confidence
    reason = f"scene:layer {layer_reason}; {day_reason}"
    return (not is_low), reason, day_scene, layer_confidence


def build_cam_layer_profiles(initial_backgrounds):
    profiles = []
    for cam_idx in range(N_CAM):
        spec = CAM_LAYER_MODE_SPECS[min(cam_idx, len(CAM_LAYER_MODE_SPECS) - 1)].lower() if CAM_LAYER_MODE_SPECS else ""
        bg_gray = initial_backgrounds[cam_idx][0] if cam_idx < len(initial_backgrounds) else None
        bg_qrange = initial_backgrounds[cam_idx][2] if cam_idx < len(initial_backgrounds) else None
        if spec in ("high", "h"):
            _auto_high, _auto_reason, day_scene, layer_confidence = estimate_layer_from_background(bg_gray, bg_qrange)
            high, reason = True, f"debug-override:high; {_auto_reason}"
        elif spec in ("low", "l"):
            _auto_high, _auto_reason, day_scene, layer_confidence = estimate_layer_from_background(bg_gray, bg_qrange)
            high, reason = False, f"debug-override:low; {_auto_reason}"
        elif spec in ("auto", "a") or ENABLE_AUTO_CAM_LAYER:
            high, reason, day_scene, layer_confidence = estimate_layer_from_background(bg_gray, bg_qrange)
        else:
            high, reason = HIGH_LAYER_MODE, f"debug-global:{LAYER_MODE}"
            day_scene = estimate_day_scene_from_bg(bg_gray)[0]
            _is_low, layer_confidence, _ = estimate_low_layer_scene(bg_qrange)
        profiles.append(make_layer_profile(
            high,
            reason,
            day_scene=day_scene,
            layer_confidence=layer_confidence,
        ))
    return profiles

# ==========================================
# 2
# ==========================================
inf_queues = [queue.Queue(maxsize=INF_QUEUE_SIZE) for _ in range(N_CAM)]
res_queues =[queue.Queue(maxsize=RES_QUEUE_SIZE) for _ in range(N_CAM)]
display_queues =[queue.Queue(maxsize=2) for _ in range(N_CAM)]

stop_event = threading.Event()
video_allowed_event = None
SYSTEM_START_TIME = time.time()
INIT_SIGNAL_SENT = False
VIDEO_STREAM_ALLOWED = False

def init_runtime_queues():
    global inf_queues, res_queues, display_queues, stop_event, video_allowed_event
    inf_queues = [queue.Queue(maxsize=INF_QUEUE_SIZE) for _ in range(N_CAM)]
    res_queues = [queue.Queue(maxsize=RES_QUEUE_SIZE) for _ in range(N_CAM)]
    display_queues = [queue.Queue(maxsize=2) for _ in range(N_CAM)]
    stop_event = threading.Event()
    video_allowed_event = threading.Event()

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


def clear_pending(q):
    cleared = []
    while True:
        try:
            cleared.append(q.get_nowait())
        except queue.Empty:
            return cleared

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
        box_is_padded_roi = len(box) >= 9
        bcx, bcy = (box[0]+box[2])/2, (box[1]+box[3])/2
        for idx, m_box in enumerate(merged):
            if box_is_padded_roi != (len(m_box) >= 9):
                continue
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
        return None, None, None
    count = len(samples)
    sum_img = np.zeros(samples[0].shape, dtype=np.float32)
    sum_sq = np.zeros(samples[0].shape, dtype=np.float32)
    for sample in samples:
        cv2.accumulate(sample, sum_img)
        cv2.accumulateSquare(sample, sum_sq)

    stack = np.stack(samples, axis=0)
    samples.clear()
    kth = count // 2
    sorted_stack = np.sort(stack, axis=0)
    bg = sorted_stack[kth].copy()
    mean = sum_img / float(count)
    variance = np.maximum(sum_sq / float(count) - mean * mean, 0.0)
    std = np.sqrt(variance)
    std_tol = LOW_BG_ABS_DELTA + LOW_BG_STD_MULT * std
    if ENABLE_BG_QRANGE_TOL and count >= 8:
        p05 = sorted_stack[max(0, int(round((count - 1) * 0.05)))].astype(np.float32)
        p95 = sorted_stack[min(count - 1, int(round((count - 1) * 0.95)))].astype(np.float32)
        qrange = np.maximum(p95 - p05, 0.0)
        qrange_tol = LOW_BG_QRANGE_BASE_DELTA + LOW_BG_QRANGE_MULT * qrange
        tol = np.maximum(std_tol, qrange_tol)
        if ENABLE_BG_UNSTABLE_EXTRA_TOL:
            tol = tol + (qrange >= LOW_BG_UNSTABLE_QRANGE).astype(np.float32) * LOW_BG_UNSTABLE_EXTRA_DELTA
        tol = np.clip(tol, min(LOW_BG_ABS_DELTA, LOW_BG_QRANGE_BASE_DELTA), LOW_BG_QRANGE_MAX_DELTA)
    else:
        qrange = None
        tol = np.clip(std_tol, LOW_BG_ABS_DELTA, 60.0)
    tol = tol.astype(np.uint8)
    qrange_u8 = np.clip(qrange, 0.0, 255.0).astype(np.uint8) if qrange is not None else None
    return bg, tol, qrange_u8


def build_initial_video_bg_model(video_path, static_seconds=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for static background: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    seconds = float(STATIC_BG_SECONDS if static_seconds is None else static_seconds)
    sample_span = max(1, int(round(seconds * fps)))
    sample_stride = max(1, int(math.ceil(sample_span / float(LOW_BG_SAMPLE_FRAMES))))
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
        if gray.shape[1] != DIFF_W or gray.shape[0] != DIFF_H:
            gray = cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)
        samples.append(gray)
    cap.release()
    if not samples:
        raise RuntimeError("no frames sampled for static background")
    sample_count = len(samples)
    bg, tol, qrange = build_static_bg_model(samples)
    return bg, tol, qrange, sample_count, sample_span

def build_static_bg_change_mask(gray, bg, tol):
    if not ENABLE_LOW_LAYER_STATIC_BG_MASK or bg is None or tol is None:
        return None
    bg_diff = cv2.absdiff(gray, bg)
    changed = (bg_diff.astype(np.uint16) > tol.astype(np.uint16)).astype(np.uint8) * 255
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, STATIC_BG_OPEN_KERNEL, iterations=1)
    changed = cv2.dilate(changed, STATIC_BG_DILATE_KERNEL, iterations=1)
    return changed

def cleanup_motion_mask_with_iters(mask, erode_iter, dilate_iter, close_iter):
    if ENABLE_MOTION_OPENING and MOTION_OPEN_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MOTION_OPEN_KERNEL, iterations=MOTION_OPEN_ITER)
    if int(erode_iter) > 0:
        mask = cv2.erode(mask, MOTION_ERODE_KERNEL, iterations=int(erode_iter))
    if int(dilate_iter) > 0:
        mask = cv2.dilate(mask, MOTION_DILATE_KERNEL, iterations=int(dilate_iter))
    if int(close_iter) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MOTION_CLOSE_KERNEL, iterations=int(close_iter))
    return mask

def cleanup_motion_mask(mask):
    return cleanup_motion_mask_with_iters(
        mask,
        MOTION_ERODE_ITER,
        MOTION_DILATE_ITER,
        MOTION_CLOSE_ITER,
    )

def track_roi_has_motion(track_roi, mask_small, full_w, full_h, min_pixels=TRACK_MOTION_MIN_PIXELS, center_size=TRACK_MOTION_CENTER_SIZE):
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

def roi_is_padded_canvas(roi):
    return len(roi) >= 9


def roi_is_zoom_crop(roi):
    return len(roi) >= 10 and roi[9] == "zoom"


def roi_seed_center(roi):
    if roi_is_padded_canvas(roi):
        return float(roi[7]), float(roi[8])
    return (float(roi[0]) + float(roi[2])) * 0.5, (float(roi[1]) + float(roi[3])) * 0.5


def make_padded_roi_canvas(frame, roi, crop_size=CROP_SIZE, fill_value=TIGHT_MOTION_ROI_FILL):
    x1, y1, x2, y2 = [int(v) for v in roi[:4]]
    pad_x, pad_y = int(roi[5]), int(roi[6])
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return patch
    canvas = np.full((crop_size, crop_size, 3), int(fill_value), dtype=np.uint8)
    dst_x2 = min(crop_size, pad_x + patch.shape[1])
    dst_y2 = min(crop_size, pad_y + patch.shape[0])
    if dst_x2 <= pad_x or dst_y2 <= pad_y:
        return patch
    canvas[pad_y:dst_y2, pad_x:dst_x2] = patch[:dst_y2 - pad_y, :dst_x2 - pad_x]
    return canvas


def pad_roi_image_to_square(roi_img, crop_size=CROP_SIZE, fill_value=TIGHT_MOTION_ROI_FILL):
    if roi_img is None or roi_img.size == 0:
        return roi_img, 0, 0
    h, w = roi_img.shape[:2]
    if h == crop_size and w == crop_size:
        return roi_img, 0, 0
    if h > crop_size or w > crop_size:
        return roi_img, 0, 0
    pad_x = (crop_size - w) // 2
    pad_y = (crop_size - h) // 2
    canvas = np.full((crop_size, crop_size, 3), int(fill_value), dtype=np.uint8)
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = roi_img
    return canvas, pad_x, pad_y

def crop_roi_from_center(cx, cy, full_w, full_h, crop_size=CROP_SIZE):
    half = crop_size // 2
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(full_w, x1 + crop_size)
    y2 = min(full_h, y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)
    return [x1, y1, x2, y2]


def _filter_isolated_gray_seed_candidates(candidates, radius, max_neighbors):
    if not candidates:
        return []
    cell = max(8, int(radius))
    radius2 = float(radius * radius)
    grid = {}
    for idx, cand in enumerate(candidates):
        key = (int(cand["cx"] // cell), int(cand["cy"] // cell))
        grid.setdefault(key, []).append(idx)

    isolated = []
    for idx, cand in enumerate(candidates):
        gx, gy = int(cand["cx"] // cell), int(cand["cy"] // cell)
        neighbor_count = 0
        for yy in range(gy - 1, gy + 2):
            for xx in range(gx - 1, gx + 2):
                for other_idx in grid.get((xx, yy), []):
                    if other_idx == idx:
                        continue
                    other = candidates[other_idx]
                    dx = float(other["cx"] - cand["cx"])
                    dy = float(other["cy"] - cand["cy"])
                    if dx * dx + dy * dy <= radius2:
                        neighbor_count += 1
                        if neighbor_count > max_neighbors:
                            break
                if neighbor_count > max_neighbors:
                    break
            if neighbor_count > max_neighbors:
                break
        if neighbor_count <= max_neighbors:
            isolated.append(cand)
    return isolated


def gray_seed_rois_from_frame(gray, scale_x, scale_y, full_w, full_h):
    profile = current_layer_profile()
    if (
        not ENABLE_GRU_GRAY_SEED_ROIS
        or not profile.get("high", False)
        or GRU_GRAY_SEED_MAX_ROIS <= 0
        or gray is None
    ):
        return []
    if len(gray.shape) != 2:
        gray = frame_to_gray(gray)
    if gray is None or gray.size == 0:
        return []

    h, w = gray.shape[:2]
    sky_y_limit = int(round(h * GRU_GRAY_SEED_SKY_Y_RATIO))
    scale_linear = math.sqrt((w * h) / (1280.0 * 720.0))
    radius = GRU_GRAY_SEED_RADIUS * max(0.25, scale_linear)
    border = min(GRU_GRAY_SEED_BORDER, max(0, min(w, h) // 4))

    all_candidates = []
    for thresh in (GRU_GRAY_SEED_THRESH, GRU_GRAY_SEED_RELAXED_THRESH):
        _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            area = int(bw * bh)
            if area < GRU_GRAY_SEED_MIN_AREA or area > GRU_GRAY_SEED_MAX_AREA:
                continue
            cx = int(x + bw * 0.5)
            cy = int(y + bh * 0.5)
            if cx < border or cy < border or cx >= w - border or cy >= h - border:
                continue
            if cy > sky_y_limit:
                continue
            patch = gray[y:y + bh, x:x + bw]
            if patch.size == 0:
                continue
            contrast = max(0, int(thresh) - int(np.min(patch)))
            compactness = float(cv2.contourArea(c)) / max(1.0, float(area))
            score = GRU_GRAY_SEED_SCORE_BONUS + contrast * max(1.0, math.sqrt(area)) + 4.0 * compactness
            candidates.append({
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "cx": cx,
                "cy": cy,
                "area": area,
                "score": float(score),
            })
        isolated = _filter_isolated_gray_seed_candidates(candidates, radius, GRU_GRAY_SEED_MAX_NEIGHBORS)
        all_candidates.extend(isolated)
        if all_candidates:
            break

    if not all_candidates:
        return []

    by_cell = {}
    for cand in sorted(all_candidates, key=lambda c: c["score"], reverse=True):
        key = (int(cand["cx"] // 24), int(cand["cy"] // 24))
        if key not in by_cell:
            by_cell[key] = cand

    rois = []
    for cand in sorted(by_cell.values(), key=lambda c: c["score"], reverse=True)[:GRU_GRAY_SEED_MAX_ROIS]:
        fx = int(round(cand["cx"] * scale_x))
        fy = int(round(cand["cy"] * scale_y))
        rx1, ry1, rx2, ry2 = crop_roi_from_center(fx, fy, full_w, full_h)
        rois.append((rx1, ry1, rx2, ry2, float(cand["score"])))
    return prioritize_diverse_rois(rois, full_w, full_h)

def frame_to_gray(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return frame
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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


def is_valid_detection_box_size(box, profile):
    if box is None or len(box) < 4:
        return False
    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    if bw <= 0.0 or bh <= 0.0:
        return False
    return (
        bw <= float(profile["max_det_box_w"])
        and bh <= float(profile["max_det_box_h"])
        and bw * bh <= float(profile["max_det_box_area"])
    )

def prioritize_diverse_rois(rois, full_w, full_h, max_rois=MAX_ROIS_PER_FRAME):
    if not ENABLE_ROI_GRID_QUOTA or not rois:
        return rois
    sorted_rois = sorted(rois, key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    selected = []
    cell_counts = {}
    cols = max(1, int(ROI_GRID_COLS))
    rows = max(1, int(ROI_GRID_ROWS))
    profile = current_layer_profile()
    max_per_cell = max(1, int(profile.get("roi_grid_max_per_cell", ROI_GRID_MAX_PER_CELL)))
    for roi in sorted_rois:
        cx = (float(roi[0]) + float(roi[2])) * 0.5
        cy = (float(roi[1]) + float(roi[3])) * 0.5
        cell_x = max(0, min(cols - 1, int(cx / max(1.0, float(full_w)) * cols)))
        cell_y = max(0, min(rows - 1, int(cy / max(1.0, float(full_h)) * rows)))
        cell = (cell_x, cell_y)
        if cell_counts.get(cell, 0) < max_per_cell:
            selected.append(roi)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if len(selected) >= max_rois:
            break
    return selected[:max_rois]

def should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
    profile = current_layer_profile()
    if not profile["enable_gray_noise_suppressor"] or gray is None:
        return False
    x1 = max(0, x - GRAY_TEXTURE_PAD)
    y1 = max(0, y - GRAY_TEXTURE_PAD)
    x2 = min(gray.shape[1], x + w + GRAY_TEXTURE_PAD)
    y2 = min(gray.shape[0], y + h + GRAY_TEXTURE_PAD)
    local_patch = gray[y1:y2, x1:x2]
    if local_patch.size == 0:
        return False

    local_std = float(np.std(local_patch))
    if local_std > profile["max_local_gray_std"]:
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
        area <= profile["bright_spot_max_area"]
        and bright_ratio >= 0.55
        and active_mean >= local_median + BRIGHT_SPOT_REL_THRESH
        and (local_std >= BRIGHT_SPOT_MIN_BG_STD or bright_ratio >= 0.80)
    )

def motion_rois_from_mask(
    mask,
    diff_img,
    gray,
    scale_x,
    scale_y,
    full_w,
    full_h,
):
    profile = current_layer_profile()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_rois = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        if profile.get("enable_vertical_strip_filter", ENABLE_VERTICAL_STRIP_FILTER) and is_vertical_strip_box((w, h)):
            continue
        mask_patch = mask[y:y+h, x:x+w]
        area = int(cv2.countNonZero(mask_patch))
        if area < profile["min_diff_area"]:
            continue
        active = mask_patch > 0
        local_diff = float(diff_img[y:y+h, x:x+w][active].mean()) if np.any(active) else 0.0
        if local_diff < profile.get("min_local_diff_mean", MIN_LOCAL_DIFF_MEAN):
            continue
        lcm_score, lcm_ratio = local_contrast_measure(diff_img, mask_patch, x, y, w, h)
        if profile.get("enable_lcm_filter", ENABLE_LCM_FILTER):
            if LCM_REQUIRE_BOTH:
                if lcm_score < LCM_MIN_SCORE or lcm_ratio < LCM_MIN_RATIO:
                    continue
            elif lcm_score < LCM_MIN_SCORE and lcm_ratio < LCM_MIN_RATIO:
                continue
        if should_suppress_gray_noise(gray, mask_patch, x, y, w, h, area):
            continue
        compactness = area / max(1.0, float(w * h))

        is_far_tiny = (
            area <= profile["max_diff_area"]
            and w <= profile["max_diff_box_w"]
            and h <= profile["max_diff_box_h"]
            and compactness >= profile["far_min_compactness"]
        )
        is_near_compact = (
            area <= profile["near_max_diff_area"]
            and w <= profile["near_max_diff_box_w"]
            and h <= profile["near_max_diff_box_h"]
            and compactness >= profile["near_min_compactness"]
        )
        if not (is_far_tiny or is_near_compact):
            continue

        if is_near_compact and not is_far_tiny:
            score = local_diff + 18.0 * compactness - 0.003 * area
        else:
            score = local_diff + 10.0 * compactness - 0.015 * area
        score += LCM_SCORE_WEIGHT * max(0.0, min(float(lcm_score), 8.0))

        fx, fy = int(math.floor(x * scale_x)), int(math.floor(y * scale_y))
        fx2 = int(math.ceil((x + w) * scale_x))
        fy2 = int(math.ceil((y + h) * scale_y))
        fw, fh = max(1, fx2 - fx), max(1, fy2 - fy)
        cx, cy = fx + fw // 2, fy + fh // 2
        rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h)
        temp_rois.append((
            rx1,
            ry1,
            rx2,
            ry2,
            score,
        ))
        if 0 < MOTION_ZOOM_CROP_SIZE <= min(full_w, full_h):
            zx1, zy1, zx2, zy2 = crop_roi_from_center(
                cx,
                cy,
                full_w,
                full_h,
                crop_size=MOTION_ZOOM_CROP_SIZE,
            )
            temp_rois.append((
                zx1,
                zy1,
                zx2,
                zy2,
                score + 1.0,
                0,
                0,
                cx,
                cy,
                "zoom",
            ))
        if profile.get("enable_tight_motion_roi", False):
            pad = int(TIGHT_MOTION_ROI_PAD)
            sx1 = max(0, fx - pad)
            sy1 = max(0, fy - pad)
            sx2 = min(full_w, fx2 + pad)
            sy2 = min(full_h, fy2 + pad)
            sw, sh = sx2 - sx1, sy2 - sy1
            if 0 < sw <= CROP_SIZE and 0 < sh <= CROP_SIZE:
                px = (CROP_SIZE - sw) // 2
                py = (CROP_SIZE - sh) // 2
                temp_rois.append((
                    sx1,
                    sy1,
                    sx2,
                    sy2,
                    score - TIGHT_MOTION_ROI_SCORE_PENALTY,
                    px,
                    py,
                    cx,
                    cy,
                ))
    temp_rois.sort(key=lambda r: r[4], reverse=True)
    rois = merge_nearby_boxes(temp_rois, dist_thresh=300)
    rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
    return prioritize_diverse_rois(rois, full_w, full_h)

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
        layer_profile=None,
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
        self.profile = layer_profile or current_layer_profile()

    def _center(self, box):
        return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)

    def _box_wh(self, box):
        return (max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1])))

    def _det_score(self, box):
        return float(box[4]) if len(box) > 4 else float(self.profile["conf_thresh"])

    def _det_background_risk(self, box):
        return self._clamp01(box[5]) if len(box) > 5 else 0.0

    def _det_seed_alignment(self, box):
        return self._clamp01(box[6]) if len(box) > 6 else 1.0

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

    def _disable_visual_lock(self, trk):
        trk['visual_lock'] = False
        trk['visual_lock_conf'] = 0.0
        trk['visual_lock_recheck_due'] = False
        trk['visual_lock_overdue_frames'] = 0

    def _visual_lock_yolo_gap(self, trk, frame_idx):
        last_yolo_frame = int(trk.get('last_yolo_frame', trk.get('last_frame', frame_idx)))
        return max(0, int(frame_idx) - last_yolo_frame)

    def _recent_yolo_hits(self, trk, count=None):
        history = list(trk.get('yolo_hit_history', trk.get('hit_history', [])))
        if count is not None and count > 0:
            history = history[-int(count):]
        return int(sum(1 for v in history if v))

    def _green_yolo_streak(self, trk):
        streak = 0
        for hit in reversed(list(trk.get('yolo_hit_history', trk.get('hit_history', [])))):
            if not hit:
                break
            streak += 1
        return streak

    def _mean_yolo_conf(self, trk):
        conf_history = list(trk.get('yolo_conf_history', trk.get('det_conf_history', [])))
        return float(np.mean(conf_history)) if conf_history else 0.0

    def _visual_lock_entry_stability(self, trk):
        boxes = list(trk.get('yolo_box_history', []))
        if len(boxes) < 3:
            return 0.0, 0.0
        keep = min(len(boxes), max(3, VISUAL_LOCK_ENTER_GREEN_STREAK))
        recent = np.array(boxes[-keep:], dtype=np.float32)
        centers = np.column_stack(((recent[:, 0] + recent[:, 2]) * 0.5, (recent[:, 1] + recent[:, 3]) * 0.5))
        median_center = np.median(centers, axis=0)
        center_jitter = float(np.max(np.linalg.norm(centers - median_center, axis=1)))
        sizes = np.column_stack((np.maximum(1.0, recent[:, 2] - recent[:, 0]), np.maximum(1.0, recent[:, 3] - recent[:, 1])))
        median_size = np.maximum(1.0, np.median(sizes, axis=0))
        box_jitter = float(np.max(np.abs(sizes - median_size) / median_size))
        return center_jitter, box_jitter

    def _yolo_box_stability(self, trk, keep):
        boxes = list(trk.get('yolo_box_history', []))
        if len(boxes) < max(2, int(keep)):
            return None
        recent = np.array(boxes[-int(keep):], dtype=np.float32)
        centers = np.column_stack(((recent[:, 0] + recent[:, 2]) * 0.5, (recent[:, 1] + recent[:, 3]) * 0.5))
        median_center = np.median(centers, axis=0)
        center_jitter = float(np.max(np.linalg.norm(centers - median_center, axis=1)))
        sizes = np.column_stack((np.maximum(1.0, recent[:, 2] - recent[:, 0]), np.maximum(1.0, recent[:, 3] - recent[:, 1])))
        median_size = np.maximum(1.0, np.median(sizes, axis=0))
        box_jitter = float(np.max(np.abs(sizes - median_size) / median_size))
        return center_jitter, box_jitter

    def _maybe_enter_visual_lock(self, trk, frame_idx):
        if not ENABLE_VISUAL_LOCK or not self.profile.get("high", False):
            return False
        if trk.get('visual_lock', False):
            return True
        if trk.get('yolo_hits', 0) < VISUAL_LOCK_ENTER_YOLO_HITS:
            return False
        if self._recent_yolo_hits(trk) < VISUAL_LOCK_ENTER_RECENT_HITS:
            return False
        if self._green_yolo_streak(trk) < VISUAL_LOCK_ENTER_GREEN_STREAK:
            return False
        if trk.get('misses', 0) > 0:
            return False
        if not self._is_confirmed(trk):
            return False
        traj_score = self._trajectory_score(trk)
        if float(trk.get('score', 0.0)) < VISUAL_LOCK_ENTER_TRACK_SCORE:
            return False
        if traj_score < VISUAL_LOCK_ENTER_TRAJ_SCORE:
            return False
        if self._mean_yolo_conf(trk) < VISUAL_LOCK_ENTER_MEAN_CONF:
            return False
        if self._net_motion_px(trk) < VISUAL_LOCK_ENTER_MIN_NET_MOTION:
            return False
        center_jitter, box_jitter = self._visual_lock_entry_stability(trk)
        if center_jitter > VISUAL_LOCK_ENTER_MAX_CENTER_JITTER:
            return False
        if box_jitter > VISUAL_LOCK_ENTER_MAX_BOX_JITTER_RATIO:
            return False
        trk['visual_lock'] = True
        trk['visual_lock_hits'] = 0
        trk['visual_lock_misses'] = 0
        trk['visual_lock_conf'] = max(float(trk.get('visual_lock_conf', 0.0)), VISUAL_LOCK_CONF_ENTER)
        trk['visual_lock_entry_score'] = float(trk.get('score', 0.0))
        trk['visual_lock_entry_traj_score'] = float(traj_score)
        trk['visual_lock_entry_mean_conf'] = float(self._mean_yolo_conf(trk))
        trk['visual_lock_enter_frame'] = int(frame_idx)
        trk['last_yolo_frame'] = int(trk.get('last_yolo_frame', trk.get('last_frame', frame_idx)))
        trk['visual_lock_recheck_due'] = False
        trk['visual_lock_overdue_frames'] = 0
        return True

    def _find_visual_lock_detection(self, trk, frame, frame_idx):
        if frame is None:
            return None
        gray = frame_to_gray(frame)
        if gray is None or gray.size == 0:
            return None
        full_h, full_w = gray.shape[:2]
        pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
        win = int(VISUAL_LOCK_WINDOW + 20 * min(3, int(trk.get('visual_lock_misses', 0))))
        half = win // 2
        x1 = max(0, int(round(pred_cx)) - half)
        y1 = max(0, int(round(pred_cy)) - half)
        x2 = min(full_w, x1 + win)
        y2 = min(full_h, y1 + win)
        x1 = max(0, x2 - win)
        y1 = max(0, y2 - win)
        patch = gray[y1:y2, x1:x2]
        if patch.shape[0] < 24 or patch.shape[1] < 24:
            return None

        patch_f = patch.astype(np.float32)
        smooth = cv2.GaussianBlur(patch_f, (0, 0), 1.2)
        bg = cv2.GaussianBlur(patch_f, (0, 0), 9.0)
        response = np.maximum(bg - smooth, smooth - bg)
        resp_mean = float(np.mean(response))
        resp_std = float(np.std(response))
        thresh = max(float(VISUAL_LOCK_MIN_RESPONSE), resp_mean + 2.2 * resp_std)
        mask = (response >= thresh).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        max_dist = max(18.0, VISUAL_LOCK_MAX_DIST_RATIO * float(win))
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            area = int(cv2.countNonZero(mask[by:by + bh, bx:bx + bw]))
            if area < 1 or area > VISUAL_LOCK_MAX_AREA:
                continue
            cx = x1 + bx + bw * 0.5
            cy = y1 + by + bh * 0.5
            dist = math.hypot(cx - pred_cx, cy - pred_cy)
            if dist > max_dist:
                continue
            active = mask[by:by + bh, bx:bx + bw] > 0
            if not np.any(active):
                continue
            resp_score = float(np.mean(response[by:by + bh, bx:bx + bw][active]))
            if resp_score < VISUAL_LOCK_MIN_RESPONSE:
                continue
            compactness = float(area) / max(1.0, float(bw * bh))
            box_side = max(
                VISUAL_LOCK_MIN_BOX,
                min(
                    VISUAL_LOCK_MAX_BOX,
                    int(round(max(float(trk.get('w', 1.0)), float(trk.get('h', 1.0))))),
                ),
            )
            det = [
                int(max(0, round(cx - box_side * 0.5))),
                int(max(0, round(cy - box_side * 0.5))),
                int(min(full_w, round(cx + box_side * 0.5))),
                int(min(full_h, round(cy + box_side * 0.5))),
                0.52,
                0.0,
                1.0,
            ]
            tmpl_score = self._template_score(trk, frame, det)
            if tmpl_score < VISUAL_LOCK_MIN_TEMPLATE:
                continue
            score = (
                resp_score
                + 7.0 * compactness
                + VISUAL_LOCK_TEMPLATE_WEIGHT * 20.0 * tmpl_score
                - 0.065 * dist
            )
            if score < VISUAL_LOCK_MIN_SCORE:
                continue
            if best is None or score > best[0]:
                best = (score, det, tmpl_score, resp_score)
        return best

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

    def _recent_net_motion_px(self, trk, window=8):
        pts = list(trk.get('history', []))
        if len(pts) < 2:
            return 0.0
        pts = pts[-max(2, int(window)):]
        return math.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)

    def _track_speed_px_per_frame(self, trk):
        return math.sqrt(float(trk.get('vx', 0.0)) ** 2 + float(trk.get('vy', 0.0)) ** 2)

    def _is_hover_candidate(self, trk, frame_idx):
        if not ENABLE_HOVER_HOLD:
            return False
        if trk.get('yolo_hits', 0) < HOVER_ENTER_YOLO_HITS:
            return False
        if self._recent_yolo_hits(trk, self.recent_window) < HOVER_ENTER_RECENT_HITS:
            return False
        if float(trk.get('score', 0.0)) < HOVER_MIN_TRACK_SCORE:
            return False
        if not self._is_confirmed(trk):
            return False
        if self._track_speed_px_per_frame(trk) > HOVER_MAX_SPEED_PX_PER_FRAME:
            return False
        if self._recent_net_motion_px(trk, HOVER_RECENT_WINDOW) > HOVER_MAX_RECENT_NET_MOTION_PX:
            return False
        yolo_gap = self._visual_lock_yolo_gap(trk, frame_idx)
        if yolo_gap > HOVER_MAX_YOLO_GAP_FRAMES:
            return False
        return True

    def _adaptive_required_hits(self, trk):
        required = int(self.min_hits)
        max_side = max(float(trk.get('w', 1.0)), float(trk.get('h', 1.0)))
        if (not self.profile["high"]) and max_side <= TRACK_CONFIRM_SMALL_BOX_MAX:
            required = max(required, int(TRACK_CONFIRM_SMALL_MIN_HITS))
        return required

    def _passes_motion_confirmation(self, trk):
        f = self._motion_confirmation_features(trk)
        duration = int(f['duration_frames'])
        if duration < 8:
            return True
        if ENABLE_BG_ANCHORED_TRACK_FILTER:
            box_w = float(trk.get('w', 0.0))
            box_h = float(trk.get('h', 0.0))
            max_side = max(box_w, box_h)
            if max_side >= BG_ANCHORED_TRACK_MIN_BOX_W:
                required_net_motion = max(
                    BG_ANCHORED_TRACK_MIN_NET_PX,
                    BG_ANCHORED_TRACK_MIN_NET_RATIO * max_side,
                )
                if float(f['net_displacement_px']) < required_net_motion:
                    return False
        if float(f['net_displacement_px']) < self.profile["track_confirm_min_net_motion_px"]:
            return False
        if float(f['straightness']) < self.profile["track_confirm_min_straightness"]:
            return False
        return True

    def _adaptive_background_evidence(self, trk):
        risk_history = list(trk.get('background_risk_history', []))
        alignment_history = list(trk.get('seed_alignment_history', []))
        background_risk = float(np.median(risk_history)) if risk_history else 0.0
        seed_alignment = float(np.median(alignment_history)) if alignment_history else 1.0
        f_motion = self._motion_confirmation_features(trk)
        box_diag = max(1.0, math.hypot(float(trk.get('w', 1.0)), float(trk.get('h', 1.0))))
        normalized_motion = 1.0 - math.exp(-float(f_motion['net_displacement_px']) / box_diag)
        trajectory_motion = normalized_motion * float(f_motion['straightness'])
        conf_history = list(trk.get('det_conf_history', []))
        mean_conf = float(np.mean(conf_history)) if conf_history else 0.0
        hit_history = list(trk.get('hit_history', []))
        repeatability = float(sum(hit_history)) / max(1.0, float(len(hit_history)))
        evidence = (
            0.45 * seed_alignment
            + 0.30 * trajectory_motion
            + 0.15 * mean_conf
            + 0.10 * repeatability
        )
        return self._clamp01(background_risk), self._clamp01(evidence)

    def _passes_adaptive_background_confirmation(self, trk):
        background_risk, evidence = self._adaptive_background_evidence(trk)
        return evidence >= background_risk

    def _motion_confirmation_features(self, trk):
        pts = np.array(list(trk.get('history', [])), dtype=np.float32)
        hit_history = list(trk.get('hit_history', []))
        features = {
            'duration_frames': len(hit_history),
            'path_length_px': 0.0,
            'net_displacement_px': 0.0,
            'straightness': 0.0,
        }
        if len(pts) < 2:
            return features
        dxy = np.diff(pts, axis=0)
        steps = np.linalg.norm(dxy, axis=1)
        path_length = float(np.sum(steps))
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        features['path_length_px'] = path_length
        features['net_displacement_px'] = net_disp
        features['straightness'] = float(net_disp / max(path_length, 1e-6))
        return features

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
            'yolo_hits': 1,
            'history': deque([(cx, cy)], maxlen=self.recent_window),
            'hit_history': deque([1], maxlen=self.recent_window),
            'det_conf_history': deque([self._det_score(det)], maxlen=self.recent_window),
            'yolo_hit_history': deque([1], maxlen=self.recent_window),
            'yolo_conf_history': deque([self._det_score(det)], maxlen=self.recent_window),
            'yolo_box_history': deque([det[:4]], maxlen=self.recent_window),
            'background_risk_history': deque([self._det_background_risk(det)], maxlen=self.recent_window),
            'seed_alignment_history': deque([self._det_seed_alignment(det)], maxlen=self.recent_window),
            'frame_h': int(frame.shape[0]) if frame is not None else CAPTURE_H,
            'template': self._crop_template(frame, det),
            'visual_lock': False,
            'visual_lock_hits': 0,
            'visual_lock_misses': 0,
            'visual_lock_conf': 0.0,
            'visual_lock_recheck_due': False,
            'visual_lock_overdue_frames': 0,
            'last_yolo_frame': int(frame_idx),
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
        trk['last_yolo_frame'] = int(frame_idx)
        trk['hits'] += 1
        trk['yolo_hits'] = trk.get('yolo_hits', 0) + 1
        trk['misses'] = 0
        trk['visual_lock_misses'] = 0
        trk['visual_lock_recheck_due'] = False
        trk['visual_lock_overdue_frames'] = 0
        trk['score'] = self._clamp01(0.75 * trk['score'] + 0.25 * match_score)
        trk['history'].append((cx, cy))
        trk['hit_history'].append(1)
        trk['det_conf_history'].append(self._det_score(det))
        trk.setdefault('yolo_hit_history', deque(maxlen=self.recent_window)).append(1)
        trk.setdefault('yolo_conf_history', deque(maxlen=self.recent_window)).append(self._det_score(det))
        trk.setdefault('yolo_box_history', deque(maxlen=self.recent_window)).append(det[:4])
        trk['background_risk_history'].append(self._det_background_risk(det))
        trk['seed_alignment_history'].append(self._det_seed_alignment(det))
        trk['visual_lock_conf'] = max(float(trk.get('visual_lock_conf', 0.0)), VISUAL_LOCK_CONF_ENTER if trk.get('visual_lock', False) else 0.0)
        if frame is not None:
            trk['frame_h'] = int(frame.shape[0])
        new_template = self._crop_template(frame, det)
        if new_template is not None:
            if trk.get('template') is None:
                trk['template'] = new_template
            elif tmpl_score >= 0.45 and match_score >= 0.45:
                trk['template'] = cv2.addWeighted(trk['template'], 0.85, new_template, 0.15, 0)

    def _update_track_visual(self, trk, det, frame, frame_idx, lock_score, tmpl_score):
        raw_cx, raw_cy = self._center(det)
        pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
        alpha_pos = float(VISUAL_LOCK_SMOOTH_ALPHA)
        cx = (1.0 - alpha_pos) * pred_cx + alpha_pos * raw_cx
        cy = (1.0 - alpha_pos) * pred_cy + alpha_pos * raw_cy
        dt = max(1, int(frame_idx) - int(trk['last_frame']))
        nvx = (cx - trk['cx']) / dt
        nvy = (cy - trk['cy']) / dt
        alpha = 0.28
        trk['vx'] = alpha * nvx + (1.0 - alpha) * trk['vx']
        trk['vy'] = alpha * nvy + (1.0 - alpha) * trk['vy']
        trk['cx'], trk['cy'] = cx, cy
        det_w, det_h = self._box_wh(det)
        trk['w'] = 0.88 * float(trk.get('w', det_w)) + 0.12 * det_w
        trk['h'] = 0.88 * float(trk.get('h', det_h)) + 0.12 * det_h
        full_h = int(frame.shape[0]) if frame is not None else int(trk.get('frame_h', CAPTURE_H))
        full_w = int(frame.shape[1]) if frame is not None else CAPTURE_W
        trk['box'] = [
            int(max(0, round(cx - trk['w'] * 0.5))),
            int(max(0, round(cy - trk['h'] * 0.5))),
            int(min(full_w, round(cx + trk['w'] * 0.5))),
            int(min(full_h, round(cy + trk['h'] * 0.5))),
        ]
        trk['last_frame'] = int(frame_idx)
        trk['hits'] += 1
        trk['misses'] = 0
        trk['visual_lock_hits'] = int(trk.get('visual_lock_hits', 0)) + 1
        trk['visual_lock_misses'] = 0
        trk['visual_lock_overdue_frames'] = max(
            0,
            self._visual_lock_yolo_gap(trk, frame_idx) - VISUAL_LOCK_RECHECK_INTERVAL,
        )
        lock_obs = self._clamp01(
            0.45 * min(1.0, lock_score / max(VISUAL_LOCK_MIN_SCORE * 2.0, 1.0))
            + 0.35 * tmpl_score
            + 0.20 * min(1.0, lock_score / 35.0)
        )
        prev_lock_conf = float(trk.get('visual_lock_conf', VISUAL_LOCK_CONF_ENTER))
        trk['visual_lock_conf'] = self._clamp01(
            (1.0 - VISUAL_LOCK_CONF_GAIN) * prev_lock_conf + VISUAL_LOCK_CONF_GAIN * lock_obs
        )
        trk['score'] = self._clamp01(0.94 * trk['score'] + 0.06 * lock_obs)
        trk['history'].append((cx, cy))
        trk['hit_history'].append(1)
        trk['background_risk_history'].append(0.0)
        trk['seed_alignment_history'].append(1.0)
        if frame is not None:
            trk['frame_h'] = int(frame.shape[0])
        new_template = self._crop_template(frame, det)
        if new_template is not None:
            if trk.get('template') is None:
                trk['template'] = new_template
            elif tmpl_score >= 0.50 and trk.get('visual_lock_conf', 0.0) >= VISUAL_LOCK_TEMPLATE_UPDATE_MIN_CONF:
                trk['template'] = cv2.addWeighted(trk['template'], 0.93, new_template, 0.07, 0)

    def _miss_track(self, trk):
        trk['misses'] += 1
        trk['vx'] *= MISS_VELOCITY_DECAY
        trk['vy'] *= MISS_VELOCITY_DECAY
        trk['score'] = self._clamp01(trk['score'] * 0.92)
        trk['hit_history'].append(0)
        trk.setdefault('yolo_hit_history', deque(maxlen=self.recent_window)).append(0)

    def _basic_is_confirmed(self, trk):
        recent_hits = sum(trk['hit_history'])
        traj_score = self._trajectory_score(trk)
        required_hits = self._adaptive_required_hits(trk)
        required_direct_hits = YOLO_DIRECT_CONFIRM_HITS
        required_direct_recent = YOLO_DIRECT_CONFIRM_RECENT_HITS
        required_direct_recent = max(required_direct_recent, min(required_direct_hits, self.recent_window))
        if trk.get('yolo_hits', 0) >= required_direct_hits:
            return (
                recent_hits >= required_direct_recent
                and trk['misses'] <= YOLO_DIRECT_CONFIRM_MAX_MISSES
                and trk['score'] >= YOLO_DIRECT_CONFIRM_SCORE
                and (not self.profile["high"] or traj_score >= TRACKER_MIN_TRAJ_SCORE)
            )
        return (
            trk['hits'] >= required_hits
            and recent_hits >= self.min_recent_hits
            and trk['misses'] <= YOLO_TRACK_MAX_CONFIRMED_MISSES
            and trk['score'] >= self.confirm_score
            and traj_score >= TRACKER_MIN_TRAJ_SCORE
        )

    def _is_static_confirmed(self, trk):
        if not ENABLE_STATIC_CONFIRM:
            return False
        if trk.get('yolo_hits', 0) < STATIC_CONFIRM_YOLO_HITS:
            return False
        if self._recent_yolo_hits(trk, self.recent_window) < STATIC_CONFIRM_RECENT_HITS:
            return False
        if trk.get('misses', 0) > STATIC_CONFIRM_MAX_MISSES:
            return False
        if float(trk.get('score', 0.0)) < STATIC_CONFIRM_MIN_SCORE:
            return False
        if self._mean_yolo_conf(trk) < STATIC_CONFIRM_MIN_MEAN_CONF:
            return False
        if self._track_speed_px_per_frame(trk) > STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME:
            return False
        if self._recent_net_motion_px(trk, STATIC_CONFIRM_RECENT_WINDOW) > STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX:
            return False
        stability = self._yolo_box_stability(trk, STATIC_CONFIRM_RECENT_HITS)
        if stability is None:
            return False
        center_jitter, box_jitter = stability
        if center_jitter > STATIC_CONFIRM_MAX_CENTER_JITTER:
            return False
        if box_jitter > STATIC_CONFIRM_MAX_BOX_JITTER_RATIO:
            return False
        risk_history = list(trk.get('background_risk_history', []))
        align_history = list(trk.get('seed_alignment_history', []))
        bg_risk = float(np.median(risk_history[-STATIC_CONFIRM_RECENT_HITS:])) if risk_history else 0.0
        seed_alignment = float(np.median(align_history[-STATIC_CONFIRM_RECENT_HITS:])) if align_history else 1.0
        if bg_risk > STATIC_CONFIRM_MAX_BG_RISK and seed_alignment < STATIC_CONFIRM_MIN_SEED_ALIGNMENT:
            return False
        return True

    def _is_confirmed(self, trk):
        if self._is_static_confirmed(trk):
            return True
        if not self._basic_is_confirmed(trk):
            return False
        if not self._passes_adaptive_background_confirmation(trk):
            return False
        return self._passes_motion_confirmation(trk)

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

    def update_visual_locks(self, frame=None, frame_idx=0):
        if not ENABLE_VISUAL_LOCK or not self.profile.get("high", False) or frame is None:
            return 0
        updated = 0
        for trk in self.trackers:
            if int(trk.get('last_frame', -1)) >= int(frame_idx):
                continue
            if not self._maybe_enter_visual_lock(trk, frame_idx):
                continue
            yolo_gap = self._visual_lock_yolo_gap(trk, frame_idx)
            overdue_frames = max(0, yolo_gap - VISUAL_LOCK_RECHECK_INTERVAL)
            recheck_due = overdue_frames > 0
            trk['visual_lock_recheck_due'] = recheck_due
            trk['visual_lock_overdue_frames'] = overdue_frames
            if (
                yolo_gap > VISUAL_LOCK_MAX_NO_YOLO_FRAMES
                or overdue_frames > VISUAL_LOCK_RECHECK_GRACE_FRAMES
            ):
                self._disable_visual_lock(trk)
                continue
            found = self._find_visual_lock_detection(trk, frame, frame_idx)
            if found is None:
                trk['visual_lock_misses'] = int(trk.get('visual_lock_misses', 0)) + 1
                prev_lock_conf = float(trk.get('visual_lock_conf', VISUAL_LOCK_CONF_ENTER))
                trk['visual_lock_conf'] = self._clamp01((1.0 - VISUAL_LOCK_CONF_DECAY) * prev_lock_conf)
                if (
                    trk['visual_lock_misses'] > VISUAL_LOCK_MAX_MISSES
                    or float(trk.get('visual_lock_conf', 0.0)) < VISUAL_LOCK_CONF_EXIT
                ):
                    self._disable_visual_lock(trk)
                continue
            lock_score, det, tmpl_score, _resp_score = found
            self._update_track_visual(trk, det, frame, frame_idx, lock_score, tmpl_score)
            if recheck_due:
                prev_lock_conf = float(trk.get('visual_lock_conf', VISUAL_LOCK_CONF_ENTER))
                trk['visual_lock_conf'] = self._clamp01((1.0 - VISUAL_LOCK_RECHECK_DECAY) * prev_lock_conf)
            if float(trk.get('visual_lock_conf', 0.0)) < VISUAL_LOCK_CONF_EXIT:
                self._disable_visual_lock(trk)
                continue
            updated += 1
        return updated

    def has_recent_yolo_track(self, frame_idx, max_gap_frames=45):
        for trk in self.trackers:
            if trk.get('yolo_hits', 0) <= 0:
                continue
            if float(trk.get('score', 0.0)) < 0.18:
                continue
            last_yolo_frame = int(trk.get('last_yolo_frame', trk.get('last_frame', frame_idx)))
            if int(frame_idx) - last_yolo_frame <= int(max_gap_frames):
                return True
        return False

    def get_yolo_seeded_search_rois(self, frame_idx, full_w, full_h, max_rois=MAX_TRACK_ROIS_PER_FRAME):
        rois = []
        max_age_frames = int(self.profile.get("track_search_max_age_frames", TRACK_SEARCH_MAX_AGE_FRAMES))
        max_predict_misses = int(self.profile.get("track_search_predict_max_misses", TRACK_SEARCH_PREDICT_MAX_MISSES))
        max_rois = int(self.profile.get("track_search_max_rois", max_rois))
        min_yolo_hits = int(self.profile.get("track_search_min_yolo_hits", TRACK_SEARCH_MIN_YOLO_HITS))
        min_recent_hits = int(self.profile.get("track_search_min_recent_hits", TRACK_SEARCH_MIN_RECENT_HITS))
        min_score = float(self.profile.get("track_search_min_score", TRACK_SEARCH_MIN_SCORE))
        recheck_min_gap_frames = int(self.profile.get("track_search_recheck_min_gap_frames", 0))
        visual_lock_only = bool(self.profile.get("track_search_visual_lock_only", False))
        for trk in self.trackers:
            track_age = max(0, int(frame_idx) - int(trk.get('last_frame', frame_idx)))
            if track_age > max_age_frames:
                continue
            if trk['misses'] > YOLO_TRACK_MAX_SEARCH_MISSES:
                continue
            if visual_lock_only and not trk.get('visual_lock', False):
                continue
            if recheck_min_gap_frames > 0:
                last_yolo_frame = int(trk.get('last_yolo_frame', trk.get('last_frame', frame_idx)))
                if int(frame_idx) - last_yolo_frame < recheck_min_gap_frames:
                    continue
            if self.profile["track_search_confirmed_only"] and not self._is_confirmed(trk):
                continue
            if self.profile["track_search_min_net_motion_px"] > 0.0 and self._net_motion_px(trk) < self.profile["track_search_min_net_motion_px"]:
                continue
            recent_hits = sum(trk.get('hit_history', []))
            if (
                trk.get('yolo_hits', 0) < min_yolo_hits
                or recent_hits < min_recent_hits
                or float(trk.get('score', 0.0)) < min_score
            ):
                continue
            box = self._predicted_box(trk, frame_idx) if 0 < trk['misses'] <= max_predict_misses else trk['box']
            cx = (float(box[0]) + float(box[2])) * 0.5
            cy = (float(box[1]) + float(box[3])) * 0.5
            crop_size = TRACK_ZOOM_CROP_SIZE if TRACK_ZOOM_CROP_SIZE > 0 else CROP_SIZE
            rx1, ry1, rx2, ry2 = crop_roi_from_center(cx, cy, full_w, full_h, crop_size=crop_size)
            priority = 1000.0 + float(trk.get('score', 0.0)) - 20.0 * float(trk['misses'])
            if TRACK_ZOOM_CROP_SIZE > 0:
                rois.append((rx1, ry1, rx2, ry2, priority, 0, 0, cx, cy, "zoom"))
            else:
                rois.append((rx1, ry1, rx2, ry2, priority))
        rois = merge_nearby_boxes(rois, dist_thresh=160)
        rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
        return rois[:max_rois]

    def get_hover_recheck_rois(self, frame_idx, full_w, full_h, max_rois=1):
        if not ENABLE_HOVER_HOLD:
            return []
        rois = []
        for trk in self.trackers:
            if not self._is_hover_candidate(trk, frame_idx):
                continue
            yolo_gap = self._visual_lock_yolo_gap(trk, frame_idx)
            if yolo_gap < HOVER_RECHECK_INTERVAL_FRAMES:
                continue
            if yolo_gap % HOVER_RECHECK_INTERVAL_FRAMES > max(1, PROCESS_EVERY_N_FRAMES):
                continue
            pred_cx, pred_cy, _ = self._predict_center(trk, frame_idx)
            rx1, ry1, rx2, ry2 = crop_roi_from_center(
                pred_cx,
                pred_cy,
                full_w,
                full_h,
                crop_size=min(HOVER_ROI_SIZE, max(full_w, full_h)),
            )
            priority = 1400.0 + float(trk.get('score', 0.0)) - 3.0 * float(yolo_gap // HOVER_RECHECK_INTERVAL_FRAMES)
            rois.append((rx1, ry1, rx2, ry2, priority, 0, 0, pred_cx, pred_cy, "zoom"))
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
            det_conf_history = list(trk.get('det_conf_history', []))
            background_risk, adaptive_evidence = self._adaptive_background_evidence(trk)
            seed_alignment_history = list(trk.get('seed_alignment_history', []))
            confirmed.append({
                'id': trk['id'],
                'box': box,
                'history': history,
                'score': trk['score'],
                'trajectory_score': self._trajectory_score(trk),
                'live': trk['misses'] == 0 and track_age <= TRACK_LIVE_MAX_AGE_FRAMES,
                'misses': trk['misses'],
                'age': track_age,
                'yolo_hits': int(trk.get('yolo_hits', 0)),
                'recent_hits': int(sum(trk.get('hit_history', []))),
                'net_motion_px': float(self._net_motion_px(trk)),
                'mean_det_conf': float(np.mean(det_conf_history)) if det_conf_history else 0.0,
                'background_risk': background_risk,
                'seed_alignment': float(np.median(seed_alignment_history)) if seed_alignment_history else 1.0,
                'adaptive_evidence': adaptive_evidence,
                'visual_lock': bool(trk.get('visual_lock', False)),
                'visual_lock_hits': int(trk.get('visual_lock_hits', 0)),
                'visual_lock_conf': float(trk.get('visual_lock_conf', 0.0)),
                'visual_lock_recheck_due': bool(trk.get('visual_lock_recheck_due', False)),
                'visual_lock_overdue_frames': int(trk.get('visual_lock_overdue_frames', 0)),
                'yolo_gap_frames': self._visual_lock_yolo_gap(trk, frame_idx),
                'static_confirmed': bool(self._is_static_confirmed(trk)),
                'last_yolo_frame': int(trk.get('last_yolo_frame', trk.get('last_frame', frame_idx))),
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


def open_camera_capture(cam_idx):
    target_hw_id = CAM_MAP[cam_idx]
    source = get_camera_node(target_hw_id) or f"/dev/video{cam_idx * 2}"
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, source


def gray_small_from_capture_frame(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        gray = frame
    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = frame_to_gray(frame)
    if gray.shape[1] == DIFF_W and gray.shape[0] == DIFF_H:
        return gray
    return cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)


def build_initial_camera_bg_model(cam_idx):
    cap, source = open_camera_capture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera source: {source}")
    samples = []
    start_ts = time.monotonic()
    next_sample_ts = start_ts
    sample_interval = STATIC_BG_SECONDS / float(max(1, LOW_BG_SAMPLE_FRAMES))
    last_read_ts = start_ts
    try:
        while time.monotonic() - start_ts <= STATIC_BG_SECONDS + 5.0 and len(samples) < LOW_BG_SAMPLE_FRAMES:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            last_read_ts = time.monotonic()
            if last_read_ts < next_sample_ts:
                continue
            gray_small = gray_small_from_capture_frame(frame)
            if gray_small is None:
                continue
            samples.append(gray_small.copy())
            next_sample_ts += sample_interval
    finally:
        cap.release()
    if len(samples) < max(10, LOW_BG_SAMPLE_FRAMES // 2):
        raise RuntimeError(f"not enough background samples from {source}: {len(samples)}")
    bg, tol, qrange = build_static_bg_model(samples)
    return bg, tol, qrange, len(samples), last_read_ts - start_ts, source

# ==========================================
# 4
# ==========================================
def inference_worker(worker_idx=0):
    yolo = None
    stale_task_count = 0
    inferred_task_count = 0
    stale_by_cam = [0 for _ in range(N_CAM)]
    inferred_by_cam = [0 for _ in range(N_CAM)]
    next_cam = worker_idx % N_CAM
    try: 
        core_index = worker_idx if NPU_WORKER_COUNT > 1 else None
        yolo = YoloRKNN(
            MODEL_PATH,
            (640, 640),
            ACTIVE_YOLO_CONF_THRESH,
            0.45,
            core_index=core_index,
        )
        print(f"--> RKNN worker {worker_idx} loaded.", flush=True)
    except Exception as e:
        print(f"[Fatal] RKNN Init Failed: {type(e).__name__}: {e}", flush=True)
        stop_event.set()
        return
    
    while not stop_event.is_set():
        did_work = False
        for offset in range(N_CAM):
            i = (next_cam + offset) % N_CAM
            try:
                item = inf_queues[i].get_nowait()
                if len(item) >= 7:
                    roi, x, y, src_frame_idx, enqueue_ts, seed_cx, seed_cy = item[:7]
                else:
                    roi, x, y, src_frame_idx, enqueue_ts = item
                    seed_cx = float(x) + roi.shape[1] * 0.5
                    seed_cy = float(y) + roi.shape[0] * 0.5
                did_work = True
                next_cam = (i + 1) % N_CAM
                if time.monotonic() - enqueue_ts > NPU_TASK_MAX_AGE_SEC:
                    stale_task_count += 1
                    stale_by_cam[i] += 1
                    if stale_task_count % 50 == 0:
                        print(f"--> RKNN scheduler dropped {stale_task_count} stale ROI tasks.", flush=True)
                    break
                res = yolo.infer(roi)
                inferred_task_count += 1
                inferred_by_cam[i] += 1
                rh, rw = roi.shape[:2]
                if res is None:
                    res = []
                put_latest(res_queues[i], (res, x, y, rw, rh, src_frame_idx, seed_cx, seed_cy))
                break
            except queue.Empty:
                pass
            except Exception as exc:
                print(
                    f"[Fatal] RKNN inference failed on cam {i}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                stop_event.set()
                break
        if not did_work: time.sleep(0.001)
    if yolo:
        yolo.release()
    print(
        f"--> RKNN worker {worker_idx} summary: inferred={inferred_task_count} "
        f"stale_dropped={stale_task_count} inferred_by_cam={inferred_by_cam} "
        f"stale_by_cam={stale_by_cam}",
        flush=True,
    )

# ==========================================
# 5
# ==========================================
def capture_job(cam_idx, initial_bg_gray=None, initial_bg_tol=None, initial_bg_qrange=None, layer_profile=None):
    if cam_idx not in CAM_MAP: return
    cam_profile = layer_profile or get_cam_layer_profile(cam_idx)
    set_current_layer_profile(cam_profile)
    if SIMULATE_BY_VIDEOS:
        source = get_video_test_source(cam_idx)
        cap = cv2.VideoCapture(source)
        print(f"--> Cam {cam_idx} video source: {source}", flush=True)
    else:
        cap, source = open_camera_capture(cam_idx)
        print(f"--> Cam {cam_idx} camera source: {source}", flush=True)
    print(
        f"--> Cam {cam_idx} layer profile: {cam_profile['name']} "
        f"conf={cam_profile['conf_thresh']:.2f} "
        f"day={cam_profile.get('day_scene')} "
        f"roi={'native640+tight' if cam_profile.get('enable_tight_motion_roi', False) else 'native640'} "
        f"layer_conf={cam_profile.get('layer_confidence', 0.0):.2f} "
        f"reason={cam_profile['reason']}",
        flush=True,
    )
    if not cap.isOpened():
        print(f"--> Cam {cam_idx} cannot open source: {source}", flush=True)
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    actual_capture_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or CAPTURE_W)
    actual_capture_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or CAPTURE_H)
    measured_fps = source_fps
    frame_arrival_times = deque(maxlen=FPS_ESTIMATE_WINDOW)
    video_frame_period = 1.0 / max(1.0, source_fps)
    video_next_frame_ts = time.monotonic()
    video_max_frames = int(round(VIDEO_TEST_MAX_SECONDS * source_fps)) if VIDEO_TEST_MAX_SECONDS > 0 else 0
    output_writer = None
    output_path = ""
    if SIMULATE_BY_VIDEOS and VIDEO_TEST_OUTPUT:
        output_root, output_ext = os.path.splitext(VIDEO_TEST_OUTPUT)
        output_ext = output_ext or ".mp4"
        output_path = (
            f"{output_root}_cam{cam_idx}{output_ext}"
            if N_CAM > 1
            else f"{output_root}{output_ext}"
        )
        output_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (640, 360),
        )
        if not output_writer.isOpened():
            print(f"--> Cam {cam_idx} cannot open output: {output_path}", flush=True)
            output_writer.release()
            output_writer = None
        else:
            print(f"--> Cam {cam_idx} output: {output_path}", flush=True)
    detection_csv_file = None
    detection_csv_writer = None
    if SIMULATE_BY_VIDEOS and VIDEO_TEST_DETECTION_CSV:
        csv_root, csv_ext = os.path.splitext(VIDEO_TEST_DETECTION_CSV)
        csv_ext = csv_ext or ".csv"
        detection_csv_path = (
            f"{csv_root}_cam{cam_idx}{csv_ext}"
            if N_CAM > 1
            else f"{csv_root}{csv_ext}"
        )
        csv_dir = os.path.dirname(os.path.abspath(detection_csv_path))
        os.makedirs(csv_dir, exist_ok=True)
        detection_csv_file = open(detection_csv_path, "w", newline="", encoding="utf-8")
        detection_csv_writer = csv.DictWriter(
            detection_csv_file,
            fieldnames=[
                "kind", "source_frame", "display_frame", "time_s", "track_id",
                "x1", "y1", "x2", "y2", "cx", "cy", "width", "height",
                "center_x_norm", "center_y_norm", "lower_half", "confidence",
                "misses", "age", "traj_score", "yolo_hits",
                "recent_hits", "net_motion_px", "mean_det_conf",
                "background_risk", "seed_alignment", "adaptive_evidence",
                "visual_lock", "visual_lock_hits", "visual_lock_conf",
                "visual_lock_recheck_due", "visual_lock_overdue_frames", "yolo_gap_frames",
                "static_confirmed",
            ],
        )
        detection_csv_writer.writeheader()
        print(f"--> Cam {cam_idx} detection CSV: {detection_csv_path}", flush=True)
    if SIMULATE_BY_VIDEOS and VIDEO_TEST_SAVE_ROIS_DIR:
        os.makedirs(VIDEO_TEST_SAVE_ROIS_DIR, exist_ok=True)

    frame_t1 = None
    tracker = TrajectoryFilter(
        max_dist=TRACKER_MAX_DIST,
        min_hits=TRACKER_MIN_HITS,
        recent_window=TRACKER_RECENT_WINDOW,
        min_recent_hits=TRACKER_MIN_RECENT_HITS,
        confirm_score=TRACKER_CONFIRM_SCORE,
        template_size=TRACKER_TEMPLATE_SIZE,
        max_gate=TRACKER_MAX_GATE,
        max_speed_mps=TRACKER_MAX_SPEED_MPS,
        fps=source_fps,
        layer_profile=cam_profile,
    )
    print(
        f"--> Cam {cam_idx} frame-diff cadence: capture={actual_capture_w}x{actual_capture_h} "
        f"diff={DIFF_W}x{DIFF_H} reported_fps={source_fps:.1f} "
        f"fixed_stride={PROCESS_EVERY_N_FRAMES} serialized=yes "
        f"direct_fullframe={'yes' if DIRECT_FULL_FRAME_INFERENCE else 'no'} "
        f"zoom={MOTION_ZOOM_CROP_SIZE}x{MOTION_ZOOM_MAX_ROIS} "
        f"fullframe_fallback_only={'yes' if FULLFRAME_FALLBACK_ONLY else 'no'}",
        flush=True,
    )
    f_idx = 0
    local_data_sender = DataSender(DATA_TARGETS, BOARD_ID)
    v_sender = VideoSender(
        VIDEO_TARGET_IP,
        VIDEO_BASE_PORT,
        width=VIDEO_STREAM_W,
        height=VIDEO_STREAM_H,
        quality=VIDEO_STREAM_QUALITY,
    )
    learning_mode = True
    current_draw_boxes =[]
    bg_samples = []
    bg_gray, bg_tol, bg_qrange = initial_bg_gray, initial_bg_tol, initial_bg_qrange
    bg_risk_model = build_background_risk_model(bg_qrange)
    bg_learning_start_ts = None
    bg_next_sample_ts = 0.0
    last_tracker_update_frame = 0
    last_grouped_result_frame = -1
    pending_result_expected = {}
    pending_result_received = {}
    pending_result_boxes = {}
    pending_result_created = {}
    pending_result_frames = {}
    submitted_task_count = 0
    replaced_task_count = 0
    submitted_group_count = 0
    complete_group_count = 0
    partial_group_count = 0
    frame_expired_group_count = 0
    tracker_group_update_count = 0
    result_lag_frame_sum = 0
    result_lag_frame_max = 0
    result_lag_sample_count = 0

    def cancel_replaced_tasks(items):
        nonlocal replaced_task_count
        for item in items:
            if len(item) < 4:
                continue
            replaced_frame_idx = int(item[3])
            replaced_task_count += 1
            expected = pending_result_expected.get(replaced_frame_idx)
            if expected is None:
                continue
            received = pending_result_received.get(replaced_frame_idx, 0)
            expected = max(received, expected - 1)
            pending_result_expected[replaced_frame_idx] = expected
            if expected == 0 and received == 0:
                pending_result_expected.pop(replaced_frame_idx, None)
                pending_result_received.pop(replaced_frame_idx, None)
                pending_result_boxes.pop(replaced_frame_idx, None)
                pending_result_created.pop(replaced_frame_idx, None)
                pending_result_frames.pop(replaced_frame_idx, None)

    def write_detection_row(kind, source_frame_idx, display_frame_idx, box, track=None):
        if detection_csv_writer is None or box is None or len(box) < 4:
            return
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        confidence = float(box[4]) if len(box) > 4 else float((track or {}).get("score", 0.0))
        detection_csv_writer.writerow({
            "kind": kind,
            "source_frame": int(source_frame_idx),
            "display_frame": int(display_frame_idx),
            "time_s": f"{float(source_frame_idx) / max(source_fps, 1.0):.3f}",
            "track_id": int((track or {}).get("id", -1)),
            "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
            "cx": f"{cx:.2f}", "cy": f"{cy:.2f}",
            "width": f"{max(0.0, x2 - x1):.2f}",
            "height": f"{max(0.0, y2 - y1):.2f}",
            "center_x_norm": f"{cx / max(1.0, float(CAPTURE_W)):.6f}",
            "center_y_norm": f"{cy / max(1.0, float(CAPTURE_H)):.6f}",
            "lower_half": int(cy >= 0.5 * float(CAPTURE_H)),
            "confidence": f"{confidence:.6f}",
            "misses": int((track or {}).get("misses", 0)),
            "age": int((track or {}).get("age", 0)),
            "traj_score": f"{float((track or {}).get('trajectory_score', 0.0)):.6f}",
            "yolo_hits": int((track or {}).get("yolo_hits", 0)),
            "recent_hits": int((track or {}).get("recent_hits", 0)),
            "net_motion_px": f"{float((track or {}).get('net_motion_px', 0.0)):.6f}",
            "mean_det_conf": f"{float((track or {}).get('mean_det_conf', 0.0)):.6f}",
            "background_risk": f"{float((track or {}).get('background_risk', box[5] if len(box) > 5 else 0.0)):.6f}",
            "seed_alignment": f"{float((track or {}).get('seed_alignment', box[6] if len(box) > 6 else 1.0)):.6f}",
            "adaptive_evidence": f"{float((track or {}).get('adaptive_evidence', 0.0)):.6f}",
            "visual_lock": int(bool((track or {}).get("visual_lock", False))),
            "visual_lock_hits": int((track or {}).get("visual_lock_hits", 0)),
            "visual_lock_conf": f"{float((track or {}).get('visual_lock_conf', 0.0)):.6f}",
            "visual_lock_recheck_due": int(bool((track or {}).get("visual_lock_recheck_due", False))),
            "visual_lock_overdue_frames": int((track or {}).get("visual_lock_overdue_frames", 0)),
            "yolo_gap_frames": int((track or {}).get("yolo_gap_frames", 0)),
            "static_confirmed": int(bool((track or {}).get("static_confirmed", False))),
        })

    def drain_inference_results(current_frame, current_frame_idx, full_w, full_h):
        nonlocal last_grouped_result_frame, last_tracker_update_frame
        nonlocal complete_group_count, partial_group_count, frame_expired_group_count
        nonlocal tracker_group_update_count
        nonlocal result_lag_frame_sum, result_lag_frame_max, result_lag_sample_count

        drained_boxes = []
        got_result = False
        while not stop_event.is_set():
            try:
                result = res_queues[cam_idx].get_nowait()
            except queue.Empty:
                break

            if len(result) >= 8:
                dets, xo, yo, roi_w, roi_h, src_frame_idx, seed_cx, seed_cy = result[:8]
            else:
                dets, xo, yo, roi_w, roi_h, src_frame_idx = result
                seed_cx = float(xo) + float(roi_w) * 0.5
                seed_cy = float(yo) + float(roi_h) * 0.5

            src_frame_idx = int(src_frame_idx)
            if src_frame_idx <= last_grouped_result_frame:
                continue
            pending_result_received[src_frame_idx] = pending_result_received.get(src_frame_idx, 0) + 1
            pending_result_boxes.setdefault(src_frame_idx, [])
            pending_result_created.setdefault(src_frame_idx, time.monotonic())
            sx = float(roi_w) / float(CROP_SIZE)
            sy = float(roi_h) / float(CROP_SIZE)
            for det in dets:
                score = float(det[4]) if len(det) > 4 else float(cam_profile["conf_thresh"])
                if score < float(cam_profile["conf_thresh"]):
                    continue
                mapped_box = [
                    int(max(0, min(full_w, det[0] * sx + xo))),
                    int(max(0, min(full_h, det[1] * sy + yo))),
                    int(max(0, min(full_w, det[2] * sx + xo))),
                    int(max(0, min(full_h, det[3] * sy + yo))),
                    score,
                ]
                background_risk = background_risk_for_box(mapped_box, bg_risk_model, full_w, full_h)
                seed_alignment = motion_seed_alignment(mapped_box, seed_cx, seed_cy)
                if background_risk > seed_alignment:
                    continue
                mapped_box.extend((background_risk, seed_alignment))
                if (
                    mapped_box[2] > mapped_box[0]
                    and mapped_box[3] > mapped_box[1]
                    and is_valid_detection_box_size(mapped_box, cam_profile)
                    and not is_vertical_strip_box(mapped_box)
                ):
                    pending_result_boxes[src_frame_idx].append(mapped_box)

        group_now = time.monotonic()
        for result_frame_idx in sorted(pending_result_expected):
            if result_frame_idx <= last_grouped_result_frame:
                continue
            expected_count = pending_result_expected.get(result_frame_idx, 0)
            received_count = pending_result_received.get(result_frame_idx, 0)
            group_wait = group_now - pending_result_created.get(result_frame_idx, group_now)
            group_complete = received_count >= expected_count
            group_expired = group_wait >= RESULT_GROUP_WAIT_SEC
            frame_expired = current_frame_idx - result_frame_idx > MAX_INFERENCE_RESULT_AGE_FRAMES
            if not group_complete and not group_expired and not frame_expired:
                break

            if frame_expired:
                frame_expired_group_count += 1
            elif group_complete:
                complete_group_count += 1
            else:
                partial_group_count += 1

            grouped_boxes = merge_nearby_boxes(
                pending_result_boxes.get(result_frame_idx, []),
                dist_thresh=100,
            )
            if not frame_expired:
                got_result = True
                drained_boxes.extend(grouped_boxes)
                for grouped_box in grouped_boxes:
                    write_detection_row("raw", result_frame_idx, current_frame_idx, grouped_box)
                result_lag_frames = max(0, int(current_frame_idx) - int(result_frame_idx))
                result_lag_frame_sum += result_lag_frames
                result_lag_frame_max = max(result_lag_frame_max, result_lag_frames)
                result_lag_sample_count += 1
                if ENABLE_TRAJECTORY_TRACKING:
                    source_frame = pending_result_frames.get(result_frame_idx, current_frame)
                    tracker.update(grouped_boxes, frame=source_frame, frame_idx=result_frame_idx)
                    tracker_group_update_count += 1
                    last_tracker_update_frame = result_frame_idx
                    if DEBUG_ASYNC_TRACE:
                        confirmed_now = tracker.get_confirmed_tracks(result_frame_idx)
                        print(
                            f"ASYNC_TRACE frame={result_frame_idx} rois={expected_count} "
                            f"raw={len(grouped_boxes)} tracks={len(tracker.trackers)} "
                            f"confirmed={len(confirmed_now)}",
                            flush=True,
                        )
            last_grouped_result_frame = result_frame_idx
            pending_result_expected.pop(result_frame_idx, None)
            pending_result_received.pop(result_frame_idx, None)
            pending_result_boxes.pop(result_frame_idx, None)
            pending_result_created.pop(result_frame_idx, None)
            pending_result_frames.pop(result_frame_idx, None)

        return drained_boxes, got_result

    while not stop_event.is_set():
        if SIMULATE_BY_VIDEOS and video_max_frames > 0 and f_idx >= video_max_frames:
            print(
                f"--> Cam {cam_idx} reached video limit: "
                f"frames={f_idx} seconds={VIDEO_TEST_MAX_SECONDS:.1f}",
                flush=True,
            )
            break
        if SIMULATE_BY_VIDEOS and VIDEO_TEST_REALTIME:
            now_ts = time.monotonic()
            if video_next_frame_ts > now_ts:
                time.sleep(video_next_frame_ts - now_ts)
            elif now_ts - video_next_frame_ts > 2.0 * video_frame_period:
                video_next_frame_ts = now_ts
            video_next_frame_ts += video_frame_period
        ret, frame = cap.read()
        if not ret:
            if SIMULATE_BY_VIDEOS and VIDEO_TEST_LOOP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            elif SIMULATE_BY_VIDEOS:
                print(f"--> Cam {cam_idx} reached video EOF after frames={f_idx}", flush=True)
                break
            if not ret:
                time.sleep(0.1)
                continue
        if (
            SIMULATE_BY_VIDEOS
            and VIDEO_TEST_RESIZE_W > 0
            and VIDEO_TEST_RESIZE_H > 0
            and (frame.shape[1] != VIDEO_TEST_RESIZE_W or frame.shape[0] != VIDEO_TEST_RESIZE_H)
        ):
            frame = cv2.resize(frame, (VIDEO_TEST_RESIZE_W, VIDEO_TEST_RESIZE_H), interpolation=cv2.INTER_AREA)
        gray_frame = None
        if len(frame.shape) == 2:
            gray_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray_frame = frame[:, :, 0]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]
        f_idx += 1
        capture_now_ts = time.monotonic()
        frame_arrival_times.append(capture_now_ts)
        if len(frame_arrival_times) >= 6:
            sample_seconds = frame_arrival_times[-1] - frame_arrival_times[0]
            if sample_seconds > 0.0:
                measured_fps = (len(frame_arrival_times) - 1) / sample_seconds
                tracker.fps = max(1.0, measured_fps)
        stream_allowed = is_video_stream_allowed()
        visual_enabled = stream_allowed or SHOW_INDIVIDUAL_WINDOWS or output_writer is not None
        send_video_this_frame = stream_allowed and ((f_idx + cam_idx) % VIDEO_SEND_EVERY_N_FRAMES == 0)
        draw_this_frame = send_video_this_frame or SHOW_INDIVIDUAL_WINDOWS or output_writer is not None
        if learning_mode and (time.time() - SYSTEM_START_TIME > INIT_TIME):
            learning_mode = False

        raw_boxes_in_this_frame, got_inference_result = drain_inference_results(frame, f_idx, W, H)

        if (f_idx + cam_idx) % PROCESS_EVERY_N_FRAMES == 0:
            gray = gray_frame if gray_frame is not None else frame_to_gray(frame)
            if gray.shape[1] == DIFF_W and gray.shape[0] == DIFF_H:
                gray_small = gray
            else:
                gray_small = cv2.resize(gray, (DIFF_W, DIFF_H), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = W / float(DIFF_W), H / float(DIFF_H)

            if ENABLE_LOW_LAYER_STATIC_BG_MASK and bg_gray is None:
                now_ts = time.monotonic()
                if bg_learning_start_ts is None:
                    bg_learning_start_ts = now_ts
                    bg_next_sample_ts = now_ts
                    print(f"--> Cam {cam_idx} learning its own static background.", flush=True)
                sample_interval = cam_profile["static_bg_seconds"] / float(max(1, LOW_BG_SAMPLE_FRAMES))
                if now_ts >= bg_next_sample_ts and len(bg_samples) < LOW_BG_SAMPLE_FRAMES:
                    bg_samples.append(gray_small.copy())
                    bg_next_sample_ts += sample_interval
                learning_elapsed = now_ts - bg_learning_start_ts
                enough_samples = len(bg_samples) >= LOW_BG_SAMPLE_FRAMES
                fallback_samples = learning_elapsed >= cam_profile["static_bg_seconds"] + 2.0 and len(bg_samples) >= 20
                if learning_elapsed >= cam_profile["static_bg_seconds"] and (enough_samples or fallback_samples):
                    bg_sample_count = len(bg_samples)
                    bg_gray, bg_tol, bg_qrange = build_static_bg_model(bg_samples)
                    bg_risk_model = build_background_risk_model(bg_qrange)
                    auto_high, auto_reason, day_scene, layer_confidence = estimate_layer_from_background(bg_gray, bg_qrange)
                    cam_profile = make_layer_profile(
                        auto_high,
                        f"online:{auto_reason}",
                        day_scene=day_scene,
                        layer_confidence=layer_confidence,
                    )
                    CAM_LAYER_PROFILES[cam_idx] = cam_profile
                    set_current_layer_profile(cam_profile)
                    tracker.profile = cam_profile
                    bg_samples = []
                    print(
                        f"--> Cam {cam_idx} own static background ready: "
                        f"seconds={learning_elapsed:.1f} samples={bg_sample_count} "
                        f"day={cam_profile.get('day_scene')} "
                        f"roi=native640+tight "
                        f"layer_conf={cam_profile.get('layer_confidence', 0.0):.2f}",
                        flush=True,
                    )
                frame_t1 = gray_small
                if not DIRECT_FULL_FRAME_INFERENCE:
                    continue
            
            rois =[]
            motion_mask = None
            track_rois = (
                tracker.get_yolo_seeded_search_rois(f_idx, W, H)
                if ENABLE_TRAJECTORY_TRACKING and ENABLE_TRACK_SEARCH_ROIS
                else []
            )
            hover_rois = (
                tracker.get_hover_recheck_rois(f_idx, W, H, max_rois=1)
                if ENABLE_TRAJECTORY_TRACKING and ENABLE_HOVER_HOLD
                else []
            )
            track_roi_keys = set()
            diff_img = None
            if ENABLE_FRAME_DIFF_ROIS and frame_t1 is not None:
                with FRAME_DIFF_LOCK:
                    diff_img = cv2.absdiff(frame_t1, gray_small)
                    _, mask = cv2.threshold(diff_img, cam_profile["diff_thresh"], 255, cv2.THRESH_BINARY)
                    bg_change_mask = build_static_bg_change_mask(gray_small, bg_gray, bg_tol)
                    if bg_change_mask is not None and cam_profile.get("enable_static_bg_change_gate", True):
                        mask = cv2.bitwise_and(mask, bg_change_mask)
                    mask = cleanup_motion_mask_with_iters(
                        mask,
                        MOTION_ERODE_ITER,
                        MOTION_DILATE_ITER,
                        MOTION_CLOSE_ITER,
                    )
                    motion_mask = cv2.bitwise_or(motion_mask, mask) if motion_mask is not None else mask
                    diff_rois = motion_rois_from_mask(
                        mask,
                        diff_img,
                        gray_small,
                        scale_x,
                        scale_y,
                        W,
                        H,
                    )
                if diff_rois:
                    rois = merge_nearby_boxes(diff_rois + rois, dist_thresh=160)
                    rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                    rois = prioritize_diverse_rois(rois, W, H)

            gray_seed_rois = gray_seed_rois_from_frame(gray_small, scale_x, scale_y, W, H)
            if gray_seed_rois:
                rois = merge_nearby_boxes(gray_seed_rois + rois, dist_thresh=160)
                rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                rois = prioritize_diverse_rois(rois, W, H)

            require_track_motion = bool(cam_profile.get("track_search_require_current_motion", TRACK_REQUIRE_CURRENT_MOTION))
            if ENABLE_FRAME_DIFF_ROIS and require_track_motion and track_rois:
                track_rois = [r for r in track_rois if track_roi_has_motion(r, motion_mask, W, H)]
            if ENABLE_FRAME_DIFF_ROIS and (track_rois or hover_rois):
                track_roi_keys = {tuple(int(v) for v in r[:4]) for r in (track_rois + hover_rois)}
                rois = merge_nearby_boxes(track_rois + hover_rois + rois, dist_thresh=160)
                rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                rois = prioritize_diverse_rois(rois, W, H)

            if (
                ENABLE_FULLFRAME_SEARCH_FALLBACK
                and not DIRECT_FULL_FRAME_INFERENCE
                and (f_idx + cam_idx) % FULLFRAME_SEARCH_INTERVAL_FRAMES == 0
                and not tracker.has_recent_yolo_track(f_idx, FULLFRAME_SEARCH_NO_YOLO_GAP_FRAMES)
            ):
                rois.append((0, 0, W, H, FULLFRAME_SEARCH_SCORE))

            if DIRECT_FULL_FRAME_INFERENCE:
                zoom_rois = [r for r in rois if roi_is_zoom_crop(r)]
                zoom_rois.sort(key=lambda r: r[4] if len(r) > 4 else 0.0, reverse=True)
                selected_zoom = zoom_rois[:MOTION_ZOOM_MAX_ROIS]
                if FULLFRAME_FALLBACK_ONLY and selected_zoom:
                    rois = selected_zoom
                else:
                    rois = [(0, 0, W, H, 1.0)] + selected_zoom
                track_roi_keys.clear()

            if ENABLE_FRAME_DIFF_ROIS:
                frame_t1 = gray_small
            if f_idx not in pending_result_expected:
                submitted_group_count += 1
                pending_result_expected[f_idx] = 0
                pending_result_received[f_idx] = 0
                pending_result_boxes[f_idx] = []
                pending_result_created[f_idx] = time.monotonic()
                pending_result_frames[f_idx] = gray.copy()
            if NPU_REPLACE_PENDING_ON_UPDATE:
                cancel_replaced_tasks(clear_pending(inf_queues[cam_idx]))
            enqueue_ts = time.monotonic()
            for roi in rois[:MAX_ENQUEUED_ROIS_PER_UPDATE]:
                x1, y1, x2, y2 = roi[:4]
                seed_cx, seed_cy = roi_seed_center(roi)
                if roi_is_zoom_crop(roi):
                    roi_img = frame[y1:y2, x1:x2].copy()
                    origin_x = int(x1)
                    origin_y = int(y1)
                elif cam_profile.get("enable_tight_motion_roi", False) and roi_is_padded_canvas(roi):
                    roi_img = make_padded_roi_canvas(frame, roi)
                    origin_x = int(x1) - int(roi[5])
                    origin_y = int(y1) - int(roi[6])
                else:
                    roi_img = frame[y1:y2, x1:x2].copy()
                    origin_x = int(x1)
                    origin_y = int(y1)
                    roi_img, pad_x, pad_y = pad_roi_image_to_square(roi_img)
                    origin_x -= int(pad_x)
                    origin_y -= int(pad_y)
                if roi_img.size == 0:
                    continue
                if put_latest(
                    inf_queues[cam_idx],
                    (roi_img, origin_x, origin_y, f_idx, enqueue_ts, seed_cx, seed_cy),
                ):
                    submitted_task_count += 1
                    pending_result_expected[f_idx] = pending_result_expected.get(f_idx, 0) + 1
                    pending_result_received.setdefault(f_idx, 0)
                    pending_result_boxes.setdefault(f_idx, [])
                    pending_result_created.setdefault(f_idx, time.monotonic())
                    pending_result_frames.setdefault(f_idx, gray.copy())
                    frame_seconds = f_idx / max(1.0, source_fps)
                    save_roi_now = (
                        VIDEO_TEST_SAVE_ROIS_DIR
                        and frame_seconds >= VIDEO_TEST_SAVE_ROIS_START
                        and (VIDEO_TEST_SAVE_ROIS_END <= 0 or frame_seconds <= VIDEO_TEST_SAVE_ROIS_END)
                    )
                    if save_roi_now:
                        roi_name = (
                            f"cam{cam_idx}_f{f_idx:06d}_t{frame_seconds:07.3f}_"
                            f"x{origin_x}_y{origin_y}.png"
                        )
                        cv2.imwrite(os.path.join(VIDEO_TEST_SAVE_ROIS_DIR, roi_name), roi_img)
                    is_track_roi = tuple(int(v) for v in roi[:4]) in track_roi_keys
                    if is_track_roi:
                        label = "TRKROI"
                        color = (0, 255, 255)
                    else:
                        label = "ROI"
                        color = (255, 255, 255)
                    if visual_enabled and DRAW_INTERMEDIATE_BOXES:
                        current_draw_boxes.append([x1, y1, x2, y2, color, label, 2])

        new_raw_boxes, got_new_result = drain_inference_results(frame, f_idx, W, H)
        raw_boxes_in_this_frame.extend(new_raw_boxes)
        got_inference_result = got_inference_result or got_new_result
        visual_lock_updates = 0
        if (
            ENABLE_TRAJECTORY_TRACKING
            and ENABLE_VISUAL_LOCK
            and (f_idx + cam_idx) % VISUAL_LOCK_EVERY_N_FRAMES == 0
        ):
            visual_lock_updates = tracker.update_visual_locks(frame=frame, frame_idx=f_idx)

        unique_raw_boxes = merge_nearby_boxes(raw_boxes_in_this_frame, dist_thresh=100)
        
        for rb in unique_raw_boxes:
            if visual_enabled and DRAW_INTERMEDIATE_BOXES:
                current_draw_boxes.append([rb[0], rb[1], rb[2], rb[3], (0, 0, 255), "Raw", 2])


        if ENABLE_TRAJECTORY_TRACKING:
            tracker_query_frame = f_idx if visual_lock_updates > 0 else (last_tracker_update_frame if last_tracker_update_frame > 0 else f_idx)
            confirmed_tracks = tracker.get_confirmed_tracks(tracker_query_frame)
        else:
            confirmed_tracks = []
            for idx, rb in enumerate(unique_raw_boxes):
                cx = (rb[0] + rb[2]) * 0.5
                cy = (rb[1] + rb[3]) * 0.5
                confirmed_tracks.append({
                    'id': idx + 1,
                    'box': rb[:4],
                    'history': [(cx, cy)],
                    'score': rb[4] if len(rb) > 4 else float(cam_profile["conf_thresh"]),
                    'live': True,
                    'misses': 0,
                })
        live_confirmed_tracks = [t for t in confirmed_tracks if t.get('live', True)]
        confirmed = [t['box'] for t in live_confirmed_tracks]
        for t in live_confirmed_tracks:
            box = t['box']
            write_detection_row("target", tracker_query_frame, f_idx, box, track=t)
            if visual_enabled:
                if t.get('static_confirmed', False):
                    label = "STATIC"
                elif (
                    t.get('visual_lock', False)
                    and int(t.get('visual_lock_hits', 0)) > 0
                    and float(t.get('visual_lock_conf', 0.0)) >= VISUAL_LOCK_CONF_EXIT
                ):
                    label = "RECHECK" if t.get('visual_lock_recheck_due', False) else "LOCK"
                else:
                    label = "TARGET"
                current_draw_boxes.append([box[0], box[1], box[2], box[3], (0, 255, 0), label, 1])

        if not learning_mode:
            if confirmed:
                gimbal_data = [[b[0], b[1], b[2], b[3], estimate_rough_range_m(b, W)] for b in confirmed]
                local_data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")
        if visual_enabled and len(current_draw_boxes) > MAX_DRAW_BOXES:
            current_draw_boxes = current_draw_boxes[-MAX_DRAW_BOXES:]

        if draw_this_frame:
            try:
                show_frame = cv2.resize(frame, (VIDEO_STREAM_W, VIDEO_STREAM_H), interpolation=cv2.INTER_AREA)
                dsx, dsy = float(VIDEO_STREAM_W) / W, float(VIDEO_STREAM_H) / H
                for x1, y1, x2, y2, color, text, _life in current_draw_boxes:
                    cv2.rectangle(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 1 if text in ("ROI", "TRKROI") else 2)
                for i in range(len(current_draw_boxes)-1, -1, -1):
                    current_draw_boxes[i][6] -= 1
                    if current_draw_boxes[i][6] <= 0: current_draw_boxes.pop(i)

                if send_video_this_frame: v_sender.send(BOARD_ID, cam_idx, show_frame)
                if output_writer is not None:
                    output_writer.write(show_frame)
                if SHOW_INDIVIDUAL_WINDOWS:
                    put_latest(display_queues[cam_idx], show_frame)
            except: pass
    cap.release()
    if output_writer is not None:
        output_writer.release()
    if detection_csv_file is not None:
        detection_csv_file.close()
    print(
        f"--> Cam {cam_idx} scheduler summary: submitted_tasks={submitted_task_count} "
        f"replaced_tasks={replaced_task_count} "
        f"submitted_groups={submitted_group_count} complete_groups={complete_group_count} "
        f"partial_groups={partial_group_count} expired_groups={frame_expired_group_count} "
        f"tracker_group_updates={tracker_group_update_count} "
        f"result_lag_avg={result_lag_frame_sum / max(1, result_lag_sample_count):.1f} "
        f"result_lag_max={result_lag_frame_max}",
        flush=True,
    )

if __name__ == '__main__':
    initial_backgrounds = [(None, None, None) for _ in range(N_CAM)]
    if SIMULATE_BY_VIDEOS and ENABLE_LOW_LAYER_STATIC_BG_MASK:
        if VIDEO_TEST_PATHS:
            built_backgrounds = []
            for cam_idx in range(N_CAM):
                source = get_video_test_source(cam_idx)
                if not os.path.isfile(source):
                    print(f"--> Cam {cam_idx} static background skipped, missing video: {source}", flush=True)
                    built_backgrounds.append((None, None, None))
                    continue
                try:
                    bg_gray, bg_tol, bg_qrange, bg_sample_count, bg_frame_span = build_initial_video_bg_model(source)
                    spec = CAM_LAYER_MODE_SPECS[min(cam_idx, len(CAM_LAYER_MODE_SPECS) - 1)].lower() if CAM_LAYER_MODE_SPECS else ""
                    estimated_high, _layer_reason, _day_scene, _layer_confidence = estimate_layer_from_background(bg_gray, bg_qrange)
                    needs_low_background = spec in ("low", "l") or (
                        (spec in ("", "auto", "a")) and ENABLE_AUTO_CAM_LAYER and not estimated_high
                    )
                    if needs_low_background and STATIC_BG_SECONDS < 30.0:
                        bg_gray, bg_tol, bg_qrange, bg_sample_count, bg_frame_span = build_initial_video_bg_model(
                            source,
                            static_seconds=30.0,
                        )
                    print(
                        f"--> Cam {cam_idx} static background ready: "
                        f"source={source} frames={bg_frame_span} samples={bg_sample_count}",
                        flush=True,
                    )
                    built_backgrounds.append((bg_gray, bg_tol, bg_qrange))
                except Exception as exc:
                    print(f"--> Cam {cam_idx} static background failed, use online learning: {exc}", flush=True)
                    built_backgrounds.append((None, None, None))
            initial_backgrounds = built_backgrounds
        elif os.path.isfile(VIDEO_TEST_PATH):
            try:
                initial_bg_gray, initial_bg_tol, initial_bg_qrange, bg_sample_count, bg_frame_span = build_initial_video_bg_model(VIDEO_TEST_PATH)
                estimated_high, _layer_reason, _day_scene, _layer_confidence = estimate_layer_from_background(
                    initial_bg_gray,
                    initial_bg_qrange,
                )
                if not estimated_high and STATIC_BG_SECONDS < 30.0:
                    initial_bg_gray, initial_bg_tol, initial_bg_qrange, bg_sample_count, bg_frame_span = build_initial_video_bg_model(
                        VIDEO_TEST_PATH,
                        static_seconds=30.0,
                    )
                print(
                    f"--> Shared static background ready: frames={bg_frame_span} samples={bg_sample_count}",
                    flush=True,
                )
                initial_backgrounds = [(initial_bg_gray, initial_bg_tol, initial_bg_qrange) for _ in range(N_CAM)]
            except Exception as exc:
                print(f"--> Shared static background failed, use online learning: {exc}", flush=True)
    elif (not SIMULATE_BY_VIDEOS) and ENABLE_LOW_LAYER_STATIC_BG_MASK and ENABLE_CAMERA_PREBUILD_BG:
        built_backgrounds = []
        for cam_idx in range(N_CAM):
            try:
                print(f"--> Cam {cam_idx} prebuilding camera background.", flush=True)
                bg_gray, bg_tol, bg_qrange, bg_sample_count, bg_seconds, source = build_initial_camera_bg_model(cam_idx)
                print(
                    f"--> Cam {cam_idx} camera background ready: "
                    f"source={source} seconds={bg_seconds:.1f} samples={bg_sample_count}",
                    flush=True,
                )
                built_backgrounds.append((bg_gray, bg_tol, bg_qrange))
            except Exception as exc:
                print(f"--> Cam {cam_idx} camera background failed, use online learning: {exc}", flush=True)
                built_backgrounds.append((None, None, None))
        initial_backgrounds = built_backgrounds
    CAM_LAYER_PROFILES = build_cam_layer_profiles(initial_backgrounds)
    ACTIVE_YOLO_CONF_THRESH = min(float(p["conf_thresh"]) for p in CAM_LAYER_PROFILES) if CAM_LAYER_PROFILES else CONF_THRESH
    init_runtime_queues()
    latest_display_frames = [None for _ in range(N_CAM)]

    print(f"--> Detection enabled. version={ALGORITHM_VERSION}", flush=True)
    print(f"--> {ALGORITHM_NOTE}", flush=True)
    print(
        f"--> Layer: {'high' if HIGH_LAYER_MODE else 'low'} "
        f"board_row={BOARD_ROW_IDX} mode={LAYER_MODE} conf={CONF_THRESH:.2f}",
        flush=True,
    )
    print(
        f"--> Scene mode: {SCENE_MODE} "
        f"capture={CAPTURE_W}x{CAPTURE_H} diff={DIFF_W}x{DIFF_H} "
        f"direct_fullframe={'yes' if DIRECT_FULL_FRAME_INFERENCE else 'no'} "
        f"motion_zoom={MOTION_ZOOM_CROP_SIZE} "
        f"track_zoom={TRACK_ZOOM_CROP_SIZE} "
        f"fallback_only={'yes' if FULLFRAME_FALLBACK_ONLY else 'no'}",
        flush=True,
    )
    print(
        f"--> Per-cam layer profiles: "
        + ", ".join(
            f"cam{i}={p['name']}({p['reason']})" for i, p in enumerate(CAM_LAYER_PROFILES)
        )
        + f"; yolo_conf={ACTIVE_YOLO_CONF_THRESH:.2f}",
        flush=True,
    )
    print(f"--> RKNN model: {os.path.abspath(MODEL_PATH)}", flush=True)
    print(f"--> Input mode: {'video-test' if SIMULATE_BY_VIDEOS else 'independent-cameras'}", flush=True)
    print(f"--> Board id: {BOARD_ID} local_ips={get_local_ipv4s()}", flush=True)
    print(
        f"--> Video stream: target={VIDEO_TARGET_IP}:{VIDEO_BASE_PORT} "
        f"size={VIDEO_STREAM_W}x{VIDEO_STREAM_H} quality={VIDEO_STREAM_QUALITY} "
        f"cams={','.join(VIDEO_STREAM_CAM_SPECS) if VIDEO_STREAM_CAM_SPECS else 'all'} "
        f"every={VIDEO_SEND_EVERY_N_FRAMES}",
        flush=True,
    )
    if not SIMULATE_BY_VIDEOS:
        print(
            f"--> Camera prebuild background: {'on' if ENABLE_CAMERA_PREBUILD_BG else 'off'} "
            f"seconds={STATIC_BG_SECONDS:.1f} samples={LOW_BG_SAMPLE_FRAMES}",
            flush=True,
        )
    if SIMULATE_BY_VIDEOS:
        video_sources = [get_video_test_source(i) for i in range(N_CAM)]
        print(
            f"--> Video test: path={VIDEO_TEST_PATH} paths={video_sources} cameras={N_CAM} "
            f"max_seconds={VIDEO_TEST_MAX_SECONDS:.1f} realtime={VIDEO_TEST_REALTIME} "
            f"resize={VIDEO_TEST_RESIZE_W}x{VIDEO_TEST_RESIZE_H if VIDEO_TEST_RESIZE_W > 0 else 0} "
            f"output={VIDEO_TEST_OUTPUT or 'off'} save_rois={VIDEO_TEST_SAVE_ROIS_DIR or 'off'} "
            f"roi_seconds={VIDEO_TEST_SAVE_ROIS_START:.1f}-{VIDEO_TEST_SAVE_ROIS_END:.1f}",
            flush=True,
        )
        print(
            f"--> RKNN scheduler: workers={NPU_WORKER_COUNT} queue_per_cam={INF_QUEUE_SIZE} "
            f"rois_per_update={MAX_ENQUEUED_ROIS_PER_UPDATE} "
            f"task_max_age={NPU_TASK_MAX_AGE_SEC:.2f}s "
            f"result_max_age_frames={MAX_INFERENCE_RESULT_AGE_FRAMES} "
            f"replace_pending={NPU_REPLACE_PENDING_ON_UPDATE}",
            flush=True,
        )
    print(f"--> Display windows: {'on' if SHOW_INDIVIDUAL_WINDOWS else 'off'}", flush=True)
    print("--> Camera workers: threads", flush=True)
    camera_workers = []
    for i in range(N_CAM):
        initial_bg_gray, initial_bg_tol, initial_bg_qrange = initial_backgrounds[i]
        worker = threading.Thread(
            target=capture_job,
            args=(i, initial_bg_gray, initial_bg_tol, initial_bg_qrange, CAM_LAYER_PROFILES[i]),
            name=f"camera-{i}",
            daemon=True,
        )
        worker.start()
        camera_workers.append(worker)
        time.sleep(0.5)

    print(
        f"--> All camera threads started; initializing {NPU_WORKER_COUNT} RKNN inference thread(s).",
        flush=True,
    )
    inference_workers = []
    for worker_idx in range(NPU_WORKER_COUNT):
        worker = threading.Thread(
            target=inference_worker,
            args=(worker_idx,),
            name=f"rknn-inference-{worker_idx}",
            daemon=True,
        )
        worker.start()
        inference_workers.append(worker)
        if NPU_WORKER_COUNT > 1:
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
        while not stop_event.is_set():
            if SIMULATE_BY_VIDEOS and all(not worker.is_alive() for worker in camera_workers):
                break
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
    except KeyboardInterrupt:
        pass
    stop_event.set()
    for worker in camera_workers:
        try:
            worker.join(timeout=1.0)
        except Exception:
            pass
    time.sleep(0.5)
    cv2.destroyAllWindows()
