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
from comms import DataSender, VideoSender

# ==========================================
# 0. 辅助算法：用 NumPy 替代 SciPy 以去除依赖


try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ==========================================
# 1. 配置参数
# ==========================================
ALGORITHM_VERSION = "cpu-gru-detect-only-20260618"
ALGORITHM_NOTE = "Pure CPU-based thresholding detection + GRU trajectory noise filtering & prediction (No YOLO, No RKNN)."
N_CAM = 5                        
INIT_TIME = 5                    

SHOW_INDIVIDUAL_WINDOWS = True
VIDEO_SEND_EVERY_N_FRAMES = 3
MAX_DRAW_BOXES = 80
CAPTURE_W, CAPTURE_H = 1920, 1080 # 采用1080p分辨率

BOARD_ID = "BOARD_2" 
BOARD_ROW_IDX = 1 

DATA_TARGETS = {
    "gimbal": ("192.168.0.100", 8888)
}
VIDEO_TARGET_IP = "192.168.0.200"       
VIDEO_BASE_PORT = 9999                  

PROCESS_EVERY_N_FRAMES = 3

ROUGH_TARGET_WIDTH_M = 0.5
CAM_H_FOV = 17.5
IMG_W, IMG_H = CAPTURE_W, CAPTURE_H

ROUGH_RANGE_MIN_M = 20
ROUGH_RANGE_MAX_M = 2000
ROUGH_RANGE_ROUND_M = 10
CAM_MAP = {i: f"00000000{i+1}" for i in range(5)}

# ==========================================
# 2. 神经网络模型定义 (已彻底移除 PyTorch 与 GRU 模型，改用纯 OpenCV 级联追踪判定)
# ==========================================

# ==========================================
# 3. 追踪与检测处理器类
# ==========================================
class Track:
    def __init__(self, track_id: int, first_pos: Tuple[float, float], w: float = 50.0, h: float = 50.0, conf: float = 1.0):
        self.track_id = track_id
        self.history_buffer = []
        self.history_buffer.append({
            'x': first_pos[0], 'y': first_pos[1],
            'px': first_pos[0], 'py': first_pos[1],
            'w': w, 'h': h, 'conf': conf
        })
        self.missing_frames = 0
        self.is_uav = False
        self.classification_prob = None
        self.pred_coords = []
        self.locked_positions = [first_pos]
        self.invalid_dist_frames = 0
        self.smooth_px = first_pos[0]
        self.smooth_py = first_pos[1]
        self.smooth_w = w
        self.smooth_h = h

    def update_smooth_filter(self, px: float, py: float, w: float, h: float, alpha: float = 0.65):
        self.smooth_px = alpha * px + (1.0 - alpha) * self.smooth_px
        self.smooth_py = alpha * py + (1.0 - alpha) * self.smooth_py
        self.smooth_w = alpha * w + (1.0 - alpha) * self.smooth_w
        self.smooth_h = alpha * h + (1.0 - alpha) * self.smooth_h

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
        
        self.tracks = {}
        self.next_track_id = 1
        self.last_valid_offset = (0.0, 0.0)
        self.offset_initialized = False
        
        self.max_offset_jump = 40.0
        self.max_drone_move = 60.0
        self.max_missing_frames = 10

    def detect_all_centroids(self, frame: np.ndarray, thresh_val: int = 120):
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = frame
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        border = 40
        centroids = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if 1 <= area < 2000:
                cX = int(x + w / 2.0)
                cY = int(y + h / 2.0)
                if border <= cX < width - border and border <= cY < height - border:
                    max_contrast = int(120 - np.min(gray[y:y+h, x:x+w]))
                    centroids.append((cX, cY, float(w), float(h), max_contrast))
                    
        if len(centroids) > 1:
            merged = []
            used = set()
            for i in range(len(centroids)):
                if i in used:
                    continue
                cX1, cY1, w1, h1, max_contrast1 = centroids[i]
                group = [centroids[i]]
                used.add(i)
                for j in range(i + 1, len(centroids)):
                    if j in used:
                        continue
                    cX2, cY2, w2, h2, max_contrast2 = centroids[j]
                    dist = np.hypot(cX1 - cX2, cY1 - cY2)
                    if dist < 20.0:
                        group.append(centroids[j])
                        used.add(j)
                        
                if len(group) == 1:
                    merged.append(centroids[i])
                else:
                    sum_x = sum(g[0] * (g[2] * g[3]) for g in group)
                    sum_y = sum(g[1] * (g[2] * g[3]) for g in group)
                    sum_area = sum(g[2] * g[3] for g in group)
                    
                    merged_cX = int(sum_x / (sum_area + 1e-8))
                    merged_cY = int(sum_y / (sum_area + 1e-8))
                    merged_w = max(g[2] for g in group)
                    merged_h = max(g[3] for g in group)
                    merged_contrast = max(g[4] for g in group)
                    merged.append((merged_cX, merged_cY, merged_w, merged_h, merged_contrast))
            centroids = merged
            
        return centroids

    def _find_csv_bound_track_id(self, csv_row):
        best_tid = None
        min_dist = float('inf')
        target_x = csv_row["x_center"] + self.last_valid_offset[0]
        target_y = csv_row["y_center"] + self.last_valid_offset[1]
        for tid, track in self.tracks.items():
            last = track.history_buffer[-1]
            dist = np.hypot(last['px'] - target_x, last['py'] - target_y)
            if dist < min_dist:
                min_dist = dist
                best_tid = tid
        return best_tid

    def _is_closest_to_csv(self, track, csv_row):
        bound_id = self._find_csv_bound_track_id(csv_row)
        return bound_id == track.track_id

    def update(self, frame: np.ndarray, csv_row = None):
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        scale_area = (width * height) / (1280.0 * 720.0)
        scale_linear = np.sqrt(scale_area)
        current_max_offset_jump = self.max_offset_jump * scale_linear
        current_max_drone_move = self.max_drone_move * scale_linear
        sky_height_limit = int(height * (1100.0 / 1440.0))
        
        centroids = self.detect_all_centroids(frame, thresh_val=120)
        if not centroids and not self.tracks:
            centroids = self.detect_all_centroids(frame, thresh_val=130)
            
        track_predictions = {}
        for tid, track in self.tracks.items():
            last_pt = track.history_buffer[-1]
            pred_x, pred_y = last_pt['px'], last_pt['py']
            if len(track.history_buffer) >= 2:
                prev_pt = track.history_buffer[-2]
                vx = last_pt['px'] - prev_pt['px']
                vy = last_pt['py'] - prev_pt['py']
                pred_x += vx
                pred_y += vy
            track_predictions[tid] = (pred_x, pred_y)
            
        matched_centroids = set()
        matched_tracks = set()
        match_candidates = []
        for tid, track in self.tracks.items():
            pred_pos = track_predictions[tid]
            for c_idx, c in enumerate(centroids):
                dist = np.hypot(c[0] - pred_pos[0], c[1] - pred_pos[1])
                if dist < current_max_drone_move:
                    match_candidates.append((dist, tid, c_idx))
                    
        match_candidates.sort(key=lambda x: x[0])
        
        for dist, tid, c_idx in match_candidates:
            if tid not in matched_tracks and c_idx not in matched_centroids:
                matched_tracks.add(tid)
                matched_centroids.add(c_idx)
                c = centroids[c_idx]
                track = self.tracks[tid]
                
                if csv_row is not None and self._is_closest_to_csv(track, csv_row):
                    current_x, current_y = csv_row["x_center"], csv_row["y_center"]
                    proposed_offset = (float(c[0] - current_x), float(c[1] - current_y))
                    if not self.offset_initialized:
                        self.last_valid_offset = proposed_offset
                        self.offset_initialized = True
                        aligned_px, aligned_py = c[0], c[1]
                    else:
                        dx = proposed_offset[0] - self.last_valid_offset[0]
                        dy = proposed_offset[1] - self.last_valid_offset[1]
                        offset_dist = np.sqrt(dx**2 + dy**2)
                        if offset_dist < current_max_offset_jump:
                            self.last_valid_offset = proposed_offset
                            aligned_px, aligned_py = c[0], c[1]
                        else:
                            aligned_px = current_x + self.last_valid_offset[0]
                            aligned_py = current_y + self.last_valid_offset[1]
                            
                    track.history_buffer.append({
                        'x': current_x, 'y': current_y,
                        'px': float(aligned_px), 'py': float(aligned_py),
                        'w': float(c[2]), 'h': float(c[3]), 'conf': csv_row.get("detector_confidence", 1.0)
                    })
                else:
                    track.history_buffer.append({
                        'x': float(c[0]), 'y': float(c[1]),
                        'px': float(c[0]), 'py': float(c[1]),
                        'w': float(c[2]), 'h': float(c[3]), 'conf': 1.0
                    })
                    
                track.missing_frames = 0
                track.invalid_dist_frames = 0
                track.locked_positions.append((c[0], c[1]))
                if len(track.locked_positions) > 50:
                    track.locked_positions.pop(0)
                
                latest = track.history_buffer[-1]
                track.update_smooth_filter(latest['px'], latest['py'], latest['w'], latest['h'], alpha=0.65)
                    
        dead_track_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing_frames:
                    dead_track_ids.append(tid)
                else:
                    pred_pos = track_predictions[tid]
                    last_pt = track.history_buffer[-1]
                    track.history_buffer.append({
                        'x': pred_pos[0] - self.last_valid_offset[0],
                        'y': pred_pos[1] - self.last_valid_offset[1],
                        'px': pred_pos[0], 'py': pred_pos[1],
                        'w': last_pt['w'], 'h': last_pt['h'], 'conf': last_pt['conf'] * 0.8
                    })
                    track.invalid_dist_frames += 1
                    track.locked_positions.append(pred_pos)
                    if len(track.locked_positions) > 50:
                        track.locked_positions.pop(0)
                        
                    latest = track.history_buffer[-1]
                    track.update_smooth_filter(latest['px'], latest['py'], latest['w'], latest['h'], alpha=0.65)
                        
        for tid in dead_track_ids:
            del self.tracks[tid]
            
        if len(self.tracks) > 1:
            tids = list(self.tracks.keys())
            merged_tids = set()
            for i in range(len(tids)):
                tid1 = tids[i]
                if tid1 in merged_tids or tid1 not in self.tracks:
                    continue
                track1 = self.tracks[tid1]
                last_pt1 = track1.history_buffer[-1]
                
                for j in range(i + 1, len(tids)):
                    tid2 = tids[j]
                    if tid2 in merged_tids or tid2 not in self.tracks:
                        continue
                    track2 = self.tracks[tid2]
                    last_pt2 = track2.history_buffer[-1]
                    
                    dist = np.hypot(last_pt1['px'] - last_pt2['px'], last_pt1['py'] - last_pt2['py'])
                    if dist < 35.0:
                        prob1 = track1.classification_prob if track1.classification_prob is not None else -1.0
                        prob2 = track2.classification_prob if track2.classification_prob is not None else -1.0
                        
                        if abs(prob1 - prob2) < 1e-4:
                            age1 = len(track1.history_buffer)
                            age2 = len(track2.history_buffer)
                            keep_tid = tid1 if age1 >= age2 else tid2
                            delete_tid = tid2 if keep_tid == tid1 else tid1
                        else:
                            keep_tid = tid1 if prob1 >= prob2 else tid2
                            delete_tid = tid2 if keep_tid == tid1 else tid1
                            
                        merged_tids.add(delete_tid)
                        
            for tid in merged_tids:
                if tid in self.tracks:
                    del self.tracks[tid]
            
        for tid, track in list(self.tracks.items()):
            is_deadlocked = False
            if len(track.locked_positions) >= 30:
                recent_30 = track.locked_positions[-30:]
                xs = [p[0] for p in recent_30]
                ys = [p[1] for p in recent_30]
                span_x = max(xs) - min(xs)
                span_y = max(ys) - min(ys)
                
                is_near_ground = track.locked_positions[-1][1] >= sky_height_limit
                if is_near_ground and span_x < 5.0 * scale_linear and span_y < 5.0 * scale_linear:
                    is_deadlocked = True
                elif len(track.locked_positions) >= 50:
                    recent_50 = track.locked_positions[-50:]
                    xs_50 = [p[0] for p in recent_50]
                    ys_50 = [p[1] for p in recent_50]
                    span_x_50 = max(xs_50) - min(xs_50)
                    span_y_50 = max(ys_50) - min(ys_50)
                    if span_x_50 < 3.0 * scale_linear and span_y_50 < 3.0 * scale_linear:
                        is_deadlocked = True
                        
            if is_deadlocked:
                del self.tracks[tid]
                
        for c_idx, c in enumerate(centroids):
            if c_idx not in matched_centroids:
                if c[1] < sky_height_limit:
                    coords = np.array([[p[0], p[1]] for p in centroids])
                    if len(coords) > 1:
                        dists = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
                        neighbors = np.sum(dists < 150.0 * scale_linear) - 1
                    else:
                        neighbors = 0
                        
                    if neighbors < 4:
                        new_track = Track(
                            track_id=self.next_track_id,
                            first_pos=(float(c[0]), float(c[1])),
                            w=float(c[2]),
                            h=float(c[3]),
                            conf=1.0
                        )
                        if csv_row is not None and self._is_closest_to_csv(new_track, csv_row):
                            current_x, current_y = csv_row["x_center"], csv_row["y_center"]
                            proposed_offset = (float(c[0] - current_x), float(c[1] - current_y))
                            self.last_valid_offset = proposed_offset
                            self.offset_initialized = True
                            new_track.history_buffer[0] = {
                                'x': current_x, 'y': current_y,
                                'px': float(c[0]), 'py': float(c[1]),
                                'w': float(c[2]), 'h': float(c[3]), 'conf': csv_row.get("detector_confidence", 1.0)
                            }
                            new_track.smooth_px = float(c[0])
                            new_track.smooth_py = float(c[1])
                            new_track.smooth_w = float(c[2])
                            new_track.smooth_h = float(c[3])
                            
                        self.tracks[self.next_track_id] = new_track
                        self.next_track_id += 1
                        
        for track in self.tracks.values():
            if len(track.history_buffer) > self.seq_len:
                track.history_buffer.pop(0)
                
        primary_track = None
        uav_tracks = [t for t in self.tracks.values() if t.is_uav]
        if uav_tracks:
            primary_track = max(uav_tracks, key=lambda t: t.classification_prob)
        elif self.tracks:
            primary_track = min(self.tracks.values(), key=lambda t: t.track_id)
            
        if primary_track is not None and primary_track.history_buffer:
            drone_pos = (int(primary_track.smooth_px), int(primary_track.smooth_py))
            is_valid = len(primary_track.history_buffer) == self.seq_len
        else:
            drone_pos = None
            is_valid = False
            
        return drone_pos, self.last_valid_offset, is_valid



# ==========================================
# 4. 辅助函数
# ==========================================
def estimate_rough_range_m(box, frame_width=IMG_W):
    pixel_width = max(1.0, float(box[2] - box[0]))
    frame_width = max(1.0, float(frame_width))
    fx_px = (frame_width / 2.0) / math.tan(math.radians(CAM_H_FOV / 2.0))
    distance = (ROUGH_TARGET_WIDTH_M * fx_px) / pixel_width
    distance = max(ROUGH_RANGE_MIN_M, min(ROUGH_RANGE_MAX_M, distance))
    return int(round(distance / ROUGH_RANGE_ROUND_M) * ROUGH_RANGE_ROUND_M)

def init_runtime_ipc():
    global display_queues, stop_event, video_allowed_event
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("--> multiprocessing fork is unavailable; fallback to camera threads.", flush=True)
        video_allowed_event = None
        return None
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
# 5. 摄像头处理任务 (全程 CPU 运行)
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

    tracker = FeatureTracker(seq_len=20, input_dim=8, smooth=True, normalize=True)

    f_idx = 0
    local_data_sender = DataSender(DATA_TARGETS, BOARD_ID)
    v_sender = VideoSender(VIDEO_TARGET_IP, VIDEO_BASE_PORT)
    learning_mode = True
    current_draw_boxes = []

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
        
        if learning_mode and (time.time() - SYSTEM_START_TIME > INIT_TIME):
            learning_mode = False

        # 核心检测追踪逻辑 (每 PROCESS_EVERY_N_FRAMES 帧执行一次)
        if (f_idx + cam_idx) % PROCESS_EVERY_N_FRAMES == 0:
            # A. 追踪器更新 (如果是灰度图，直接传入原始灰度图，避免多余转换)
            input_frame = gray_frame if gray_frame is not None else frame
            drone_pos, _, is_valid = tracker.update(input_frame)
            
            # B. 对所有的活跃轨迹基于持续帧数进行 UAV 认定 (3588 CPU 极简直接判定，不使用 GRU 模型)
            for tid, track in tracker.tracks.items():
                if len(track.history_buffer) >= 5:
                    track.classification_prob = 1.0
                    track.is_uav = True
                else:
                    track.classification_prob = None
                    track.is_uav = False
                track.pred_coords = []
                        
            # C. 选取唯一黄金主轨迹进行高亮展示和云台发数
            primary_track = None
            uav_tracks = [t for t in tracker.tracks.values() if t.is_uav]
            if uav_tracks:
                primary_track = max(uav_tracks, key=lambda t: t.classification_prob)
                
            # D. 清空并生成当前帧可视化覆盖数据
            current_draw_boxes = []
            
            # 渲染黄金轨迹
            if primary_track is not None and primary_track.is_uav:
                # 绘制历史轨迹 (橙色)
                if len(primary_track.history_buffer) > 1:
                    for i in range(1, len(primary_track.history_buffer)):
                        p0 = primary_track.history_buffer[i - 1]
                        p1 = primary_track.history_buffer[i]
                        pt0 = (int(p0["px"]), int(p0["py"]))
                        pt1 = (int(p1["px"]), int(p1["py"]))
                        current_draw_boxes.append([pt0[0], pt0[1], pt1[0], pt1[1], (0, 69, 255), "LINE", 1])
                
                # 绘制未来 5 步预测轨迹 (黄色)
                if primary_track.pred_coords:
                    pred_pts = primary_track.pred_coords
                    for i in range(1, len(pred_pts)):
                        current_draw_boxes.append([pred_pts[i - 1][0], pred_pts[i - 1][1], pred_pts[i][0], pred_pts[i][1], (0, 255, 255), "LINE", 1])
                    for coord in pred_pts:
                        current_draw_boxes.append([coord[0], coord[1], 0, 0, (0, 255, 255), "PREDPT", 1])
                
                # 绘制目标框 (黄色)
                cX = int(primary_track.smooth_px)
                cY = int(primary_track.smooth_py)
                w = max(30, int(primary_track.smooth_w))
                h = max(30, int(primary_track.smooth_h))
                x1 = cX - w // 2
                y1 = cY - h // 2
                x2 = cX + w // 2
                y2 = cY + h // 2
                
                prob = primary_track.classification_prob if primary_track.classification_prob is not None else 1.0
                label = f"UAV #{primary_track.track_id} (conf: {prob:.2f})"
                current_draw_boxes.append([x1, y1, x2, y2, (0, 255, 255), label, 1])
                
                # E. 向云台发送 UDP 电传数据
                if not learning_mode:
                    gimbal_data = [[x1, y1, x2, y2, estimate_rough_range_m([x1, y1, x2, y2], W)]]
                    local_data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")
            else:
                # 若无主轨迹，但有其他轨迹段处于分类判定阶段，可以在画面上以小灰色虚框表现
                for tid, t in tracker.tracks.items():
                    if len(t.history_buffer) >= 5:
                        tcX = int(t.smooth_px)
                        tcY = int(t.smooth_py)
                        tw = max(16, int(t.smooth_w))
                        th = max(16, int(t.smooth_h))
                        current_draw_boxes.append([tcX - tw//2, tcY - th//2, tcX + tw//2, tcY + th//2, (128, 128, 128), "TRACK", 1])

        # F. 绘制推流画面与 HUD 信息
        if draw_this_frame:
            try:
                show_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                dsx, dsy = 640.0/W, 360.0/H
                
                # 绘制框和折线
                for i in range(len(current_draw_boxes)-1, -1, -1):
                    x1, y1, x2, y2, color, text, life = current_draw_boxes[i]
                    if text == "LINE":
                        cv2.line(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 2, cv2.LINE_AA)
                    elif text == "PREDPT":
                        cv2.circle(show_frame, (int(x1*dsx), int(y1*dsy)), 3, color, -1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 1 if text == "TRACK" else 2)
                        if text != "TRACK":
                            cv2.putText(show_frame, text, (int(x1*dsx), int(y1*dsy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                
                # 绘制精致 HUD 面板
                active_uavs = sum(1 for t in tracker.tracks.values() if t.is_uav)
                filtered_noises = sum(1 for t in tracker.tracks.values() if not t.is_uav and len(t.history_buffer) >= 5)
                
                # 绘制磨砂半透明黑背景
                overlay = show_frame.copy()
                cv2.rectangle(overlay, (8, 8), (210, 85), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.65, show_frame, 0.35, 0, show_frame)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(show_frame, f"Cam {cam_idx} Frame: {f_idx}", (15, 23), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(show_frame, f"Active UAVs: {active_uavs}", (15, 43), font, 0.40, (0, 255, 0) if active_uavs > 0 else (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(show_frame, f"Noise Filtered: {filtered_noises}", (15, 63), font, 0.40, (0, 69, 255) if filtered_noises > 0 else (180, 180, 180), 1, cv2.LINE_AA)

                if send_video_this_frame: v_sender.send(BOARD_ID, cam_idx, show_frame)
                if SHOW_INDIVIDUAL_WINDOWS:
                    put_latest(display_queues[cam_idx], show_frame)
            except Exception as e:
                pass

    cap.release()

# ==========================================
# 6. 主程序启动入口
# ==========================================
if __name__ == '__main__':
    runtime_ctx = init_runtime_ipc()
    use_camera_processes = runtime_ctx is not None
    latest_display_frames = [None for _ in range(N_CAM)]
    
    print(f"--> [UAV Trajectory System] version={ALGORITHM_VERSION}", flush=True)
    print(f"--> {ALGORITHM_NOTE}", flush=True)
    print(f"--> Camera workers: {'processes' if use_camera_processes else 'threads'}", flush=True)
    
    camera_workers = []
    SYSTEM_START_TIME = time.time()
    INIT_SIGNAL_SENT = False
    VIDEO_STREAM_ALLOWED = False
    
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
