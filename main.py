#!/usr/bin/env python3
import cv2
import threading
import queue
import time
import socket
import struct
import os
import numpy as np
import subprocess
import math
from rknnlite.api import RKNNLite
from yololib import YoloRKNN 
from comms import DataSender, VideoSender

# ==========================================
# 1. 全局配置 (测试版)
# ==========================================
# 在这里指定你的测试视频文件路径，如果没有5个，可以重复使用同一个
TEST_VIDEO_FILES = [
    "./4.mp4",
    "./4.mp4",
    "./4.mp4",
    "./4.mp4",
    "./4.mp4"
]

N_CAM = len(TEST_VIDEO_FILES)
INIT_TIME = 2                    # 测试时缩短初始化时间

# 🚀 调优后的超参数 🚀
ENABLE_SUPPRESSOR = False        
TRACKER_MIN_HITS = 3             # 视频测试可稍微降低门槛
TRACKER_MAX_AGE = 12
TRACKER_MAX_DIST = 100           
TRACKER_GREEN_HOLD_FRAMES = 4
MAX_INFERENCE_RESULT_AGE_FRAMES = 6
CONF_THRESH = 0.40               

BOARD_ID = "BOARD_9" 
BOARD_ROW_IDX = 0  

# 模拟发送目标（如果不需要网络发送，可以指向127.0.0.1）
DATA_TARGETS = {
    "gimbal": ("192.168.2.8", 8888),
    "strike": ("127.0.0.1", 8889)
}
VIDEO_TARGET_IP = "192.168.2.200"       
VIDEO_BASE_PORT = 9999                  

CROP_SIZE = 640      
TIGHT_ROI_PAD = 32
TIGHT_ROI_FILL = 114
DIFF_THRESH = 20                 
MIN_OBJ_SIZE = 20                
MAX_ROIS_PER_FRAME = 6           
DIFF_W, DIFF_H = 1920, 1080
MODEL_PATH = '../model/yolov5s.rknn'

UAV_REAL_WIDTH_M = 0.5   
CAM_H_FOV = 17.5         
CAM_V_FOV = 9.9          
IMG_W, IMG_H = 3840, 2160 # 假设视频也是这个分辨率或进行比例换算
FX = (IMG_W / 2) / math.tan(math.radians(CAM_H_FOV / 2))
FY = (IMG_H / 2) / math.tan(math.radians(CAM_V_FOV / 2))

HW_ID_TO_COL_OFFSET = {"CAM_0": -2, "CAM_1": -1, "CAM_2": 0, "CAM_3": 1, "CAM_4": 2}
CAM_MAP = {i: f"CAM_{i}" for i in range(5)}

# ==========================================
# 2. 全局对象
# ==========================================
data_sender = DataSender(DATA_TARGETS, BOARD_ID)
video_senders =[VideoSender(VIDEO_TARGET_IP, VIDEO_BASE_PORT) for i in range(N_CAM)]

inf_queues = [queue.Queue(maxsize=3) for _ in range(N_CAM)]
res_queues =[queue.Queue(maxsize=5) for _ in range(N_CAM)]
display_queues =[queue.Queue(maxsize=2) for _ in range(N_CAM)]

stop_event = threading.Event()
SYSTEM_START_TIME = time.time()
VIDEO_STREAM_ALLOWED = True 

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

def make_padded_roi_canvas(frame, roi_info, crop_size=CROP_SIZE, fill_value=TIGHT_ROI_FILL):
    x1, y1, x2, y2, pad_x, pad_y = [int(v) for v in roi_info[:6]]
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return patch

    canvas = np.full((crop_size, crop_size, 3), np.uint8(fill_value), dtype=np.uint8)
    dst_x2 = min(crop_size, pad_x + patch.shape[1])
    dst_y2 = min(crop_size, pad_y + patch.shape[0])
    canvas[pad_y:dst_y2, pad_x:dst_x2] = patch[:dst_y2-pad_y, :dst_x2-pad_x]
    return canvas

def map_padded_detection_to_frame(det, roi_info, full_w, full_h):
    x1, y1, x2, y2, pad_x, pad_y = [int(v) for v in roi_info[:6]]
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
    ]

# ==========================================
# 3. 核心辅助函数 (保持不变)
# ==========================================
def calculate_spatial_status(hw_id, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pixel_width = max(1.0, float(x2 - x1))
    distance = (UAV_REAL_WIDTH_M * FX) / pixel_width
    delta_az = math.degrees(math.atan((cx - (IMG_W / 2)) / FX))
    delta_el = math.degrees(math.atan(((IMG_H / 2) - cy) / FY)) 
    col_offset = HW_ID_TO_COL_OFFSET.get(hw_id, 0)
    base_az = (col_offset * 18.0 + delta_az) % 360.0
    base_el = delta_el
    return round(base_az, 2), round(base_el, 2), round(distance, 2)

def merge_nearby_boxes(boxes, dist_thresh=120):
    if not boxes: return []
    merged = []
    for box in boxes:
        is_dup = False
        bcx, bcy = (box[0]+box[2])/2, (box[1]+box[3])/2
        for m_box in merged:
            mcx, mcy = (m_box[0]+m_box[2])/2, (m_box[1]+m_box[3])/2
            dist = math.sqrt((bcx-mcx)**2 + (bcy-mcy)**2)
            if dist < dist_thresh:
                is_dup = True; break
        if not is_dup: merged.append(box)
    return merged

class TrajectoryFilter:
    def __init__(self, max_dist=100, min_hits=4, max_age=15, hold_age=4):
        self.trackers =[] 
        self.max_dist, self.min_hits, self.max_age = max_dist, min_hits, max_age
        self.hold_age = hold_age

    @staticmethod
    def _predicted_box(trk):
        box = trk['box']
        age = trk['age']
        dx = trk.get('vx', 0.0) * age
        dy = trk.get('vy', 0.0) * age
        return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

    def update(self, detections):
        for trk in self.trackers: trk['age'] += 1
        matched_indices =[]
        for det in detections:
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            best_dist, best_idx = float('inf'), -1
            for i, trk in enumerate(self.trackers):
                if i in matched_indices: continue
                predicted = self._predicted_box(trk)
                tcx, tcy = (predicted[0]+predicted[2])/2, (predicted[1]+predicted[3])/2
                dist = math.sqrt((cx-tcx)**2 + (cy-tcy)**2)
                if dist < self.max_dist and dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx != -1:
                trk = self.trackers[best_idx]
                old_cx = (trk['box'][0] + trk['box'][2]) / 2.0
                old_cy = (trk['box'][1] + trk['box'][3]) / 2.0
                dt = max(1, trk['age'])
                observed_vx = (cx - old_cx) / dt
                observed_vy = (cy - old_cy) / dt
                trk.update({
                    'box': det,
                    'hits': trk['hits'] + 1,
                    'age': 0,
                    'vx': 0.5 * trk.get('vx', 0.0) + 0.5 * observed_vx,
                    'vy': 0.5 * trk.get('vy', 0.0) + 0.5 * observed_vy,
                })
                matched_indices.append(best_idx)
            else:
                self.trackers.append({'box': det, 'hits': 1, 'age': 0, 'vx': 0.0, 'vy': 0.0})
        self.trackers = [t for t in self.trackers if t['age'] <= self.max_age]
        return [
            self._predicted_box(t)
            for t in self.trackers
            if t['hits'] >= self.min_hits and t['age'] <= self.hold_age
        ]

# ==========================================
# 4. 推理线程 (保持不变)
# ==========================================
def inference_worker():
    try: 
        yolo = YoloRKNN(MODEL_PATH, (640, 640), CONF_THRESH, 0.45)
        print("--> [AI] RKNN Model Loaded successfully.", flush=True)
    except Exception as e:
        print(f"[Fatal] RKNN Init Failed: {e}"); return
    
    while not stop_event.is_set():
        did_work = False
        for i in range(N_CAM):
            try:
                roi, roi_info, src_frame_idx = inf_queues[i].get_nowait()
                did_work = True
                res = yolo.infer(roi)
                if res is not None:
                    put_latest(res_queues[i], (res, roi_info, src_frame_idx))
            except queue.Empty: pass
        if not did_work: time.sleep(0.001)
    if yolo: yolo.release()

# ==========================================
# 5. 视频文件读取测试线程
# ==========================================
def video_test_job(cam_idx, video_path):
    print(f"--> [Thread] Starting Video Test for Cam {cam_idx}: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_t1, frame_t2 = None, None
    tracker = TrajectoryFilter(
        max_dist=TRACKER_MAX_DIST,
        min_hits=TRACKER_MIN_HITS,
        max_age=TRACKER_MAX_AGE,
        hold_age=TRACKER_GREEN_HOLD_FRAMES,
    )
    
    f_idx = 0 
    v_sender = video_senders[cam_idx]
    current_draw_boxes = [] 
    target_hw_id = CAM_MAP[cam_idx]

    while not stop_event.is_set():
        ret, frame = cap.read()
        
        # --- 视频循环逻辑 ---
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 视频结束，重置到第一帧
            continue

        # 控制测试频率，模拟真实15-20fps
        time.sleep(0.05) 

        H, W = frame.shape[:2]
        f_idx += 1

        # 1. 运动检测
        if f_idx % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (DIFF_W, DIFF_H))
            scale_x, scale_y = W / float(DIFF_W), H / float(DIFF_H)
            
            rois =[]
            if frame_t1 is not None and frame_t2 is not None:
                diff1 = cv2.absdiff(frame_t1, gray_small)
                diff2 = cv2.absdiff(frame_t2, gray_small)
                _, m1 = cv2.threshold(diff1, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                _, m2 = cv2.threshold(diff2, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_and(m1, m2)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                temp_rois = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 10: continue
                    x, y, w, h = cv2.boundingRect(c)
                    fx, fy, fw, fh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                    sx1 = max(0, fx - TIGHT_ROI_PAD)
                    sy1 = max(0, fy - TIGHT_ROI_PAD)
                    sx2 = min(W, fx + fw + TIGHT_ROI_PAD)
                    sy2 = min(H, fy + fh + TIGHT_ROI_PAD)
                    sw, sh = sx2 - sx1, sy2 - sy1
                    if sw <= CROP_SIZE and sh <= CROP_SIZE:
                        pad_x = (CROP_SIZE - sw) // 2
                        pad_y = (CROP_SIZE - sh) // 2
                        temp_rois.append((sx1, sy1, sx2, sy2, pad_x, pad_y))
                    else:
                        cx, cy = fx + fw//2, fy + fh//2
                        half = CROP_SIZE//2
                        rx1 = max(0, min(W - CROP_SIZE, cx - half))
                        ry1 = max(0, min(H - CROP_SIZE, cy - half))
                        rx2 = min(W, rx1 + CROP_SIZE)
                        ry2 = min(H, ry1 + CROP_SIZE)
                        pad_x = (CROP_SIZE - (rx2 - rx1)) // 2
                        pad_y = (CROP_SIZE - (ry2 - ry1)) // 2
                        temp_rois.append((rx1, ry1, rx2, ry2, pad_x, pad_y))
                
                rois = merge_nearby_boxes(temp_rois, dist_thresh=300)
            
            frame_t2, frame_t1 = frame_t1, gray_small
            for roi_info in rois[:MAX_ROIS_PER_FRAME]:
                x1, y1, x2, y2 = roi_info[:4]
                current_draw_boxes.append([x1, y1, x2, y2, (255, 255, 255), "ROI", 2])
                roi_img = make_padded_roi_canvas(frame, roi_info)
                if roi_img.size != 0:
                    put_latest(inf_queues[cam_idx], (roi_img, roi_info, f_idx))

        # 2. 接收推理结果
        raw_boxes_in_this_frame =[]
        while True:
            try:
                dets, roi_info, src_frame_idx = res_queues[cam_idx].get_nowait()
                if f_idx - src_frame_idx > MAX_INFERENCE_RESULT_AGE_FRAMES:
                    continue
                for d in dets:
                    mapped = map_padded_detection_to_frame(d, roi_info, W, H)
                    if mapped is not None:
                        raw_boxes_in_this_frame.append(mapped)
            except queue.Empty: break

        unique_raw_boxes = merge_nearby_boxes(raw_boxes_in_this_frame, dist_thresh=100)
        for rb in unique_raw_boxes:
            current_draw_boxes.append([rb[0], rb[1], rb[2], rb[3], (0, 0, 255), "Raw", 2])

        # 3. 轨迹滤波与数据发送 (🚀 已添加发送逻辑)
        confirmed = tracker.update(unique_raw_boxes)
        valid_objs = []
        for box in confirmed:
            current_draw_boxes.append([box[0], box[1], box[2], box[3], (0, 255, 0), "TARGET", 1])
            # 计算空间坐标
            az, el, dist = calculate_spatial_status(target_hw_id, box)
            valid_objs.append({"azimuth": az, "elevation": el, "distance_m": dist})
        
        # 发送给云台 (Gimbal)
        if unique_raw_boxes:
            gimbal_data = [[b[0], b[1], b[2], b[3], round((UAV_REAL_WIDTH_M * FX)/(max(1, b[2]-b[0])), 2)] for b in unique_raw_boxes]
            data_sender.send_packet("data", cam_idx, gimbal_data, target="gimbal")

        # 发送给打击端 (Strike)
        if valid_objs:
            data_sender.send_packet("data", cam_idx, valid_objs, target="strike")

        # 4. 渲染可视化
        show_frame = cv2.resize(frame, (640, 360))
        dsx, dsy = 640.0/W, 360.0/H
        for i in range(len(current_draw_boxes)-1, -1, -1):
            x1, y1, x2, y2, color, text, life = current_draw_boxes[i]
            cv2.rectangle(show_frame, (int(x1*dsx), int(y1*dsy)), (int(x2*dsx), int(y2*dsy)), color, 1 if text=="ROI" else 2)
            current_draw_boxes[i][6] -= 1
            if current_draw_boxes[i][6] <= 0: current_draw_boxes.pop(i)
        
        # 发送至视频传输模块
        if VIDEO_STREAM_ALLOWED: v_sender.send(BOARD_ID, cam_idx, show_frame)
        
        # 存入显示队列
        put_latest(display_queues[cam_idx], show_frame)

    cap.release()

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 检查视频路径
    for p in TEST_VIDEO_FILES:
        if not os.path.exists(p):
            print(f"Warning: Video file {p} not found!")

    # 创建显示窗口
    for i in range(N_CAM):
        cv2.namedWindow(f"Test Cam {i}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Test Cam {i}", 640, 360)
    
    # 启动推理线程
    t_inf = threading.Thread(target=inference_worker)
    t_inf.daemon = True
    t_inf.start()

    # 启动视频测试读取线程
    for i in range(N_CAM):
        v_path = TEST_VIDEO_FILES[i] if i < len(TEST_VIDEO_FILES) else TEST_VIDEO_FILES[0]
        t = threading.Thread(target=video_test_job, args=(i, v_path))
        t.daemon = True
        t.start()
    
    print(">>> Video Test Running. Press 'q' to exit.")

    try:
        while True:
            for i in range(N_CAM):
                try: 
                    frame_to_show = display_queues[i].get_nowait()
                    cv2.imshow(f"Test Cam {i}", frame_to_show)
                except queue.Empty: pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    print(">>> Shutting down...")
    stop_event.set()
    time.sleep(1.0)
    cv2.destroyAllWindows()
