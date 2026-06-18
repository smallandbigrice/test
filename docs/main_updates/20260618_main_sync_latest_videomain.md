# 2026-06-18 main.py 同步最新 videomain

## 修改目标

将用户重新提供并确认能够找到目标的 `F:\videomain.py` 完整同步到 `main.py`。

## 与上一版的差异

- RKNN 模型由 `../model/yolov5s_fp16.rknn` 改为 `../model/yolov5s.rknn`。
- YOLO 置信度阈值由 `0.50` 放宽为 `0.40`。
- 其余检测、帧差、ROI、轨迹和五路视频逻辑保持 `videomain.py` 原样。

## 当前关键流程

- 五路读取 `./4.mp4`。
- 每三帧处理一次。
- 当前灰度帧分别与前两次处理帧做帧差，两个二值结果取交集。
- 使用 15×15 矩形核膨胀一次。
- 根据运动轮廓中心裁剪原始 640×640 ROI。
- ROI 不做特征融合和填充，直接送入 `YoloRKNN`。
- YOLO 置信度阈值为 `0.40`，轨迹累计命中 3 次后确认绿框。

## 验证

- `main.py` 与 `F:\videomain.py` 文件内容比较无差异。
- 已执行 `python -m py_compile main.py`，语法检查通过。
- 尚未在 RK3588 上执行实际推理。
