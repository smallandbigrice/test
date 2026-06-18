# 2026-06-18 main.py 使用 videomain 检测链路

## 修改目标

当前复杂检测版在板端视频测试中召回不稳定，因此不再继续叠加修补，直接使用已验证能够正常检测的 `F:\videomain.py` 覆盖 `main.py`。

## 覆盖范围

- `main.py` 的检测、推理、轨迹、显示和主程序结构均来自 `videomain.py`。
- 五路输入统一设置为 `./4.mp4`，视频播放结束后自动循环。
- 模型恢复为 `../model/yolov5s_fp16.rknn`。
- 取消静态背景蒙版、动态背景蒙版、LCM、复杂形态过滤、tight ROI、特征融合和多进程采集。

## 当前检测流程

1. 五个线程分别读取同一个 `./4.mp4`。
2. 每三帧执行一次 1920×1080 灰度运动检测。
3. 当前帧分别与前两次处理帧做绝对帧差，两个二值结果取交集。
4. 使用 15×15 矩形核膨胀一次，轮廓面积小于 10 的候选被过滤。
5. 按候选中心在原始帧中裁剪最大 640×640 ROI。
6. ROI 不做融合、不填充，直接送入 `YoloRKNN`。
7. 一个 RKNN 推理线程轮询五路 ROI 队列。
8. YOLO 结果按中心距离关联，轨迹累计命中 3 次后显示绿框。

## 关键参数

- `DIFF_THRESH = 20`
- `DIFF_W, DIFF_H = 1920, 1080`
- `MAX_ROIS_PER_FRAME = 6`
- `TRACKER_MIN_HITS = 3`
- `TRACKER_MAX_AGE = 20`
- `TRACKER_MAX_DIST = 100`
- `CONF_THRESH = 0.50`
- `MODEL_PATH = '../model/yolov5s_fp16.rknn'`

## 注意事项

- 这是五路视频测试版，不是五路 V4L2 摄像头采集版。
- 保留了源 `videomain.py` 中的云台和打击端数据发送代码。
- 当前轨迹关联较简单，没有速度预测和多目标全局匹配。
- 已执行 `python -m py_compile main.py`，语法检查通过。
- 尚未在 RK3588 上完成实际推理验证。

## 备份

覆盖前版本保存在：

`main.py.bak_before_videomain_20260618_231701`
