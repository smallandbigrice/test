# 板端视频回归入口

此目录只存放板端视频推理回归入口，不参与正式摄像头自启动。

- `videomain.py`：用于板端视频回归、拼接预览和 CSV 输出。
- `videoyololib.py`：视频回归使用的 RKNN 适配层。

运行前先停止正式服务，避免两套 RKNN 同时占用 NPU：

```bash
sudo systemctl stop python_autostar.service
cd /home/Tronlong/rknn_model_zoo/examples/yolov5/python
UAV_VIDEO_CAM_COUNT=1 UAV_VIDEO_PATH=./5.mp4 UAV_LAYER_MODE=low \
UAV_VIDEO_MAX_SECONDS=20 UAV_VIDEO_REALTIME=0 UAV_SHOW_WINDOWS=0 \
python3 -u apps/video_inference/videomain.py
```

昼夜模型规则与正式 `main.py` 保持一致：

- `UAV_SCENE_MODE=day`：优先加载 `yolov5s_day_20260626.rknn`。
- `UAV_SCENE_MODE=night`：优先加载 `yolov5s_night_latest.rknn`。
- `UAV_RKNN_MODEL` 可手动覆盖默认模型。

视频、CSV、截图和 ROI 应写入根目录下的 `artifacts/<实验名>/`，不要直接放在仓库根目录。
