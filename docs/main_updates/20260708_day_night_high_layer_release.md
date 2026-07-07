# 20260708 白天高层与夜间高层检测版本说明

## 版本目标

本次版本用于板端高层场景的实测运行，重点保留白天高层已经验证稳定的检测链路，同时将夜间高层调整为低分辨率输入、帧差触发、小窗口放大送检的轻量方案，避免夜间 2K 输入和全帧兜底导致传输与检测延迟升高。

## 白天高层方案

白天高层仍采用 26 号白天模型和较完整的帧差 ROI 检测流程：

- 场景模式：`UAV_SCENE_MODE=day`
- 模型路径：`/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_day_20260626.rknn`
- 摄像头输入：`2560x1440`
- 帧差分辨率：`1920x1080`
- 帧差间隔：`UAV_PROCESS_EVERY_N_FRAMES=3`
- 全帧直接送检：关闭
- 运动点放大 ROI：关闭
- 轨迹搜索 ROI：关闭
- 悬停保持：开启

白天的基本逻辑是：先建立静态背景，再做帧差和形态学清理，提取运动候选区域；候选区域按原图 640 ROI 裁剪后送入 YOLO/RKNN；后处理再结合近期 YOLO 命中、丢失次数、轨迹分数和确认分数输出绿框。

## 夜间高层方案

夜间高层使用最新夜间模型，并且不再做全帧兜底，也不再用轨迹搜索 ROI 去持续送检。只有当前帧差产生运动 ROI 时，才围绕运动点裁剪 `160x160` 小窗口，再放大到 YOLO 输入尺寸进行推理。

- 场景模式：`UAV_SCENE_MODE=night`
- 模型路径：`/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_night_latest.rknn`
- 摄像头输入：`640x480`
- 帧差分辨率：`640x480`
- 帧差间隔：`UAV_PROCESS_EVERY_N_FRAMES=2`
- 运动点放大 ROI：`UAV_MOTION_ZOOM_CROP_SIZE=160`
- 每帧最多运动放大 ROI：`UAV_MOTION_ZOOM_MAX_ROIS=1`
- 仅使用运动放大 ROI：`UAV_MOTION_ZOOM_ONLY=1`

夜间明确关闭以下兜底和保持逻辑：

- `UAV_DIRECT_FULL_FRAME_INFERENCE=0`
- `UAV_FULLFRAME_FALLBACK_ONLY=0`
- `UAV_FULLFRAME_SEARCH_FALLBACK=0`
- `UAV_TRACK_SEARCH_ROIS=0`
- `UAV_HOVER_HOLD=0`

这样做的目的是保证夜间没有运动点时不送 YOLO，减少无效推理和画面延迟；有运动点时只送最关键的小区域，保持 400 米到 500 米范围内目标仍能被放大观察。

## 切换脚本

本地切换脚本为：

```powershell
.\scripts\switch_board_scene_mode.ps1 -BoardIp 192.168.0.3 -Mode day
.\scripts\switch_board_scene_mode.ps1 -BoardIp 192.168.0.3 -Mode night
```

如模型已经在板端，只想切换参数，可加：

```powershell
.\scripts\switch_board_scene_mode.ps1 -BoardIp 192.168.0.3 -Mode night -SkipModelDeploy
```

## 已同步的代码位置

- 主入口：`main.py`
- 板端视频测试入口：`apps/video_inference/videomain.py`
- 白天 systemd 模板：`tools/systemd/30-day-capture.conf`
- 夜间 systemd 模板：`tools/systemd/30-night-capture.conf`
- 白天/夜间切换脚本：`scripts/switch_board_scene_mode.ps1`

## 验证记录

- `python -m py_compile main.py apps\video_inference\videomain.py comms.py yololib.py tools\board_udp_video_receiver.py` 通过。
- PowerShell 切换脚本语法检查通过。
- 已在在线板端同步夜间配置，日志显示：`Scene mode: night capture=640x480 diff=640x480 direct_fullframe=no motion_zoom=160 motion_zoom_only=yes`。

## 使用提醒

白天实测优先使用 `day` 模式和 26 号白天模型；夜间实测使用 `night` 模式和最新夜间模型。夜间若发现目标悬停不再送检，这是当前策略的预期行为，因为夜间已关闭全帧兜底、轨迹搜索 ROI 和悬停保持。
