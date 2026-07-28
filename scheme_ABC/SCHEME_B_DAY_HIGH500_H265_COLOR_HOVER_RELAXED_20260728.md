# 方案B白天高层500米封档说明

封档日期：2026-07-28

## 封档结论

方案B正式命名为“方案B白天高层500米检测”。本次封档版本采用 H265 摄像头输入，按摄像头序列号绑定设备顺序，板端硬解 NV12 后使用彩色图像进入白天高层检测链路。该版本保留白天高层 2K 采集、三帧帧差、远距小目标送检和悬停保持，并将悬停确认参数放宽，用于降低目标短时间悬停后绿框断开的情况。

## 适用范围

- 适用板端：板3到板9。
- 适用场景：白天高层、天空占比高、复杂背景少、远距无人机慢速移动或悬停。
- 当前同步验证：2026-07-28 已同步并验证板3到板7服务启动正常，板号对应 `BOARD_3` 到 `BOARD_7`。

## 固定配置

- 场景模式：`UAV_SCENE_MODE=day`
- 检测层级：`UAV_LAYER_MODE=high`
- 摄像头数量：`UAV_VIDEO_CAM_COUNT=5`
- 输入链路：`UAV_H265_YPLANE_CAPTURE=1`
- 彩色输入：`UAV_H265_COLOR_CAPTURE=1`
- H265设备绑定：`UAV_H265_DEVICES=by-id`
- 采集尺寸：`UAV_CAPTURE_W=2560`，`UAV_CAPTURE_H=1440`
- 帧差尺寸：`UAV_DIFF_W=1920`，`UAV_DIFF_H=1080`
- 帧差间隔：`UAV_PROCESS_EVERY_N_FRAMES=3`
- 模型：`yolov5s_day_20260626.rknn`
- 置信度：`UAV_YOLO_CONF=0.40`
- 视频回传：`640x480`，`UAV_VIDEO_SEND_EVERY_N_FRAMES=2`

## 悬停保持参数

- `UAV_HOVER_HOLD=1`
- `UAV_TRACK_SEARCH_ROIS=1`
- `UAV_HIGH_TRACK_SEARCH_MAX_ROIS=4`
- `UAV_HOVER_ENTER_YOLO_HITS=2`
- `UAV_HOVER_ENTER_RECENT_HITS=1`
- `UAV_HOVER_RECHECK_INTERVAL_FRAMES=9`
- `UAV_HOVER_MAX_RECHECK_MISSES=16`
- `UAV_HOVER_MAX_YOLO_GAP_FRAMES=240`
- `UAV_HOVER_MIN_TRACK_SCORE=0.28`
- `UAV_HOVER_MAX_SPEED_PX_PER_FRAME=6.0`
- `UAV_HOVER_MAX_RECENT_NET_MOTION_PX=80`

## 静止确认参数

- `UAV_STATIC_CONFIRM=1`
- `UAV_STATIC_CONFIRM_YOLO_HITS=2`
- `UAV_STATIC_CONFIRM_RECENT_HITS=2`
- `UAV_STATIC_CONFIRM_MAX_MISSES=6`
- `UAV_STATIC_CONFIRM_KEEP_MISSES=48`
- `UAV_STATIC_CONFIRM_MAX_AGE_FRAMES=240`
- `UAV_STATIC_CONFIRM_MAX_YOLO_GAP_FRAMES=240`
- `UAV_STATIC_CONFIRM_MIN_SCORE=0.28`
- `UAV_STATIC_CONFIRM_MIN_MEAN_CONF=0.28`
- `UAV_STATIC_CONFIRM_MAX_CENTER_JITTER=110`
- `UAV_STATIC_CONFIRM_MAX_BOX_JITTER_RATIO=1.5`

## 对应文件

- 主代码：`E:\detect uav\scheme_ABC\code\main.py`
- 方案配置：`E:\detect uav\scheme_ABC\configs\scheme_B_day_high500.conf`
- 批量切换：`E:\detect uav\scheme_ABC\scripts\switch_high_layer_profiles.ps1 -Mode day`
- 桌面入口：`E:\detect uav\scheme_ABC\scripts\desktop_wrappers\一键切换_方案B_白天高层500米_板3到板9.bat`
- 模型文件：`E:\detect uav\scheme_ABC\models\yolov5s_day_20260626.rknn`
- 模型 MD5：`30261610759aec78c94cdf7d6f6510ce`

## 切换说明

日常切换使用桌面入口或直接运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\detect uav\scheme_ABC\scripts\switch_high_layer_profiles.ps1" -Mode day
```

默认切换不重复上传模型。需要重传模型时，显式增加 `-DeployModels` 参数。
