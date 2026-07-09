# 20260709 板3到板8高层双方案清理与夜间400切换

## 本次目标

板3到板8只保留两套高层检测方案：

- 白天高层500米：使用白天6月26号模型，高层远距离检测，保留悬停保持。
- 夜间400米：使用夜间最新模型，640x480输入，160 ROI放大，仅帧差触发送检。

低层复杂背景、自动高低层、边缘区域切换、旧视频低清回传等历史配置不再保留在板3到板8。

## 板端清理结果

板3到板8的 `python_autostar.service.d` 已清理为：

- `20-board-id.conf`
- `zzzzz-uav-scene-mode.conf`

最终生效参数由 `zzzzz-uav-scene-mode.conf` 统一管理，避免旧 `30-*`、`35-*`、`99-*`、`zz-*` 配置覆盖昼夜方案。

## 当前夜间400米配置

- `UAV_SCENE_MODE=night`
- `UAV_LAYER_MODE=high`
- `UAV_RKNN_MODEL=/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_night_latest.rknn`
- `UAV_CAPTURE_W=640`
- `UAV_CAPTURE_H=480`
- `UAV_DIFF_W=640`
- `UAV_DIFF_H=480`
- `UAV_PROCESS_EVERY_N_FRAMES=2`
- `UAV_MOTION_ZOOM_CROP_SIZE=160`
- `UAV_MOTION_ZOOM_MAX_ROIS=1`
- `UAV_MOTION_ZOOM_ONLY=1`
- `UAV_HOVER_HOLD=0`
- `UAV_FULLFRAME_SEARCH_FALLBACK=0`
- `UAV_FULLFRAME_FALLBACK_ONLY=0`
- `UAV_NIGHT_STATIC_POINT_SUPPRESS=1`

## 夜间静止光点屏蔽

夜间初始化阶段会基于每路摄像头自己的静态背景，生成静止亮点/暗点屏蔽图。该屏蔽图来自背景中值图、局部对比度和初始化期间的稳定性统计。

本次补充后，静止光点屏蔽不只过滤 ROI 和 YOLO 框，还会在帧差 mask 生成后直接扣除固定星点、固定灯点等区域，减少它们反复触发候选框。

## 视频回传实验

正式回传参数：

- 目标地址：`192.168.0.200`
- 端口：`9999`
- 回传分辨率：`640x480`
- `UAV_VIDEO_SEND_EVERY_N_FRAMES=1`
- `UAV_VIDEO_JPEG_QUALITY=25`

调试时曾临时指向本机 `192.168.0.30` 做20秒UDP接收测试。实际收到 `BOARD_3` 5路、`BOARD_4` 4路，共9路视频流；每路约 `10 fps`，最大帧间隔约 `0.125s` 到 `0.219s`，末帧年龄均小于 `0.2s`。当前接入摄像头数量下，640x480单帧回传满足小于 `0.5s` 的观察要求。

## 一键脚本

夜间400米：

```powershell
.\scripts\switch_board3_8_night400.bat
```

白天高层500米：

```powershell
.\scripts\switch_board3_8_day500.bat
```

通用脚本：

```powershell
.\scripts\switch_high_layer_profiles.ps1 -Mode night
.\scripts\switch_high_layer_profiles.ps1 -Mode day
```

640x480接收端：

```powershell
.\tools\start_640x480_receiver.bat
```
