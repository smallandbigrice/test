# 方案C夜间无光全场景检测封档说明

封档日期：2026-07-25

## 封档结论

方案C正式命名为“方案C夜间无光全场景检测”。本次现场验证使用板8单路 H265 摄像头完成，验证结论为：彩色 H265 输入经板端硬解后只取 NV12 Y 平面，能够保持与灰度相机方案一致的夜间检测输入形态。

正式部署仍按原方案C的5摄像头结构执行，不改成单摄方案。单摄测试仅用于确认 H265/Y 平面输入链路。

## 固定配置

- 场景模式：`UAV_SCENE_MODE=night`
- 检测层级：`UAV_LAYER_MODE=high`
- 摄像头数量：`UAV_VIDEO_CAM_COUNT=5`
- 输入链路：`UAV_H265_YPLANE_CAPTURE=1`
- H265设备：`UAV_H265_DEVICES=auto`
- 采集尺寸：`UAV_CAPTURE_W=640`，`UAV_CAPTURE_H=480`
- 帧差尺寸：`UAV_DIFF_W=640`，`UAV_DIFF_H=480`
- 帧差间隔：`UAV_PROCESS_EVERY_N_FRAMES=2`
- 运动ROI：`UAV_MOTION_ZOOM_CROP_SIZE=160`
- 单帧ROI上限：`UAV_MOTION_ZOOM_MAX_ROIS=1`
- 送检方式：`UAV_MOTION_ZOOM_ONLY=1`
- 全帧兜底：`UAV_FULLFRAME_SEARCH_FALLBACK=0`
- 悬停保持：`UAV_HOVER_HOLD=0`
- 模型：`yolov5s_night_latest.rknn`

## 启动星点屏蔽

方案C默认开启启动星点屏蔽：

- `UAV_NIGHT_STARTUP_STAR_SUPPRESS=1`
- `UAV_NIGHT_STARTUP_STAR_SECONDS=60`
- `UAV_NIGHT_STARTUP_STAR_PAD=6`

背景初始化完成后，前60秒内由YOLO命中的小目标会登记为静止星点区域。登记后，该区域会直接写入屏蔽掩码，后续帧差候选框和YOLO结果只要命中该掩码区域就会被过滤。

## 对应文件

- 主代码：`E:\detect uav\scheme_ABC\code\main.py`
- 方案配置：`E:\detect uav\scheme_ABC\configs\scheme_C_night_no_light_all_scene.conf`
- 批量切换：`E:\detect uav\scheme_ABC\scripts\switch_high_layer_profiles.ps1 -Mode night`
- 单板切换：`E:\detect uav\scheme_ABC\scripts\switch_board_scene_mode.ps1 -Mode night`
- 桌面入口：`E:\detect uav\scheme_ABC\scripts\desktop_wrappers\一键切换_方案C_夜间无光全场景检测_板1到板9.bat`
