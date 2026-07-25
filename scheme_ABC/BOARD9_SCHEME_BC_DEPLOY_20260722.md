# 板9方案 B/C 部署记录

时间：2026-07-22

板端：board9，`192.168.0.9`

## 操作目的

将板9从复杂背景低层检测实验状态恢复到封档方案 B/C：

- 方案 B：白天高层 500 米检测。
- 方案 C：夜间无光全场景检测。

## 已保存的复杂背景实验状态

本地归档目录：

`E:\detect uav\complex_background_lowlayer\board9_complex_state_backup_before_BC_20260722_110415`

归档包含：

- `/home/Tronlong/h265_main_test/`
- `/home/Tronlong/run_board9_h265_complex_frame3_bg_gate.sh*`
- `/home/Tronlong/run_board9_replay*`
- `/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_run014_run015_board9manual_roi640_20260721_fp16.rknn`
- `/home/Tronlong/uav_debug/board9_300m_track_replay_board9manual_100e_best_dual030_age24_20240826_051345/`
- 切换前生产路径代码和 `python_autostar` systemd 配置。

远端备份目录：

`/home/Tronlong/backup_before_schemeBC_20260722_110415`

## 部署到板9的封档文件

生产代码路径：

`/home/Tronlong/rknn_model_zoo/examples/yolov5/python/`

已部署文件：

- `main.py`
- `yololib.py`
- `comms.py`

模型路径：

`/home/Tronlong/rknn_model_zoo/examples/yolov5/model/`

已部署模型：

- `yolov5s_day_20260626.rknn`
- `yolov5s_night_latest.rknn`

## 当前运行状态

当前已启动方案 B：

- `UAV_BOARD_ID=BOARD_9`
- `UAV_SCENE_MODE=day`
- `UAV_LAYER_MODE=high`
- `UAV_RKNN_MODEL=/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_day_20260626.rknn`
- `UAV_YOLO_CONF=0.40`
- `UAV_CAPTURE_W=2560`
- `UAV_CAPTURE_H=1440`
- `UAV_DIFF_W=1920`
- `UAV_DIFF_H=1080`
- `UAV_PROCESS_EVERY_N_FRAMES=3`
- `UAV_HOVER_HOLD=1`
- `UAV_VIDEO_TARGET_IP=192.168.0.200`
- `UAV_VIDEO_STREAM_W=640`
- `UAV_VIDEO_STREAM_H=480`
- `UAV_VIDEO_SEND_EVERY_N_FRAMES=2`
- `UAV_VIDEO_JPEG_QUALITY=25`

服务状态：

- `python_autostar.service`：`active`
- `NRestarts=0`

## 文件校验

- `main.py`：`90c2fed3d2b9c243086c75c47af03643bea0435196f37b281772579fd00ba13d`
- `yololib.py`：`4bbc3cb67097b706de68f8929d696523a158815c9d01cc0f252b908b738cf0df`
- `comms.py`：`15f42b53765efb26fac1f61c6e415317068b30f7387f3cb04de69804dd756127`
- `yolov5s_day_20260626.rknn`：`273a2162528e9fd743f8dfcb882c9ffe3dcf51e04e0f75a3ec749b5acd708941`
- `yolov5s_night_latest.rknn`：`fb0d7395ca934b14a6f8779b413c7733d7aacacaf998876d757b0207d44f2779`

方案 C 当前输入口径为 H265 彩色摄像头，板端硬解后只取 NV12 的 Y 平面进入夜间检测流程；启动后 1 分钟内命中的小目标登记为星点屏蔽区域。

## 说明

板9当前自启入口仍为：

`/home/Tronlong/rknn_model_zoo/examples/yolov5/python/main.py`

复杂背景实验代码已本地保存，不作为板9当前自启入口。
