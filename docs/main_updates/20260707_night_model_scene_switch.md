# 20260707 夜间模型与昼夜切换脚本同步

## 修改目标

把最新夜间 RKNN 模型纳入当前代码版本，并让正式 `main.py`、板端视频测试入口和本地昼夜切换脚本使用同一套模型选择规则。

## 模型文件

- 新增夜间模型：`model/yolov5s_night_latest.rknn`
- 来源文件：`E:\download\yolov5s (7).rknn`
- MD5：`003214966000be2605e8a0148790b931`
- 该模型对应此前夜间视频 9 效果较好的 7 月 2 日 RKNN 模型。

## 代码调整

- `main.py` 增加按 `UAV_SCENE_MODE` 选择模型：
  - `day` 优先查找 `yolov5s_day_20260626.rknn`
  - `night` 优先查找 `yolov5s_night_latest.rknn`
  - `UAV_RKNN_MODEL` 仍保留最高优先级，可手动覆盖
- `apps/video_inference/videomain.py` 同步相同模型选择逻辑，避免视频测试和正式入口加载不同模型。
- 夜间默认 `UAV_PROCESS_EVERY_N_FRAMES=2`，白天默认 `3`；仍可通过环境变量覆盖。

## 本地切换脚本

更新 `scripts/switch_board_scene_mode.ps1`：

- 白天写入 26 号白天模型路径。
- 夜间写入 `yolov5s_night_latest.rknn` 路径。
- 默认会把本机对应模型上传到板端模型目录；如只想切环境变量，可加 `-SkipModelDeploy`。

示例：

```powershell
.\scripts\switch_board_scene_mode.ps1 -BoardIp 192.168.0.3 -Mode night
.\scripts\switch_board_scene_mode.ps1 -BoardIp 192.168.0.3 -Mode day -SkipModelDeploy
```

## systemd 模板

- `tools/systemd/30-night-capture.conf` 补齐夜间 640x480、两帧帧差、160 ROI 放大和夜间模型路径。
- `tools/systemd/30-day-capture.conf` 补齐白天 2K、三帧帧差和 26 号白天模型路径。

## 验证

- `python -m py_compile main.py apps/video_inference/videomain.py comms.py yololib.py` 通过。
- PowerShell 脚本语法检查通过。
