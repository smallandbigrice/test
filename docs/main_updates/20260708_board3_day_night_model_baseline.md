# 20260708 板3昼夜模型作为板端基准

## 目的

根据现场确认，以板 3 当前可用的白天高层模型和夜间高层模型作为后续板端统一基准。板 3、板 4 已先切换为白天高层运行并验证通过，后续板 3 到板 8 均按该基准统一。

## 白天高层基准模型

- 文件名：`model/yolov5s_day_20260626.rknn`
- 来源：从板 3 当前白天封板可用模型 `/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s.rknn` 拉取并固定命名。
- MD5：`30261610759aec78c94cdf7d6f6510ce`
- 大小：`15884417` bytes

说明：该模型是板 3/板 4 现场白天高层封板效果对应的模型。后续虽然沿用 `yolov5s_day_20260626.rknn` 文件名以兼容现有切换脚本，但实际基准以 MD5 为准。

## 夜间高层基准模型

- 文件名：`model/yolov5s_night_latest.rknn`
- 来源：板 3 当前夜间高层模型，与本地 `E:\download\yolov5s (7).rknn` 一致。
- MD5：`003214966000be2605e8a0148790b931`
- 大小：`15881601` bytes

## 当前运行策略

白天高层：

- `UAV_SCENE_MODE=day`
- `UAV_RKNN_MODEL=/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_day_20260626.rknn`
- `UAV_CAPTURE_W=2560`
- `UAV_CAPTURE_H=1440`
- `UAV_DIFF_W=1920`
- `UAV_DIFF_H=1080`
- `UAV_PROCESS_EVERY_N_FRAMES=3`
- `UAV_MOTION_ZOOM_ONLY=0`
- `UAV_HOVER_HOLD=1`

夜间高层：

- `UAV_SCENE_MODE=night`
- `UAV_RKNN_MODEL=/home/Tronlong/rknn_model_zoo/examples/yolov5/model/yolov5s_night_latest.rknn`
- `UAV_CAPTURE_W=640`
- `UAV_CAPTURE_H=480`
- `UAV_DIFF_W=640`
- `UAV_DIFF_H=480`
- `UAV_PROCESS_EVERY_N_FRAMES=2`
- `UAV_MOTION_ZOOM_CROP_SIZE=160`
- `UAV_MOTION_ZOOM_ONLY=1`
- `UAV_HOVER_HOLD=0`

## 板3和板4预验证

已将板 3、板 4 切换到白天高层，并验证：

- `yolov5s_day_20260626.rknn` MD5 为 `30261610759aec78c94cdf7d6f6510ce`
- `yolov5s_night_latest.rknn` MD5 为 `003214966000be2605e8a0148790b931`
- `python_autostar.service` 为 `active`
- 启动日志显示白天高层参数：`capture=2560x1440 diff=1920x1080 fixed_stride=3 direct_fullframe=no zoom=0x0`

## 后续约束

不要再用模糊的 `yolov5s.rknn` 判断白天模型版本。白天和夜间都应优先看固定文件名和 MD5：

- 白天：`yolov5s_day_20260626.rknn` / `30261610759aec78c94cdf7d6f6510ce`
- 夜间：`yolov5s_night_latest.rknn` / `003214966000be2605e8a0148790b931`
