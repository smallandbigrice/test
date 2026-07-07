# 20260707 板端高层固定方案备份

## 修改目标

本次用于备份当前板端可用代码版本。根据现场测试结果，撤回“边缘目标特殊 ROI 裁剪/填充”实验，保留当天确认过的高层检测方案，用于板 3 及以上高层摄像头。

## 本次代码状态

- `main.py` 当前版本号：`frame-diff-rknn-static-confirm-lowmotion-20260707`
- 算法主线仍为静态背景蒙版 + 帧差候选 + 640 ROI 送入 RKNN YOLO + 近期命中/轨迹/低速静止确认。
- 已移除本次边缘 ROI 实验代码，代码中不再保留 `EDGE_CENTERED` 相关逻辑。
- 板端运行时固定使用白天高层方案：
  - `UAV_SCENE_MODE=day`
  - `UAV_LAYER_MODE=high`
  - `UAV_BOARD_ROW_IDX=1`
  - `UAV_CAPTURE_W=2560`
  - `UAV_CAPTURE_H=1440`
  - `UAV_DIFF_W=1920`
  - `UAV_DIFF_H=1080`
- 关闭运动放大、小框 zoom、全帧直接推理等实验入口，避免和当前高层方案冲突。

## 相关依赖

- `comms.py` 保留 UDP 视频分片发送能力，用于板端多路画面回传。
- `yololib.py` 保留 RKNN NPU core 选择和调试输出能力。
- `.gitignore` 增加 `artifacts/`，避免后续把推理视频、CSV、截图等大文件误提交。

## 板端同步记录

- 已同步并验证：board3、board4、board5、board6、board7、board8。
- board9 当时网络未联通，需联通后再同步同一版本。
- 本次 GitHub 备份只记录代码状态，不提交板端生成的视频、日志、数据集和临时脚本。

## 验证

- 本地执行 `python -m py_compile main.py comms.py yololib.py` 通过。
- 本地确认 `main.py` 中无 `EDGE_CENTERED` 相关残留。

## 回退方式

如后续板端效果变差，可从本次 GitHub 分支恢复 `main.py`、`comms.py` 和 `yololib.py`，再按上述环境变量重新部署到对应板端。
