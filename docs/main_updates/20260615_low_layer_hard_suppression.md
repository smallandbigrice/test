# 20260615 低层复杂背景强压版本

## 修改目标

上一版将边缘密度过滤从硬过滤改成降权，目的是避免无人机经过复杂背景附近时被直接误杀。

但视频 5 实测反馈显示，高楼边缘又被 YOLO 识别出来，说明低层复杂背景不能只做软降权，需要恢复更强的高纹理硬过滤。

本版目标：

- 低层复杂背景恢复边缘密度硬过滤。
- 高层天空仍保留软降权，避免远距离小目标被过度过滤。
- 不修改 NPU 推理结构。
- 不修改模型。

## 本次改动

### 1. 版本号

```text
ALGORITHM_VERSION = frame-diff-low-layer-hard-suppress-20260615
```

### 2. 新增低层边缘硬过滤开关

```text
ENABLE_LOW_LAYER_EDGE_HARD_FILTER = not HIGH_LAYER_MODE
```

含义：

- 低层模式：启用硬过滤。
- 高层模式：不启用硬过滤，只保留原来的边缘密度降权。

### 3. 恢复低层 ROI 的纹理硬门槛

在 `motion_rois_from_mask()` 中，低层候选必须满足：

```text
far tiny:
texture <= EDGE_DENSITY_THRESH

near compact:
texture <= NEAR_EDGE_DENSITY_THRESH
```

这样高楼、树叶、窗格等高纹理背景不会只靠降权继续进入 YOLO。

## 算法影响

- 帧差：未修改。
- 静态背景蒙版：未修改。
- 形态学 opening：保留。
- 空间混乱度抑制：保留。
- 边缘密度：低层恢复硬过滤，高层保持软降权。
- YOLO/RKNN：未修改。
- 轨迹确认：未修改。

## 视频 5 PC 推理验证

使用模型：

```text
E:\download\best only big.pt
```

输入视频：

```text
E:\detect uav\record_2k_pure\5.mp4
```

输出：

```text
E:\detect uav\record_2k_pure\5_hardsuppress_only_big_infer.mp4
E:\detect uav\record_2k_pure\5_hardsuppress_only_big_mask.mp4
E:\detect uav\record_2k_pure\5_hardsuppress_only_big_infer.csv
```

统计对比：

```text
旧 only big：raw 183，target 360，轨迹 11 条
上一版低层参数：raw 142，target 171，轨迹 8 条
本版强压：raw 79，target 156，轨迹 8 条
```

说明：

- raw 红框明显减少，说明高纹理候选被压掉一部分。
- target 绿框数量略降，但仍有部分轨迹保留。
- 后续需要人工查看视频，确认真实目标是否被过度压制。

## 已知风险

- 低层高楼/树叶虚警会减少，但无人机如果贴近复杂背景飞行，也可能更难进入 YOLO。
- 高层天空不使用该硬过滤，避免影响 400 到 500 米小目标。
- 如果视频 5 中仍有高楼误检，需要继续提高低层 `EDGE_DENSITY_THRESH` 或引入更长时间动态背景统计。

## 验证

已执行：

```text
python -m py_compile main.py
```
