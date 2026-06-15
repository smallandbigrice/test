# 2026-06-15 main.py 低层回退到视频验证版

## 修改目标

把 `main.py` 的低层复杂背景检测逻辑，同步回本地 `single_video_main_pt.py` 视频推理验证后确认可用的一版。

这次同步的重点不是继续加强背景压制，而是恢复真实远距小目标的送检机会，避免高楼附近弱小目标在候选筛选阶段被提前卡掉。

## 本次修改

### 1. 低层局部差分阈值放宽

- `MIN_LOCAL_DIFF_MEAN: 16.0 -> 10.0`

作用：

- 放宽低层候选区域对局部差分强度的要求；
- 让复杂背景中较弱但真实存在的远距小目标更容易通过首轮候选筛选。

### 2. 关闭灰度噪声抑制

- `ENABLE_GRAY_NOISE_SUPPRESSOR = False`

作用：

- 停止在低层场景中用局部灰度纹理和亮点规则提前压掉候选；
- 避免高楼/建筑附近的小目标被当作灰度噪声误杀。

### 3. 放宽 LCM 约束

- `LCM_MIN_SCORE: 2.4 -> 1.8`
- `LCM_MIN_RATIO: 1.45 -> 1.25`
- `LCM_REQUIRE_BOTH: True -> False`

作用：

- 恢复低层复杂背景下弱响应目标的通过率；
- 不再要求 LCM 两个指标同时严格满足，避免对高楼附近碎块型小目标筛得过狠。

### 4. 进一步减弱形态学压制

- `MOTION_ERODE_ITER: 1 -> 0`
- `MOTION_OPEN_ITER: low=1 -> 0`

保留：

- `MOTION_DILATE_ITER = 1`
- `MOTION_CLOSE_ITER = 1`

作用：

- 尽量保留碎块化的小目标响应；
- 减少腐蚀和开运算把远距小目标打散、打没的问题。

### 5. 撤回低层强压制逻辑

- `ENABLE_EDGE_DENSITY_FILTER: True -> False`
- `ENABLE_LOW_LAYER_EDGE_HARD_FILTER: True -> False`
- `ENABLE_SPATIAL_CHAOS_FILTER: True -> False`
- `ENABLE_ADAPTIVE_TRACK_CONFIRM: True -> False`

作用：

- 撤回 2026-06-15 那版偏强的低层建筑/植被压制策略；
- 恢复到更接近 6 月中旬视频验证可用的筛选强度。

### 6. 轨迹搜索门槛同步回较宽松版本

- `TRACK_SEARCH_MIN_YOLO_HITS = 2`
- `TRACK_SEARCH_MIN_RECENT_HITS = 2`

作用：

- 让低层候选在后续轨迹搜索阶段不过早失去延续机会；
- 与本地 PT 推理验证使用的轨迹搜索门槛保持一致。

## 影响范围

本次修改影响：

- 帧差候选筛选；
- 形态学后处理；
- LCM 局部对比度过滤；
- 灰度噪声抑制；
- 低层轨迹搜索门槛。

本次未修改：

- 5 路相机并发结构；
- RKNN 推理结构；
- 发送链路；
- 高层核心检测路线。

## 验证依据

本次 `main.py` 同步依据来自 PC 端视频 `5.mp4` 的本地验证结果。

用于验证的脚本版本特征：

- 更弱形态学；
- 关闭灰度噪声抑制；
- 放宽 LCM；
- 不启用 `lcm require both`。

已生成对应推理产物：

- `E:\detect uav\record_2k_pure\5_low_relaxed_lcm_grayoff_include_small_infer.mp4`
- `E:\detect uav\record_2k_pure\5_low_relaxed_lcm_grayoff_include_small_mask.mp4`
- `E:\detect uav\record_2k_pure\5_low_relaxed_lcm_grayoff_include_small_infer.csv`

## 风险说明

这版回退后，真实小目标送检率会上升，但复杂背景噪声也可能同步增加。

因此这次同步的核心含义是：

- 先恢复“能送检”；
- 后续再通过更细的候选合并、局部规则或训练数据优化，继续压虚警。
