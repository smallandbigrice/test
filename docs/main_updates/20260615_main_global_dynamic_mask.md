# 2026-06-15 main.py 接入全局 1 分钟动态蒙版

## 修改目标

把视频 `5` 验证中效果较好的“动态蒙版抑制持续晃动背景”思路同步到 `main.py`。

这次设计成：

- 全局范围生效；
- 每路相机在启动后的前 `1` 分钟在线统计；
- 不缓存整分钟所有帧，改为在线累计高频运动像素；
- 1 分钟结束后生成一张动态运动蒙版；
- 后续在帧差候选生成前，先把这部分高频背景运动区域抠掉。

## 本次改动

### 1. 新增全局动态蒙版参数

在 `main.py` 中新增：

- `ENABLE_DYNAMIC_MOTION_MASK = True`
- `DYNAMIC_MOTION_MASK_SECONDS = 60.0`
- `DYNAMIC_MOTION_MASK_SAMPLE_INTERVAL = 0.5`
- `DYNAMIC_MOTION_MASK_DIFF_THRESH = 8`
- `DYNAMIC_MOTION_MASK_HIT_RATIO = 0.12`
- `DYNAMIC_MOTION_MASK_MIN_PAIRS = 20`

含义：

- 前 1 分钟内每隔 `0.5s` 抽一次当前灰度帧做统计；
- 统计相邻采样帧之间的帧差；
- 若某像素在足够多次采样中都表现为运动，则认为它属于持续晃动背景；
- 后续推理时，这些像素不再进入帧差候选。

### 2. 新增动态蒙版辅助函数

新增函数：

- `update_dynamic_motion_counts(...)`
- `finalize_dynamic_motion_mask(...)`
- `apply_dynamic_motion_mask(...)`

作用：

- 在线累计高频运动像素；
- 1 分钟结束后生成最终动态蒙版；
- 在正式帧差候选生成前应用该蒙版。

### 3. 在 `camera_worker()` 中接入在线统计

每路相机各自维护：

- `dynamic_motion_hits`
- `dynamic_motion_prev_gray`
- `dynamic_motion_pair_count`
- `dynamic_motion_start_ts`
- `dynamic_motion_next_sample_ts`
- `dynamic_motion_mask`

这样每路相机都能独立学习自己的长期动态背景，而不会互相干扰。

### 4. 在帧差候选入口增加动态蒙版过滤

原先低层候选链路是：

1. `absdiff`
2. `threshold`
3. `apply_static_bg_mask`
4. `cleanup_motion_mask`
5. 候选 ROI

现在变成：

1. `absdiff`
2. `threshold`
3. `apply_static_bg_mask`
4. `apply_dynamic_motion_mask`
5. `cleanup_motion_mask`
6. 候选 ROI

也就是把动态蒙版放在形态学之前，尽早挡掉长期晃动区域。

## 设计取舍

这次没有采用“缓存前 1 分钟全部帧后一次性建模”的方式，原因是那样在 5 路 2K/1080p 场景下内存压力太大。

改成在线累计后：

- 内存开销稳定；
- 各路相机都能独立生成动态蒙版；
- 启动 1 分钟后即可进入稳定抑制状态。

## 影响范围

本次修改影响：

- 低层/高层帧差候选入口；
- 每路相机启动后前 1 分钟的背景学习逻辑；
- 后续 ROI 送检总量。

本次未修改：

- RKNN 推理结构；
- 轨迹跟踪核心逻辑；
- 网络发送链路；
- 5 路进程/线程结构。

## 风险说明

全局动态蒙版会压掉“长期高频运动”的区域，因此对树叶、草木、水面反光这类背景通常有效。

但如果真实目标长时间反复经过同一区域，且该区域在前 1 分钟内被统计为高频运动区域，后续也可能被一起压掉。

因此这版重点是先降低持续晃动背景误检，后续仍需结合实机画面继续观察：

- 是否把下方植被误检压到了可接受范围；
- 是否对真实远距小目标送检率造成过大影响。
