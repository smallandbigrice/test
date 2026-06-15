# 20260615 植被与复杂边缘虚警抑制

## 修改目标

针对风吹柳条、整棵树晃动、高纹理建筑边缘等复杂背景虚警，给 `main.py` 增加更稳的前端 ROI 抑制和后端轨迹确认约束。

本次只实现前 5 项策略：

1. 形态学开运算滤波。
2. 空间混乱度抑制。
3. 边缘密度滤波器改为降权机制。
4. 轨迹直达率与最小净位移过滤。
5. 确认帧数自适应拉长。

未改 NPU 多核绑定，也未开启 GMC。

## 本次改动

### 1. 形态学开运算

在 `cleanup_motion_mask()` 中加入 `cv2.MORPH_OPEN`：

```text
3x3 ellipse opening
```

当前默认：

```text
ENABLE_MOTION_OPENING = True
MOTION_OPEN_ITER = 0 if HIGH_LAYER_MODE else 1
```

也就是低层复杂背景启用一次轻量开运算，高层天空默认不做开运算，避免远距离 1-2 像素小目标被腐蚀掉。

### 2. 空间混乱度抑制

新增 `apply_spatial_chaos_filter()`。

逻辑：

- 以 `CHAOS_CELL_PX = 220` 像素为局部统计单元。
- 如果一个局部区域附近 ROI 数量达到 `CHAOS_LOCAL_ROI_LIMIT = 5`，认为该区域疑似植被/复杂背景震荡。
- 对该区域 ROI 做降权。
- 每个混乱 cell 只保留分数最高的 `CHAOS_KEEP_PER_CELL = 1` 个 ROI。

这样可以避免树叶晃动时产生大量 ROI，把 RKNN/NPU 推理队列堵住。

### 3. 边缘密度降权

开启：

```text
ENABLE_EDGE_DENSITY_FILTER = True
```

但不再把高边缘密度 ROI 一票否决，而是计算风险分数：

```text
edge_texture_risk()
```

高纹理区域会降低 ROI 排序分数，并把风险传递给后续 YOLO 结果和轨迹确认。

这样做的目的：

- 树叶、建筑窗格、楼边缘优先级下降。
- 无人机如果恰好经过复杂背景附近，仍然保留被 YOLO 确认的机会。

### 4. 轨迹净位移与直达率确认

新增 `_passes_motion_confirmation()`。

轨迹确认时会检查：

- `net_displacement_px`：首尾净位移。
- `straightness`：直达率。

默认阈值：

```text
TRACK_CONFIRM_MIN_NET_MOTION_PX = 6.0 if HIGH_LAYER_MODE else 10.0
TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX = 10.0 if HIGH_LAYER_MODE else 16.0
TRACK_CONFIRM_MIN_STRAIGHTNESS = 0.24 if HIGH_LAYER_MODE else 0.35
TRACK_CONFIRM_RISK_MIN_STRAIGHTNESS = 0.45
```

来自复杂背景的高风险轨迹必须有更明显的净位移和更直的运动趋势，才能出绿框。

### 5. 自适应确认帧数

新增 `_adaptive_required_hits()`。

默认策略：

- 高风险背景轨迹：确认命中数提高到 `TRACK_CONFIRM_RISK_MIN_HITS = 7`。
- 低层小目标：确认命中数提高到 `TRACK_CONFIRM_SMALL_MIN_HITS = 6`。
- 高层低风险小目标不全局拉长确认帧数，保留远距离检测实时性。

## 算法逻辑影响

- 帧差：增加低层轻量 opening。
- 蒙版：未修改。
- ROI：新增空间混乱度限流与边缘密度降权。
- YOLO/RKNN：推理方式未修改。
- 轨迹：新增 ROI 风险传递、净位移、直达率、自适应确认帧数。
- 绿框确认：复杂背景来源的轨迹更难出绿框，纯天空目标尽量保持原实时性。

## 参数变化

```text
ALGORITHM_VERSION = frame-diff-vegetation-filter-20260615
ENABLE_EDGE_DENSITY_FILTER = True
ENABLE_SPATIAL_CHAOS_FILTER = True
ENABLE_ADAPTIVE_TRACK_CONFIRM = True
REF_TRAJ_MIN_STRAIGHTNESS = 0.30 if HIGH_LAYER_MODE else 0.45
```

## 验证情况

已执行语法检查：

```text
python -m py_compile main.py
```

当前电脑环境没有 RK3588 板端 5 路摄像头，未做板端实机推理验证。

## 已知风险

- 边缘密度计算会增加少量 CPU 开销。
- 复杂背景轨迹确认更严格，绿框出现时间可能比上一版略晚。
- 如果无人机真实悬停或短距离慢速移动，净位移过滤可能导致确认变慢。
- 后续需要用 5、8 等复杂背景视频和板端实机画面继续调：
  - `CHAOS_LOCAL_ROI_LIMIT`
  - `EDGE_DENSITY_SCORE_PENALTY`
  - `TRACK_CONFIRM_RISK_MIN_HITS`
  - `TRACK_CONFIRM_RISK_MIN_NET_MOTION_PX`
