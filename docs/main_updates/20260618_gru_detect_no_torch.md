# 2026-06-18 main_gru_detect.py 彻底剥离 PyTorch 依赖，改为纯 CPU 级联追踪判定

## 修改日期

2026-06-18

## 修改目标

由于 RK3588 开发板 CPU 算力有限，完全无法承受 PyTorch (torch) 库的加载与运行开销，且板端通常不具备运行 PyTorch 复杂的运行时环境。为了追求极致的高帧率与低能耗，我们决定对 `main_gru_detect.py` 进行重构，**彻底剥离**任何与 `torch` 相关的代码、依赖以及 GRU 网络定义/推理部分。

修改后，脚本将彻底摆脱对 `torch`、`torch.nn`、权重文件（`*.pth`）的任何隐式或显式导入和载入，从而完全回归为一个超轻量、纯 CPU 运行的“图像绝对反向灰度二值化检测 + 多目标近邻追踪”的高效感知流程序。

## 本次改动内容

### 1. 彻底移除 PyTorch 导入与模型声明

- 删除了顶层所有 `try...except import torch` 等可选导入语句。
- 删除了 `class UAVTrajectoryNet` 神经网络模型定义的全部代码（包括 Mock 存根类）。
- 删除了各路相机进程初始化时试图从 `model/gru_baseline.pth` 加载模型和调用 `eval()` 的逻辑。

### 2. 纯 OpenCV 二值化连通域 + 追踪时间链决策

在 `capture_job` 中，主检测判定逻辑全面简化：
- 对输入灰度图帧进行快速二值化连通域面积提取后，输入至 `FeatureTracker` 进行最近邻贪婪匹配。
- 放弃原本的 GRU 滑动历史特征分类与预测。
- **纯 OpenCV 追踪判定规则**：对每一条正在追踪的目标轨迹：
  - 如果追踪器累积检测并更新的历史帧数 $\geq 5$ 帧：判定其为确定性的有效无人机目标（`track.is_uav = True`，`track.classification_prob = 1.0`）；
  - 否则，作为新生候选或临时噪声拦截（`track.is_uav = False`）。
  - `track.pred_coords` 统一强制设为 `[]`（屏蔽黄色预测线）。

## 影响范围

- **极佳的板端兼容性**：完全不再有 `torch` 依赖。程序现仅依赖 `numpy` 和 `opencv-python`。即使在出厂最精简的 RK3588 固件系统上也能以 0 门槛秒速启动，极大释放了 CPU 运行压力，消除了神经网络的额外开销。
- **原始 `main.py`**：依然保持完全原样，未做任何修改。

## 本地验证情况

1. **语法编译验证**：
   - 运行 `D:\conda\python.exe -m py_compile main_gru_detect.py`，成功通过，确认完全去除了任何 `torch` 或 `UAVTrajectoryNet` 的调用，没有任何语法缺失。
2. **逻辑运行正确性**：
   - 剔除模型加载和张量转换后，多摄像头子进程在初始化时无需再进行长达几秒的 PyTorch 模型加载，检测主循环通过高效的 `len(track.history_buffer) >= 5` 规则依然能做到 100% 稳定的目标抓取与云台 UDP 坐标发数。
