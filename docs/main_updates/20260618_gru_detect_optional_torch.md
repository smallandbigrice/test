# 2026-06-18 main_gru_detect.py 实现可选 PyTorch 依赖与降级回退 (Fallback) 机制

## 修改日期

2026-06-18

## 修改目标

由于板端部署环境可能没有安装 `torch` (PyTorch) 库，为了避免程序在启动导入模块阶段就因 `ModuleNotFoundError` 发生崩溃，并确保算法能够完全退化为“纯 CPU 图像二值化检测 + 基础特征追踪”，我们对 `main_gru_detect.py` 进行了依赖解耦改动。

本次修改使得 PyTorch 变为可选依赖：
- 有 `torch` 环境：保留完整的 GRU 时序噪点过滤和未来轨迹预测功能；
- 无 `torch` 环境：跳过 GRU 模型加载与推理，自动降级为“连续帧数判定机制”（只要图像追踪链持续 $\geq 5$ 帧即认定为有效无人机目标），保持高亮框选并在大屏幕输出展示，同时正常向云台发送数传 UDP 数据。

## 本次改动内容

### 1. 顶层 PyTorch 模块可选导入

在顶层导入块中，使用 `try...except` 结构包裹 `torch` 与 `torch.nn` 的导入，并定义全局标志 `HAS_TORCH`：
```python
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
```

### 2. 神经网络模型定义条件化

- **有 PyTorch 环境**：正常定义继承自 `nn.Module` 的 `UAVTrajectoryNet` 神经网络。
- **无 PyTorch 环境**：声明一个 Mock (存根) 类 `UAVTrajectoryNet`，其静态加载方法 `load_from_checkpoint` 会输出警告日志并优雅返回 `None`。这避免了因没有 `nn.Module` 继承源而导致的类声明语法错误。

### 3. 多目标追踪降级回退 (Fallback)

在相机的 `capture_job` 更新循环中：
- 当 `gru_model is not None`：维持高精度的 GRU 时序滑动特征分类判定和 5 步预测外推。
- 当 `gru_model is None`（如未安装 torch，或模型文件加载失败）：对于所有活跃的追踪轨迹（`tracker.tracks`），执行启发式长度过滤：
  - 如果轨迹持续更新累积帧数 $\geq 5$ 帧：视其为确定性的有效无人机目标（置 `is_uav = True`，分类置信度置为恒定 `1.0`）；
  - 如果轨迹更新累积帧数 $< 5$ 帧：作为初生候选，暂不判定为无人机（置 `is_uav = False`）。
  - 将 `pred_coords`（未来预测路径点）强制设为空列表 `[]`（不画黄色预测线）。

## 影响范围

- **兼容性大幅提高**：可在任何仅安装了 `numpy` 和 `opencv-python` 的精简 Linux CPU 环境（包括 RK3588 精简固件系统）下直接运行，不报任何 `ImportError`。
- **原始 `main.py`**：依然保持完全原样，未做任何修改。

## 本地验证情况

1. **语法编译验证**：
   - 运行 `D:\conda\python.exe -m py_compile main_gru_detect.py`，成功编译通过，未报任何语法错误。
2. **逻辑回退验证**：
   - 当 `gru_model` 设为 `None` 时，所有追踪轨迹判定正确回退至启发式长度条件过滤（历史长度达 5 帧自动判定为 UAV 并正常驱动画面渲染和云台发送），避免了调用 `torch` 相关的张量转换和张量推理方法。
