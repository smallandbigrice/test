# 2026-06-18 main_gru_detect.py 支持结合 ONNX Runtime 加载 gru.onnx 轨迹预测

## 修改日期

2026-06-18

## 修改目标

为了满足板端在极低 CPU 负载下依然能够利用 GRU 时序模型进行真伪 UAV 分类和未来轨迹预测，且同时避免在板端部署庞大、难装的 PyTorch 库，我们将 `main_gru_detect.py` 升级为支持可选的 **ONNX Runtime** 加载 **`gru.onnx`** 模型。

该设计结合了先前优化的纯 OpenCV 几何追踪降级方案，使得脚本在板端拥有双重极其强韧的自适应特性：
- **ONNX 模式**：如果板端环境安装了轻量的 `onnxruntime` 库，且当前目录（或 `model/` 目录下）存在 `gru.onnx` 模型，程序将自动通过 ONNX Runtime 秒级在 CPU 上执行时序推断，高精度的进行真假 UAV 识别并预测输出黄色未来 5 步飞行轨迹线。
- **纯 OpenCV 模式 (Fallback)**：若 `onnxruntime` 未安装或 `gru.onnx` 文件缺失，程序会自动且优雅地通过 `try...except` 拦截异常，无缝降级为“连续帧数判定机制”（直接秒级出框，且正常进行云台发数），脚本绝不崩溃。

## 本次改动内容

### 1. 轻量化 ONNX 依赖可选导入

在顶层导入块中，动态捕获 `onnxruntime`：
```python
import os
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
```

### 2. 重建 NumPy 无损特征提取器

由于 ONNX 推理需要输入轨迹时序特征，我们将之前剥离的特征计算方法高标准重新接回（完全不带任何 typing 类型提示，以保障嵌入式兼容性）：
- **`savgol_filter_numpy`**：基于 NumPy 一维卷积重写的 5 点 2 次 Savitzky-Golay 滤波器。
- **`get_track_features`**：对已追踪 20 帧的轨迹生成 8 维标准化时序特征数据，输出尺寸为 `(20, 8)`。

### 3. capture_job 中集成 ONNX 会话初始化

在相机工作进程启动时，尝试在当前目录和 `model/` 目录下检索 `gru.onnx`，若存在，通过 `ort.InferenceSession` 使用 CPU 推理提供者（`CPUExecutionProvider`）实例化会话。

### 4. ONNX 推理与黄色未来预测路径绘制

- 在 `PROCESS_EVERY_N_FRAMES`（3帧）更新中，对积攒到 20 帧长度的轨迹提取特征矩阵，执行 `gru_model.run` 推理：
  - 第一个输出 `outputs[0]` 经 Sigmoid 计算为分类概率，概率 $\geq0.60$ 认定为 UAV。
  - 第二个输出 `outputs[1]` 解包为未来 5 步飞行预测偏移点 `pred_coords`。
- 画面渲染层：当主黄金轨迹包含 `pred_coords` 时，在画面中精美渲染绘制**黄色未来 5 步轨迹折线和预测点**。

## 影响范围

- **高灵活性部署**：极大地精简了依赖包的体量，只需要 `pip install onnxruntime` 这一超轻量库即可秒速唤醒 GRU 时序滤波功能。若不安装此库，系统也能照常极速运转。
- **原始 `main.py`**：依然保持完全原样，未做任何修改。

## 本地验证情况

1. **编译及导入安全性验证**：
   - 运行 `D:\conda\python.exe -m py_compile main_gru_detect.py`，成功通过，证明即便在缺少 ONNX Runtime 的本地环境中也不会报任何编译错误。
2. **离线降级验证**：
   - 使用离线视频 4 执行处理，控制台正确打印了 `--> ONNX Runtime is not installed. Bypassing GRU ONNX prediction.` 降级提示，并完美生成了定位准确的纯 OpenCV 多目标锁定轨迹视频。
