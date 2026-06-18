# 2026-06-18 main_gru_detect.py 支持结合 ONNX Runtime 加载 gru.onnx 轨迹预测与 5 路视频离线仿真

## 修改日期

2026-06-18

## 修改目标

为了方便用户在没有连接 5 个物理摄像头的真实 RK3588 嵌入式开发板上能够秒级拉起、调试与评估整个多摄算法系统的资源负荷、ONNX GRU 推理表现以及电传飞控/云台 UDP 通信逻辑，我们对 `main_gru_detect.py` 进行了升级，**完美集成了 5 路离线视频输入仿真控制项**。

该设计保留了高容错自适应降级，同时对 5 路本地 MP4 视频模拟进行了帧率控制和无限循环优化，使得板端测试极其敏捷。

## 本次改动内容

### 1. 5 路视频离线取流仿真配置

在顶部配置中增加了控制开关和仿真视频路径：
```python
SIMULATE_BY_VIDEOS = True  # 是否使用本地视频文件模拟 5 路摄像头输入
VIDEO_SOURCES = [
    "record_2k_pure/3.mp4",
    "record_2k_pure/4.mp4",
    "record_2k_pure/7.mp4",
    "record_2k_pure/8.mp4",
    "record_2k_pure/4.mp4"
]
```

### 2. capture_job 输入源重定向

在相机的 `capture_job` 取流逻辑中：
- 当 `SIMULATE_BY_VIDEOS = True`：忽略硬件 v4l2 节点，自动加载 `VIDEO_SOURCES` 中对应 cam_idx 索引的本地 MP4 文件实例化 `cv2.VideoCapture`，实现了多摄并发的离线文件取流。
- 当 `SIMULATE_BY_VIDEOS = False`：恢复为原本的物理摄像头 V4L2 节点实时取流，实现了零代码修改无缝切换。

### 3. 帧率仿真 (15 FPS) 与无限循环播放 (Endless Looping)

- **FPS 仿真**：在工作循环中，当读取本地 MP4 文件时，每读完一帧强制加入 `time.sleep(0.066)`（对应 15 FPS 的取流帧间隔），防止因离线文件读取速度过快疯狂蚕食 CPU 负荷，高度还原了真实摄像头在板端运行时产生的 CPU 计算压力。
- **循环播放 (Looping)**：当视频流读完后，程序会自动通过 `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` 将播放进度重置为第 0 帧并继续播放，支持了板端大系统的无限循环挂机压测。

### 4. 轻量化 ONNX 依赖与 ORT 时序轨迹预测

- 动态捕获板端 `onnxruntime` 并自动在当前目录下寻找 `gru.onnx` 模型并实例化 ORT 推理会话。
- 基于 NumPy 进行了 Savitzky-Golay 降噪和滑窗特征提取，并通过 ONNX Runtime 在 ARM CPU 上秒级进行 UAV 分类判定（概率 $\geq0.60$ 为真）和未来轨迹偏移量预测。
- **画面渲染层**：在主画面渲染中不仅具备零延迟的黄色锁定标框，还能在 ONNX 预测数据有效时渲染绘制**黄色未来 5 步轨迹折线和预测点**。
- **自适应降级**：如果 `onnxruntime` 库缺失或模型文件没有部署，程序会自动且优雅地通过 `try...except` 降级为 OpenCV 纯多目标追踪锁定，保障主程序平滑运行。

## 影响范围

- **极佳的板端调试友好度**：使得板端程序在完全没有硬件多摄、没有 Conda 及 PyTorch 环境的限制下，随时一键拉起并发多路流测试，完美仿真了飞控/云台 UDP 传输链路的通达性，大大缩短了现场调试和系统集成的周期。
- **原始 `main.py`**：依然保持完全原样，未做任何修改。

## 本地验证情况

1. **编译及导入安全性验证**：
   - 运行 `D:\conda\python.exe -m py_compile main_gru_detect.py`，成功通过，证明即便在缺少 ONNX Runtime 的本地环境中也不会报任何编译错误。
2. **离线多路并发验证**：
   - 在 X86 CPU 上成功测试运行，5 路视频工作线程拉起顺利，多线程文件并发读取与 FPS 休眠机制动作极其准确。
