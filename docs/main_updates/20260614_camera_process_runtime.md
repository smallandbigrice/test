# 20260614 摄像头进程化运行结构

## 修改目标

将 `main.py` 的 5 路摄像头采集与帧差预处理从线程模式调整为“每路摄像头一个独立进程”，降低多路图像预处理在 Python 调度层互相影响的概率。

RKNN/YOLO 推理仍然保持集中式，只启动一个统一推理 worker，避免多个进程同时抢占 NPU。

## 本次改动

- 新增 `USE_CAMERA_PROCESSES = True`。
- 新增 `multiprocessing` 运行初始化函数 `init_runtime_ipc()`。
- 将 `inf_queues`、`res_queues`、`display_queues` 在进程模式下替换为 `multiprocessing.Queue`。
- 将 `stop_event` 在进程模式下替换为 `multiprocessing.Event`。
- 新增 `video_allowed_event`，用于父进程通知各摄像头进程开始视频发送。
- `capture_job()` 内部为每个摄像头进程重新创建 `DataSender` 和 `VideoSender`，避免直接继承父进程 socket。
- 主入口中根据平台能力启动：
  - Linux/RK3588 支持 `fork` 时：5 路摄像头使用独立进程。
  - 不支持 `fork` 时：自动回退到原线程模式。
- OpenCV 显示窗口挪到摄像头 worker 启动之后创建，减少子进程继承 GUI 状态的风险。

## 算法逻辑影响

本次不修改核心检测算法，只调整运行结构。

- 帧差逻辑：未修改。
- 静态背景蒙版：未修改。
- LCM/噪声过滤：未修改。
- ROI 生成与融合：未修改。
- YOLO/RKNN 推理：仍然集中推理，未改模型和阈值。
- 轨迹判断：仍在每路摄像头 worker 内部独立维护，未改确认条件。
- 绿框确认逻辑：未修改。

## 当前运行结构

```text
Cam 0 process -> frame diff / ROI / tracking -> inference queue
Cam 1 process -> frame diff / ROI / tracking -> inference queue
Cam 2 process -> frame diff / ROI / tracking -> inference queue
Cam 3 process -> frame diff / ROI / tracking -> inference queue
Cam 4 process -> frame diff / ROI / tracking -> inference queue

central RKNN thread -> YOLO inference -> result queues

main process -> display windows / lifecycle control
```

## 参数变化

- `ALGORITHM_VERSION` 更新为 `frame-diff-multiproc-20260614`。
- 新增 `USE_CAMERA_PROCESSES = True`。
- 其余检测阈值未调整。

## 验证情况

- 已执行 Python 语法检查：

```text
python -m py_compile main.py
```

- 当前电脑环境没有 RK3588 板端摄像头，因此未做 5 路实机推理验证。

## 已知风险

- 进程间传递 ROI 图像仍然存在内存拷贝开销，但比传整帧低很多。
- 如果实际瓶颈在 RKNN 推理或窗口显示，进程化摄像头预处理不会明显提升速度。
- 如果某些环境不支持 `fork`，代码会自动回退线程模式。
- 板端首次运行需要观察：
  - 5 路摄像头是否都能正常打开。
  - RKNN 推理队列是否堆积。
  - 显示窗口是否卡顿。
  - CPU 使用率是否比线程版更均衡。
