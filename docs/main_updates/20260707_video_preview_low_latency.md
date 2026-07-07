# 20260707 视频回传低延迟调整

## 问题

现场接收端预览出现约 1 到 2 秒延迟。主要原因不是检测模型，而是视频回传链路容易积压：

- 板端发送 20 路 `640x360 quality=50` 预览，码率偏高。
- 本地接收端 UDP 缓冲设置为 64MB，旧包会在系统缓冲里排队。
- 接收端每轮最多处理 40 到 80 个包，20 路画面时不能及时排空。
- OpenCV 窗口刷新较慢时，显示的是已经排队的旧帧。

## 调整

### 板端发送端

- 默认预览尺寸从 `640x360` 改为 `320x180`。
- 默认 JPEG 质量从 `50` 改为 `35`。
- 默认发送间隔：
  - 白天：每 3 帧发送一次。
  - 夜间：每 2 帧发送一次。
- `tools/board_video_receiver_systemd.conf` 同步为 `320x180 quality=35`。

### 本地接收端

`tools/board_udp_video_receiver.py` 改为低延迟优先：

- UDP 接收缓冲默认从 64MB 降为 4MB。
- 默认显示帧率提高到 15 FPS。
- 每轮最多处理 2000 个包，并增加 30ms 解码时间预算。
- 分片帧只保留同一路最新 frame id，旧分片直接丢弃。
- 未完成分片帧 0.5 秒后丢弃。
- `cv2.waitKey` 默认 1ms，减少窗口刷新阻塞。

### 本地启动脚本

`tools/start_20_lowres_receiver.bat` 已同步上述低延迟参数。

## 使用

本地接收端直接运行：

```bat
tools\start_20_lowres_receiver.bat
```

如果板端已有旧 drop-in，需要把 `tools/board_video_receiver_systemd.conf` 同步到板端
`/etc/systemd/system/python_autostar.service.d/30-video-lowres20.conf` 后重启服务。

## 验证

- `python -m py_compile main.py comms.py tools\board_udp_video_receiver.py` 通过。
