# 2026-06-19 main.py 低层关闭灰度噪声抑制以恢复弱小目标送检

## 修改目标

为了恢复低层场景下（如视频 5）在 9-12s 左右位于右下角的弱小无人机目标的识别率，我们将 `main.py` 和 `single_video_main_pt.py` 的低层（`low`）配置调整为自动关闭灰度噪声抑制。

这对应了本地视频推理效果最好的一版配置（`5_low_relaxed_lcm_grayoff_include_small_infer`）。

## 本次修改

### 1. 灰度噪声抑制自适应关闭

- **`main.py`**：将 `ENABLE_GRAY_NOISE_SUPPRESSOR` 更改为自适应设置：
  ```python
  ENABLE_GRAY_NOISE_SUPPRESSOR = True if HIGH_LAYER_MODE else False
  ```
  从而在低层（`low`）模式下默认将其关闭（`False`）。
  
- **`single_video_main_pt.py`**：在 `if args.layer_mode == "low":` 配置块中自动将 `ENABLE_GRAY_NOISE_SUPPRESSOR` 覆写为 `False`：
  ```python
  ENABLE_GRAY_NOISE_SUPPRESSOR = False
  ```

### 2. 作用与效果

- 避免高楼/复杂背景附近的弱小目标因图像灰度局部质地规则或亮点检测被误判为灰度噪声，从而在第一阶段被提前过滤掉。
- 经过前 20s (300帧) 视频推理验证，低层有效目标检测帧数从原来的 **151 帧** 提升到了 **172 帧**，成功在 9-12s 区域恢复了对右下角弱小目标的连续锁定，且主程序与单视频推理脚本输出仍保持 100% 精确对齐。

## 验证依据

- 本地 `single_video_main_pt.py` 运行通过，生成 CSV 和视频正常。
- 本地 `main_video_pt_test.py` 运行通过，结果与单视频推理完全匹配。
