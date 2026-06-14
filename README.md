# UAV Detection Runtime

Current runtime version: `frame-diff-20260614`

## Main Pipeline

`main.py` runs the board-side detection pipeline:

1. Capture 5 camera streams.
2. Resize grayscale frames to `1920x1080` for frame-difference analysis.
3. Build a static background model during startup and use it to suppress fixed background pixels.
4. Generate motion ROIs from frame difference, morphology, area/shape filters, and optional LCM filtering.
5. Crop/fuse `640x640` ROIs and send them to RKNN YOLO.
6. Confirm targets with YOLO hit history and trajectory rules.
7. Display/send the full-frame preview with ROI/raw/confirmed boxes.

The sky-gray gray-threshold candidate path has been removed from `main.py` in this version.
