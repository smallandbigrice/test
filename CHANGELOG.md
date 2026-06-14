# Changelog

## frame-diff-20260614

- Switched `main.py` back to the frame-difference ROI pipeline.
- Removed the sky-gray gray-threshold candidate path from `main.py`.
- Tracker confirmation now uses YOLO detections plus trajectory rules only.
- Kept static background masking in the frame-difference mask stage.
- Kept 1920x1080 frame-difference analysis resolution and full-frame display.
