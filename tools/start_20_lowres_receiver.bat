@echo off
setlocal
cd /d "%~dp0\.."

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set OUT_DIR=E:\detect uav\record_2k_pure\board_video_receiver\lowres20_%STAMP%

python tools\board_udp_video_receiver.py ^
  --host 0.0.0.0 ^
  --port 9999 ^
  --tile-w 320 ^
  --tile-h 180 ^
  --cols 5 ^
  --max-streams 20 ^
  --display-fps 15 ^
  --wait-ms 1 ^
  --rcvbuf-mb 4 ^
  --decode-budget-ms 30 ^
  --chunk-ttl-sec 0.5 ^
  --max-packets-per-loop 2000 ^
  --fps 8 ^
  --codec MJPG ^
  --ext avi ^
  --out-dir "%OUT_DIR%"

pause
