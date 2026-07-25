@echo off
chcp 65001 >nul
setlocal
set REPO=E:\detect uav\scheme_ABC
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set OUT_DIR=E:\detect uav\record_2k_pure\board_video_receiver\vga_%STAMP%

python "%REPO%\code\board_udp_video_receiver.py" ^
  --host 0.0.0.0 ^
  --port 9999 ^
  --tile-w 320 ^
  --tile-h 240 ^
  --cols 5 ^
  --max-streams 30 ^
  --display-fps 10 ^
  --wait-ms 1 ^
  --rcvbuf-mb 8 ^
  --decode-budget-ms 20 ^
  --chunk-ttl-sec 0.25 ^
  --max-packets-per-loop 1600 ^
  --fps 10 ^
  --codec MJPG ^
  --ext avi ^
  --out-dir "%OUT_DIR%"

pause
