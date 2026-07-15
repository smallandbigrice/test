@echo off
setlocal
cd /d "%~dp0\.."
if "%VIDEO_TARGET%"=="" set VIDEO_TARGET=192.168.0.200

powershell -ExecutionPolicy Bypass -File ".\scripts\switch_high_layer_profiles.ps1" ^
  -Mode day ^
  -BoardIps 192.168.0.1 192.168.0.2 ^
  -VideoTargetIp %VIDEO_TARGET% ^
  -VideoWidth 640 ^
  -VideoHeight 480 ^
  -VideoSendEvery 1 ^
  -VideoJpegQuality 25 %*

pause
