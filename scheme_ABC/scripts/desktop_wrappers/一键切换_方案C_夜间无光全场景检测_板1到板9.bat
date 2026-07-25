@echo off
chcp 65001 >nul
set REPO=E:\detect uav\scheme_ABC
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\switch_high_layer_profiles.ps1" -Mode night -BoardIps 192.168.0.1,192.168.0.2,192.168.0.3,192.168.0.4,192.168.0.5,192.168.0.6,192.168.0.7,192.168.0.8,192.168.0.9
pause
