@echo off
chcp 65001 >nul
set REPO=E:\detect uav\scheme_ABC
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\switch_scheme_a_complex500.ps1" -BoardIps 192.168.0.1,192.168.0.2
pause
