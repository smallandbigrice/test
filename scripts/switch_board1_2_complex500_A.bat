@echo off
setlocal
cd /d "%~dp0\.."

powershell -ExecutionPolicy Bypass -File ".\scripts\switch_board1_2_complex500_A.ps1" %*

pause
