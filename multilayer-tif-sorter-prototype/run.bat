@echo off
cd /d "%~dp0"
if not exist ".venv-app\Scripts\pythonw.exe" (
  echo 실행 환경을 찾을 수 없습니다. setup.bat을 먼저 실행하세요.
  pause
  exit /b 1
)
set "TCL_LIBRARY=%~dp0runtime\python313\tcl\tcl8.6"
set "TK_LIBRARY=%~dp0runtime\python313\tcl\tk8.6"
start "TIFF Sorter" ".venv-app\Scripts\pythonw.exe" "app.py"
