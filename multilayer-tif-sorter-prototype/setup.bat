@echo off
cd /d "%~dp0"
if not exist "runtime\python313\python.exe" (
  echo 프로젝트 Python runtime이 없습니다.
  pause
  exit /b 1
)
if not exist ".venv-app\Scripts\python.exe" runtime\python313\python.exe -m venv .venv-app
.venv-app\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 pause
