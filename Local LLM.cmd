@echo off
cd /d "%~dp0"
set "LOCAL_LLM_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%LOCAL_LLM_PYTHON%" set "LOCAL_LLM_PYTHON=%~dp0localmodel-env\pythonw.exe"
if not exist "%LOCAL_LLM_PYTHON%" (
  echo Local Model is not installed. Run "Install Local LLM.cmd" first.
  pause
  exit /b 1
)
start "" "%LOCAL_LLM_PYTHON%" "%~dp0launcher.py"
