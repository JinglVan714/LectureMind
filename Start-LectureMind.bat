@echo off
setlocal
cd /d "%~dp0"

set "CONDA_ENV_NAME=myagent"
set "PYTHON_EXE="

echo [LectureMind] Starting from %CD%

if exist "%CD%\.venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if exist "D:\anaconda\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=D:\anaconda\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if exist "D:\Anaconda\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=D:\Anaconda\envs\%CONDA_ENV_NAME%\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\anaconda3\envs\%CONDA_ENV_NAME%\python.exe" set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\%CONDA_ENV_NAME%\python.exe"

if defined PYTHON_EXE (
    echo [LectureMind] Using Python: %PYTHON_EXE%
) else (
    echo [LectureMind] Could not find conda env '%CONDA_ENV_NAME%' directly. Falling back to PATH python.
    set "PYTHON_EXE=python"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev_up.ps1" -NoConda -PythonExe "%PYTHON_EXE%"
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% NEQ 0 (
    echo [LectureMind] Startup failed with exit code %EXIT_CODE%.
) else (
    echo [LectureMind] Server stopped.
)
pause
exit /b %EXIT_CODE%
