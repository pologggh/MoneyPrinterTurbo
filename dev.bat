@echo off
setlocal
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"
cd /d "%CURRENT_DIR%"
echo ***** MoneyPrinterTurbo Unified Development Environment *****
echo ***** Project directory: %CURRENT_DIR% *****
set "PYTHONPATH=%CURRENT_DIR%"

set "PYTHON_CMD="
if exist "%CURRENT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD="%CURRENT_DIR%\.venv\Scripts\python.exe""
) else if exist "%CURRENT_DIR%\lib\python\python.exe" (
    set "PYTHON_CMD="%CURRENT_DIR%\lib\python\python.exe""
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    ) else (
        where uv >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_CMD=uv run python"
        )
    )
)

if not defined PYTHON_CMD (
    echo ***** Neither project Python, uv, nor system Python was found. Please install dependencies first. *****
    pause
    exit /b 1
)

%PYTHON_CMD% dev.py %*
