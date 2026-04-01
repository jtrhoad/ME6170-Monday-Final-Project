@echo off
:: ============================================================
::  setup_env.bat  -  Run ONCE to bootstrap your dev environment
::  ME4170 Raspbot V2 - Color Tracking (Local Desktop Dev)
:: ============================================================

echo [1/4] Checking Python version...
python --version
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Install from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating virtual environment in .\venv ...
python -m venv venv
IF ERRORLEVEL 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo [3/4] Activating venv and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [4/4] Installing project dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ============================================================
echo  Setup complete! Run start_dev.bat for future sessions.
echo ============================================================
pause
