@echo off
:: ============================================================
::  start_dev.bat  -  Run every session
::  Activates venv, pulls latest code, opens a ready shell
::  ME4170 Raspbot V2 - Color Tracking (Local Desktop Dev)
:: ============================================================

:: --- CONFIG: set this to your project folder path ---
:: Example: set PROJECT_DIR=C:\Users\YourName\Documents\ME4170_project
set PROJECT_DIR=%~dp0
:: (%~dp0 means "same folder as this script" - convenient default)

echo Navigating to project directory...
cd /d "%PROJECT_DIR%"

echo.
echo Checking for virtual environment...
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run setup_env.bat first!
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Pulling latest code from Git...
git pull
IF ERRORLEVEL 1 (
    echo WARNING: Git pull failed. Check your repo URL or network connection.
    echo          Continuing with local files...
)

echo.
echo ============================================================
echo  Environment ready! Python is:
python --version
echo  Active packages:
pip list --format=columns
echo ============================================================
echo.
echo Type 'python your_script.py' to run a script.
echo Type 'deactivate' to exit the virtual environment.
echo.

:: Keep the terminal open and in the activated environment
cmd /k
