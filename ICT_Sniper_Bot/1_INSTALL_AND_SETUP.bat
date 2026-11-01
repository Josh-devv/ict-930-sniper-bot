@echo off
title ICT 9:30 VWAP Sniper Bot — Setup
color 0A
echo.
echo  ============================================
echo   ICT 9:30 VWAP SNIPER BOT — FIRST TIME SETUP
echo  ============================================
echo.
echo  Installing required components...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)

:: Install dependencies
pip install aiohttp websockets python-dotenv --quiet
echo  Dependencies installed OK.
echo.

:: Run setup wizard
echo  Starting configuration wizard...
echo.
python SETUP_WIZARD.py

echo.
echo  Setup complete! You can now double-click START_BOT.bat to run the bot.
pause

