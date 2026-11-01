@echo off
title ICT 9:30 VWAP Sniper Bot — RUNNING
color 0A
echo.
echo  ============================================
echo   ICT 9:30 VWAP SNIPER BOT — STARTING
echo  ============================================
echo.

if not exist ".env" (
    echo  ERROR: Bot not configured yet.
    echo  Please run 1_INSTALL_AND_SETUP.bat first.
    pause
    exit /b
)

echo  Bot is starting... check the logs folder for activity.
echo  DO NOT close this window while trading.
echo  To stop the bot, press CTRL+C
echo.
python scripts\bot_tradovate.py
pause

