@echo off
title ICT 9:30 Sniper — Test Tradovate Credentials
color 0B
echo.
echo  ======================================================
echo   ICT 9:30 SNIPER BOT — TRADOVATE CONNECTION TEST
echo  ======================================================
echo.

if not exist ".env" (
    echo  [!] No .env file found.
    echo  Please run 1_INSTALL_AND_SETUP.bat first!
    echo.
    pause
    exit /b
)

python scripts\test_connection.py
pause

