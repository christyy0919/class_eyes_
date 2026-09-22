@echo off
title Classroom Expression Analysis UI
cd /d "%~dp0"
echo Starting server... keep this window open.
echo Log file: %~dp0ui\server.log
python -u ui\app.py > "%~dp0ui\server.log" 2>&1
pause
