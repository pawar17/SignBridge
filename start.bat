@echo off
title SignBridge Server
cd /d "%~dp0"

echo.
echo  ====================================
echo   SignBridge v2 - Starting Backend
echo  ====================================
echo.

REM Optional: load .env file
if exist .env (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
    )
    echo Loaded .env
)

echo  Starting API on http://localhost:8000
echo  Open frontend/index.html in your browser
echo.
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
