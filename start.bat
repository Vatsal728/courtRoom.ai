@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CourtRoom.ai - Setup, Check and Run

echo ============================================================
echo   CourtRoom.ai - Setup, Check and Run
echo ============================================================
echo.
echo   This script AUTO-INSTALLS all project dependencies.
echo   It never overwrites existing files: venv, node_modules,
echo   .env, chroma_db and the knowledge base are reused if present.
echo.

echo   --- EXTERNAL prerequisites (install these separately) ---
echo   1. Python 3.10-3.12     https://www.python.org/downloads/
echo   2. Node.js 18+ (LTS 22) https://nodejs.org/en/download
echo   3. MongoDB Community    https://www.mongodb.com/try/download/community
echo   4. Ollama               https://ollama.com/download
echo.
echo   --- Groq API key (for cloud generation) ---
echo   Sign up at https://console.groq.com (free tier), create an
echo   API key and set it in this project's .env file:
echo       GROQ_API_KEY=your_key_here
echo   Primary answer engine: llama-3.3-70b-versatile.
echo   If the key is missing or unreachable the app automatically
echo   falls back to local Ollama, so the system still works.
echo.
echo   --- Ollama models (runtime, NOT pip/npm dependencies) ---
echo   Run these once after installing Ollama:
echo       ollama pull nomic-embed-text
echo       ollama pull qwen2.5:3b
echo       ollama pull qwen2.5:3b-8k    (optional, long-context)
echo.
echo   --- Optional: BGE reranker (torch stack) ---
echo       pip install -r requirements-reranker.txt
echo   then set reranker_enabled: True in backend\config.py
echo ============================================================
echo.

goto menu

rem ==========================================================
rem Interactive menu
rem ==========================================================
:menu
echo.
echo  ============================================================
echo    What would you like to do?
echo  ============================================================
echo    0) Run everything, in order  1-2-3-4-5-6
echo    1) Backend environment      create venv + install deps
echo    2) Frontend environment     install npm deps
echo    3) Preflight checks         MongoDB :27017 / Ollama :11434
echo    4) Start backend            reuse if :8000 already running
echo    5) Backend health test      GET /health
echo    6) Launch frontend UI       Vite :3000 + open browser
echo    Q) Quit
echo.
set "CHOICE="
set /p "CHOICE=Enter your choice: "
if /i "%CHOICE%"=="Q" goto END
if /i "%CHOICE%"=="ALL" goto RUN_ALL
set "MATCH="
if "%CHOICE%"=="0" set "MATCH=1"
if "%CHOICE%"=="1" set "MATCH=1"
if "%CHOICE%"=="2" set "MATCH=1"
if "%CHOICE%"=="3" set "MATCH=1"
if "%CHOICE%"=="4" set "MATCH=1"
if "%CHOICE%"=="5" set "MATCH=1"
if "%CHOICE%"=="6" set "MATCH=1"
if not defined MATCH (
    echo   Unknown option: "%CHOICE%"
    goto menu
)
if "%CHOICE%"=="0" goto RUN_ALL
if "%CHOICE%"=="1" call :step_backend
if "%CHOICE%"=="2" call :step_frontend
if "%CHOICE%"=="3" call :step_preflight
if "%CHOICE%"=="4" call :step_run_backend
if "%CHOICE%"=="5" call :step_health
if "%CHOICE%"=="6" call :step_launch_frontend
goto menu

rem ==========================================================
rem Step 1 - Backend environment
rem ==========================================================
:step_backend
echo.
echo  [Step 1] Backend environment
if not exist "venv\Scripts\python.exe" (
    where python >nul 2>nul || (
        echo  [ERROR] Python not found. Install Python 3.10-3.12 first.
        exit /b 1
    )
    echo    Creating virtual environment...
    python -m venv venv || (
        echo  [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)
echo    Using: venv\Scripts\python.exe
venv\Scripts\python.exe -c "import fastapi,uvicorn,langchain,langchain_community,chromadb,pymongo,rank_bm25,sklearn,groq,dotenv,jose,passlib,structlog,slowapi,fitz,reportlab,joblib,numpy,pandas,httpx,requests,matplotlib" >nul 2>nul
if errorlevel 1 (
    echo    Installing backend dependencies from requirements.txt...
    venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Backend dependency install failed.
        exit /b 1
    )
) else (
    echo    Backend dependencies already satisfied - skipping install.
)
echo  [Step 1] Done.
exit /b 0

rem ==========================================================
rem Step 2 - Frontend environment
rem ==========================================================
:step_frontend
echo.
echo  [Step 2] Frontend environment
if not exist "frontend" (
    echo  [ERROR] Frontend folder not found: frontend
    exit /b 1
)
if exist "frontend\node_modules" (
    echo    node_modules already present - skipping npm install.
    exit /b 0
)
where node >nul 2>nul || (
    echo  [ERROR] Node.js not found. Install Node.js 18+ first.
    exit /b 1
)
echo    Installing frontend dependencies via npm install...
pushd "frontend"
call npm install
if errorlevel 1 (
    popd
    echo  [ERROR] npm install failed.
    exit /b 1
)
popd
echo  [Step 2] Done.
exit /b 0

rem ==========================================================
rem Step 3 - Preflight checks (warnings only)
rem ==========================================================
:step_preflight
echo.
echo  [Step 3] Preflight checks - warnings only
call :check_port 27017 MongoDB
call :check_port 11434 Ollama
echo  [Step 3] Done.
exit /b 0

rem ==========================================================
rem Step 4 - Start backend
rem ==========================================================
:step_run_backend
echo.
echo  [Step 4] Backend
call :check_port 8000 "API backend"
if errorlevel 1 (
    if not exist "logs" mkdir "logs"
    echo    Starting backend on http://127.0.0.1:8000 ...
    start "CourtRoom-AI-Backend" /min cmd /c "venv\Scripts\python.exe -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000 >> logs\api_server.log 2>&1"
) else (
    echo    Backend already running on :8000 - reusing it.
)
echo  [Step 4] Done.
exit /b 0

rem ==========================================================
rem Step 5 - Backend health test
rem ==========================================================
:step_health
echo.
echo  [Step 5] Backend health test  GET /health
call :check_port 8000 "API backend"
if errorlevel 1 (
    echo  [ERROR] Backend is not running. Run Step 4 first.
    exit /b 1
)
set /a TRIES=0
:WAIT_HEALTH
set /a TRIES+=1
if %TRIES% gtr 60 (
    echo  [ERROR] Backend did not become healthy after 60 seconds.
    echo  Check logs\api_server.log for errors.
    exit /b 1
)
venv\Scripts\python.exe -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read()" >nul 2>nul
if not errorlevel 1 goto HEALTH_OK
ping -n 2 127.0.0.1 >nul
goto WAIT_HEALTH

:HEALTH_OK
venv\Scripts\python.exe -c "import urllib.request,json;d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=10));print('    status  :',d.get('status'));print('    mongodb :',d.get('mongodb'));print('    ollama  :',d.get('ollama'));print('    rag     :',d.get('rag'))"
venv\Scripts\python.exe -c "import urllib.request,json;d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=10));exit(0 if d.get('status')=='ok' else 1)"
if errorlevel 1 (
    echo.
    echo    [WARN] Health is DEGRADED - see details above.
    echo    Fix MongoDB or Ollama and re-run, or check logs\api_server.log.
) else (
    echo    [PASS] Backend health check OK.
)
echo  [Step 5] Done.
exit /b 0

rem ==========================================================
rem Step 6 - Launch frontend UI
rem ==========================================================
:step_launch_frontend
echo.
echo  [Step 6] Frontend UI
call :check_port 3000 "Frontend Vite"
if errorlevel 1 (
    echo    Frontend already running on :3000 - reusing it.
    goto FE_OPEN
)
if not exist "frontend\node_modules" (
    echo  [ERROR] Frontend dependencies missing. Run Step 2 first.
    exit /b 1
)
echo    Starting Vite dev server on http://localhost:3000 ...
start "CourtRoom-AI-Frontend" /min cmd /c "cd /d frontend && npm run dev >> vite.log 2>&1"

set /a TRIES=0
:WAIT_FE
set /a TRIES+=1
if %TRIES% gtr 45 (
    echo  [WARN] Frontend did not come up on :3000 within 45 seconds.
    echo  Check frontend/vite.log for errors.
    exit /b 1
)
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName localhost -Port 3000 -InformationLevel Quiet) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 goto FE_OPEN
ping -n 2 127.0.0.1 >nul
goto WAIT_FE

:FE_OPEN
echo.
echo  ============================================================
echo    Backend  : http://127.0.0.1:8000    API docs: /docs
echo    Frontend : http://localhost:3000
echo    Logs     : logs\api_server.log  and  frontend\vite.log
echo    To stop  : close the two minimized windows.
echo  ============================================================
start "" "http://localhost:3000"
echo  [Step 6] Done.
exit /b 0

rem ==========================================================
rem Run everything in order
rem ==========================================================
:RUN_ALL
call :step_backend
if errorlevel 1 goto FINISH
call :step_frontend
if errorlevel 1 goto FINISH
call :step_preflight
call :step_run_backend
call :step_health
call :step_launch_frontend
goto FINISH

:FINISH
echo.
echo  ============================================================
echo    Done. Press any key to close.
echo  ============================================================
pause >nul
goto END

rem ==========================================================
rem Helper: check if a TCP port is listening
rem Usage: call :check_port PORT NAME
rem ==========================================================
:check_port
set "CP_PORT=%~1"
set "CP_NAME=%~2"
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName localhost -Port %CP_PORT% -InformationLevel Quiet) { exit 0 } else { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo    [WARN] %CP_NAME% not reachable on port %CP_PORT%.
    exit /b 1
)
echo    [OK]   %CP_NAME% reachable on port %CP_PORT%.
exit /b 0

:END
endlocal
exit /b 0
