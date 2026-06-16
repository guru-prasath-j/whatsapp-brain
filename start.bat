@echo off
setlocal EnableDelayedExpansion
title WhatsApp Brain — RAG Server
color 0A

echo.
echo  ==========================================
echo   WhatsApp Brain - RAG Server Startup
echo  ==========================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo  [OK] Python found

:: ── Check pip ─────────────────────────────────────────────────────────────────
pip --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] pip not found. Re-install Python with pip included.
    pause
    exit /b 1
)

:: ── Check .env ────────────────────────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo  [SETUP] No .env file found. Creating from template...
    (
        echo # OpenAI API key — set this to use ChatGPT instead of Ollama
        echo OPENAI_API_KEY=
        echo OPENAI_MODEL=gpt-4o-mini
        echo.
        echo # Ollama settings — used as fallback when OPENAI_API_KEY is not set
        echo OLLAMA_BASE_URL=http://localhost:11434
        echo OLLAMA_MODEL=llama3.2:latest
        echo OLLAMA_EMBED_MODEL=nomic-embed-text
        echo.
        echo DOCS_DIR=docs
        echo VECTOR_STORE_PATH=vector_store
    ) > .env
    echo  [OK] .env created with defaults. Edit it if needed.
)

:: ── Install dependencies ──────────────────────────────────────────────────────
echo.
echo  [SETUP] Installing Python dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo  [OK] Dependencies ready

:: ── Detect LLM provider ──────────────────────────────────────────────────────
set USE_OPENAI=0
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /I "%%A"=="OPENAI_API_KEY" (
            if not "%%B"=="" if not "%%B"=="sk-..." (
                set USE_OPENAI=1
            )
        )
    )
)

if "!USE_OPENAI!"=="1" (
    echo.
    echo  [OK] OPENAI_API_KEY found — using ChatGPT ^(Ollama not required^)
    goto :skip_ollama
)

:: ── Check Ollama ──────────────────────────────────────────────────────────────
echo.
echo  [CHECK] Checking Ollama... ^(no OPENAI_API_KEY set^)
curl -s http://localhost:11434/ >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Ollama is not running. Starting it...
    start /B ollama serve
    echo  [WAIT] Waiting for Ollama to be ready...
    set OLLAMA_READY=0
    for /L %%i in (1,1,20) do (
        if "!OLLAMA_READY!"=="0" (
            timeout /t 2 /nobreak >nul
            curl -s http://localhost:11434/ >nul 2>&1
            if not errorlevel 1 set OLLAMA_READY=1
        )
    )
    if "!OLLAMA_READY!"=="0" (
        echo  [WARNING] Could not connect to Ollama after 40 seconds.
        echo  Tip: set OPENAI_API_KEY in .env to use ChatGPT instead of Ollama.
        echo  Or start Ollama manually: https://ollama.com
        pause
        exit /b 1
    )
)
echo  [OK] Ollama is running

:: ── Check required models ─────────────────────────────────────────────────────
echo.
echo  [CHECK] Checking for required Ollama models...

ollama list | findstr "nomic-embed-text" >nul 2>&1
if errorlevel 1 (
    echo  [PULL] Downloading nomic-embed-text ~274MB...
    ollama pull nomic-embed-text
)
echo  [OK] nomic-embed-text ready

ollama list | findstr "llama3.2" >nul 2>&1
if errorlevel 1 (
    ollama list | findstr "llama3.1" >nul 2>&1
    if errorlevel 1 (
        echo  [PULL] Downloading llama3.2 ~2GB... (this may take a while)
        ollama pull llama3.2
    )
)
echo  [OK] LLM model ready

:skip_ollama

:: ── Create docs folder if missing ────────────────────────────────────────────
if not exist "docs" (
    mkdir docs
    echo  [OK] Created docs/ folder. Add your business documents here.
)

:: ── Start the server ──────────────────────────────────────────────────────────
echo.
echo  ==========================================
echo   Starting RAG server on port 8000...
echo   Dashboard: http://localhost:8000/docs
echo   Press Ctrl+C to stop
echo  ==========================================
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
