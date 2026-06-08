@echo off
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
        echo OLLAMA_BASE_URL=http://localhost:11434
        echo OLLAMA_MODEL=llama3.2:latest
        echo OLLAMA_EMBED_MODEL=nomic-embed-text
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

:: ── Check Ollama ──────────────────────────────────────────────────────────────
echo.
echo  [CHECK] Checking Ollama...
curl -s http://localhost:11434/ >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Ollama is not running. Starting it...
    start /B ollama serve
    timeout /t 4 /nobreak >nul
    curl -s http://localhost:11434/ >nul 2>&1
    if errorlevel 1 (
        echo  [WARNING] Could not start Ollama automatically.
        echo  Please start Ollama manually and re-run this script.
        echo  Download from: https://ollama.com
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
