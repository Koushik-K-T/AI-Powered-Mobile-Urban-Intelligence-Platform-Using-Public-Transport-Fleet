@echo off
echo =============================================
echo  Road Defect Detection System — Setup
echo =============================================
echo.
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python is in PATH.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating directories...
mkdir uploads 2>nul
mkdir models 2>nul
mkdir frontend 2>nul

echo.
echo [3/4] Downloading YOLO road damage model (~50MB)...
python download_model.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Model download failed. Will use fallback generic model.
)

echo.
echo [4/4] Verifying database...
python -c "import asyncio; import database; asyncio.run(database.init_db()); print('Database OK')"

echo.
echo =============================================
echo  Setup complete!
echo.
echo  To start the system:
echo    Terminal 1: uvicorn main:app --reload
echo    Terminal 2: python simulate_bus.py
echo    Browser:    http://localhost:8000
echo =============================================
pause
