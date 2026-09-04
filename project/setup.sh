#!/bin/bash
echo "============================================="
echo " Road Defect Detection System — Setup"
echo "============================================="

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt || { echo "ERROR: pip install failed"; exit 1; }

echo "[2/4] Creating directories..."
mkdir -p uploads models frontend

echo "[3/4] Downloading YOLO road damage model (~50MB)..."
python download_model.py || echo "WARNING: Model download failed — will use fallback"

echo "[4/4] Verifying database..."
python -c "import asyncio; import database; asyncio.run(database.init_db()); print('Database OK')"

echo ""
echo "============================================="
echo " Setup complete!"
echo ""
echo " To start the system:"
echo "   Terminal 1: uvicorn main:app --reload"
echo "   Terminal 2: python simulate_bus.py"
echo "   Browser:    http://localhost:8000"
echo "============================================="
