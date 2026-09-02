#!/bin/bash
set -e

# Default to port 7860 for Hugging Face Spaces Docker
export PORT="${PORT:-7860}"

echo "=================================================="
echo "Starting Enterprise HR AI Platform Container"
echo "Target Frontend Port: ${PORT}"
echo "=================================================="

# 1. Start FastAPI backend on internal loopback 127.0.0.1:8000 in background
echo "Starting FastAPI backend server on 127.0.0.1:8000..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# 2. Wait for FastAPI backend to become ready
echo "Waiting for FastAPI backend to initialize..."
python -c "
import time, urllib.request
ready = False
for i in range(40):
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:8000/', timeout=2)
        if resp.status == 200:
            ready = True
            break
    except Exception:
        time.sleep(1)
if ready:
    print('FastAPI backend is online and ready!')
else:
    print('Warning: FastAPI backend did not respond in 40s')
"

# 3. Start Streamlit frontend in foreground on 0.0.0.0:${PORT:-7860}
echo "Starting Streamlit UI on 0.0.0.0:${PORT}..."
exec python -m streamlit run frontend/dashboard.py \
    --server.port "${PORT}" \
    --server.address "0.0.0.0" \
    --server.headless true
