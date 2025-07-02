@echo off
REM 一鍵啟動 SakuraLLM 伺服器與 WebUI (單一終端)
cd /d d:\RushiaMode\scripts\llama.cpp

echo [1/3] 啟動 SakuraLLM API 伺服器...
start "" llama-server.exe -m d:\RushiaMode\models\sakura-14b-qwen2.5-v1.0-iq4xs.gguf --host 127.0.0.1 --port 8000 --gpu-layers 60 --threads 12

cd /d d:\RushiaMode\scripts

echo [2/3] 檢查 Python venv 環境...
if not exist venv (
    echo 尚未建立 venv，正在建立...
    python -m venv venv
)

call venv\Scripts\activate.bat

pip install --upgrade pip
pip install flask flask_cors requests

echo [3/3] 啟動 Sakura WebUI (虛擬環境)...
start /b python sakura_webui.py

REM 自動開啟瀏覽器
start http://127.0.0.1:5000

echo 已啟動 SakuraLLM API 與 WebUI，請於本終端視窗觀察 log。
pause
