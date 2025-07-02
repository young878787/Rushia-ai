@echo off
REM 啟動 SakuraLLM API 伺服器 (llama.cpp 版本)
cd /d d:\RushiaMode\scripts\llama.cpp
start cmd /k "llama-server.exe -m d:\RushiaMode\models\sakura-14b-qwen2.5-v1.0-iq4xs.gguf --host 127.0.0.1 --port 8000 --gpu-layers 40 --threads 8"

echo SakuraLLM 伺服器已啟動於 http://127.0.0.1:8000
pause
