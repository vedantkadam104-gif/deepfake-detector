@echo off
cd /d "%~dp0"
echo Starting DeepShield AI Web Studio on http://127.0.0.1:8000 ...
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
.\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
pause
