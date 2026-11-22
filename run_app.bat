@echo off
cd /d "%~dp0"
echo ============================================================
echo   Starting DeepShield AI Web Studio on http://127.0.0.1:8005
echo   Please keep this window open!
echo   Loading PyTorch + RTX 3050 model (takes ~15-60s)...
echo   Your browser will open automatically as soon as it is ready.
echo ============================================================
start "" powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 180; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8005/api/status' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8005'; break } } catch {}; Start-Sleep -Seconds 1 }"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8005
pause
