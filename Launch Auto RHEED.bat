@echo off
title Auto RHEED
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

echo ==============================================================
echo    Auto RHEED is starting...
echo.
echo    A browser tab will open automatically in a few seconds.
echo    If it does not, open:  http://127.0.0.1:5000/
echo.
echo    KEEP THIS WINDOW OPEN while using the app.
echo    Close this window to stop the server.
echo ==============================================================
echo.

rem Open the default browser a few seconds after the server starts booting.
start "" cmd /c "timeout /t 4 /nobreak >nul & start "" http://127.0.0.1:5000/"

rem Launch from the already-synced uv environment without removing optional AI adapters.
uv run --frozen --no-sync python -m rheed_webapp.app

echo.
echo Auto RHEED has stopped. Press any key to close this window.
pause >nul
