@echo off
setlocal

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 or 3.12 and select "Add Python to PATH".
  exit /b 1
)

if not exist .venv (
  py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_install.py

echo.
echo Setup complete. Run run_app.bat to launch FraudNet ML.
endlocal
