@echo off
color 0A

echo.
echo          ______ _     ______       _
echo    10100 | ___ \ |    | ___ \     | |   00101
echo   110011 | |_/ / |    | |_/ / ___ | |_  110011
echo 00110110 |    /| |    | ___ \/ _ \| __| 01101100
echo   010010 | |\ \| |____| |_/ / (_) | |_  010010
echo    10010 \_| \_\_____\/____/ \___/ \__| 01001
echo                   GUI LAUNCHER
echo.
echo [92m==========================================[0m
echo [92m    FightIQ MMA Prediction System[0m
echo [92m==========================================[0m
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [91mPython is not installed! Please install Python 3.10 or higher.[0m
    pause
    exit /b
)

REM Create and activate virtual environment
if not exist .venv (
    echo [93mCreating virtual environment...[0m
    python -m venv .venv
)

echo [93mActivating virtual environment...[0m
call .venv\Scripts\activate.bat

REM Install requirements
echo [93mInstalling requirements...[0m
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Run data pipeline
echo [92mRunning data pipeline...[0m
python scripts\scrape_rankings.py
python scripts\scrape_fighter_stats.py
python scripts\scrape_events.py
python scripts\build_dataset.py

REM Train model
echo [92mTraining prediction model...[0m
python train.py

REM Launch dashboard
echo [96mLaunching FightIQ Dashboard...[0m
streamlit run dashboard_app.py

pause
