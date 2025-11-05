# FightIQ MMA Prediction System Launcher
# PowerShell script to set up environment and launch the dashboard

Write-Host ""
Write-Host "     ______ _       _     _   _____  ____  " -ForegroundColor Green
Write-Host "    |  ____(_)     | |   | | |_   _|/ __ \ " -ForegroundColor Green
Write-Host "    | |__   _  __ _| |__ | |_  | | | |  | |" -ForegroundColor Green
Write-Host "    |  __| | |/ _  | '_ \| __| | | | |  | |" -ForegroundColor Green
Write-Host "    | |    | | (_| | | | | |_ _| |_| |__| |" -ForegroundColor Green
Write-Host "    |_|    |_|\__, |_| |_|\__|_____|\___\_\" -ForegroundColor Green
Write-Host "              __/ |                        " -ForegroundColor Green
Write-Host "             |___/                         " -ForegroundColor Green
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "    FightIQ MMA Prediction System" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed! Please install Python 3.10 or higher." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

# Run data pipeline
Write-Host ""
Write-Host "========== DATA PIPELINE ==========" -ForegroundColor Cyan
Write-Host "Scraping UFC rankings..." -ForegroundColor Yellow
python scripts\scrape_rankings.py

Write-Host "Scraping fighter stats..." -ForegroundColor Yellow
python scripts\scrape_fighter_stats.py

Write-Host "Scraping event results..." -ForegroundColor Yellow
python scripts\scrape_events.py

Write-Host "Building dataset..." -ForegroundColor Yellow
python scripts\build_dataset.py

# Update ELO ratings
Write-Host "Updating ELO ratings..." -ForegroundColor Yellow
python elo_pipeline.py

# Train model
Write-Host ""
Write-Host "========== MODEL TRAINING ==========" -ForegroundColor Cyan
Write-Host "Training prediction model..." -ForegroundColor Yellow
python train.py

# Launch dashboard
Write-Host ""
Write-Host "========== LAUNCHING DASHBOARD ==========" -ForegroundColor Cyan
Write-Host "Starting FightIQ Dashboard..." -ForegroundColor Magenta
streamlit run dashboard_app.py

Write-Host ""
Write-Host "Dashboard closed. Thank you for using FightIQ!" -ForegroundColor Green
Read-Host "Press Enter to exit"
