# PowerShell script to set up virtual environment

Write-Host "Setting up Python Virtual Environment..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error creating virtual environment!" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully!" -ForegroundColor Green

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error installing dependencies!" -ForegroundColor Red
    exit 1
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan

Write-Host "`nTo activate the virtual environment, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White

Write-Host "`nTo generate example datasets, run:" -ForegroundColor Yellow
Write-Host "  python example_dataset.py" -ForegroundColor White

Write-Host "`nTo test the GA implementation, run:" -ForegroundColor Yellow
Write-Host "  python test_ga.py" -ForegroundColor White

Write-Host "`nTo start the web application, run:" -ForegroundColor Yellow
Write-Host "  python run.py" -ForegroundColor White

Write-Host "`nThen open your browser and navigate to:" -ForegroundColor Yellow
Write-Host "  http://localhost:5000" -ForegroundColor White
Write-Host ""
