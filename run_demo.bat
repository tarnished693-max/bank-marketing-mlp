@echo off
echo ========================================================
echo   Bank Marketing MLP - Experimentation Engineer Demo
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or 3.11 from python.org
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [3/3] Running Reference Experiment...
python src\experiment_runner.py --config configs\base_reference.json --run-id EXP-REF-001

echo.
echo ========================================================
echo   Experiment Finished! 
echo   Results saved to: experiments\experiments.csv
echo   Curves saved to:  experiments\curves\
echo ========================================================
pause
