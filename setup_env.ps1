# PowerShell script to set up the Python 3.11 environment for the Bank-Marketing-MLP project
# ------------------------------------------------------------
# Prerequisites:
#   • Windows PowerShell (5.1 or later)
#   • Python 3.11 installed and accessible via the py launcher (run `py install 3.11` if needed)
#   • Internet connectivity for pip to download packages
#   • ExecutionPolicy that allows script execution (see note below)
# ------------------------------------------------------------

# 1. Ensure the script can run (temporarily bypass execution policy for this session)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 2. Change to the project root folder
Set-Location "D:\Downloads\NN porject3.0\bank-marketing-mlp-starter"

# 3. Create a fresh virtual environment using Python 3.11
py -3.11 -m venv .venv

# 4. Activate the virtual environment (PowerShell activation script)
& .\.venv\Scripts\Activate.ps1

# 5. Upgrade pip to the latest version (required for many‑linux wheels)
python -m pip install --upgrade pip

# 6. Install the project dependencies (this will download from PyPI)
python -m pip install -r requirements.txt

# 7. Run the test suite to verify everything works
python -m pytest -v

# 8. Run the baseline pipeline (produces model artefacts and CSV)
python -m src.baselines

# 9. Execute the exploratory notebook so that charts and tables are saved in the notebook file
#    Requires Jupyter; install it if you don’t have it already (already in requirements.txt)
python -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb

# 10. Commit the executed notebook (optional, if you want the outputs to appear on GitHub)
#    Uncomment the lines below if you want to push automatically.
# git add notebooks/01_eda.ipynb
# git commit -m "chore: execute 01_eda notebook with output cells"
# git push

Write-Host "Setup complete! Verify that the following files exist:" -ForegroundColor Green
Write-Host "  - models/preprocessor.joblib"
Write-Host "  - models/baseline_logreg.joblib"
Write-Host "  - experiments/baselines_val.csv"
Write-Host "  - notebooks/01_eda.ipynb (with output cells)"

# End of script