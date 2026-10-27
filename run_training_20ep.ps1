$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:MPLCONFIGDIR = ".\.matplotlib"

Write-Host "Starting insulin 20 epoch training..."
.\.venv\Scripts\python.exe -W ignore::FutureWarning .\train.py `
  --manifest data\manifest.csv `
  --target insulin `
  --output-dir runs\insulin_model_20ep `
  --epochs 20 `
  --batch-size 32 `
  --num-workers 4 `
  --preview-images 100 `
  --threshold-mode balanced_accuracy

Write-Host "Starting glucagon 20 epoch training..."
.\.venv\Scripts\python.exe -W ignore::FutureWarning .\train.py `
  --manifest data\manifest.csv `
  --target glucagon `
  --output-dir runs\glucagon_model_20ep `
  --epochs 20 `
  --batch-size 32 `
  --num-workers 4 `
  --preview-images 100 `
  --threshold-mode balanced_accuracy

Write-Host "All training complete."
