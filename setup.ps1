# Windows setup: virtual environment, dependencies, GPU check.
# powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Datasets are downloaded manually into data\ (see README).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) { python -m venv .venv }
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, results | Out-Null
& $py check_gpu.py
