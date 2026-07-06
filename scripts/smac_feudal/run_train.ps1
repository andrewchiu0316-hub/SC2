$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-smac\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "SMAC environment is missing. Run .\setup_smac.ps1 first."
}
Set-Location $ProjectRoot
& $Python (Join-Path $PSScriptRoot "train.py")
if ($LASTEXITCODE -eq 0) {
    & $Python (Join-Path $PSScriptRoot "chart_gui.py")
}
