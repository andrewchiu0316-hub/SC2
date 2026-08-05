$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-smac\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "SMAC environment is missing. Run .\setup_smac.ps1 first."
}
$DefaultSC2Path = "E:\StarCraft II"
if (-not $env:SC2PATH -and (Test-Path (Join-Path $DefaultSC2Path "Versions"))) {
    $env:SC2PATH = $DefaultSC2Path
}
Set-Location $ProjectRoot
& $Python (Join-Path $PSScriptRoot "train.py")
if ($LASTEXITCODE -eq 0) {
    & $Python (Join-Path $PSScriptRoot "chart_gui.py")
}
