[CmdletBinding()]
param(
    [int]$Port = 6006
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvRoot = Join-Path $ProjectRoot ".venv-smac"
$Python = Join-Path $VenvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "SMAC environment is missing. Run .\setup_smac.ps1 first."
}

$env:PATH = "$VenvRoot;$VenvRoot\Scripts;$env:PATH"
$LogDir = Join-Path $ProjectRoot "runs"

Write-Host "TensorBoard: http://localhost:$Port"
Write-Host "Press Ctrl+C to stop TensorBoard."
& $Python -m tensorboard.main --logdir $LogDir --port $Port
