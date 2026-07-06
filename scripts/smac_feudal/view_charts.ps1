$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-smac\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python 環境不存在，請先執行 .\setup_smac.ps1"
}
& $Python (Join-Path $PSScriptRoot "chart_gui.py")
