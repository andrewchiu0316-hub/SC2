[CmdletBinding()]
param(
    [string]$SC2Path = "",
    [switch]$SkipPythonInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $ProjectRoot ".venv-smac"
$Python = Join-Path $Venv "Scripts\python.exe"
$Vendor = Join-Path $ProjectRoot "vendor"
$PySC2Source = Join-Path $Vendor "pysc2"
$SMACSource = Join-Path $Vendor "smac"

New-Item -ItemType Directory -Force -Path $Vendor | Out-Null
if (-not (Test-Path $PySC2Source)) {
    git clone --depth 1 https://github.com/google-deepmind/pysc2.git $PySC2Source
}
if (-not (Test-Path $SMACSource)) {
    git clone --depth 1 https://github.com/oxwhirl/smac.git $SMACSource
}

if (-not $SkipPythonInstall) {
    if (-not (Test-Path $Python)) {
        py -3.12 -m venv --system-site-packages $Venv
    }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -e $PySC2Source
    & $Python -m pip install -e $SMACSource
    & $Python -m pip install matplotlib
}

if (-not $SC2Path) {
    $Candidates = @(
        ${env:SC2PATH},
        "C:\Program Files (x86)\StarCraft II",
        "C:\Program Files\StarCraft II"
    ) | Where-Object { $_ -and (Test-Path $_) }
    $SC2Path = $Candidates | Select-Object -First 1
}

if (-not $SC2Path) {
    Write-Warning "StarCraft II was not found. Install the free Starter Edition with Battle.net, then rerun:"
    Write-Warning ".\setup_smac.ps1 -SC2Path 'D:\StarCraft II'"
    exit 2
}

$SourceMaps = Join-Path $SMACSource "smac\env\starcraft2\maps\SMAC_Maps"
$TargetMaps = Join-Path $SC2Path "Maps\SMAC_Maps"
New-Item -ItemType Directory -Force -Path $TargetMaps | Out-Null
Copy-Item -Path (Join-Path $SourceMaps "*") -Destination $TargetMaps -Recurse -Force
[Environment]::SetEnvironmentVariable("SC2PATH", $SC2Path, "User")
$env:SC2PATH = $SC2Path

Write-Host "SMAC setup complete: $SC2Path"
& $Python -m smac.bin.map_list
