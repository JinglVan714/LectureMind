# scripts/dev_up.ps1 — one-shot local dev launcher.
#
# Steps:
#   1. ensure .env exists (auto-copy from .env.example with a warning if not)
#   2. mkdir data/ + run scripts/init_db.py to materialise the SQLite schema
#   3. start uvicorn on $APP_HOST:$APP_PORT (defaults to 127.0.0.1:8000)
#
# Usage (PowerShell, repo root):
#   ./scripts/dev_up.ps1                              # myagent conda env, foreground
#   ./scripts/dev_up.ps1 -CondaEnv foo
#   ./scripts/dev_up.ps1 -NoConda                     # uses python on PATH
#   ./scripts/dev_up.ps1 -Reload                      # uvicorn --reload for dev
#
# This script does NOT modify .env if it already exists.

[CmdletBinding()]
param(
    [string]$CondaEnv = 'myagent',
    [string]$PythonExe = 'python',
    [switch]$NoConda,
    [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 1. .env bootstrap (do not overwrite)
if (-not (Test-Path '.env')) {
    if (Test-Path '.env.example') {
        Copy-Item '.env.example' '.env'
        Write-Host '[bootstrap] .env created from .env.example — fill DASHSCOPE_API_KEY etc. before real runs' -ForegroundColor Yellow
    } else {
        Write-Host '[error] neither .env nor .env.example exists' -ForegroundColor Red
        exit 1
    }
}

# 2. data dir + DB init
New-Item -ItemType Directory -Force -Path 'data' | Out-Null

function Invoke-Python {
    param([string[]]$PythonArgs)
    if ($NoConda) {
        & $PythonExe @PythonArgs
    } else {
        & conda run --no-capture-output -n $CondaEnv python @PythonArgs
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host '==> python -m scripts.init_db' -ForegroundColor Cyan
Invoke-Python @('-m', 'scripts.init_db')

# 3. uvicorn
$uvArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000')
if ($Reload) { $uvArgs += '--reload' }

Write-Host "==> uvicorn app.main:app  (host/port read from .env, override via APP_HOST/APP_PORT env)" -ForegroundColor Cyan
Invoke-Python $uvArgs
