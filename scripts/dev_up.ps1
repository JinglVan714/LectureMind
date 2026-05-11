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

# 0. PATH bootstrap for -NoConda -PythonExe <abs path>:
#    Without `conda activate`, the child python inherits the caller's PATH,
#    which usually lacks the env's Scripts/Library\bin dirs. yt-dlp.exe and
#    ffmpeg.exe (installed via pip/conda into the env) then fall outside
#    `shutil.which` lookup. Mirror what `conda activate` would prepend.
if ($NoConda -and $PythonExe -and (Test-Path $PythonExe)) {
    $envRoot = Split-Path -Parent (Resolve-Path $PythonExe)
    $candidatePaths = @(
        $envRoot,
        (Join-Path $envRoot 'Scripts'),
        (Join-Path $envRoot 'Library\bin'),
        (Join-Path $envRoot 'Library\mingw-w64\bin'),
        (Join-Path $envRoot 'Library\usr\bin')
    ) | Where-Object { Test-Path $_ }
    $existing = $env:PATH -split ';'
    $missing = $candidatePaths | Where-Object { $existing -notcontains $_ }
    if ($missing.Count -gt 0) {
        $env:PATH = ($missing -join ';') + ';' + $env:PATH
        Write-Host "[bootstrap] prepended env bins to PATH: $($missing -join '; ')" -ForegroundColor DarkGray
    }
}

# 0b. tooling smoke: warn (do not fail) if external binaries are missing.
#     The lecturize pipeline shells out to yt-dlp + ffmpeg; Copilot Q&A does
#     not, so a missing tool surfaces only when the user clicks summarize.
foreach ($tool in @('yt-dlp', 'ffmpeg')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "[warn] $tool not found on PATH -- video summarization will fail. " -ForegroundColor Yellow -NoNewline
        Write-Host "Install via 'pip install yt-dlp' / 'conda install -c conda-forge ffmpeg' inside the env." -ForegroundColor Yellow
    }
}

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
