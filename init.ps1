[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$tmpRoot = Join-Path $root ".tmp\pytest"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot

function Resolve-Python {
    $candidates = @()
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += "python"
    }

    $fallback = "D:\anaconda\envs\myagent\python.exe"
    if (Test-Path $fallback) {
        $candidates += $fallback
    }

    foreach ($candidate in $candidates) {
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $candidate -c "import fastapi, pytest" *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } finally {
            $ErrorActionPreference = $previousPreference
        }
    }

    throw "No suitable Python interpreter with fastapi and pytest was found."
}

$python = Resolve-Python

Write-Host "==> LectureMind agent init" -ForegroundColor Cyan
Write-Host "==> repo root: $root" -ForegroundColor DarkGray
Write-Host "==> temp root: $tmpRoot" -ForegroundColor DarkGray
Write-Host "==> python executable: $python" -ForegroundColor DarkGray
Write-Host "==> python version: $(& $python --version)" -ForegroundColor DarkGray

if (-not (Test-Path ".env") -and -not (Test-Path ".env.example")) {
    throw "Missing both .env and .env.example"
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Write-Warning ".env is missing; copy .env.example before running real model-backed flows."
}

foreach ($tool in @("ffmpeg", "yt-dlp")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Warning "$tool not found on PATH"
    }
}

Write-Host "==> python -m compileall app tests scripts" -ForegroundColor Cyan
& $python -m compileall app tests scripts
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> python -m pytest tests/test_harness_workspace.py -q" -ForegroundColor Cyan
& $python -m pytest tests/test_harness_workspace.py -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> python -m pytest tests/test_skill_harness.py -q" -ForegroundColor Cyan
& $python -m pytest tests/test_skill_harness.py -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> python -m pytest tests/test_smoke.py -q" -ForegroundColor Cyan
& $python -m pytest tests/test_smoke.py -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> init complete" -ForegroundColor Green
