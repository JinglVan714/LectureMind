[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$tmpRoot = Join-Path $root ".tmp\pytest"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
$env:TEMP = $tmpRoot
$env:TMP = $tmpRoot

Write-Host "==> Media skill harness init" -ForegroundColor Cyan
Write-Host "==> temp root: $tmpRoot" -ForegroundColor DarkGray

powershell -ExecutionPolicy Bypass -File .\init.ps1
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$python = "D:\anaconda\envs\myagent\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host "==> python scripts/skill_harness_check.py" -ForegroundColor Cyan
& $python scripts\skill_harness_check.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> media skill harness init complete" -ForegroundColor Green
