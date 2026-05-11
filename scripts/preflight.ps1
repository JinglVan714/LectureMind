# scripts/preflight.ps1 — pre-publish hygiene check before pushing to GitHub.
#
# What it verifies:
#   1. .gitignore covers .env, .env.*, data/, caches
#   2. .env is NOT tracked by git (warns loudly if it is)
#   3. DASHSCOPE_API_KEY / BASIC_AUTH_PASSWORD in .env are not the placeholders
#   4. .env.example exists and contains no real-looking secret
#   5. data/ is not tracked by git
#   6. (optional) full automated regression via scripts/qa_full.ps1
#
# Usage (PowerShell, repo root):
#   ./scripts/preflight.ps1
#   ./scripts/preflight.ps1 -SkipTests        # skip pytest if you just ran qa_full
#   ./scripts/preflight.ps1 -NoConda          # use python on PATH

[CmdletBinding()]
param(
    [string]$CondaEnv = 'myagent',
    [switch]$NoConda,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$warnings = New-Object System.Collections.Generic.List[string]
$errors   = New-Object System.Collections.Generic.List[string]

function Add-Warn { param($Msg) $warnings.Add($Msg) | Out-Null; Write-Host "[warn] $Msg" -ForegroundColor Yellow }
function Add-Err  { param($Msg) $errors.Add($Msg)   | Out-Null; Write-Host "[err]  $Msg" -ForegroundColor Red }
function Add-Ok   { param($Msg) Write-Host "[ok]   $Msg" -ForegroundColor Green }

# 1. .gitignore covers the dangerous paths
if (Test-Path .gitignore) {
    $gi = Get-Content .gitignore
    foreach ($pat in @('.env', 'data/')) {
        $needle = [regex]::Escape($pat)
        if ($gi -match "^$needle$") {
            Add-Ok ".gitignore covers $pat"
        } else {
            Add-Err ".gitignore does NOT cover $pat"
        }
    }
} else {
    Add-Err '.gitignore missing'
}

# 2. .env hygiene
if (Test-Path .env) {
    $envBody = Get-Content .env -Raw
    if ($envBody -match 'DASHSCOPE_API_KEY=sk-replace-me') {
        Add-Warn '.env still has the placeholder DASHSCOPE_API_KEY -- real Copilot/lecturize calls will fail'
    } else {
        Add-Ok '.env has a non-placeholder DASHSCOPE_API_KEY'
    }
    if ($envBody -match 'BASIC_AUTH_PASSWORD=change-me-please') {
        Add-Warn '.env still has the default BASIC_AUTH_PASSWORD -- change it before any public exposure'
    }
} else {
    Add-Warn '.env not present (dev_up.ps1 will bootstrap it from .env.example)'
}

# 3. .env.example must exist and stay placeholder-only
if (Test-Path .env.example) {
    $exBody = Get-Content .env.example -Raw
    if ($exBody -match 'DASHSCOPE_API_KEY=sk-(?!replace-me)[A-Za-z0-9]{20,}') {
        Add-Err '.env.example appears to contain a real DASHSCOPE key -- replace with sk-replace-me'
    } else {
        Add-Ok '.env.example uses placeholder secret values'
    }
} else {
    Add-Err '.env.example missing'
}

# 4. git tracking sanity (only if repo is initialised)
if (Test-Path .git) {
    # Native git non-zero exits + 2>$null still trip $ErrorActionPreference='Stop' in
    # PowerShell, so temporarily relax it around the probes and rely on $LASTEXITCODE.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        git ls-files --error-unmatch .env *> $null
        if ($LASTEXITCODE -eq 0) { Add-Err '.env is tracked by git -- fix with: git rm --cached .env' }
        else                     { Add-Ok '.env is not tracked by git' }

        git ls-files --error-unmatch data *> $null
        if ($LASTEXITCODE -eq 0) { Add-Err 'data/ is tracked by git -- fix with: git rm -r --cached data' }
        else                     { Add-Ok 'data/ is not tracked by git' }
    } finally {
        $ErrorActionPreference = $prevEAP
        $global:LASTEXITCODE = 0
    }
} else {
    Add-Warn 'no .git directory -- repo not initialised yet (skip tracking checks)'
}

# 5. optional: run the full regression
if (-not $SkipTests) {
    Write-Host '==> running scripts/qa_full.ps1 (use -SkipTests to bypass)' -ForegroundColor Cyan
    # Hashtable splat → PowerShell binds these as -Named parameters reliably
    # (positional array splat misbinds when scripts share param names like -CondaEnv).
    $qaArgs = @{}
    if ($NoConda) { $qaArgs.NoConda = $true } else { $qaArgs.CondaEnv = $CondaEnv }
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & "$PSScriptRoot/qa_full.ps1" @qaArgs
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($LASTEXITCODE -ne 0) { Add-Err "qa_full.ps1 failed (exit $LASTEXITCODE)" }
    else                     { Add-Ok 'qa_full.ps1 passed' }
}

# Summary
Write-Host ''
Write-Host "== preflight summary ==" -ForegroundColor Cyan
Write-Host ("warnings: {0}" -f $warnings.Count)
Write-Host ("errors:   {0}" -f $errors.Count)
if ($errors.Count -gt 0) {
    Write-Host 'preflight FAILED -- fix the [err] items above before pushing to GitHub.' -ForegroundColor Red
    exit 1
}
Write-Host 'preflight OK -- safe to push.' -ForegroundColor Green
