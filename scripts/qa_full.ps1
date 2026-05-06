# scripts/qa_full.ps1 — full automated regression for LectureMind.
#
# Runs the same triple that gates every release:
#   1. node --check on the Copilot front-end bundle (catches JS syntax regressions)
#   2. python -m compileall on app/ tests/ scripts/
#   3. pytest tests/ -v
#
# Usage (from repo root, PowerShell):
#   ./scripts/qa_full.ps1                 # uses 'myagent' conda env by default
#   ./scripts/qa_full.ps1 -CondaEnv foo   # different conda env name
#   ./scripts/qa_full.ps1 -NoConda        # use the python on PATH directly
#
# Exits non-zero on the first failure so CI / shell can fail fast.

[CmdletBinding()]
param(
    [string]$CondaEnv = 'myagent',
    [switch]$NoConda
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Name] failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Invoke-Python {
    param([string[]]$Args)
    if ($NoConda) {
        & python @Args
    } else {
        & conda run --no-capture-output -n $CondaEnv python @Args
    }
}

# 1. Front-end syntax (cheap — catches regressions in copilot.js fast)
if (Get-Command node -ErrorAction SilentlyContinue) {
    Invoke-Step 'node --check app/static/copilot.js' { node --check app/static/copilot.js }
} else {
    Write-Host '[warn] node not found on PATH; skipping copilot.js syntax check' -ForegroundColor Yellow
}

# 2. Python bytecode sanity
Invoke-Step 'python -m compileall app tests scripts' {
    Invoke-Python @('-m', 'compileall', 'app', 'tests', 'scripts')
}

# 3. Full pytest sweep
Invoke-Step 'pytest tests/ -v' {
    Invoke-Python @('-m', 'pytest', 'tests/', '-v')
}

Write-Host 'OK · full automated regression passed' -ForegroundColor Green
