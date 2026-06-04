<#
.SYNOPSIS
  Windows task runner for invoice-iq — mirrors the Makefile so local (Windows)
  and CI (Linux) run identical commands.

.EXAMPLE
  .\tasks.ps1 install
  .\tasks.ps1 test
  .\tasks.ps1 check
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'lint', 'format', 'type', 'test', 'check', 'sync-metrics', 'clean')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'

# Prefer the project venv interpreter if present.
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$py = if (Test-Path $venvPy) { $venvPy } else { 'python' }

function Invoke-Step($cmd) {
    Write-Host "> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed (exit $LASTEXITCODE): $cmd" }
}

switch ($Task) {
    'help' {
        Write-Host "Tasks: install | lint | format | type | test | check | sync-metrics | clean"
    }
    'install' {
        Invoke-Step "& '$py' -m pip install --upgrade pip"
        Invoke-Step "& '$py' -m pip install -e '.[dev]'"
    }
    'lint'   { Invoke-Step "& '$py' -m ruff check src tests" }
    'format' {
        Invoke-Step "& '$py' -m ruff format src tests"
        Invoke-Step "& '$py' -m ruff check --fix src tests"
    }
    'type'   { Invoke-Step "& '$py' -m mypy" }
    'test'   { Invoke-Step "& '$py' -m pytest" }
    'check'  {
        Invoke-Step "& '$py' -m ruff check src tests"
        Invoke-Step "& '$py' -m mypy"
        Invoke-Step "& '$py' -m pytest"
    }
    'sync-metrics' { Invoke-Step "& '$py' scripts/sync_metrics_readme.py" }
    'clean' {
        foreach ($d in '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist') {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        if (Test-Path '.coverage') { Remove-Item -Force '.coverage' }
    }
}
