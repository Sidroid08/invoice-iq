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
    [ValidateSet('help', 'install', 'lint', 'format', 'type', 'test', 'check', 'demo', 'serve', 'docker-build', 'docker-up', 'sync-metrics', 'clean')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'

# Prefer the project venv interpreter if present.
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$py = if (Test-Path $venvPy) { $venvPy } else { 'python' }

# This machine intercepts TLS; point SSL-using libs (httpx/huggingface, requests)
# at the exported Windows CA bundle so model downloads work. Local-only; on a
# normal machine the bundle is absent and this is a no-op. See env memory note.
$caBundle = Join-Path $PSScriptRoot 'win-ca-bundle.pem'
if (Test-Path $caBundle) {
    $env:SSL_CERT_FILE = $caBundle
    $env:REQUESTS_CA_BUNDLE = $caBundle
    $env:CURL_CA_BUNDLE = $caBundle
}

function Invoke-Step($cmd) {
    Write-Host "> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed (exit $LASTEXITCODE): $cmd" }
}

switch ($Task) {
    'help' {
        Write-Host "Tasks: install | lint | format | type | test | check | demo | serve | docker-build | docker-up | sync-metrics | clean"
    }
    'install' {
        Invoke-Step "& '$py' -m pip install --upgrade pip"
        Invoke-Step "& '$py' -m pip install -e '.[dev]'"
    }
    'lint'   { Invoke-Step "& '$py' -m ruff check src tests scripts" }
    'format' {
        Invoke-Step "& '$py' -m ruff format src tests scripts"
        Invoke-Step "& '$py' -m ruff check --fix src tests scripts"
    }
    'type'   { Invoke-Step "& '$py' -m mypy" }
    'test'   { Invoke-Step "& '$py' -m pytest" }
    'check'  {
        Invoke-Step "& '$py' -m ruff check src tests scripts"
        Invoke-Step "& '$py' -m mypy"
        Invoke-Step "& '$py' -m pytest"
    }
    'demo' { Invoke-Step "& '$py' scripts/demo.py" }
    'serve' {
        Invoke-Step "& '$py' -m uvicorn invoice_iq.serving.app:create_app --factory --host 0.0.0.0 --port 8000 --reload"
    }
    'docker-build' { Invoke-Step "docker build -t invoice-iq:local ." }
    'docker-up' { Invoke-Step "docker compose up --build" }
    'sync-metrics' { Invoke-Step "& '$py' scripts/sync_metrics_readme.py" }
    'clean' {
        foreach ($d in '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist') {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        if (Test-Path '.coverage') { Remove-Item -Force '.coverage' }
    }
}
