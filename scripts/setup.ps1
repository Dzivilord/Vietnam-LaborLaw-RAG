[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
Set-Location $ProjectRoot

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $Hint"
    }
}

Require-Command $Python "Activate Conda labor first, or pass -Python with the path to your Python executable."
Require-Command "node" "Install Node.js 20.9+ and reopen PowerShell."
Require-Command "npm" "Install npm with Node.js and reopen PowerShell."
Require-Command "docker" "Install and start Docker Desktop."

$pythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$versionParts = $pythonVersion.Trim().Split('.') | ForEach-Object { [int]$_ }
if (($versionParts[0] -lt 3) -or (($versionParts[0] -eq 3) -and ($versionParts[1] -lt 10))) {
    throw "Python 3.10+ is required; detected $pythonVersion."
}

if (-not (Test-Path "configs/labor_rag.yaml")) { throw "Missing configs/labor_rag.yaml" }
if (-not (Test-Path "data/processed/chunks/labor_law_legal.jsonl")) { throw "Missing processed chunk data." }

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt
& $Python -m pip install -e .

if (-not $SkipFrontend) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location "app/frontend"
    try { npm ci } finally { Pop-Location }
}

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama detected. Required model: qwen3.5:4b" -ForegroundColor Green
} else {
    Write-Warning "Ollama was not found. Install it before starting the product."
}

Write-Host "Setup dependencies are complete. Next run the Qdrant/indexing commands in README.md." -ForegroundColor Green