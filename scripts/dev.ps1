[CmdletBinding()]
param(
    [switch]$SkipQdrant
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
Set-Location $ProjectRoot

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not $SkipQdrant) {
    if (-not (Test-Port 6333)) {
        docker compose up -d qdrant
    } else {
        Write-Host "Qdrant is already listening on port 6333." -ForegroundColor Green
    }
}

if (-not (Test-Port 8000)) {
    $backendCommand = "Set-Location '$ProjectRoot'; python -m uvicorn app.backend.api.main:app --reload"
    if (Test-Path (Join-Path $ProjectRoot ".env")) {
        $backendCommand += " --env-file .env"
    }
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit", "-NoProfile", "-Command",
        $backendCommand
    )
} else {
    Write-Host "Backend is already listening on port 8000." -ForegroundColor Green
}

if (-not (Test-Port 3000)) {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit", "-NoProfile", "-Command",
        "Set-Location '$ProjectRoot\app\frontend'; npm run dev"
    )
} else {
    Write-Host "Frontend is already listening on port 3000." -ForegroundColor Green
}

Write-Host "Services requested. Backend: http://127.0.0.1:8000/docs; Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "The launcher does not index data automatically. Stop services by closing their windows and run 'docker compose down' when needed."