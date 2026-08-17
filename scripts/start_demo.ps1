param(
    [switch]$Setup,
    [switch]$Regenerate,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$demoRoot = Join-Path $projectRoot "demo"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$exportScript = Join-Path $projectRoot "scripts\export_demo_data.py"
$demoData = Join-Path $demoRoot "public\demo-data.json"
$nodeModules = Join-Path $demoRoot "node_modules"
$npmCache = Join-Path $projectRoot ".cache\npm"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found. Follow docs\REPRODUCIBILITY.md first."
}

$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $nodeCommand -or -not $npmCommand) {
    throw "Node.js/npm not found. The demo requires Node.js 22.13 or newer."
}

$nodeVersion = (& $nodeCommand.Source --version).TrimStart("v")
$nodeMajor = [int]($nodeVersion.Split(".")[0])
$nodeMinor = [int]($nodeVersion.Split(".")[1])
if ($nodeMajor -lt 22 -or ($nodeMajor -eq 22 -and $nodeMinor -lt 13)) {
    throw "Node.js $nodeVersion found; version 22.13 or newer is required."
}

if ($Regenerate -or -not (Test-Path -LiteralPath $demoData)) {
    Write-Host "Generating anonymized Validation samples..."
    & $pythonPath $exportScript
    if ($LASTEXITCODE -ne 0) {
        throw "Demo sample export failed."
    }
}

if (-not (Test-Path -LiteralPath $nodeModules)) {
    if (-not $Setup) {
        throw "Demo dependencies are missing. Run this script once with -Setup."
    }
    New-Item -ItemType Directory -Force -Path $npmCache | Out-Null
    $env:npm_config_cache = $npmCache
    Write-Host "Installing demo dependencies inside the D-drive project..."
    Push-Location $demoRoot
    try {
        & $npmCommand.Source ci --ignore-scripts --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) {
            throw "Demo dependency installation failed."
        }
    }
    finally {
        Pop-Location
    }
}

if ($CheckOnly) {
    Write-Host "Demo preflight passed. Node.js $nodeVersion, sample data, and dependencies are ready."
    exit 0
}

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:3000/"
    } | Out-Null
}

Write-Host "ColdLens Demo is starting at http://localhost:3000/."
Write-Host "Press Ctrl+C to stop."
Push-Location $demoRoot
try {
    & $npmCommand.Source run dev
}
finally {
    Pop-Location
}
