param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentRoot = Join-Path $ProjectRoot '.venv'
$InstalledPython = Join-Path $EnvironmentRoot 'Scripts\python.exe'
$WorkspacePython = Join-Path $ProjectRoot 'localmodel-env\python.exe'
$Python = if (Test-Path -LiteralPath $InstalledPython) {
    $InstalledPython
} elseif (Test-Path -LiteralPath $WorkspacePython) {
    $WorkspacePython
} else {
    $InstalledPython
}

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        & $Launcher.Source -3 -m venv $EnvironmentRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Local Model environment.' }
    } else {
        $SystemPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $SystemPython) {
            throw 'Python 3 is required. Install Python, then run this installer again.'
        }
        & $SystemPython.Source -m venv $EnvironmentRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Local Model environment.' }
    }
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install Local Model dependencies.' }

$Settings = Join-Path $ProjectRoot '.env'
if (-not (Test-Path -LiteralPath $Settings)) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot '.env.example') -Destination $Settings
}

$SettingsLines = @(Get-Content -LiteralPath $Settings)
$GridLine = $SettingsLines | Where-Object { $_ -match '^GRID_WORKSHOP_ROOT=' } | Select-Object -First 1
$GridRoot = if ($GridLine) { ($GridLine -replace '^GRID_WORKSHOP_ROOT=', '').Trim().Trim('"').Trim("'") } else { '' }
if (-not $GridRoot -or $GridRoot -match 'your-name') {
    $GridCandidate = Join-Path (Split-Path -Parent $ProjectRoot) 'Grid-Workshop'
    if (Test-Path -LiteralPath (Join-Path $GridCandidate 'powerworld-aux-agent\pyproject.toml')) {
        $GridRoot = $GridCandidate.Replace('\', '/')
        if ($GridLine) {
            $SettingsLines = $SettingsLines | ForEach-Object {
                if ($_ -match '^GRID_WORKSHOP_ROOT=') { "GRID_WORKSHOP_ROOT=$GridRoot" } else { $_ }
            }
        } else {
            $SettingsLines += "GRID_WORKSHOP_ROOT=$GridRoot"
        }
        Set-Content -LiteralPath $Settings -Value $SettingsLines -Encoding utf8
    }
}

if ($GridRoot) {
    $GridAgent = Join-Path $GridRoot 'powerworld-aux-agent'
    if (Test-Path -LiteralPath (Join-Path $GridAgent 'pyproject.toml')) {
        & $Python -m pip install -e $GridAgent
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the Grid Workshop tool environment.' }
    }
}

Write-Host ''
Write-Host 'Local Model is installed.' -ForegroundColor Green
Write-Host "Choose a model in $Settings, then double-click 'Local LLM.cmd'."
