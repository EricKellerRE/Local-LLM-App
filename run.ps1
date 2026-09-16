param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstalledPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$WorkspacePython = Join-Path $ProjectRoot 'localmodel-env\python.exe'
$Python = if (Test-Path -LiteralPath $InstalledPython) { $InstalledPython } else { $WorkspacePython }

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Local Model is not installed. Run 'Install Local LLM.cmd' first."
}

& $Python (Join-Path $ProjectRoot 'launcher.py') @Arguments
