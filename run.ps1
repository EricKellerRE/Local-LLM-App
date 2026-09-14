param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot 'localmodel-env\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Could not find the project environment at $Python. Recreate the localmodel-env junction."
}

& $Python (Join-Path $ProjectRoot 'main.py') @Arguments
