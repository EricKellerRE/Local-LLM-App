param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot 'localmodel-env\python.exe'
$env:PYTHONPATH = $ProjectRoot

Set-Location -LiteralPath $ProjectRoot
& $Python -m uvicorn local_model_app.server:app --host 127.0.0.1 --port 8765
