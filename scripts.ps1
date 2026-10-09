# Usage: .\scripts.ps1 test | setup
param([Parameter(Mandatory)][ValidateSet("setup","test")][string]$Task)
$py = Join-Path $env:USERPROFILE ".venvs\fada\Scripts\python.exe"
switch ($Task) {
  "setup" { python -m venv (Split-Path (Split-Path $py)); & $py -m pip install -r requirements.txt }
  "test"  { & $py -m pytest -q }
}
