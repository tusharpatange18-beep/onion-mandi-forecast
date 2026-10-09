# Usage: .\scripts.ps1 setup | test | clean
param([Parameter(Mandatory)][ValidateSet("setup","test","clean")][string]$Task)
$py = Join-Path $env:USERPROFILE ".venvs\fada\Scripts\python.exe"
switch ($Task) {
  "setup" { python -m venv (Split-Path (Split-Path $py)); & $py -m pip install -r requirements.txt }
  "test"  { & $py -m pytest -q }
  "clean" { $env:PYTHONPATH = "src"; & $py -m mandi.run_phase1 }
}
