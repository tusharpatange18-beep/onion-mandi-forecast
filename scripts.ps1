# Usage: .\scripts.ps1 setup | test | clean | demo | report     (Linux / macOS: ./run.sh or make)
param([Parameter(Mandatory)][ValidateSet("setup","test","clean","demo","report")][string]$Task)
$py = Join-Path $env:USERPROFILE ".venvs\fada\Scripts\python.exe"
$env:PYTHONPATH = "src;."
switch ($Task) {
  "setup"  { python -m venv (Split-Path (Split-Path $py)); & $py -m pip install -r requirements.txt }
  "test"   { & $py -m pytest -q }
  "clean"  { & $py -m mandi.run_phase1 }
  "demo"   { & $py -m mandi.demo }
  "report" { & $py -m mandi.report_risk; & $py -m mandi.readme_tables }
}
