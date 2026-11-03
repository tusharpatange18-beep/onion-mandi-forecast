#!/usr/bin/env bash
# Usage: ./run.sh setup | test | demo | check | report
#   setup   create .venv and install the pinned requirements (Python 3.12 recommended)
#   test    run the tests (those needing the downloaded data skip and list the missing files)
#   demo    run the backtesting framework end to end on synthetic prices (no data download)
#   check   tests + README result blocks match results/*.csv (what CI does)
#   report  regenerate the tail-risk / trading tables and README blocks (needs results/fair_forecasts.parquet)
# The full pipeline on the real data is described in the README ("Reproduce") and in the Makefile (`make all`).
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"   # no venv yet (e.g. CI): use the python on the PATH
export PYTHONPATH="src:."

case "${1:-}" in
  setup)  python3 -m venv "$VENV" && "$VENV/bin/python" -m pip install -r requirements.txt ;;
  test)   "$PY" -m pytest -rs ;;
  demo)   "$PY" -m mandi.demo ;;
  check)  "$PY" -m pytest -rs && "$PY" -m mandi.readme_tables --check ;;
  report) "$PY" -m mandi.report_risk && "$PY" -m mandi.readme_tables ;;
  *) echo "usage: $0 setup|test|demo|check|report" >&2; exit 2 ;;
esac
