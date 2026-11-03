# One-command entry points for Linux / macOS (and Windows with make). Windows without make: scripts.ps1.
#   make setup    create .venv and install the pinned requirements (Python 3.12 recommended)
#   make test     run the test suite (tests that need the downloaded data skip and say which files are missing)
#   make demo     run the backtesting framework end to end on SYNTHETIC prices: no data download needed
#   make check    tests + README result blocks match results/*.csv (what CI does)
#   make report   regenerate tail-risk and trading tables and the README blocks from results/fair_forecasts.parquet
#   make all      the full pipeline on the real data (see README "Reproduce"; the data must be downloaded by hand first)
VENV   ?= .venv
PY      = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
export PYTHONPATH := src:.

.PHONY: setup test demo check report all clean-cache

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -rs

demo:
	$(PY) -m mandi.demo

check: test
	$(PY) -m mandi.readme_tables --check

report:
	$(PY) -m mandi.report_risk
	$(PY) -m mandi.readme_tables

# The tuning run must come first and its selection file must be committed before `final` (the final run refuses otherwise).
all:
	$(PY) -m mandi.run_phase1
	$(PY) -m mandi.run_eda
	$(PY) -m mandi.run_phase3 tune
	@echo "Now commit configs/selected_models.yaml, then run: make all-final"

all-final:
	$(PY) -m mandi.run_phase3 final
	$(PY) -m mandi.run_phase3 final --variant error
	$(PY) -m mandi.run_phase3 ablation
	$(PY) -m mandi.posthoc
	$(PY) -m mandi.report_tuning
	$(PY) -m mandi.report_model
	$(PY) -m mandi.intervals
	$(PY) -m mandi.anomalies
	$(PY) -m mandi.report4
	$(PY) -m mandi.forecast
	$(PY) -m mandi.report_risk
	$(PY) -m mandi.readme_tables

clean-cache:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
