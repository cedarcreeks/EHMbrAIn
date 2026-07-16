# EHMbrAIn — gate H6: full replication with one command.
# Foreground required (torch-MPS). Reference machine: Apple M5 (~15 min + tuning).
PY := uv run python

all: model fleet audits pipelines f5 evidence report
full: all f8   # includes the F8 limitations program (L1/L2/L6)

model:
	$(PY) scripts/run_design_point.py
	$(PY) scripts/run_anchors.py
	$(PY) scripts/make_decks.py
	$(PY) scripts/make_corrected_baseline.py
	$(PY) scripts/make_icm.py

fleet:
	$(PY) scripts/make_fleet.py

audits:
	$(PY) scripts/audit_dataset.py
	$(PY) scripts/audit_nonlinearity.py

pipelines:
	$(PY) scripts/run_trad.py
	$(PY) scripts/run_ai.py
	$(PY) scripts/run_hybrid.py
	$(PY) scripts/run_pcs.py

f5:
	$(PY) scripts/tune_f5.py trad 50
	$(PY) scripts/tune_f5.py ai 50
	$(PY) scripts/f5_confirm.py
	$(PY) scripts/dump_optuna_history.py
	$(PY) scripts/sim_to_real.py

f8:
	$(PY) scripts/f8_surrogate_data.py 2400
	$(PY) scripts/f8_surrogate_data.py 2400 takeoff
	$(PY) scripts/f8_surrogate.py cruise
	$(PY) scripts/f8_surrogate.py takeoff
	$(PY) scripts/make_fleet.py surrogate
	$(PY) scripts/audit_dataset.py fleet_v2
	$(PY) scripts/audit_v2_fidelity.py 60
	$(PY) scripts/f8_l6_hybrid.py
	$(PY) scripts/f8_l4_recoverable.py
	$(PY) scripts/f8_l5_arch.py
	$(PY) scripts/f8_l7_drift.py
	$(PY) scripts/f8_lh2_wall.py
	$(PY) scripts/f8_lrul_advanced.py
	$(PY) scripts/f8_l9_pcs.py
	$(PY) scripts/f10_certificate.py
	$(PY) scripts/f11_prognostic_floor.py
	$(PY) scripts/f_ops_conversion.py
	$(PY) scripts/econ_impact.py

evidence:
	$(PY) scripts/make_case_studies.py
	$(PY) scripts/fig_rul_distribution.py
	$(PY) scripts/fig_isolation.py
	$(PY) scripts/benchmark_pipeline.py model decks fleet audits trad
	$(PY) scripts/make_report_assets.py

report:
	cd paper/report && latexmk -pdf -outdir=build report.tex && cp build/report.pdf report.pdf

onepager:
	cd paper/onepager && latexmk -pdf -outdir=build onepager.tex && cp build/onepager.pdf onepager.pdf

test:
	uv run pytest -q

.PHONY: all model fleet audits pipelines f5 f8 evidence report onepager test
