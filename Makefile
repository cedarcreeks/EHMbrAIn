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
	$(PY) scripts/f_icm_robustness.py
	$(PY) scripts/f_noise_sweep.py
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
	cd paper/onepager && latexmk -pdf -outdir=build EHMbrAIn.tex && cp build/EHMbrAIn.pdf EHMbrAIn.pdf

slides:
	cd paper/slides && latexmk -pdf -outdir=build EHMbrAIn-slides.tex && cp build/EHMbrAIn-slides.pdf EHMbrAIn-slides.pdf
	cd paper/slides && latexmk -pdf -outdir=build guion-presentacion.tex && cp build/guion-presentacion.pdf guion-presentacion.pdf

test:
	uv run pytest -q

# ===================== PENDING: ch11 (Overcoming) IS NOT WIRED =======================
# `make all` does NOT regenerate chapter 11 — about 35 of the report's 146 pages, and
# the chapter that holds the entire adversarial audit. Thirteen drivers have no target,
# so a clean clone cannot reproduce them. paper/report/chapters/03-methodology.tex still
# claims "the complete replication path"; that claim is false until this is fixed.
#
# Needs a `make overcoming` target covering, in dependency order:
#   scripts/f_uq_reattribution.py            sec:f-uq
#   scripts/f13_gate1_mechanism.py           sec:f13-mech        (Optuna 25 trials/family)
#   scripts/f14_bilstm_replication.py        sec:f14-ext
#   scripts/f15_h158_instrument_vs_engine.py sec:f15-instrument  (sharded)
#   scripts/f16_gate_t_transient.py          sec:gate-t
#   scripts/f17_certificate_hybrid.py        sec:f17-hybrid
#   F18_CELL=gru  scripts/f18_bidirectionality.py   sec:f18-bidir
#   F18_CELL=lstm scripts/f18_bidirectionality.py   sec:f18-lstm   <- BOTH are required
#   scripts/f19_certificate_isolated.py      sec:f19-cert        (sharded)
#   scripts/f20_ncmapss_icm.py               sec:f20-ncmapss     ) need the N-CMAPSS
#   scripts/f21_ncmapss_cert_floor.py        sec:f21-port        ) download first, 14.68 GB
#   scripts/f22_f10_shuffle_control.py       sec:f21-port
#   scripts/f23_decoupled_certificate.py     sec:f23-decoupled   (writes the cache f24 reads)
#   scripts/f24_crb_scale_magnitude.py       sec:f24-scale       <- MUST run after f23
#
# Ordering constraints that a naive target would get wrong:
#   * f24 reads data/processed/f23/preds_*.npz and f23/cache.npz — f23 first, always.
#   * f13 writes gate1_verdict.json, whose best_params_sequence f23 and f24 both load.
#   * f21/f22 depend on f20's estimated ICM.
# Sharded scripts honour F18_SHARDS / F23_SHARDS (default 4 = performance cores). Four is
# for heat and responsiveness, not speed — see docs/TODO.md standing engineering notes.
#
# COST: this is hours, not minutes. F18 alone measured 112.2 min for the LSTM pass. The
# "657 seconds / about eleven minutes" figure in sec:replication covers only the wired
# stages and will need a scope qualifier once this target exists.
# Tracked: docs/TODO.md section 0, defect D2.
# =====================================================================================

.PHONY: all model fleet audits pipelines f5 f8 evidence report onepager slides test
