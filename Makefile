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

# ===================== ch11: the adversarial audit (F13-F24) =========================
# HOURS, not minutes — this retrains every sequence model in the audit. `make all` does
# NOT run it; it rebuilds the report from the verdict artifacts these produce.
#
# Ordering is load-bearing, not stylistic:
#   * f13 writes gate1_verdict.json, whose best_params_sequence f23 AND f24 both load;
#   * f23 writes f23/cache.npz and f23/preds_*.npz, which f24 reads — f23 always first;
#   * F18 needs BOTH cells: one pass writes bidir_verdict_gru.json, the other _lstm.json,
#     and sec:f18-lstm has no data without the second.
# Shard count via F18_SHARDS / F23_SHARDS (default 4 = performance cores). Four is for
# heat and responsiveness, not speed — see docs/TODO.md, standing engineering notes.
overcoming: f13
	$(PY) scripts/f_uq_reattribution.py                # sec:f-uq
	$(PY) scripts/f14_bilstm_replication.py            # sec:f14-ext
	$(PY) scripts/f15_h158_instrument_vs_engine.py     # sec:f15-instrument (sharded)
	$(PY) scripts/f16_gate_t_transient.py              # sec:gate-t
	$(PY) scripts/f17_certificate_hybrid.py            # sec:f17-hybrid
	F18_CELL=gru  $(PY) scripts/f18_bidirectionality.py   # sec:f18-bidir
	F18_CELL=lstm $(PY) scripts/f18_bidirectionality.py   # sec:f18-lstm  (~112 min)
	$(PY) scripts/f19_certificate_isolated.py          # sec:f19-cert (sharded)
	$(PY) scripts/f22_f10_shuffle_control.py           # sec:f21-port, the F10 control
	$(PY) scripts/f23_decoupled_certificate.py         # sec:f23-decoupled
	$(PY) scripts/f24_crb_scale_magnitude.py           # sec:f24-scale (needs f23)

f13:
	$(PY) scripts/f13_gate1_mechanism.py               # sec:f13-mech (Optuna, 25 trials/family)

# External validation. Needs the N-CMAPSS dataset (14.7 GB) downloaded first; f20 estimates
# the influence matrix that f21 then certifies against.
ncmapss:
	$(PY) scripts/f20_ncmapss_icm.py                   # sec:f20-ncmapss
	$(PY) scripts/f21_ncmapss_cert_floor.py            # sec:f21-port

.PHONY: all full model fleet audits pipelines f5 f8 evidence report onepager slides test \
        overcoming f13 ncmapss
