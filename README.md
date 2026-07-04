# EHMbrAIn — AI-based vs Traditional Engine Health Monitoring on the CFM56-7B

**EHM + brAIn**: the EHM brain. Engine Health Monitoring re-thought with AI, benchmarked
head-to-head against the traditional approach.

Reproducible testbed comparing **traditional Engine Health Monitoring** (linear Gas Path
Analysis, Kalman tracking, expert rules, trend monitoring) against **AI-based EHM**
(anomaly detection, fault diagnosis, RUL prognosis, physics-informed hybrids) on the same
synthetic CFM56-7B fleet with full ground truth.

See [PLAN.md](PLAN.md) for the full project plan (Spanish): hypotheses, work packages,
milestones H0–H6 and the academic contributions (SynCFM56 open benchmark, pre-registered
comparison protocol, conformal RUL intervals, Physics-Consistency Score).

## Quick start

```bash
uv sync                                  # Python 3.11 env with pyCycle 4.4
uv run python scripts/hello_pycycle.py   # H0 hello world: turbojet design + off-design
uv run pytest                            # smoke tests
```

> Note: pyCycle 4.4 requires `numpy<2` (pinned in `pyproject.toml`).

## Layout

```
conf/          Hydra configs (engine, datagen, models, eval)
src/ehmbrain/
  perf/        F1: pyCycle CFM56-7B model, calibration, ICM, baseline decks
  datagen/     F2: degradation, fleet simulation, sensor model, ACARS snapshots
  trad/        F3: traditional EHM (baselines, trending, WLS/Kalman GPA, rules)
  ai/          F4: detection, diagnosis, RUL, physics-informed hybrid, UQ, XAI
  eval/        F5: common metrics, statistical tests, ablations
  common/      units, corrections, IO, data schema
tests/         pytest (physics sign checks, numerics, regression, smoke)
docs/          engineering specs (F1 model spec with calibration status)
data/          DVC-tracked datasets (raw/interim/processed)
dashboard/     Streamlit demo
paper/         LaTeX sources, generated figures
```

## Project report

`paper/report/` holds a living LaTeX report documenting every milestone in full detail,
written to be self-contained (no gas-turbine background assumed). All result figures and
tables are **generated from the model** — never hand-copied:

```bash
uv run python scripts/make_report_assets.py        # regenerate figures + tables
cd paper/report && latexmk -pdf -outdir=build report.tex
```

TeXstudio users: set the bibliography tool to Biber (Options → Configure → Build).

## Replicate everything (~11 min on an Apple M5)

```bash
uv sync && uv run pytest                          # environment + 36 gate tests
uv run python scripts/run_design_point.py         # F1: design point
uv run python scripts/run_anchors.py              # F1: calibration vs TCDS/EEDB
uv run python scripts/make_decks.py               # F1: baseline decks
uv run python scripts/make_corrected_baseline.py  # F1: corrected-space baseline
uv run python scripts/make_icm.py                 # F1: ICM grid + observability
uv run python scripts/make_fleet.py               # F2: SynCFM56 fleet
uv run python scripts/audit_dataset.py            # F2: difficulty + realism gates
uv run python scripts/audit_nonlinearity.py       # F2: linearization audit
uv run python scripts/run_trad.py                 # F3: traditional EHM metrics
uv run python scripts/run_ai.py                   # F4: AI suite (MPS; run in FOREGROUND)
uv run python scripts/run_hybrid.py               # F4: hybrid ablation (foreground)
uv run python scripts/run_pcs.py                  # F4: Physics-Consistency Score
uv run python scripts/benchmark_pipeline.py       # norm N5: compute times
uv run python scripts/make_report_assets.py       # regenerate ALL report evidence
```

macOS notes: torch-MPS runs must be foreground (backgrounded runs segfault);
XGBoost is intentionally absent (OpenMP clash with torch-MPS — sklearn HistGB instead).
Full mapping of scripts to report tables/figures: report ch. 3, "Replication guide".

## Status

- [x] **H0** — environment runs an end-to-end pyCycle cycle; repo skeleton in place
- [ ] **H1** — calibrated CFM56-7B performance model + influence coefficient matrix
- [ ] **H2** — SynCFM56 synthetic fleet dataset v1.0
- [ ] **H3** — traditional EHM baseline pipeline
- [ ] **H4** — AI EHM suite
- [ ] **H5** — pre-registered comparative evaluation
- [ ] **H6** — case studies, dashboard, paper
