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

## Project report (memoria)

`paper/memoria/` holds a living LaTeX report (Spanish) documenting every milestone with
full detail. All result figures and tables are **generated from the model** — never
hand-copied:

```bash
uv run python scripts/make_memoria_assets.py       # regenerate figures + tables
cd paper/memoria && latexmk -pdf -outdir=build memoria.tex
```

## Status

- [x] **H0** — environment runs an end-to-end pyCycle cycle; repo skeleton in place
- [ ] **H1** — calibrated CFM56-7B performance model + influence coefficient matrix
- [ ] **H2** — SynCFM56 synthetic fleet dataset v1.0
- [ ] **H3** — traditional EHM baseline pipeline
- [ ] **H4** — AI EHM suite
- [ ] **H5** — pre-registered comparative evaluation
- [ ] **H6** — case studies, dashboard, paper
