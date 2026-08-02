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

## Replicate everything (one command)

```bash
make all         # core pipeline -> verdicts -> case studies -> report PDF (~15 min)
make full        # adds the F8 limitations program
make overcoming  # F13-F24: the adversarial audit behind report ch. 11 (HOURS, not minutes)
make test        # 44 gate tests
uv run streamlit run dashboard/app.py   # interactive fleet/engine/verdicts views
```

`make all` rebuilds the report from cached verdicts; it does **not** re-run F13–F24. To
regenerate chapter 11's evidence from scratch, run `make overcoming` first — it retrains
every sequence model in the audit and costs hours (F18 alone measured 112 min). Two of its
stages (`f20`, `f21`) additionally need the N-CMAPSS dataset downloaded (14.7 GB).

Or stage by stage (~15 min on an Apple M5, plus tuning):

```bash
uv sync && uv run pytest                          # environment + 44 gate tests
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
uv run python scripts/tune_f5.py trad 50          # F5: tuning campaigns (repeat: ai 50)
uv run python scripts/f5_confirm.py               # F5: single confirmatory pass -> verdicts
uv run python scripts/sim_to_real.py              # F5: C-MAPSS FD001 ranking check
uv run python scripts/benchmark_pipeline.py       # norm N5: compute times
uv run python scripts/make_report_assets.py       # regenerate ALL report evidence
```

F8 limitations program (nonlinear v2 fleet, H4 re-refutation): `make f8` (or `make full`).

macOS notes: torch-MPS runs must be foreground (backgrounded runs segfault);
XGBoost is intentionally absent (OpenMP clash with torch-MPS — sklearn HistGB instead).
Full mapping of scripts to report tables/figures: report ch. 3, "Replication guide".

## What it found

Every number below is against the **advanced** classical baseline — similarity matching,
what a shop actually uses — not against a naive linear extrapolation. That distinction is
most of the result: against Theil-Sen the AI advantage looks two to six times larger, and
this project spent a full re-audit phase deflating its own headlines to the figures here.

**Where AI wins, and it is real:**

| task | traditional | AI | test |
|---|---|---|---|
| detection recall (matched false alarms) | 0.130 | **0.478** | McNemar *p* = 0.0039 |
| detection delay | 6033 cy | **499 cy** | 8 AI-only wins, 0 the other way |
| RUL RMSE @ 90 % life | 1118 cy | **834 cy** | Wilcoxon *p* = 0.0018 |
| conformal interval half-width | 2717 cy | **2028 cy** | at matched ~90 % coverage |
| unscheduled → scheduled conversion | 15 % net | **35 % net** | within 7.5 pt of the irreducible floor |

**Where it does not, and the controls say why:**

- **Fault isolation of the confusable pair: 0.308 vs 0.308.** Identical. The η_HPC/η_HPT
  pair sits 1.3° apart in noise-whitened signature space, and no architecture invents
  information the sensors never carried. Reproduced on an external engine model.
- **RUL before ~70 % of life: a tie** (1.005× at 70 %; 1.18× at *p* = 0.205 at 50 %).
- **The identifiability certificate as an AI feature actively hurts** (−0.119, *t* = −3.72).
- **The certificate's own honesty test is under-powered on this fleet** — not refuted,
  unmeasurable: ranking is capped at ten health directions, and the magnitude route fails
  its pre-declared gate because the per-engine bound varies only 1.15 % across a fleet
  where every engine flies the same profile. `docs/TODO.md` §3 names the one experiment
  that would settle it.

## Status

- [x] **H0** — environment runs an end-to-end pyCycle cycle; repo skeleton in place
- [x] **H1** — calibrated CFM56-7B26 model + baseline decks + influence coefficient matrix
- [x] **H2** — SynCFM56 synthetic fleet (v1.1: multi-episode, twice-hardened difficulty gate)
- [x] **H3** — traditional EHM pipeline with test-fleet metrics (floor numbers, pre-tuning)
- [x] **H4** — AI suite (detection, diagnosis, RUL, hybrids, conformal, PCS); **refuted** at
  the pre-registered gate — the physics-informed hybrid does not beat the pure learner
- [x] **H5** — pre-registered verdicts: H1/H3/H5 confirmed, H2/H4 refuted; FD001 ranking check passed
- [x] **H6** — five case studies, Streamlit dashboard, `make all` replication, conclusions
- [x] **F7–F8** — GPA tomography; the limitations program (surrogate twin, nonlinear v2 fleet,
  recoverable fraction, architecture breadth, drift, the real-vs-virtual sensor wall)
- [x] **F10–F12** — identifiability certificate (C8), prognostic floor (C9), operational
  conversion; F12 withdrawn as tautological, with the reason on record
- [x] **F13–F19** — the overcoming program: mechanism attribution, published Bi-LSTM
  replication, instrument-vs-engine, Gate T on transients, certificate hybrid and isolation
- [x] **F20–F24** — external validation on N-CMAPSS, and a four-control audit of this
  project's own most-promoted result, which changed the claim

All verdicts, including the refutations and the withdrawn lines, are in
`paper/report/report.pdf` (146 pp). Pre-registration tags `prereg-v1`…`prereg-v27` are
indexed in [docs/prereg-index.md](docs/prereg-index.md); `docs/TODO.md` carries what is
still open.
