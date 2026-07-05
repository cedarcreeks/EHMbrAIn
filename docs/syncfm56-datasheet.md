# SynCFM56 v1.1 — Dataset datasheet

Following the *Datasheets for Datasets* structure (Gebru et al.).

## Motivation

**Purpose.** Benchmark for comparing traditional gas-path-analysis EHM against AI-based EHM on
the same data with complete ground truth: per-flight component health state, EGT margin,
remaining useful life, fault labels and event log. Built because no public dataset offers
GPA-style ground truth for a specific commercial engine with ACARS-style reporting.

**Creator/funding.** The EHMbrAIn project (independent research promoting industry adoption of AI-based engine health monitoring). No external funding.

## Composition

- **Instances.** 1 551 071 flight-cycle rows from 100 synthetic CFM56-7B26 engines run to
  failure (median life 15 139 cycles, range 9 972–20 682, none censored). Each row carries a
  takeoff and a cruise snapshot: 8 measured channels each (N1, N2, WF, EGT, P25, T25, PS3, T3)
  plus the true (noise-free) values, operating conditions, and ground truth (10 health
  parameters, EGT margin, RUL, isolation label, chronic label, drift flags).
- **Events table.** Washes (with recovery fractions), foreign-object damage, acute fault
  episodes (onset, parameter, magnitude). v1.1: up to three non-overlapping episodes per
  engine (79 engines, 114 episodes, all 6 fault classes with ≥9 training instances).
- **Index.** Per-engine life, split, new-engine EGT margin, severity multipliers, drift channel.
- **Splits.** 70/10/20 engines (train/val/test); no engine in two splits (CI-checked).
- **Missing data.** ~4 % of snapshots lost (isolated + ACARS outage bursts), by design.

## Collection / generation process

Generated, not collected. Generator: `z(u,x) = baseline(u) · (1 + H(u)·x/100)` where the
baseline and influence-coefficient matrices come from a pyCycle CFM56-7B26 model calibrated
against EASA TCDS E.004 and ICAO EEDB measured data (report ch. 4). Health trajectories follow
literature degradation profiles with hierarchical engine-to-engine variability; sensors add
noise, quantization, drift and dropout per `conf/fault_catalog.yaml`. Fully deterministic from
the catalog and seed (`fleet.seed`); regenerate with `uv run python scripts/make_fleet.py`.

## Validation (gate H2 audits, report ch. 5)

- **Nonlinearity vs. full physics:** per-channel error medians 0.01–0.33 %, P95 ≤ 1.0 % against
  full pyCycle solves at 80 fleet-sampled states — at the sensor noise floor.
- **Difficulty:** trivial classifier (logistic on raw channels) scores 58.1 % on the gated
  acute-fault isolation task (< 60 % criterion). NOTE: this gate has now failed twice and
  been fixed physically both times — v1.0 first design at 81.6 % (chronic labels were an age
  proxy → acute episodes added), and v1.1 at 62.2 % (more episodes fed the trivial
  classifier too → magnitudes scaled ×0.75). Chronic labels remain a secondary,
  explicitly-not-gated signal.
- **Realism:** wash sawtooth (+2.8 °C mean EGTM recovery, 99.75 % positive), end-of-life
  deterioration signs correct in 100 % of engines, measured noise matches the catalog.

## Known limitations

Linearized generation (audited above); EGT is a station-4.5 proxy of the certified T49.5 with
its display shunt out of scope; two snapshot conditions with ambient/power scatter, not a full
mission mix; chronic labels correlate with age by nature; at most three acute episodes and one
sensor drift family per engine; N1/N2 absolute labels approximate (generic component maps).

## Distribution & license

Regenerable from the public repository (https://github.com/cedarcreeks/EHMbrAIn), MIT-licensed
code; the dataset itself (if redistributed as files) under CC-BY 4.0. Parquet files are not
committed (250 MB); the 48 KB ICM artifacts required for regeneration are versioned.

## Maintenance

Versioned with the repository. Changes to the catalog or generator bump the dataset version;
v1.0 was the H2-gated release (2026-07-03); v1.1 (2026-07-04, multi-episode + rescaled
magnitudes, re-audited) is the FROZEN evaluation dataset for phases F3–F7 (its hashes anchor
prereg-v1/v2). v2 (2026-07-05, F8/L2) regenerates the same trajectories through the
differentiable neural-twin emitter — nonlinear snapshot physics, 3–7× closer to full pyCycle
than v1's linearization (EGT median 0.04 % vs 0.31 %), difficulty gate re-passed at 58.0 %.
v2 lives at data/processed/fleet_v2/; v1.1 is never overwritten.
