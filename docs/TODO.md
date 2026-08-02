# Pending work

Updated 2026-07-30. Report at **141 pp**, **44 tests** green, 0 undefined references,
`origin/main` at `35f85c6`, tags through `prereg-v22`. Working tree clean, nothing running.

---

## 1. ~~Open defect~~ — RESOLVED, and it changed the claim

L-BIDIR generalised a GRU experiment to a statement about LSTM. Repeated with LSTM cells
(`F18_CELL=lstm`), everything else identical:

| task | arm | GRU | LSTM |
|---|---|---|---|
| attribution ($R^2$) | uni | +0.181 | **−0.220** |
| | bi_equal | +0.230 | −0.055 |
| | bi_double | +0.167 | −0.312 |
| prognosis (cy) | uni | 1490.9 | **1251.3** |
| | bi_equal | 1475.6 | 1417.7 (+166 worse) |
| | bi_double | **1396.7** (−94, p=0.016) | 1380.5 (+129 worse) |

**The finding does not transfer.** On GRU the doubled-parameter arm improved prognosis; on
LSTM it worsens it. Sign reversed. The claim is now restricted to GRU in the report.

**And the LSTM comparison cannot settle it either.** Hyperparameters came from F13's Optuna
search *for a GRU*, and all three LSTM attribution arms return negative $R^2$ — a model not
converging, not a cell that is bad at the task. So it measures tuning, not architecture, which
is exactly this chapter's objection to a fifteen-way published ranking.

**If this is ever to become a claim about architecture** it needs a symmetric tuning budget per
cell — an Optuna run for LSTM matching F13's for GRU, then the three arms again. That is the
only version worth doing, and it is a study, not a re-run.

---

## 2. N-CMAPSS — gate RUN, passed; see `docs/f20-ncmapss-gate.md`

Answered on documentation alone, without downloading (path A). Full write-up in that file;
the short version:

- **14.68 GB**, one zip for DS01–DS08, no per-subset download. My "roughly 1.2 GB" was wrong by
  twelve times. The report's existing reason for substituting FD001 is understated, not
  overstated, and stands.
- **No Jacobian or influence coefficients are published anywhere.** Everyone treats the C-MAPSS
  model as a black box. So F10 cannot simply be ported — the original worry was right.
- **But the ICM does not have to come from NASA.** Each sample pairs `θ` with `x_s` at a known
  `w`, so regressing sensor deviations on `θ` at matched `w` estimates the influence matrix
  from the data. Ordinary way to get a Jacobian from a closed model.
- **Caveat to gate on:** degradation is coordinated within a unit, so `θ` columns correlate.
  The seven failure modes are spread across subsets, so pooling buys the variation — check
  conditioning/rank before using any estimated ICM, and if it fails, that is the finding.
- Also corrected: N-CMAPSS ships `θ` in the **repository** distribution (which is where DS02
  lives), not in the **Challenge** subset, which strips it to `W`, `X_s`, `Y`, `A`.

**The by-product is worth more than the original goal.** An estimated ICM allows measuring the
confusable-pair angle on a completely independent engine model — different engine, different
authors, 14 sensors instead of 3. That tests whether the $1.3^{\circ}$ wall is a fact about
turbofans or an artifact of our deck, which is the strongest available answer to the
circularity objection. Compare on the shared sensor subset, report the full 14-channel angle
alongside as the analogue of our extended set (already $25$–$30^{\circ}$, §sec:gate-t).

**Order:** download outside the reproducible pipeline → estimate ICM and gate on conditioning →
angle comparison → only then consider porting F10/F11. Roughly a day to the ICM, hours for the
angle, open-ended after that.

---

## 3. Remaining plan lines, each a study of its own

| Line | What | Note |
|---|---|---|
| L3 | Real station EGT (T49.5 channel + display shunt as a known bias) | Does the confusability map change with the real station? |
| L10 | Calibrate map *shape* against the full EEDB curve | How much map shape is recoverable from public data alone? |
| M3 | Physics-constrained loss — the third H4 mechanism | Deferred behind two H4 refutations; needs its own pre-registration against that record |
| C5 | PCS evaluated on the competent F7 isolation model | The validation the metric never got |
| — | Public release of SynCFM56 (v1.1 frozen + v2 nonlinear) with a DOI | |

---

## 4. Standing engineering notes

- **Parallel over serial, always.** Independent jobs go to shard subprocesses.
  `multiprocessing` does not work here: `spawn` hangs re-importing the module, `fork`
  deadlocks on the parent's BLAS/torch threads. Pattern that works:
  `f18_bidirectionality.py` / `f19_certificate_isolated.py` — parent builds and caches to
  `npz`, launches `--shard i n` subprocesses, collects.
- **Shard count: 4 (= performance cores) is COOLER, not faster.** This machine is 4P + 6E.
  Measured on an LSTM training, single thread: P-core 1.96 s/epoch, E-core (`taskpolicy -b`)
  12.83 — **6.5× slower**, so eight shards park half the work on slow cores. But measured
  end-to-end, four shards took **112.2 min** for 60 LSTM trainings where eight would have taken
  roughly 104 (75.7 min for GRU × 1.37 for LSTM's extra gate). So the straggler reasoning was
  incomplete: E-cores are slow but still add net throughput. Four shards runs at 410 % CPU and
  load 5.5 against 691 % and 9.7 — **buy it for heat and responsiveness, not for speed**.
- **To run the laptop cooler, cut shards — do not use `taskpolicy -b`.** E-cores would put 60
  trainings at 4.4 h. `nice` does not reduce heat when nothing competes.
- **MPS is not the bottleneck** for these models: 1.43 s/epoch against 1.56 on a single CPU
  thread, because 1507 samples at batch 128 is twelve batches per epoch.
- **Measure, do not estimate.** Time estimates were wrong six times on 2026-07-30: 75 min for
  what was 3 h (twice); 25 min for what was 52; an E-core recommendation that measured 6.5×
  worse; 60 min for what was 112; and "four shards will be faster", which measured ~8% slower.
  A single-training benchmark understates ~50 % because it misses contention. Timing one epoch
  costs two minutes and changed the decision every time.
- **Cross-run numbers are not comparable.** The `uni` attribution arm scored +0.181, +0.248 and
  +0.414 across three runs of a nominally identical configuration (per-seed sd 0.114). Only
  paired comparisons within a single run are reliable.
- **Bidirectional pooling.** Read forward-at-last concatenated with backward-at-first. `o[:, -1]`
  gives the backward pass a single sample and silently cripples the model — it cost two void
  runs (F18, F15).

---

## Done this session, for context

In the report: **L-MECH** (mechanism attribution from trajectory, 3/5 recoverable),
**L-INST** (instrument vs engine, refuted, fraud check passes), **L-EXT** (published Bi-LSTM
replicated; 5/10 runs collapse; ranking inside seed noise), **Gate T** (transients do not open
the confusable angle via operating points; the dynamic requirement is priced), **L-HYB**
(physics injection is task-dependent — fails for prognosis, helps attribution), **L-BIDIR**
(the edge is capacity, not direction — see defect 1), **L-CERT** (the certificate is right about
what it certifies and wrong as a filter).

Withdrawn with reasons on record: F12 (tautological gate), the ambiguity clock (its own stated
risk materialised), K3 and the KPI attribution channel (degenerate — `hot_section` dominates
100/100 engines, now a declared SynCFM56 limitation).

Also added: `docs/safety-case-boundaries.md` with a summary section in the conclusions,
`tests/test_time_axis.py` (6 tests guarding the time-axis rule), and the OEM briefing set
(`paper/oem-brief/`: 2-page brief, 17-slide deck, 8-page Spanish speaker guide).
