# Pending work

Updated 2026-08-02. Report at **145 pp**, **44 tests** green, 0 undefined references,
tags through `prereg-v25`. Working tree clean, nothing running.

> **Read first:** the project's most-promoted result is now qualified everywhere it is quoted.
> F10's $\rho = 0.70$ never had a control. Decomposed by §sec:f21-port and §sec:f23-decoupled:
> **$\approx0.24$ is the matrix the bound shares with the estimator, $\approx0.46$ survives an
> estimator that never sees it, and neither clears significance** — ten health directions is
> too few for a rank test, and ten is fixed by the physics. The certificate is not shown to be
> dishonest; it is shown to be **unprovable by the test F10 chose**.

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

## 2. ~~N-CMAPSS~~ — DONE, and it found a defect in our own F10

Three lines, all in the report.

**L-EXT2 (`sec:f20-ncmapss`, prereg-v23) — the wall is not ours.** ICM estimated from
N-CMAPSS (condition 5.6, rank 10/10, 699 012 rows). Confusable pair **0.66°** on cockpit-class
channels and **20.92°** with full instrumentation, against our **1.30°** and **26.71°**. Both
the wall and its cure reproduce on a different engine, in different software, by other people.

**L-CERT2 (`sec:f21-port`, prereg-v24/25) — the floor ports, the certificate does not.**

- **F11 confirms externally**: irreducible share 0.68 / 0.72 / 0.82 against our 0.87 / 0.85 /
  0.88. Needs no influence matrix, so it is clean.
- **F10 does not port**: ρ = 0.842 on N-CMAPSS, but a column-shuffled matrix — no physics,
  same coupling — reaches 0.830. The correlation measures the shared matrix.
- **And the same control on our own F10**: real ρ = **0.697** (faithful to the published 0.70)
  against a null median **0.242**, p95 **0.722**, **p = 0.085**. Real signal well above the
  null median, never significant against it. **Under-powered, not refuted** — a rank test over
  ten directions cannot resolve it.

**Why it was missed:** F10 predates the control-arm discipline. Every later line has one
(H15.3's fired, H15.11's inverted a post-hoc, H15.8's validated a null), and in each case the
control changed the verdict.

**That was run** (`sec:f23-decoupled`, prereg-v26): a learned estimator that never sees
$\mathbf{H}$ collapses the null from 0.242 to **0.006**, confirming the coupling is gone, and
the certificate still ranks at $\rho = 0.455$ on all ten seeds — but $p = 0.100$. The design
worked; the statistic is the wall.

**What would actually settle it:** not more engines and not a better estimator, but a different
*statistic*. Compare certified precision against achieved error in **physical units** across
engines — thousands of paired observations instead of ten — which is regression, not ranking.
That needs the CRB's absolute scale to be trustworthy, and §sec:bt-honesty already shows it is
not (H10.2 refuted, repaired only by conformal). **Fix the scale first, then test magnitude.**
A study of its own.

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
- **Measure, do not estimate.** Wrong seven times across 2026-07-30/08-02: 75 min for what was
  3 h (twice); 25 min for what was 52; an E-core recommendation that measured 6.5× worse; 60 min
  for what was 112; "four shards will be faster", which measured ~8 % slower; and a download
  ETA of 14 h read off a 0.3 MB/s sample that was S3 warming up — it finished at 10 MB/s. A
  single-training benchmark understates ~50 % because it misses contention, and a transfer rate
  sampled in the first minute is not the transfer rate.
- **A pipeline swallows exit codes.** `curl ... | tail` returned 0 while curl had died with
  `(56) Recv failure` at 8.78 of 14.68 GB. The truncated archive would have surfaced three
  steps later, inside the analysis. Always verify a download against its `Content-Length`, and
  wrap long transfers in a resume loop with `--speed-limit`/`--speed-time` — `--retry` alone
  does not cover a mid-transfer stall.
- **Cross-run numbers are not comparable.** The `uni` attribution arm scored +0.181, +0.248 and
  +0.414 across three runs of a nominally identical configuration (per-seed sd 0.114). Only
  paired comparisons within a single run are reliable.
- **Every claim needs a null, not just a p-value.** F10's ρ = 0.70 looked decisive for months
  because nobody asked what a physics-free matrix would score. It scores 0.242 on average and
  0.722 at the 95th percentile. Where an estimator and its bound share a component, shuffle
  that component: it keeps the coupling and destroys the meaning.
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
