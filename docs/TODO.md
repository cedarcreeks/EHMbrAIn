# Pending work

Updated 2026-08-02. Report at **146 pp**, **44 tests** green, 0 undefined references,
tags through `prereg-v27`. Working tree clean, nothing running.

> **The science is closed. Four documentation defects are not** — see section 0. None of
> them changes a number; all four are places where the repository misrepresents work that
> was actually done, which for this project is the expensive kind of error.

> **Read first:** the project's most-promoted result is qualified everywhere it is quoted, and
> after four tests its status is precise. Of F10's $\rho = 0.70$, **$\approx0.24$ is the matrix
> the bound shares with the estimator and $\approx0.46$ survives an estimator that never sees
> it** (§sec:f21-port, §sec:f23-decoupled). Testing magnitude instead of ranking is blocked too:
> the bound understates error $8.9\times$ with exponent $0.382$ — a *shape* error, so no single
> constant repairs it — and the powered statistic fails its own gate because the CRB varies only
> **1.15 %** between engines (§sec:f24-scale).
>
> **The certificate is not shown to be dishonest. It is shown to be unmeasurable *on this
> fleet*.** Both walls are properties of SynCFM56, not of the instrument — see item 1.

---

## 0. Documentation defects — OPEN, and they are what stands between here and "done"

Found by a full-repo consistency audit on 2026-08-02, after L-SCALE. All four are marked
**in situ** with a comment block at the exact site, so they cannot be lost. Total ≈ 1.5 h.

### D1 — the contributions list still reads as if C8 passed
**Site:** `paper/report/chapters/01-introduction.tex`, entry `C8` (marked in file).

It is the **last place in the repository** quoting $\rho = 0.70$ unqualified. Every other
site already carries the decomposition. The contributions list is what a reader meets
first, so this contradicts chapters 11 and 12 in the most visible position available.

Restate as a certificate whose honesty test is *under-powered on SynCFM56*, and point at
`sec:future-c8`. The $45\times$ acquisition figure **survives** — it is a ratio between
sensor configurations, and F24 disqualified only absolute magnitude — but needs a pointer.

### D2 — `make all` cannot rebuild chapter 11
**Sites:** `Makefile` (comment block before `.PHONY`) and
`paper/report/chapters/03-methodology.tex`, `sec:replication` (comment after the
extension table).

Thirteen drivers have no target: `f_uq_reattribution`, `f13`–`f24`. That is ~35 of 146
pages and the entire adversarial audit. Two statements in `sec:replication` are currently
**false**: that it gives "the complete replication path", and that `make all` "runs
everything below". Ordering constraints a naive target would break are written into the
Makefile comment (f23 before f24; f13 before both; f20 before f21/f22; F18 needs *two*
runs, `gru` and `lstm`).

Also: the "657 s / about eleven minutes" cost figure covers only the wired stages and will
need a scope qualifier — the unwired thirteen are hours (F18 alone measured 112.2 min).

### D3 — the pre-registration looks abandoned at v14
**Site:** `docs/prereg-index.md` (written; it records the gap and indexes all 27 tags).

`docs/` holds `prereg-v1.md`…`v14.md`, no `v4`, nothing after `v14`. The natural reading is
that the discipline was dropped **exactly where the adversarial audit begins**. Wrong, but
the repository supports it. From `v15` the pre-registration is the driver docstring
(32–49 lines: hypothesis, design, numeric gate, meaning of a null) sealed by an annotated
tag. Arguably *stronger* — `MIN_CV = 0.05` as a branching constant cannot be quietly moved.

**Still to do:** a sentence in `sec:safeguards`/`sec:replication` stating the medium change
and pointing at the index; and closing the `v4` hole (content is in `docs/f10-proposal.md`).
Of the four defects this one has the worst optics-to-effort ratio — fix it first.

### D4 — the OEM set stops at F23 (minor, judgement call)
**Site:** `paper/oem-brief/oem-brief.tex` (marked); same wording in `oem-slides.tex` and
`guia-presentacion-oem.tex`.

Not wrong — "does not clear a physics-free null" is still true and still the operative
sentence. Incomplete: F24 sharpens the status to "unmeasurable *on this fleet*, and here is
the fleet that would measure it". Deliberately left open: a named next experiment plays
better with a sceptic than an open caveat, but this is a 2-page brief, not an audit trail.

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

**And the magnitude route was run too** (`sec:f24-scale`, prereg-v27), which was supposed to be
the way past that wall. Two findings, both negative and both useful:

- Calibrating on validation gives $\log|\mathrm{err}| = -2.188 + 0.382 \log \mathrm{CRB}$. The
  bound understates achievable error $8.9\times$ — expected for a bound on *unbiased* estimators
  — but the exponent is **0.382, not 1**, so the distortion is a **shape** error and no single
  constant repairs it. After a two-parameter fit, 90 % of test errors still sit within a factor
  of **5.6**.
- The powered within-direction test (n = 200 instead of 10) **fails its pre-declared gate**: the
  CRB varies only **1.15 %** between engines, because every engine in SynCFM56 flies a
  near-identical N1 profile. It returned $p = 0.041$ and $p = 0.009$ and **those are not
  reported as evidence** — with a predictor that moves 1 %, that is what n = 200 manufactures.

---

## 3. The one experiment that would settle C8 — a mission-diverse fleet

Both walls are properties of the fleet, not of the certificate:

- the **ranking** statistic is capped at ten health directions, fixed by the physics;
- the **magnitude** statistic needs the bound to vary between engines, and it varies 1.15 %.

**Regenerate SynCFM56 with deliberate mission diversity** — route mixes different enough that
engines earn materially different bounds. Then certified precision can be regressed against
achieved error across engines, direction effects differenced out, with a predictor that
actually moves. Generator change plus a re-run, not new theory.

The leverage is known to exist: §ch:tomography moved separability by **79 %** by changing a
single report condition, so operating history has first-order influence on this geometry.

This is the only outstanding experiment that would resolve the project's most-promoted claim,
and it is now the top of the queue.

---

## 4. Remaining plan lines, each a study of its own

| Line | What | Note |
|---|---|---|
| L3 | Real station EGT (T49.5 channel + display shunt as a known bias) | Does the confusability map change with the real station? |
| L10 | Calibrate map *shape* against the full EEDB curve | How much map shape is recoverable from public data alone? |
| M3 | Physics-constrained loss — the third H4 mechanism | Deferred behind two H4 refutations; needs its own pre-registration against that record |
| C5 | PCS evaluated on the competent F7 isolation model | The validation the metric never got |
| — | Public release of SynCFM56 (v1.1 frozen + v2 nonlinear) with a DOI | |

---

## 5. Standing engineering notes

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
