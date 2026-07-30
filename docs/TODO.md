# Pending work

Updated 2026-07-30. Report at **140 pp**, **44 tests** green, 0 undefined references,
`origin/main` at `35f85c6`, tags through `prereg-v22`. Working tree clean, nothing running.

---

## 1. Open defect — a claim in the report overreaches

**L-BIDIR's keyidea generalises a GRU experiment to a statement about LSTM.**
`sec:f18-bidir` says *"the published Bi-LSTM's edge over an LSTM is capacity, not direction"*,
but `scripts/f18_bidirectionality.py:92` uses `nn.GRU`. The paper's cell is LSTM, and L-EXT
replicated it with `nn.LSTM` — the mechanism test did not.

The parameter-doubling argument is cell-agnostic in principle, but "in principle" is not a
measurement, and this is the same species of error the chapter exists to correct.

**Fix by running it, not by hedging.** Add a `cell` parameter, repeat the three arms
(`uni`, `bi_equal`, `bi_double`) with LSTM cells, both tasks, same ten seeds. 60 trainings,
sharded, ~55 min.

- matches GRU → the finding is cell-agnostic and the sentence stands
- does not match → restrict the claim to GRU and rewrite; interesting in itself

Until then the sentence should be read as untested for LSTM.

---

## 2. N-CMAPSS DS02 (plan line L8) — feasibility gate first

Not on disk; only FD001 (5.5 MB) is. So this is data acquisition plus a new pipeline, not a
re-run.

**The plan undersells why it matters.** N-CMAPSS ships the *true health parameters* in its `T`
group — the only external dataset that does. That means the F10 certificate and the F11 floor,
the two results this project could only validate against its own generator, might be
recomputable on data that did not come from our simulator. That attacks the circularity
objection directly, and it is a better reason than L8's stated one ("transfer with real
per-cycle flight physics").

**The obstacle:** F10 needs an influence coefficient matrix and N-CMAPSS does not ship one.
Recovering sensitivities numerically, or from their model description, is research rather than
a port — days, and possibly not feasible.

**Gate, ~2 h, before committing anything:**
1. Download DS02; verify actual size and schema. *Do not trust remembered figures* — roughly
   1.2 GB and about nine units is a belief, not a measurement.
2. Inspect whether `T` is usable as truth in the form F10/F11 need.
3. Check whether anything ICM-equivalent is recoverable.

Gate passes → the most valuable line left in the plan. Gate fails → L8 reduces to re-running
C7 on more data to confirm a ranking FD001 already confirmed, and is not worth 1–2 days.

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
- **MPS is not the bottleneck** for these models: 1.43 s/epoch against 1.56 on a single CPU
  thread, because 1507 samples at batch 128 is twelve batches per epoch. Eight CPU shards beat
  one MPS process by roughly 3×.
- **Measure, do not estimate.** Time estimates were wrong three times on 2026-07-30 (75 min
  for what was 3 h, twice; then 25 min for what was 52). Timing one epoch costs two minutes.
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
