# Pre-registration v1 — confirmatory evaluation of hypotheses H1–H5

**Frozen at git tag `prereg-v1` (2026-07-04).** After this tag: no changes to the dataset,
splits, hypotheses, operationalizations, tuning budget or statistical machinery. Any
post-freeze change requires a `prereg-v2` with explicit disclosure of what changed and why.
Refutation of any hypothesis is a reportable outcome, not a defect.

## 1. Frozen dataset

SynCFM56 **v1.1**, regenerated deterministically from `conf/fault_catalog.yaml`
(seed 20260703). SHA-256 prefixes at freeze:

| Artifact | sha256[:16] |
|---|---|
| `data/processed/fleet/snapshots.parquet` | `9c9b9bc86c9cf34e` |
| `data/processed/fleet/events.parquet` | `b1ca076933ec3fe8` |
| `data/processed/fleet/fleet_index.json` | `3901f217125c09e5` |
| `conf/fault_catalog.yaml` | `23e0d4d9abae3951` |

Splits: by engine, 70 train / 10 val / 20 test, as recorded in `fleet_index.json`
(CI-checked leakage-free). The 20 test engines are touched exactly once more: the single
confirmatory pass of §5.

## 2. Disclosure of prior test-set exposure

The test split was evaluated during exploratory v0/v1.0/v1.1 development (report chs. 6–7):
traditional and AI pipelines at engineering defaults, the hybrid ablation, the PCS study and
one detection threshold sweep (selection on val, one application to test). All published in
the report and git history. Consequences accepted: (a) the confirmatory evaluation uses the
tuned pipelines selected on train+val only; (b) no hypothesis threshold below was chosen to
fit an observed test number; (c) the v0 test numbers are labeled exploratory in the report
and will not be pooled with confirmatory results.

## 3. Hypotheses — exact operationalization

Common machinery: test statistics computed on the 20 test engines (or their 23 acute
episodes); paired tests as specified; Holm correction across the five hypothesis families at
α = 0.05; BCa bootstrap (10 000 resamples, resampling engines) for confidence intervals;
AI models trained with seeds {0, 1, 2} and metrics reported as mean across seeds (per-seed
values in the artifact).

**H1 — detection.** Operating point per family: threshold(s) selected on validation
maximizing episode recall subject to ≤ 1 false-alarm engine on val. Metrics on test:
episode recall (alarm in [onset, next-onset)), median detection delay (cycles), false-alarm
engines (alarm before first onset or on a clean engine). *Confirmed iff* AI episode recall
≥ traditional's AND AI median delay ≤ 0.8 × traditional's AND AI false alarms ≤
traditional's; paired per-episode detection outcomes compared with McNemar's exact test,
p < 0.05. (Amended from the plan's "at fixed recall 0.9": v0 showed neither family reaches
0.9 recall on v1.1 fault magnitudes; amendment made at freeze time, before any tuned run.)

**H2 — isolation on confusable faults.** Oracle-timed protocol: both families answer at
onset+500 (or the episode's end if earlier). Confusable subset: episodes with true parameter
in {hpc.eta, hpt.eta, hpt.flow}. *Confirmed iff* AI accuracy on the confusable subset ≥
traditional's + 10 percentage points AND McNemar exact p < 0.05 on the paired
correct/incorrect outcomes.

**H3 — RUL prognosis.** Predictions at 50/70/90 % of each test engine's life; both families
capped at 12 000 cycles. Metrics: RMSE and the NASA PHM08 score with d in units of 100
cycles (declared scaling). *Confirmed iff* AI improves BOTH metrics at ALL three life
fractions, Wilcoxon signed-rank paired by engine on absolute errors (pooled across
fractions), p < 0.05.

**H4 — physics-informed hybrid under data scarcity.** Stacking hybrid (Kalman-GPA channels)
vs pure GRU at 10/25/100 % of training engines, seeds {0, 1, 2}, engine-subset draws fixed
by seed. *Confirmed iff* hybrid test RMSE ≤ pure at 100 % AND strictly lower at 10 % and
25 % (mean across seeds), Wilcoxon on per-engine absolute errors at 10 %, p < 0.05. (v0
exploratory evidence points to refutation; recorded in report §7.4 before this freeze.)

**H5 — uncertainty calibration.** AI: split-conformal 90 % intervals calibrated on val.
Traditional: 90 % interval from the 5th–95th percentile band of Theil–Sen pairwise-slope
projections to margin exhaustion (spec in `trad/pipeline.py` at freeze). Metrics on the
pooled 60 test predictions (20 engines × 3 fractions): |empirical coverage − 0.90| and mean
interval half-width. *Confirmed iff* AI coverage error < traditional's AND AI half-width ≤
1.2 × traditional's.

## 4. Symmetric tuning budget

**50 Optuna trials per family** (TPE sampler, seed 0), objective evaluated on validation
only. Search spaces (full list in `scripts/tune_f5.py` at freeze):

- Traditional: Holt α/β, CUSUM k/h, gap-detector EWMAs and n-sigma, persistence k/n, Kalman
  q, WLS λ and prior σ, RUL window and smoothing.
- AI: GRU hidden/layers/lr/epochs/window/downsample, detector feature windows (short/long,
  the v0 structural gap lever), Mahalanobis percentile and persistence, classifier depth /
  learning rate / iterations / sample counts.

Selection metric per task, on val: detection = episode recall subject to FA ≤ 1; isolation =
confusable accuracy; RUL = RMSE pooled over fractions; H5 uses the H3-selected model.

## 5. Confirmatory procedure

1. Run both tuning campaigns (train+val only).
2. Freeze selected configurations (recorded in `data/processed/f5/selected_configs.json`).
3. **One** evaluation pass of every tuned pipeline on the 20 test engines.
4. Compute the H1–H5 verdicts with the machinery of §3.
5. Publish verdicts, effect sizes (Cliff's delta), CIs and per-seed detail in the report —
   confirmations and refutations alike.

## 6. Compute (norm N5)

All stages benchmarked on the reference machine (Apple M5, 10 cores, 16 GB, torch MPS);
timings merged into `data/processed/compute_times.json` and published in the report.
