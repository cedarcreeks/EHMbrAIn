# Pre-registration v14 — the noise axis of the determinability map, C6 (frozen 2026-07-17, tag prereg-v14)

Contribution C6 promised a **sensors x noise x data-size** determinability map. Two of the
three axes were delivered: sensors (cockpit vs extended, throughout the document) and data
size (the H4 ablation at 10/25/100 % of training engines). The **noise axis was never swept**
-- every result in the document lives at one single sensor-quality setting, the catalog's [L]
literature values. This line closes it, and states in advance what would count as the map
holding or breaking.

## The sweep

The fleet is regenerated with every sensor sigma in `conf/fault_catalog.yaml` scaled by
`m` in {0.5, 1, 2, 4}. Everything else is held byte-identical: the same seed, the same health
trajectories, the same engine splits, the same fault magnitudes and onsets. Only measurement
quality moves, so any change in a score is attributable to it alone. `m = 1` reproduces the
frozen v1.1 fleet.

Both families are evaluated at their **F5-tuned configurations, unchanged** (the per-task
winners of `data/processed/f5/selected_*.json`), on the test split. This is deliberate and is
the honest reading of the result: the map measures how the systems an operator would actually
have deployed degrade as sensors get worse or better, not the best achievable score at each
noise level. Re-tuning per level would cost 8 more Optuna campaigns and would answer a
different question; the choice is disclosed rather than hidden.

## Frozen decision rule (honest either way)

- **H-N.1 (the prognosis advantage is noise-robust) CONFIRMED iff** the AI's RUL RMSE at
  90 % life is lower than the traditional pipeline's at **every** noise level in the sweep.
  If the AI's lead disappears at 2x or 4x, H3's operational claim needs a sensor-quality
  precondition attached, and we say so.
- **H-N.2 (the isolation wall is noise-invariant) CONFIRMED iff** the absolute gap between the
  two families' confusable-episode isolation accuracy stays <= 10 percentage points at every
  noise level -- i.e. neither family pulls ahead as sensors change. This is the noise-axis
  version of the H2 refutation.
- **H-N.3 (the wall is geometric, not noise-driven) CONFIRMED iff** at the **halved**-noise
  level (m = 0.5) neither family's confusable isolation accuracy exceeds 50 %. Rationale: if
  merely buying quieter sensors of the same *set* lifted confusable isolation past a coin
  flip, the H2 conclusion ("buy different sensors, not better models") would need rewriting as
  "buy better sensors of the same kind" -- a materially different purchase order. The
  signature-angle geometry predicts it will not.

A refutation of any of the three is the more useful outcome and is reported as such. No
threshold moves after the run.

Output: `data/processed/f5/noise_sweep.json`; driver `scripts/f_noise_sweep.py`.
