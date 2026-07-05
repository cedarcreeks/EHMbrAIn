# Pre-registration v7 — F8/L4 confirmatory (frozen 2026-07-05, tag prereg-v7)

Novel, actionable task: estimate the RECOVERABLE FRACTION of an engine's lost EGT margin --
the share attributable to fouling (which a wash restores) versus permanent wear (erosion,
tip-clearance, hot-section, FOD, acute). If estimable from the deviation trajectory, an
operator can predict "a wash will recover X % of your lost margin" before washing.

Dataset: SynCFM56 v1.1 (frozen). Ground truth = per-mechanism contributions replayed
deterministically from each engine's seed (generate_engine); recoverable fraction at cycle n
= (H_EGT . x_fouling) / (H_EGT . x_total), clipped [0,1], with H_EGT the takeoff-hot ICM EGT
row. Features = trailing-window statistics (level, slope, curvature) of the measured cockpit
deviations already in the fleet (the past wash sawtooth is a legitimate observable).

Model: HistGB regressor, train split; evaluated once on test. Baseline = predict the
train-set mean recoverable fraction.

## Frozen decision rule

- **H4L.1 (estimable) CONFIRMED iff** test R^2 > 0.30 AND Spearman(pred, truth) > 0.5,
  p < 0.05. Otherwise REFUTED (the recoverable fraction cannot be read from trajectory shape
  -- you must wash to know), which is itself a reportable operational finding.

Output: data/processed/f8/recoverable_verdict.json
