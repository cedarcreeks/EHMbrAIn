# Pre-registration v9 — L-H2 / L-H2b confirmatory (frozen 2026-07-05, tag prereg-v9)

Does breaking the confusable-isolation wall need real sensors (L-H2), and can a virtual
(model-predicted) sensor substitute (L-H2b)? Feasibility measured: on the ICM the
hpc.eta~hpt.eta signature angle goes 1.3deg (cockpit) -> 19.6deg (extended), above the 15deg
confusable threshold; a virtual PS3 (predicted from cockpit) keeps the rank at 3 (adds no
information). This confirms whether that geometry translates into isolation ACCURACY.

Dataset: SynCFM56 v1.1 (frozen), test split, confusable episodes (true param in
{hpc.eta, hpt.eta, hpt.flow}). Method: WLS-GPA nearest-signature isolation (the traditional
isolator, which reads the geometry directly), oracle-timed at onset+500. Three sensor
conditions:
- COCKPIT: N2, WF, EGT (the H2 baseline).
- VIRTUAL: cockpit + model-predicted P25/T25/PS3/T3, where the virtual channels are
  H_extra . x_hat_cockpit (x_hat the cockpit WLS estimate) -- a function of the cockpit
  measurements, adding no independent information.
- REAL: cockpit + measured P25/T25/PS3/T3 (the extended set).

## Frozen decision rule (honest either way)

- **H-H2.1 (real sensors break the wall) CONFIRMED iff** confusable isolation accuracy with
  REAL extended sensors exceeds cockpit by >= 25 percentage points, McNemar one-sided p<0.05.
- **H-H2.2 (virtual cannot substitute) CONFIRMED iff** VIRTUAL accuracy does not significantly
  exceed cockpit (McNemar p>0.05) while REAL does -- demonstrating the wall is informational,
  not algorithmic: a synthetic sensor derived from existing data cannot rescue it.

If real sensors do NOT lift accuracy despite the angle opening, the bottleneck is the
estimator/noise not the geometry -- reported honestly.

Output: data/processed/f8/wall_verdict.json
