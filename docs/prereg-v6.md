# Pre-registration v6 — F8/L7 confirmatory (frozen 2026-07-05, tag prereg-v6)

Resolves Case C (the drifting thermocouple). A plain Kalman-GPA attributes a drifting EGT
sensor to a phantom hot-section fault. The classical remedy is an AUGMENTED-STATE Kalman that
estimates the sensor bias b alongside health x (the S.b term of report eq. 2.4). Question:
does it work on the cockpit (rank-3) sensor set, or is b too confounded with health?

Dataset: SynCFM56 v1.1 (frozen), test split, engines whose drift_channel is EGT (no acute
fault, so the true health is chronic-only and any large health estimate on the hot section is
a phantom). Ground-truth bias = smoothed (measured - true) EGT deviation [%].

Augmented Kalman: state [x(10); b(1)], random walk (q_x = 2e-4, q_b frozen at 1e-5), obs
dz = [H(u) | e_EGT] [x; b] + v. Compared to the plain 10-state Kalman.

## Frozen decision rule (honest either way)

- **H7L.1 (bias recovery) CONFIRMED iff** across drifting-EGT test engines, the augmented
  estimate of b correlates with the true bias with Spearman rho >= 0.6, p < 0.05.
- **H7L.2 (phantom suppression) CONFIRMED iff** the augmented Kalman's late-life spurious
  health magnitude on the hot-section/HPT parameters is lower than the plain Kalman's
  (median across drifting engines).

If both fail: the honest finding is that the cockpit set cannot separate drift from health
even with the classical remedy -- reinforcing the sensor-hygiene message of Case C.

Output: data/processed/f8/drift_verdict.json
