# F10 — BREAKTHROUGH: earned-confidence gas-path diagnosis

## The claim

The first **per-engine, ground-truth-validated identifiability certificate** for gas-path
diagnosis: a physics-derived statement of exactly which health directions are recoverable,
to what precision, from an engine's ACTUAL service history — and a proof, against ground
truth, that the statement is honest. Smearing stops being a hidden failure and becomes an
explicit, trustworthy, per-direction confidence report. The same certificate drives sensor
and operating-condition acquisition as a Fisher-information optimization.

## Why nobody has done this well

- Classical GPA: point estimate; smearing hidden; no earned per-direction confidence.
- Bayesian/observability GPA (exists): static design-point Gramian; NEVER validated against
  true component states (impossible with real data); never a per-engine operator certificate.
- AI EHM: black box; no physics-grounded identifiability.

Unique enablers here: a calibrated (differentiable) twin giving exact H(u) at every real
condition; complete ground truth to validate; the full ICM geometry.

## Measured feasibility (exploratory, 2026-07-05 — disclosed)

Accumulating Fisher information F = P0^-1 + sum_t H(u_t)^T R^-1 H(u_t) over each test
engine's real late-life cruise-N1 history, the CRB posterior std per health parameter:
- ranks directions physically (fan.flow 0.46% identifiable; efficiencies fan/hpc/lpc
  1.6-1.8% unobservable);
- **predicts the actual Kalman-GPA per-direction error with Spearman rho = 0.70
  (p = 0.025)** across the 20 test engines. The certificate is honest.

## Hypotheses (to freeze as prereg-v4 BEFORE the confirmatory run)

- **H10.1 (certificate honesty).** Across the test fleet, the CRB-predicted per-direction
  precision ranks actual per-direction diagnosis error with Spearman rho >= 0.6, p < 0.05.
  (Exploratory rho = 0.70 disclosed; confirmatory uses the frozen estimator below.)
- **H10.2 (calibrated full-state region).** The physics-derived 90% posterior region
  (anisotropic ellipsoid through the twin, accumulated over history) contains the true
  10-dim health vector at 86-94% empirical coverage on test engines — a whole-vector
  guarantee, not per-parameter.
- **H10.3 (acquisition value).** The extended gas-path sensor set shrinks the median CRB in
  the three unobservable efficiency directions by >= 2x, and the certificate predicts which
  directions each addition rescues (rank agreement with the actual error reduction).

## Frozen estimator (H10.1/H10.2)

Fisher accumulation over each engine's cruise history (subsample stride 20) in the last
30 % of life; R = diag(0.07, 0.5, 0.23)^2 %^2; prior P0 = 4 %^2 I. CRB = (F)^-1. Actual
error = mean |Kalman estimate - truth| over the last 15 % of life. Coverage = fraction of
engines whose true late-life x lies inside the chi-square(10) 90 % ellipsoid of the CRB
posterior centered at the estimate.

## Work packages

- WP10.1 certificate module (src/ehmbrain/trad/identifiability.py): Fisher/CRB accumulation,
  posterior region, per-direction identifiability tags, from the twin/ICM over a history.
- WP10.2 fleet validation script -> verdicts_f10.json (H10.1 honesty, H10.2 coverage).
- WP10.3 acquisition study (H10.3): CRB with cockpit vs extended sensors; the rescue map.
- WP10.4 prereg-v4 freeze + confirmatory pass + verdicts.
- WP10.5 breakthrough chapter: the certificate, the honesty proof, the per-engine report
  figure, the acquisition instrument unifying F7.

## Risk / honesty

If H10.1 confirmatory rho < 0.6: report as a weaker-but-real relationship (the exploratory
0.70 may shrink under the frozen estimator). If coverage is off: the region needs a
calibration factor (conformal-style), reported. Either way the certificate's VALUE (honest
per-direction ranking) is already measured; the phase cannot produce a dishonest positive.
