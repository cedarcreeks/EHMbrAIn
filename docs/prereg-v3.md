# Pre-registration v3 — F8/L6 confirmatory (frozen 2026-07-05, tag prereg-v3)

Re-opens hypothesis H4 (does physics injection help RUL under data scarcity?) on the
nonlinear SynCFM56 v2 fleet, where twin-residuals carry genuine information (the generator
is the surrogate, not the linear GPA the residual is computed against).

Dataset: SynCFM56 v2 (data/processed/fleet_v2/), same splits as v1.1 (70/10/20 by engine).
Thresholds frozen NOW, before any confirmatory run (lesson from H7.3: never calibrate
margins on a small dev sample).

## Mechanisms tested (both are the ones the F5 stacking hybrid never tried)

- **M2 twin-residual features.** Per snapshot, the linear WLS residual
  r = (I - H_ref M) Δz, where M is the regularized WLS operator at the engine's reference
  H — the part of the measurement the linear GPA cannot explain. On v2 this is nonzero and
  informative; on v1 it was near-zero by construction. The RUL GRU input is augmented from
  4 to 7 channels (4 deviations + 3 residuals).
- **M3 physics-constrained loss.** (Implemented if M2 is non-negative; otherwise reported
  as deferred with M2's result standing as the H4 update.)

## Protocol (identical machinery to F5 H4)

Pure GRU vs hybrid GRU at 10 %, 25 %, 100 % of training engines; seeds {0,1,2}; RUL RMSE on
the 20 test engines at 50/70/90 % life, capped 12 000 cycles. Confirmatory pass = one run
per configuration.

## Frozen decision rule

**H4-v2 CONFIRMED iff** hybrid mean-across-seeds test RMSE is:
(a) <= pure at 100 % data, AND
(b) strictly lower at BOTH 10 % and 25 %, AND
(c) Wilcoxon signed-rank (paired per test engine, abs error) p < 0.05 at the 10 % fraction.
Otherwise REFUTED. A partial pattern (e.g. helps only at 10 %) is reported as such but does
not confirm.

Output: data/processed/f8/h4_v2_verdict.json
