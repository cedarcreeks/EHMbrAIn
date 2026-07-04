# Pre-registration v2 — F7 confirmatory (frozen 2026-07-04, tag prereg-v2)

Same dataset freeze as prereg-v1 (SynCFM56 v1.1, hashes therein). Development used a
15-engine holdout FROM TRAIN (dev split); val used only for conformal calibration; the 23
test episodes are evaluated once, below thresholds frozen now.

Disclosed exposure: test episodes were previously evaluated in F5 (different methods).
The F7 learner and stacked-WLS baseline have NEVER seen test.

Methods frozen at this commit: scripts/f7_learner.py (block-mean windows DS=10;
stacked-WLS lam=2, thr=0.25; GRU-over-projections hidden 64, 80 epochs, class-weighted CE,
standardized inputs; APS conformal calibrated on val at 90 %).

- **H7.2' (fusion beats classical stacking).** Short window (300 cy), test episodes:
  GRU accuracy >= stacked-WLS + 10 pp AND one-sided McNemar p < 0.05.
- **H7.3 (drift robustness — core claim).** Long window (2000 cy): stacked-WLS degrades
  >= 20 pp vs its short-window accuracy; GRU degrades <= 10 pp. Both conditions required.
- **H7.4' (calibrated ambiguity).** APS sets on test: coverage >= 0.85 AND median set size
  <= 4 overall AND median size(fundamental hpc.eta) > median size(other classes) at the
  short window (physics-tracking signaling).

Seeds: learner seed 0 (single seed disclosed as a limitation; multi-seed in the F7 report
if any verdict is marginal). Verdicts to data/processed/f7/verdicts_f7.json.
