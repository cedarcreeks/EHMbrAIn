# Pre-registration v5 — F8/L9 confirmatory (frozen 2026-07-05, tag prereg-v5)

Validates contribution C5 (the Physics-Consistency Score, PCS). The v0 PCS was a null result
on a WEAK cockpit classifier. Question: when a classifier genuinely reasons physically, does
PCS reflect it? If PCS cannot separate a physical reasoner from a random one, the metric is
useless; if it can, C5 is validated.

Dataset: SynCFM56 v1.1 (frozen), test split, oracle-timed acute episodes.

PCS = | cos( H_S^+ m_shap , e_true ) |, where m_shap is the step-block SHAP attribution of the
predicted class, H_S the ICM rows for the classifier's sensor set S, e_true the true fault's
unit direction. Computed for three classifiers:
- COMPETENT: HistGB on EXTENDED sensors (N2,WF,EGT,P25,T25,PS3,T3) -- where the ICM makes
  faults separable (F10 showed HPC efficiency 45x more identifiable with station probes).
- CONFUSED: HistGB on cockpit sensors (N2,WF,EGT) -- the v0 setting.
- CONTROL: cockpit HistGB trained on SHUFFLED labels (reasons about nothing physical).

## Frozen decision rule

- **H9.1 (PCS validity) CONFIRMED iff** mean PCS(competent) > mean PCS(control) AND
  Mann-Whitney one-sided p < 0.05 (competent vs control). Ordering competent > confused >
  control is the expected pattern, reported.
- **H9.2 (PCS tracks correctness) CONFIRMED iff** for the competent classifier, mean PCS on
  correctly-classified episodes > mean PCS on incorrect ones.

Output: data/processed/f8/pcs_validation.json
