# Pre-registration v13 — L-ICM: is the observability verdict robust to calibration? (frozen 2026-07-17, tag prereg-v13)

The whole benchmark leans on one untested assertion, stated in the report's model chapter:
"None of these limitations affects the validity of the AI-versus-traditional comparison,
which rests on the model's sensitivity structure (the ICM), not on its absolute values."
That is a claim about the ICM's *robustness*, and it has never been measured. This line
measures it — the classical GPA sensitivity study the project owes its own premise.

## What is perturbed

The calibration knobs, each moved within the tolerance its own contract declares
(conf/cfm56_7b_targets.yaml, appendix A1 decision register), one at a time (OAT) plus a
joint worst-case draw:

| Knob | Nominal | Perturbation | Why that size |
|---|---|---|---|
| `DESIGN.hpc.PR` | 9.35 | +-3 % | calibrated to the measured SLS OPR 27.61, contract tol 3 % |
| `DESIGN.HP_Nmech` | 13940 rpm | +-3 % | map speed reference; anchored to the TCDS redline, not measured |
| `DESIGN.Fn_DES` | 5480 lbf | +-3 % | class [A], status TO_VERIFY, contract tol 3 % |
| `DESIGN.T4_MAX` | 2857 degR | +-3 % | class [D], derived from the takeoff anchor |
| `DESIGN.splitter.BPR` | 5.1 | +-0.2 | EEDB tol_abs |
| all five component `eff` | 0.87-0.92 | +-2 % rel. | generic-map quality, the declared limitation (iv) |
| joint | — | all of the above at their adverse sign | worst case, not average case |

Separately, the finite-difference step of the ICM itself (class [C], reasoned but never
swept): step in {0.25 %, 0.5 % (nominal), 1.0 %}.

## What is measured

Cockpit ICM (N2, WF, EGT) at cruise and takeoff-hot, for every perturbation:
numerical rank; minimum signature angle; the number and identity of confusable pairs
(< 15 deg); condition number; and the per-column direction shift — the angle between each
perturbed fault signature and its nominal counterpart.

## Frozen decision rule (honest either way)

- **H-ICM.1 (the verdict is calibration-invariant) CONFIRMED iff**, for **every** perturbation
  in the table: cockpit rank stays 3, the minimum signature angle stays below the project's
  own 15 deg confusability threshold, and the most-confusable pair remains
  `hpc.eta ~ hpt.eta`.
- **H-ICM.2 (signatures barely move) CONFIRMED iff** the median per-column direction shift
  across all perturbations is <= 5 deg **and** the confusable-pair set is unchanged in
  >= 90 % of perturbations. Rationale for 5 deg: it must be small against the 15 deg line
  that defines the H2 subset, otherwise calibration error could reclassify a pair.
- **H-ICM.3 (the +-0.5 % step is in the linear regime) CONFIRMED iff** ICM columns at
  0.25 % and 1.0 % differ from the 0.5 % baseline by < 2 deg in direction and < 5 % in
  magnitude.

A refutation is the more interesting outcome and is reported as such: it would mean the
observability geometry — and therefore the H2 wall, the confusable subset, and the F10
certificate that inverts the same matrix — inherits the twin's calibration uncertainty,
and every downstream claim would need that caveat attached. No threshold moves after the
run.

Output: `data/processed/icm/icm_robustness.json`; driver `scripts/f_icm_robustness.py`.
