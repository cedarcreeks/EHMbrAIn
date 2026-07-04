# F7 proposal — Operating-point tomography: learned opportunistic MOPA

**The idea in one line.** The H2 verdict said the confusable-fault wall is informational
("buy sensors"). But the influence matrix H(u) *changes with the flight condition u* — so a
sequence of ordinary in-service snapshots at scattered conditions (hot day, derate, altitude)
is a set of *different projections of the same health state*: a tomography problem. Teach a
model to fuse those projections and part of the wall falls **without new hardware**.

## Feasibility (already measured on our ICM grid, 2026-07-04)

Stacked cockpit ICM `[H(u1);…;H(uk)]`, cockpit channels only:

| Stack | rank /10 | η_HPC~η_HPT | η_HPT~Γ_HPT | η_fan~η_LPT |
|---|---|---|---|---|
| cruise only | 3 | 1.32° | 6.02° | 4.32° |
| + low-power cruise | 6 | 1.60° | 11.13° | 4.41° |
| all 6 grid points | **10** | 1.66° | **13.07°** | 4.47° |

→ Full-rank recovery; some pairs' separation ~2×; and a **map of which ambiguities are
u-breakable and which are fundamental** (η_HPC~η_HPT barely moves — its signatures co-vary
with u). Both halves are knowledge.

## Novelty positioning (to be hardened in WP7.0)

Multi-operating-point analysis (MOPA) exists classically: deliberately selected steady
points, constant-health assumption, classical estimation (Pinelli & Spina line; "artificial
operating points"). The open gaps this phase attacks:

1. **Opportunistic** u-diversity — the ambient/derate scatter an airline gets for free,
   not scheduled test points.
2. **Health drifts during the fusion window** in service — classical MOPA's constant-health
   assumption breaks over weeks of snapshots; a learned temporal model can separate slow
   drift from a fault step *while* fusing projections. This is the core technical claim.
3. **Ground-truth adjudication** — nobody can measure observability recovery against true
   component states; SynCFM56 can.
4. **Calibrated ambiguity output** — conformal *isolation sets* ("{HPC, HPT} at 90 %")
   sized by the physics: small sets on u-breakable pairs, honestly larger on fundamental
   ones. Ambiguity becomes a measured deliverable, not a failure mode.

## Hypotheses (to be frozen as prereg-v2 before confirmatory runs)

- **H7.1 (physics).** Over the fleet's realistic u-scatter, the stacked linear estimator
  restores identifiable rank ≥ 8/10 and improves the signature angle of u-breakable pairs
  by ≥ 2× vs single-point. (Analytical half already holds on the grid; the fleet-scatter
  version must hold with realistic, non-designed u draws.)
- **H7.2 (learning).** A u-aware sequence model (input: deviation ⊕ operating condition per
  snapshot) beats BOTH the snapshot AI and the classical stacked-WLS/MOPA baseline on
  isolation of u-breakable confusable episodes by ≥ 15 pp (McNemar, Holm).
- **H7.3 (drift robustness — the differentiator).** With health drifting at chronic rates
  inside the fusion window, the learned model's isolation degrades ≤ 5 pp while
  constant-health stacked-WLS degrades ≥ 15 pp.
- **H7.4 (calibrated ambiguity).** Conformal isolation sets achieve ≥ 88 % empirical
  coverage with median set size ≤ 2 on u-breakable pairs, and signal fundamental pairs with
  systematically larger sets.

## Work packages

- **WP7.0** Literature hardening: MOPA line, artificial-operating-point methods, any
  learned multi-point fusion. Kill/adjust claims accordingly. *(gate: novelty memo)*
- **WP7.1** u-scatter audit + (if needed) enrich the generator's per-flight condition draw;
  record u alongside every snapshot (already present: dTs, N1 command).
- **WP7.2** Classical baseline: stacked-WLS MOPA adapted to opportunistic windows — the
  fair straw-man-proof comparator. Fleet-scatter H7.1 check.
- **WP7.3** The learner: sequence net over (Δz_t, u_t) windows → health/fault posterior;
  same budgeted tuning discipline as F5.
- **WP7.4** Conformal isolation sets on top of both.
- **WP7.5** prereg-v2 freeze → confirmatory pass → verdicts (same machinery as F5).
- **WP7.6** Report chapter + u-breakability map figure (the headline artifact).

## Cost & risks

All on the M5 (nothing exceeds F5-scale compute). Risks: (a) WP7.0 finds closer prior art →
narrow to the drift-robustness + ground-truth-map claims (still standing); (b) realistic
u-scatter too narrow to help → that *negative* quantifies how much deliberate condition
diversity (e.g., requesting one derated takeoff per week) buys — an operational
recommendation either way; (c) η_HPC~η_HPT stays unbreakable → already framed as the
fundamental-ambiguity finding.


---

## WP7.0 + WP7.1 executed (2026-07-04): novelty memo and the honest pivot

**Literature sweep.** Multi-point appears in ML work as a *training-data generation*
device (feature-fusion cascade NN; dual-transformer spatiotemporal fusion), not as an
identifiability instrument with drifting health. Conformal prediction sets exist for fault
detection generically (risk-guaranteed prediction sets, arXiv:2508.01208; calibrated
classifiers for launcher engines, arXiv:2507.13022) but not GPA-ambiguity-aware. The drift-
during-fusion-window claim found no prior art. Claims stand, repositioned.

**H7.1 strong form FAILED on realistic free scatter** (scripts/f7_observability.py,
200 draws/window): rank recovers trivially (10/10 from K=2 flights — today's
takeoff+cruise dual-report structure already spans the space), but signature angles
saturate at ~1.4x (hpt.eta~hpt.flow plateaus at 8.7 deg vs 13.1 deg for the designed
grid): ambient/derate scatter is too narrow. Recorded as the phase's load-bearing negative.

**The pivot — report-schedule design (stronger than the original claim).** The 8.7->13.1
gap is purchasable procedurally. Greedy design over the ICM grid, starting from today's
schedule (takeoff+cruise):

| Addition to the ACARS schedule | hpc.eta~hpt.eta | hpt.eta~hpt.flow |
|---|---|---|
| (baseline) | 1.33 deg | 6.16 deg |
| + low-power stabilized cruise | 1.64 | **11.03 (+79 %)** |
| + climb report | 1.36 | 7.33 |
| + FL390 cruise | 1.41 | 8.04 |
| best pair: low-power cruise + hot-day takeoff | 1.71 | **12.65** |

New headline contribution: **the snapshot report schedule as an observability design
variable** — selecting in-service report *trigger conditions* from influence-matrix
geometry. One periodic stabilized low-power cruise report nearly doubles the separability
of the HPT ambiguity at zero hardware cost. hpc.eta~hpt.eta is confirmed *fundamental*
for cockpit sensors under any schedule in the envelope.

**Reshaped hypotheses for prereg-v2** (learning phase):
- H7.2': the u-aware sequence learner under the DESIGNED schedule beats snapshot-AI and
  stacked-WLS on u-breakable episodes by >=15 pp (free-scatter arm reported alongside).
- H7.3 (drift robustness) unchanged — now evaluated under both schedules.
- H7.4' conformal sets with set size tracking the per-pair angle map (physics-explained
  ambiguity signaling).


---

## Confirmatory verdicts (2026-07-04, prereg-v2, single test pass, 23 episodes)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H7.2' fusion beats classical stacking | **CONFIRMED** | GRU-over-projections 0.65 vs stacked-WLS 0.17 (+48 pp); McNemar p <= 0.0096 under any pairing |
| H7.3 drift robustness (thresholds) | **REFUTED as frozen** | WLS degraded 17 pp (needed >= 20); GRU 13 pp (needed <= 10). Qualitative pattern present (WLS -> 0.00, GRU holds 0.52); thresholds miscalibrated from the 16-episode dev sample. Discipline over narrative. |
| H7.4' calibrated ambiguity | **CONFIRMED** | coverage 0.96; median sets 2-3.5; fundamental (3.0) > other (2.0) at short window |

**F7's contributions, final form**: (1) the report-schedule-as-design-variable analysis with
its +79 % procedural gain and the fundamental-pair map; (2) learned MOPA — a GRU fusing
per-block physics projections — more than tripling classical multi-point isolation on
identical data; (3) conformal isolation sets whose size tracks influence-matrix geometry;
(4) two honest negatives (free scatter saturates; the drift-robustness thresholds as frozen).
