# F1 — CFM56-7B Performance Model: Implementation Spec

Binding spec for WP1.1–WP1.4. Purpose: make every modeling decision explicit **before** coding so
the model cannot drift into vagueness. Anything not listed under *Non-goals* and not pinned here
must be raised as an issue before implementation.

## 1. Strategy: morph, don't build from scratch

pyCycle ships a validated two-spool, separate-flow, cooled high-bypass turbofan example
(`example_cycles/high_bypass_turbofan.py`) with the full element graph, bleed network, shaft
balances and off-design solver wiring already working. Building the CFM56-7B from a blank file
reinvents that wiring and typically dies in Newton non-convergence.

**Protocol:**

1. Vendor the example verbatim as `scripts/hbtf_reference.py`; run it; record its outputs as the
   *known-good state* (regression anchor).
2. Copy into `src/ehmbrain/perf/cycle.py` and re-target design variables toward the CFM56-7B in
   **at most 3 parameters per step** (e.g. BPR, then FPR+LPC PR, then HPC PR+T4). Every step must
   converge before the next; each converged step is a commit. If a step diverges, bisect the
   parameter change.
3. Only after the design point matches §3 targets, wire the off-design points (WP1.2) one at a
   time, highest power first (best-conditioned), reusing the previous point's solution as the
   initial guess.

## 2. Fixed design decisions

| # | Decision | Value | Rationale |
|---|----------|-------|-----------|
| D1 | Variant | CFM56-7B26 (26.3 klbf, 737-800) | most common -7B; richest public data |
| D2 | Thermo | TABULAR (`AIR_JETA_TAB_SPEC`, FAR fuel) | 5–10× faster than CEA; CEA cross-check once at the end of WP1.2 (report delta, expect <0.5 %) |
| D3 | Design point | M0.78 / 35 000 ft / ISA / max-cruise thrust | aero design point convention; SLS points are off-design |
| D4 | Element graph | inlet → fan → splitter → duct → booster(LPC) → duct → HPC → bld3 → burner → HPT → duct → LPT → duct → core nozzle; bypass duct → bypass nozzle; 2 shafts + power extraction | mirrors HBTF example; CFM56-7B is separate-flow (no mixer) |
| D5 | Cooling network | HPC interstage: cool1 (HPT vane, non-chargeable, works in rotor), cool2 (HPT blade, chargeable); bld3 (HPC exit): cool3/cool4 (LPT); customer bleed port + HP power extraction | keep HBTF topology; tune fractions in §3 window |
| D6 | Maps | pyCycle generic `FanMap, LPCMap, HPCMap, HPTMap, LPTMap`, `map_extrap=True` | no public CFM56 maps exist; scaling to our design point is standard practice; documented as limitation |
| D7 | Power-setting parameter | N1c (corrected fan speed) at off-design | CFM56-7B is N1-rated (not EPR) |
| D8 | Control/rating logic | not modeled in F1; ratings enter as target N1/thrust tables per anchor point | FADEC rating structure is F2's concern (mission sampling) |
| D9 | Units | pyCycle native (English) internally; SI at the `common/` API boundary only | fighting pyCycle units invites bugs |
| D10 | Solver policy | Newton per point, `maxiter≥50`, `solve_subsystems=True`; archive every converged case to CaseReader file; initial guesses always seeded from nearest converged point | non-convergence is the #1 schedule risk |

## 3. Calibration contract

Numeric targets live in `conf/cfm56_7b_targets.yaml` — single source of truth, loaded by both the
model runner and the tests. Every entry carries `value, tol_pct, source, status(VERIFIED|TO_VERIFY)`.

**Anchor points (WP1.2):**

| Anchor | Condition | Primary targets | Source |
|--------|-----------|-----------------|--------|
| A1 cruise | M0.78 / 35 kft / ISA | TSFC ≈ 0.627 lb/lbf/h at Fn ≈ 5.48 klbf; BPR ≈ 5.1; OPR ≈ 32 | public type data (TO_VERIFY) |
| A2 takeoff | SLS / ISA / 100 % F00 | Fn = 26 300 lbf; WF from ICAO EEDB; N1 ≤ 5175 rpm, N2 ≤ 14 460 rpm; EGT < 950 °C | EASA TCDS E.004 + ICAO EEDB (TO_VERIFY) |
| A3 climb-out | SLS / 85 % F00 | WF from ICAO EEDB | ICAO EEDB |
| A4 approach | SLS / 30 % F00 | WF from ICAO EEDB | ICAO EEDB |
| A5 idle | SLS / 7 % F00 | WF from ICAO EEDB | ICAO EEDB |

ICAO Emissions Databank gives **measured SLS fuel flows at 4 thrust settings** for the exact
-7B26 — free, authoritative off-design calibration data most theses ignore. Acceptance (gate H1):
|error| ≤ 3 % on WF, N2, EGT, Fn at A1–A3; ≤ 5 % at A4–A5 (low-power points are map-extrapolation
territory; documented if missed).

**Design-point windows (WP1.1 DoD):** BPR 5.1 ± 0.2 · cruise OPR 30–34 · T4 within 2800–3000 °R ·
total cooling+leakage 18–25 % of core flow · fan corrected flow consistent with 61 in fan
(~1550 lbm/s class at takeoff). Cycle must converge with all bleed flows > 0 and all map operating
points inside (not clipped to) their scaled maps.

## 4. Verification (automated, from day one)

- `tests/test_design_point.py`: DP converges (Newton residual < 1e-6), §3 windows hold.
- `tests/test_regression_hbtf.py`: vendored HBTF reference still reproduces its recorded outputs
  (catches dependency-upgrade breakage, like the numpy 2 incident).
- `tests/test_od_anchors.py` (WP1.2): each anchor within tolerance — this test **is** gate H1.
- `tests/test_icm_physics.py` (WP1.4): ICM sign checks (↓η_HPT ⇒ ↑EGT, ↑WF; ↓Γ_HPC ⇒ ↑N2 …) and
  SVD conditioning report.
- Every test also runs in CI (already wired: GitHub Actions runs pytest).

## 5. Non-goals (declared limitations, cite in paper)

Transient behavior; VBV/VSV geometry (implicit in maps); Reynolds/humidity corrections; inlet
distortion; nozzle variable geometry; secondary air system detail beyond D5; noise/emissions;
certifiable accuracy. The deliverable is a **representative** -7B26: correct architecture, correct
sensitivities (ICM), anchored absolute levels — not a certified deck.

## 6. Known pyCycle failure modes and counters

| Failure | Counter |
|---------|---------|
| Newton divergence after large retarget | §1 morphing protocol, ≤3 params/step, bisect on failure |
| Idle (A5) won't converge | continuation: solve 100→85→60→45→30→15→7 % power, seed each from previous; accept 5 % tol; as last resort declare idle out of calibration set (documented) |
| Map extrapolation nonsense at low power | `map_extrap=True` but assert operating point within map bounds in tests; inspect map plots at anchors |
| Tabular thermo range exceeded at high T4 | assert T4 < table limit; fall back to CEA for that point if needed |
| Silent unit mistakes | pyCycle native units internally (D9); conversions only in `common/units.py` with round-trip tests |
