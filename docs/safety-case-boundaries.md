# Safety-case boundaries

**What this document is for.** This project produces diagnostic and prognostic estimates for a
gas-turbine engine, and one of its lines recommends *which data to collect next*
(§sec:f7-schedule, §sec:bt-honesty). Both of those sit close to activities that are regulated.
This document draws the line explicitly, so that no reader, and no future extension of the
work, has to infer it.

The posture is adopted from the external ACTIVE EHM proposal reviewed in §17 of
`docs/f14-proposal.md`, whose safety section was better written than anything this project had
at the time.

---

## 1. What this work is

- An **offline engineering study**. Every result is computed after the fact from recorded or
  synthetic data, on a desktop machine, with no connection to any engine or aircraft system.
- A **benchmark and a measurement method**: a testbed where the true engine condition is known,
  used to score diagnostic and prognostic methods against the answer rather than against
  another estimate.
- A **decision-support instrument for engineering**, specifically: which health directions a
  given sensor set can resolve, how much of a remaining-life error is irreducible, and which
  additional measurement would resolve the most.

## 2. What this work is not, and must not become without a separate case

This list is prohibitive, not aspirational. Nothing in this repository is authorised for any of
the following, and no extension should add them without a safety case of its own.

- **It does not control the engine.** No output of this work is an actuation command, a schedule
  change, a limit adjustment, or a setpoint.
- **It does not interface with the FADEC**, nor with any certified control, protection or
  monitoring function. There is no path, direct or indirect, from these scripts to engine
  control.
- **It does not design new operational manoeuvres.** Where a line recommends an *observation*
  (§sec:f7-schedule: adding a cruise report at a second power condition), the recommendation is
  restricted to **procedures that are already authorised and already flown**. It selects among
  existing conditions; it does not invent flight-test points, and it does not ask a crew to do
  anything they would not otherwise do.
- **It does not modify limits, protections, or certified logic.** Redlines, EGT margin
  definitions and removal criteria are taken as given inputs, never as outputs.
- **It does not issue airworthiness or continued-operation determinations.** Nothing here
  releases an engine to service, extends an interval, or clears a finding.

## 3. The rule that matters most

> **A statistical prediction is never presented as a certified conclusion.**

Every quantitative claim in this work carries the uncertainty of the method that produced it,
and several of the project's own findings exist precisely to bound what may be claimed:

- $87\,\%$ of mid-life remaining-life spread is irreducible (§sec:bt-prognostic-floor). A
  point RUL estimate is not a failure date.
- The confusable pair is unresolvable from cockpit sensors, by four independent measurements
  (§sec:res-h2, §sec:noise-sweep, §sec:f8-lh2, §sec:gate-t). An isolation output naming one of
  those components is not evidence that the other is healthy.
- A drifting sensor cannot be told from real degradation on cockpit data
  (§sec:f15-instrument). Every diagnosis here is conditional on instrument health that the
  method cannot itself verify.

Where a method's uncertainty statement has been validated against ground truth, that is stated
and the validation is named (§sec:bt-honesty). Where it has not, that is stated too.

## 4. Human oversight

- Every output is **advisory to a qualified engineer**, who remains the decision-maker.
- Outputs are **traceable**: each verdict carries the script, the pre-registration tag, the
  data version and the seeds that produced it (§sec:safeguards, and the milestone log).
- Refutations are recorded with the same weight as confirmations, so an engineer reading a
  result can see what the method has been shown *not* to do.
- No output is designed to be consumed automatically. There is no alerting path, no
  threshold-crossing action, and no interface intended for unattended use.

## 5. Provenance and its limits

- The engine model is calibrated from **public certification data only** (type-certificate data
  sheet, ICAO emissions databank). No proprietary or export-controlled input is used.
- The fleet is **synthetic**. Absolute numbers are not claims about any real engine; what the
  study defends is the *structure* — rank, signature geometry, method ranking, and the
  irreducible floors.
- Cross-simulator transfer is checked only for the prognosis ranking, on C-MAPSS FD001
  (§sec:sim-to-real, §sec:f14-ext). Nothing else has been shown to transfer.

## 6. Before any operational use

Each of the following would be a prerequisite, and none has been done:

1. Validation on real engine data from the target configuration, with the instrument-health
   assumption of §3 addressed rather than declared.
2. A verification and validation case for the software, at a level appropriate to the intended
   use, including the failure modes this project already documented — silent training collapse
   (§sec:f14-ext) and seed-dependent instability (§sec:f18-bidir).
3. An assessment against the applicable regulatory framework for the intended function, agreed
   with the responsible airworthiness authority.
4. Human-factors work on how an advisory estimate with a stated interval is presented so it is
   not read as a certainty.

Until those exist, the results here are evidence for engineering judgement and for research
direction. They are not a basis for action on an engine.
