# F12 — The fleet-prior dividend: where AI's advantage in EHM actually comes from

**Status:** proposed, not run. To freeze as `prereg-v16` before any confirmatory pass.
**Prerequisite reading:** `docs/f10-proposal.md` (the certificate this builds on).

---

## 0. Why this milestone is mandatory

The project's own re-audit (`prereg-v15`) shaved every headline AI claim down to size:

| Claim as first stated | What survived scrutiny |
|---|---|
| Detection 12× earlier | Real, but the win came from **window search**, not from depth (§sec:res-industry) |
| Isolation: AI better | **Refuted.** 31 % = 31 %. Wall is informational |
| RUL 2.3–4.4× better | Only against *fielded practice*. Against the advanced classical: **1.34× at 90 % of life, a tie at 70 %** (§sec:f8-lrul-sweep) |
| Intervals 6× tighter | **Mostly conformal calibration**, which is model-agnostic. 1.34× once both sides are calibrated alike (§sec:f-uq) |
| Physics-informed hybrid | **Refuted twice** |

Strip it down and the honest residue is: *AI buys a late-life shop-visit horizon, worth about
34 %.* That is a real result and an honest one. It is not a breakthrough, and no amount of
presentation makes it one.

So the question this milestone must answer is not "can we get a bigger number." It is:

> **Is there anything AI can do in engine health monitoring that a competent classical
> method structurally cannot — and can we prove it, in physical units, against ground truth?**

If the answer is no, the project should say so plainly; that is itself a publishable and
industrially valuable finding. If the answer is yes, it has to be a *mechanism*, not a
benchmark score. This proposal argues the answer is yes, names the mechanism, and specifies
the experiment that would prove or kill it.

---

## 1. The claim

**Classical gas-path analysis and AI do not fail at the same thing, because they do not draw
on the same information.**

Classical GPA solves an underdetermined inverse problem, per engine, per snapshot,
independently of every other engine that ever flew. The cockpit measurement has rank 3
against 10 health parameters, so 7 dimensions of the health state are unconstrained by the
data. Practice closes that gap with a **regularizer** — ridge, an a-priori covariance, a
Kalman process-noise matrix. That regularizer *is* a prior. It is Gaussian, near-diagonal,
hand-set, identical for every engine, and constant over life.

The real fleet prior is none of those things. Degradation is **correlated** across components
(a hot section deteriorates together), **non-Gaussian** (episodic events on a chronic drift),
**non-stationary** (accelerating), and **engine-specific** (mission mix, environment,
build standard).

> **The claim:** the one thing a learner can do that classical GPA structurally cannot is
> carry a *learned, high-dimensional, non-Gaussian prior over degradation trajectories* into
> the null space of the measurement. AI's advantage in EHM is not better inversion. It is
> **access to information the measurement does not contain and the classical regularizer
> cannot express.**

This reframes every result above. AI failed at confusable isolation (H2) because it was given
the same per-snapshot framing as GPA — a framing in which there is no prior to exploit. It won
at prognosis *late in life* because that is where trajectory shape, a fleet-level statistic,
becomes informative. Both facts are predictions of the claim, made after the fact here, but
made *before* the F12 experiment.

---

## 2. Why this is a breakthrough and the earlier ones were not

The differentiator is that the claim can be measured against an **absolute reference** instead
of a baseline.

The F10 certificate computes, per engine and per health direction, the **Cramér–Rao bound**:
the precision floor for any *unbiased* estimator using that engine's data alone. It is a
theorem, not a competitor. It cannot be under-tuned, strawmanned, or accused of being an
unfair baseline — the objection that ate the H3 and H5 headlines.

A biased or Bayesian estimator *is* allowed to beat the CRB. That is textbook. What is not
textbook is being able to *verify* it, because verification needs the true health state, which
no real fleet has. This project has it.

Therefore:

> **An estimator that beats the CRB in a certified-unidentifiable direction is provably using
> information from outside that engine's measurement. The size of the violation is the size of
> the borrowed information — per engine, per direction, in physical units.**

That is an **information-provenance decomposition**: how much of a diagnosis comes from this
engine's sensors, and how much from the fleet it belongs to. Expressed as a single scalar per
engine,

```
D_prior = ½ · log₂ ( det Σ_CRB / det Σ_achieved )     [bits]
```

— the bits of health-state information the fleet supplies that the sensors do not. It is
comparable across engines, across sensor sets, and across fleets. Nobody has this number
because nobody has both the bound and the truth.

**What it fixes about the earlier "breakthroughs":**

| Weakness of F10/F11 | How F12 answers it |
|---|---|
| Validated against the same simulator that made the data | Gate G3 tests the prior against *deliberately mismatched* fleets; the milestone fails if the advantage needs a self-consistent generator |
| Headline rests on n = 10 (rank correlation over directions) | 20 engines × 10 directions = **200 paired points**, plus a structural prediction (G5) that a coincidence cannot satisfy |
| Result is a measurement, not a mechanism | The claim *is* the mechanism, and it predicts where the advantage must appear and where it must not |

---

## 3. The experiment

All required machinery already exists in the repository. This is weeks of work, not a new
project.

| Piece | Status |
|---|---|
| Per-flight 10-dim health truth | `data/processed/fleet/snapshots.parquet`, `x_*` columns |
| Per-engine per-direction CRB | `src/ehmbrain/trad/identifiability.py` → `Certificate.certify()` |
| Classical estimator to beat | `kalman_gpa`, `trad/pipeline.py` |
| Nonlinear fleet (avoids the linear-generator objection) | `data/processed/fleet_v2/` (L2) |
| Fleet-variant generation | `datagen/fleet.py`, pattern proven by the C6 noise sweep |

**The prior-carrying estimator.** A sequence model mapping an engine's *entire* measurement
history to the 10-dim health state, trained on train-split engines with truth labels. Same GRU
family as F5 — deliberately, so that no one can attribute the result to architecture. It is
not solving an inverse problem; it is regressing a state, and everything it knows about the
null space it learned from other engines.

**The classical steelman (the control that makes or breaks the interpretation).** Give
classical GPA the *same* empirical prior: fleet-empirical health covariance as the WLS
a-priori matrix and the Kalman process-noise covariance, estimated from the train split only.
This is roughly twenty lines of linear algebra. It is the difference between "AI wins" and
"the prior wins."

**Prior-mismatch fleets (G3).** Regenerate the fleet with the degradation-mode mixture,
inter-component correlation, and rate distribution deliberately altered, at three mismatch
magnitudes. Train the prior on fleet A, evaluate on fleet B.

---

## 4. Pre-registered gates

Each gate is allowed to fail, and each failure yields a stated result. Holm correction across
the 10 health directions; paired over the 20 test engines; all fits on train only.

- **G1 — existence.** The prior-carrying estimator beats the per-engine CRB in **at least 3**
  directions the certificate tags `unobservable`, Holm-corrected, on the v2 nonlinear fleet.
  *Fails → the claim is dead, cheaply, and the project reports that classical GPA's fixed
  regularizer already extracts what the fleet has to offer. That is a strong negative result.*

- **G2 — attribution (measurement, not pass/fail).** Report the fraction of G1's gain
  recovered by the classical steelman with the same empirical prior.
  - **≥ 70 % recovered → headline: "the advantage is the prior, and it costs twenty lines of
    linear algebra, not a neural network."** Industrially the most valuable outcome in this
    document, and a direct hit on vendor hype.
  - **< 70 % → the residue is prior structure a Gaussian cannot express.** Then quantify
    which: does a Gaussian-mixture prior close it? A low-rank one? That residue is the honest
    measure of what learning uniquely buys.

- **G3 — transfer (the anti-self-simulation gate).** The G1 advantage must survive a
  pre-declared prior mismatch: the training fleet's dominant degradation mode halved in
  prevalence in the test fleet. Report the full decay curve of advantage versus mismatch
  magnitude and versus number of training engines.
  *Fails → the milestone is downgraded, honestly, to "the learned-prior advantage is a
  function of fleet homogeneity", and reports the break-even. Still the number an airline
  actually needs: **how many engines of your own history before the prior pays, and how fast
  does it decay when your fleet drifts.***

- **G4 — the wall, re-opened.** Re-run the H2 confusable-isolation task with the
  prior-carrying estimator. Exploratory-then-confirmatory, disclosed. This is the first
  re-test of the project's most prominent refutation that comes with a *mechanism* rather
  than a bigger model.

- **G5 — the structural prediction (the built-in fraud detector).** The gain must appear
  **where the certificate says it should and not where it says it should not**: large in
  `unobservable` directions, negligible in `identifiable` ones. Pre-registered criterion:
  rank correlation between per-direction CRB and per-direction CRB-violation, ρ ≥ 0.5.
  *Uniform gains across all directions indicate leakage, not a prior, and void G1.*

G5 is what separates this from a benchmark score. A leak, an overfit, or a bug produces gains
everywhere. Only a genuine prior produces gains *shaped like the null space* — a shape
predicted in advance, from physics, by an object computed without ever looking at the
estimator.

---

## 5. Why there is no way to lose

Every branch terminates in a real, quotable, honest result:

| Branch | Result |
|---|---|
| G1 fails | Classical regularization already captures the available fleet information. Negative, clean, cheap, and worth stating. |
| G1 passes, G2 ≥ 70 % | The advantage is the prior, not the network — implementable classically. The most useful anti-hype finding available. |
| G1 passes, G2 < 70 %, G3 passes | AI carries prior structure no Gaussian can express, and it survives fleet mismatch. **This is the breakthrough**, with an absolute reference, a mechanism, and 200 paired points. |
| G1 passes, G3 fails | The advantage is real but fleet-homogeneity-bound. Report the break-even curve — the number operators need before buying. |
| G5 fails | Something leaked. Found by the project's own instrument, before publication. |

---

## 6. What would be claimable if it lands

Not "AI beats traditional EHM by X ×". That claim is now measured, small, and late-life-only.

Instead: **a per-engine accounting of where a diagnosis's information comes from** — sensors
versus fleet — validated against truth, in bits, with a physics-derived prediction of its
shape. It converts the AI-versus-traditional question from a scoreboard into a budget, and it
tells an operator which half of the budget they can buy with hardware and which half with
history.

That is the contribution most likely to matter outside this document, for the same reason F10
was: **it is the one no real-data study can perform.** The difference is that this one also
survives the objection F10 could not — because G3 makes the simulator's self-consistency an
explicit experimental variable rather than an unexamined assumption.
