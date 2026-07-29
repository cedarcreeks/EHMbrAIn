# F14 — A Bi-LSTM line: replicate the published result, then take it where GPA cannot go

**Status:** planned, NOT run. Implementation of Part A drafted at
`scripts/f14_bilstm_replication.py` (written, never executed). To freeze as `prereg-v17`
before the first confirmatory run.

**Two parts.** **Part A** replicates a published Bi-LSTM FD001 result and audits it against
this project's standards — a bounded, cheap, external check. **Part B** is the research line:
put the same architecture where classical gas-path analysis is *provably* blind, and measure
the consequence in fleet operating KPIs rather than in RMSE. Part A is the calibration; Part B
is the claim.

**Source:** Mukherjee, Hazra, Das, Datta — *"Development of Bidirectional-LSTM Model for
Prognostic Health Monitoring (PHM) of NASA Turbofan Engine"*, Springer Proceedings in
Information and Communication Technologies, 2026. doi 10.1007/978-981-96-9650-5_44.
PDF at `docs/bidirectional-LSTM.pdf`.

---

## 1. Why this line is worth running

The paper is a clean, well-written instance of exactly the artifact this project exists to
measure: a deep model reported as beating "traditional" techniques on C-MAPSS FD001, with
the comparison structured so that the win is nearly guaranteed. Replicating it is cheap (the
benchmark is already on disk at `data/external/cmapss/`, and `scripts/sim_to_real.py`
already runs FD001 end to end for contribution C7), and it converts an abstract methodological
criticism into a measured one on a specific, citable, recent publication.

It also gives the project something it currently lacks: an **external** test of its own
central claim. This document argues that AI's margin in prognosis is real but small once the
baseline is competent (1.34× at 90 % of life, a tie at 70 % — §sec:f8-lrul-sweep) and that
published margins are inflated by weak comparators. F14 checks that claim against a paper the
project did not write, on a dataset the project did not generate.

## 2. What the paper reports

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Bi-LSTM** | **10.149** | **199.409** | **14.121** | **0.885** |
| LSTM | 10.678 | 202.804 | 14.241 | 0.883 |
| TCN | 11.024 | 213.204 | 14.601 | 0.877 |
| GRU | 10.848 | 216.810 | 14.724 | 0.874 |
| … | | | | |
| Random Forest | 15.628 | 416.567 | 20.410 | 0.759 |
| Linear Regression | 18.249 | 525.064 | 22.914 | 0.696 |

Protocol: FD001; 12 sensors selected by |Pearson corr with RUL| ≥ 0.5 (2, 3, 4, 7, 8, 11, 12,
13, 15, 17, 20, 21); the `cycle` column explicitly dropped as leakage; min-max scaling;
30-cycle sliding window; piecewise-linear RUL capped at 125; two stacked Bi-LSTM layers
(64 then 128 units, recurrent dropout 0.3, ReLU), Dense 128, Dense 1; Adam lr 1e-3, MSE loss,
batch 32, 25 epochs; ModelCheckpoint and ReduceLROnPlateau(patience=2, factor=0.01,
min_lr=1e-5).

Two features of the paper's own text drive this study:

- **§4.2** states ModelCheckpoint retained the weights at minimum *validation* loss, and the
  **Fig. 7 / Fig. 8 captions** label that curve **"validation (test)"**. On the paper's own
  description, the reported 14.121 is selected on the same 100 units it is reported on.
- **Fifteen architectures are ranked on single runs**, with no seeds and no intervals, on gaps
  of 0.12–0.60 RMSE. Whether that ranking is a result or seed noise is not addressed.

Neither observation requires assuming an error. Both are checkable, and F14 checks them.

## 3. What existing infrastructure this reuses

| Piece | Where |
|---|---|
| FD001 train/test/RUL files | `data/external/cmapss/` |
| FD001 loader, health-index classical prognostic | `scripts/sim_to_real.py` (`load`, `health_index`) |
| Torch device selection, prediction helper | `src/ehmbrain/ai/models.py` (`device`, `predict_torch`) |
| Verdict-JSON + figure-regeneration conventions | `scripts/make_report_assets.py` (N4) |

Nothing new needs generating. A custom training loop is required (the project's `train_torch`
has no validation monitoring, checkpointing or LR scheduling), and it is already drafted.

## 4. Pre-registered hypotheses

Freeze before the first run. Five seeds (0–4) throughout; all aggregates reported as
mean ± sd; the paper's protocol is reproduced before anything is changed.

- **H14.1 — the result replicates.** Under the paper's protocol exactly (monitoring included),
  the Bi-LSTM reaches RMSE within 15 % of 14.121 on at least one seed.
  *Confirms:* the paper is reproducible and the audit that follows is about method, not
  about arithmetic. *Refutes:* something in the specification is underdetermined — report
  which, since that is itself a finding about the paper.

- **H14.2 — the headline survives clean model selection.** Repeating the run with the
  checkpoint monitored on a validation split of 20 TRAIN units (test never touched during
  training or selection) leaves RMSE no more than 5 % worse.
  *Refutes:* the gap is the size of the test-set-selection advantage, in cycles. This is the
  number the field should see, and it is the study's most likely contribution.

- **H14.3 — it beats a fielded classical prognostic.** Under H14.2's clean protocol, the
  Bi-LSTM beats this project's health-index projection (`sim_to_real.py`, PCA composite,
  Holt-smoothed, linearly extrapolated to the fleet failure level) evaluated at the same
  cap 125. Report the ratio next to the paper's own 22.914 / 14.121 = 1.62× against linear
  regression.
  *Note:* the direction is expected to confirm — this project's own C7 found AI 14.9 vs
  traditional 42.1 on FD001. The point is the **magnitude and the comparator**, not the sign.

- **H14.4 — the architecture ranking is not seed noise.** The seed sd of RMSE under the clean
  protocol is smaller than the paper's reported gaps between Bi-LSTM and LSTM (0.120), TCN
  (0.480) and GRU (0.603).
  *Refutes:* the paper's central claim — that Bi-LSTM specifically is the best of fifteen —
  is not supported by single runs, independent of whether Bi-LSTM is a good model.

## 5. Disclosed replication gaps

PyTorch cannot express two Keras options in the paper's Table 3, and both are recorded in the
output rather than glossed:

1. Keras `recurrent_dropout=0.3` applies dropout inside the recurrence; `torch.nn.LSTM`'s
   `dropout` applies between stacked layers. The latter is used.
2. Table 3 lists **ReLU** as the Bi-LSTM activation; the fused `torch.nn.LSTM` cell is
   tanh-only. The cell stays tanh.

If H14.1 fails, these two are the first suspects and the run should be repeated with a custom
(slow) cell before concluding anything about the paper.

A third item is worth flagging but not "fixing": `ReduceLROnPlateau(factor=0.01)` cuts the
learning rate 100× per plateau, so it reaches `min_lr=1e-5` after two plateaus and the model
is effectively frozen for most of the 25 epochs. This is reproduced as specified.

## 6. What each outcome yields

| Outcome | What gets written |
|---|---|
| H14.1 confirms, H14.2 confirms | The paper's result is robust; the project reports a published external result that survives its standards, and says so. A clean positive for the field. |
| H14.1 confirms, H14.2 refutes | The size of the test-set-selection premium on FD001, in cycles, measured on a specific recent paper. The most likely outcome and the most useful. |
| H14.3 confirms with a small ratio | Direct external corroboration of §sec:f8-lrul-sweep: AI beats classical prognosis, but by far less than the against-a-weak-baseline factor that gets published. |
| H14.4 refutes | Fifteen-way architecture rankings on single runs are noise. Generalizes well beyond this paper. |

Every branch is reportable. None of them requires the paper to be wrong.

## 7. Where it lands in the document

New section in `paper/report/chapters/11-overcoming.tex` (the F8 limitations-overcoming
programme), as **L-EXT: an external replication** — the natural home, since C7's sim-to-real
line already lives near there and this extends it from "does our ranking transfer" to "does a
published result survive our standards". Add a row to `tab:f8-summary`, a milestone entry in
`A2-milestone-log.tex`, and a figure comparing paper-reported / R1 / R2 / traditional RMSE
with seed spread as error bars.

Add the citation to `paper/report/references.bib`.

## 8. Cost

FD001 train is ~20 600 rows → ~17 700 windows at stride 1. Two protocols × 5 seeds = 10 runs
of 25 epochs on a bidirectional 2-layer LSTM. Estimated 45–90 min total on the M5 at MPS,
depending on contention. Run **after** the F13 gate-one job finishes, so the two do not
compete for the GPU.

## 9. Order of execution (Part A)

1. Freeze this document as `prereg-v17`, tag before running.
2. R1 (paper protocol, 5 seeds) → H14.1.
3. R2 (clean val split, 5 seeds) → H14.2, H14.4.
4. Traditional at cap 125 → H14.3.
5. Verdict JSON, figure, report section, A2 entry, bib entry.
6. Commit and push.

---

# Part B — Where GPA cannot go, and what it is worth to a fleet

## 10. The argument

Classical gas-path analysis is a **per-snapshot inverse solver**. It maps a deviation vector
to a health state through a regularized inverse of the influence matrix, one report at a time.
Everything this project measured about its limits follows from that one structural fact:

- rank 3 against 10 unknowns → 7 unconstrained directions (\cref{sec:icm});
- $\eta_{\mathrm{HPC}}$ and $\eta_{\mathrm{HPT}}$ 1.3° apart → confusable at any precision
  (H2, and halving the noise does not help, §sec:noise-sweep);
- the F10 certificate stamps three efficiencies **unobservable** at 1.6–1.8 % CRB even after
  accumulating Fisher information over the engine's whole flown history.

None of that is a statement about *history*. It is a statement about an **instantaneous
estimand**. And the generator's chronic degradation is not instantaneous: it is five
mechanisms with sharply distinct time signatures (`conf/fault_catalog.yaml`) — fouling
saturating with a wash sawtooth, erosion linear, clearance bilinear with a break-in knee,
hot-section accelerating with efficiency down *and flow capacity up*, LPT wear linear. Per
engine the true latent is **five scalars**, not a free 10-vector per cycle.

F13 gate one measured the consequence directly (`data/processed/f13/gate1_verdict.json`,
symmetric 25-trial budget per family, selected on val, reported on 450 cuts from held-out
engines, truth replayed from seeds):

| mechanism | truth mean/sd | R² hand features | R² sequence | gain |
|---|---|---|---|---|
| hot_section | 0.47 / 0.13 | **0.780** | 0.648 | −0.132 |
| clearance | 0.22 / 0.09 | 0.286 | **0.735** | **+0.449** |
| fouling | 0.08 / 0.04 | 0.451 | **0.512** | +0.061 |
| lpt_wear | 0.04 / 0.01 | 0.217 | 0.190 | −0.026 |
| erosion | 0.09 / 0.03 | 0.130 | −0.018 | −0.148 |
| *mean* | | *0.373* | *0.414* | *+0.041* |

**G1a confirmed:** three of five mechanisms recover above R² 0.30 from the deviation
trajectory alone, and `hot_section` — the mechanism that loads the confusable pair — is
recovered at R² 0.78. Instantaneously unidentifiable; over a history, readable. Part B's
premise holds.

**G1b refuted as specified**, and the way it fails is the useful part. The criterion was that
the sequence model beat hand features on ≥ 3 of 5 mechanisms; it manages 2, at a mean gain of
+0.041. But this is neither "hand features suffice" nor "learning wins" — it is a **split that
tracks how parameterizable each mechanism's temporal signature is**:

- **`clearance` (bilinear: fast break-in over 800 cycles, then slow) — sequence wins by
  +0.449.** The signature is a *knee*, and a knee resists closed-form description; the hand
  set only carries a crude break-in-dominance ratio, while the sequence model reads the shape.
- **`hot_section` (linear-accelerating, exponent 1.6, efficiency down with flow capacity up) —
  hand features win by 0.132.** That signature has a simple parametric form, and the hand set
  encodes it directly as a quadratic term plus cross-channel slope ratios.
- **`erosion` and `lpt_wear` (both plain linear, small margin shares) — neither family works.**
  Linear mechanisms competing against other linear mechanisms, contributing 4–9 % of margin
  loss each. Probably genuinely unidentifiable rather than badly modelled.

> **Revised Part B claim.** Mechanism attribution *is* possible from trajectory shape where the
> instantaneous state is certified unidentifiable — that part stands. But learning is not
> generically the tool. It earns its keep specifically where the temporal signature is a shape
> that does not reduce to a few parameters, and physics-designed features match or beat it
> everywhere else. The deliverable is the *mechanism-by-mechanism map of which is which*, not
> a blanket claim for either family.

**Two caveats, both against the sequence side.** First, the hand features were designed by
someone who had read `fault_catalog.yaml` — one descriptor per mechanism in the catalogue —
so that baseline is stronger than a practitioner without the generator source would build.
Second, selection was thin: validation is 245 cuts from ~10 engines, and the val/test gap is
large (val: hand 0.166 vs sequence 0.507; test: 0.373 vs 0.414). The test numbers are the ones
quoted; the val ranking should not be.

> **Part B's claim.** The place a sequence model earns its keep in EHM is not RUL, where the
> margin over a competent classical prognostic is 1.34× and only late in life. It is
> **mechanism and module attribution over a full history** — the question GPA cannot pose,
> let alone answer — and the operational payoff is not a smaller RMSE but a **shorter, better
> planned shop visit**.

## 11. Why bidirectional, specifically — and a falsifiable prediction

A bidirectional layer runs the sequence forward and backward and concatenates the states. What
that buys depends entirely on the task:

- **Retrospective attribution** (the shop-visit decision): at removal you hold the engine's
  *entire* history. The backward pass is then legitimate and genuinely informative — a wash
  sawtooth observed late tells you fouling was present early; a flow-capacity rise at end of
  life re-labels an ambiguous mid-life efficiency drop as hot-section creep. Context flows
  from the end of the record to its middle. This is smoothing, not forecasting, and no future
  is being peeked at.
- **Online prognosis** (RUL): the backward pass runs over the *input window*, which is
  entirely past. Not leakage — but there is no future context inside the window to exploit, so
  it should buy little.

This yields a sharp, pre-registrable prediction that the source paper's own numbers already
hint at (`bi_lstm` 14.121 vs `lstm` 14.241 — a 0.12 RMSE gap, almost certainly seed noise):

> **Bidirectionality should help attribution materially and RUL barely.** If the measured
> pattern is the reverse, Part B's reading of the mechanism is wrong.

## 12. The hybrid — and what it must NOT be

This project has refuted physics-into-learner injection **twice**: H4 (stacking GPA estimates
as features) and L6 (twin-residual features on the nonlinear fleet). Part B does not re-run
either. The refutation's own diagnosis (§sec:res-h4) is the design constraint:

> *"smeared state estimates are noise-bearing, mutually correlated features whose information
> the learner already had. Physics injection needs an information channel the raw data
> lacks."*

The F10 certificate **is** such a channel. Per engine and per direction it is computed from
the influence-matrix geometry and the engine's actual flown N1 history — from the *experiment
design*, not from the measured values. A learner reading the deviation trajectory cannot
derive it, because it is not in the trajectory.

**Proposed architecture — certificate-gated division of labour:**

| Component | Owns | Rationale |
|---|---|---|
| GPA / Kalman | the certified-**identifiable** subspace, per snapshot | it is optimal there and the certificate proves the estimate is trustworthy ($\rho=0.70$, H10.1) |
| Bi-LSTM over full history | the certified-**unobservable** directions, via mechanism attribution | trajectory shape is the only remaining evidence, and gate one says it carries signal |
| The certificate | the referee: per-direction CRB weights routed to the learner as input | tells the model *which* GPA outputs deserve weight — information absent from the raw data |

Concretely: project the GPA state estimate onto the identifiable subspace, feed that projection
**plus the per-direction CRB vector** alongside the raw deviation sequence, and let the
sequence model own what physics has certified it cannot recover instantaneously.

This is a third hybrid attempt after two failures. It is *motivated by* those failures rather
than in spite of them, but it may fail identically, and the pre-registration must say so.

## 13. Operational KPIs

RMSE is not a fleet decision. These are, each computable from artefacts this project already
produces. $N$ engines over $T$ days; removal event $i$ has downtime $d_i$ days.

**K1 — Net unscheduled→scheduled conversion $C_{\mathrm{net}}$.** Already defined and measured
(§sec:ops-conversion, `prereg-v12`): a removal converts iff the RUL error satisfies
$-W \le e \le L$, with logistics horizon $L=400$ and wasteful-pull guard $W=800$ cycles.
Current: AI 35–50 %, traditional 0–25 %. *Baseline KPI, unchanged.*

**K2 — Unscheduled removal rate.**
$$\mathrm{URR} = \frac{n_{\text{unscheduled}}}{\text{engine-cycles}} \times 1000$$
Directly the operator's pain metric, and the one AOG cost is proportional to.

**K3 — Workscope hit rate at horizon $H$. WITHDRAWN (§16): degenerate — `hot_section` dominates 100/100 engines at 95 % of life, so a constant predictor scores 100 %.** Original definition retained below for the record.
$$\mathrm{WHR}(H) = \frac{\#\{\text{removals where the dominant mechanism was correctly called} \ge H \text{ cycles early}\}}{\#\text{removals}}$$
The dominant mechanism is the argmax of the true share vector (ground truth from the seed
replay); the call is the model's argmax at cut $= t_{\text{removal}} - H$. $H$ is the parts
procurement lead time, swept, not chosen (nominal 300–800 cycles ≈ one to three months).
**GPA's structural score here is the confusable-pair accuracy, i.e. near chance on exactly the
modules that matter.** This is the KPI that only exists because of the identifiability wall.

**K4 — Fleet availability.**
$$A = 1 - \frac{\sum_i d_i}{N \cdot T}, \qquad
d_i = \underbrace{t_{\mathrm{AOG}} \cdot \mathbb{1}[\text{unscheduled}]}_{\text{no slot, no spare staged}} + t_{\mathrm{transport}} + t_{\mathrm{shop}}(\text{workscope known?})$$
Two levers, and Part B moves both: fewer unscheduled events (K1/K2, the RUL channel) and
shorter shop turnaround when the workscope was pre-positioned (K3, the attribution channel).
The second lever is unavailable to any RUL-only method — it is the operational expression of
"where GPA cannot go".

**K5 — Spare-engine ratio.**
$$S = \frac{\text{spares required}}{N} \approx \frac{\mathrm{removal\ rate} \times \overline{\mathrm{TAT}}}{365}\;+\;\text{safety stock}$$
Capital-intensive (a spare CFM56-class engine is an eight-figure asset), and it falls with both
removal rate and turnaround time. The most credible way to state Part B's value to a CFO.

**K6 — Wasteful removal rate $W_r$.** Engines pulled early that had life left — the
false-alarm analog, already netted out in K1. Reported alongside every other KPI so no gain is
quoted gross.

### Where the numbers come from, and the honesty rule

K1, K2, K3, K6 are **measured** in the testbed against ground truth. K4 and K5 additionally
require turnaround and AOG day-counts, which the testbed cannot produce — they are an
**assumption layer**, exactly like the cost anchors in \cref{ch:economics}. They inherit that
chapter's discipline verbatim: anchor to public MRO commentary, state the range, **sweep the
softest assumptions rather than choosing them**, and report the interval with its honest
downside. A KPI improvement that survives only at the optimistic end of the sweep gets
reported as such.

## 14. Pre-registered hypotheses (Part B)

- **H15.1 — attribution beats GPA where GPA is blind.** At the nominal horizon $H$, the
  sequence model's WHR on engines whose dominant mechanism loads the confusable pair exceeds
  the GPA/Kalman rule's by ≥ 20 percentage points, Holm-corrected.
  *This is the direct operational cash-out of the identifiability wall.*

- **H15.2 — bidirectionality helps attribution, not RUL.** Bi-LSTM minus unidirectional LSTM
  is ≥ 5 points of WHR, and < 1 seed-sd of RUL RMSE. *Refutes Part B's mechanism if reversed.*

- **H15.3 — the certificate-gated hybrid beats both pure families.** On mechanism attribution,
  the hybrid of §12 beats both standalone GPA and the standalone sequence model.
  *Two prior hybrids failed; this one is allowed to fail the same way, and the negative would
  be the strongest possible statement about physics-informed EHM on this benchmark.*

- **H15.4 — availability moves, and by how much.** Report $\Delta A$ and $\Delta S$ across the
  full sweep of turnaround assumptions, with the fraction of the sweep in which the gain
  survives. *Not a pass/fail: a measurement with an interval. A confirmed-sounding
  availability number quoted without its sweep would be the exact failure mode this project
  spent `prereg-v15` correcting.*

- **H15.5 — the attribution channel is worth more than the RUL channel.** Decompose $\Delta A$
  into the K1/K2 contribution and the K3 contribution. If the attribution channel dominates,
  the project's headline claim changes from "AI predicts life better" to **"AI answers the
  question GPA cannot, and that is where the availability is"**.

## 15. Honest risks

1. **F13 gate one is closed, and it partly landed on this risk.** G1a confirmed (attribution
   is possible); G1b refuted as specified (the sequence model wins 2 of 5 mechanisms, mean
   gain +0.041). So Part B's KPIs must be attributed to **trajectory analysis** in general,
   with learning credited only on `clearance`-like signatures. Every KPI below that reads
   "sequence model" should read "trajectory-shape estimator, learned or hand-built as the
   mechanism warrants". This weakens the AI framing and does not weaken the operational claim.
2. **Third hybrid attempt.** H4 and L6 both refuted. The certificate channel is better
   motivated than either, and may still fail.
3. **K4/K5 are modelled, not measured.** Stated above; the sweep is mandatory, not optional.
4. **Bi-LSTM is probably not special.** H14.4 is likely to show the paper's 15-way ranking is
   seed noise. Part B must therefore claim *sequence models over full history*, and treat
   bidirectionality as a mechanism to be tested (H15.2), never as a brand.
5. **Circularity, stated once more.** The five mechanisms are literature-motivated but
   implemented by this project. Part B measures how much mechanism information survives
   cockpit-sensor limits — a legitimate simulation question — and must not be written as
   though it discovered that mechanisms differ.

## 16. Second target: the attribution horizon

### What this replaces, and why

An earlier draft of this section proposed treating GPA ambiguity as a *clock*: fire a trigger
when the certificate tags two directions non-identifiable, then measure the "ambiguity
resolution latency" until a calibrated model names the true mechanism and holds it. That
target is **withdrawn**, on three grounds that the project's own results supplied:

1. **Its stated main risk materialised.** The draft said the clock argument fails if the
   ambiguity is driven by *acute* episodes, which ramp over 50–500 cycles and then hold with no
   discriminating future event. H2's confusable pair is exactly that population
   (`acute_hpc.eta`, `acute_hpt.eta`, `acute_hpt.flow` in the catalogue). No event, no clock.
2. **H15.8 is corroborating evidence** (§17.1). Sensor drift was a case where two
   snapshot-confusable things were expected to diverge over a history, and they did not,
   because a slow additive ramp has no shape. The rule that emerged from F13 and H15.8 together
   — *trajectory shape separates what has shape* — predicts acute ramps behave the same way.
3. **Its refutation branch is already in the report.** The draft argued a null would mean "buy
   the station probe rather than a model". That is L-H2, measured at $0.15 \to 0.92$
   (§sec:f8-lh2). A study whose negative branch reproduces an existing result buys nothing.

The abstention and calibration apparatus is likewise dropped: F7 already delivers calibrated,
physics-tracking ambiguity sets (§ch:tomography). What survives is the one question underneath
it that is both cheap and load-bearing.

### The surviving question

> **How early before removal can the dominant degradation mechanism be called, and how
> reliably?**

This is K3 restated, and K3 is what the entire Part B availability argument runs through:
workscope pre-positioning happens at procurement lead time or it does not happen at all.

**K3 — attribution horizon curve.** For each engine, score the dominant-mechanism call at cuts
indexed by *remaining* life rather than pooled across life, giving

$$P\big[\text{correct dominant mechanism} \;\big|\; H \text{ cycles before removal}\big]$$

swept over $H$. The reportable object is the curve, not a point: an operator reads their own
lead time off the $x$-axis. **K7 is folded into K3** — one horizon axis, not two.

### Why this is nearly free

F13 gate one already generates the data. `build()` samples cuts from 2 500 cycles to end of
life and retains `(engine, cut)` metadata; the only change is to bucket accuracy by remaining
life instead of pooling. One script change and one run.

It also needs none of the withdrawn machinery: no trigger definition, no abstention
calibration, no leakage check — there is no "resolution moment" that could be faked, only a
conditional accuracy at a stated horizon.

### MEASURED FIRST, AND IT KILLS THE KPI (2026-07-29)

Before writing the horizon experiment, one cheap check: how often is each mechanism the
dominant one at the moment a workscope is decided?

**At 95 % of life, `hot_section` is the dominant contributor to lost EGT margin for 100 of 100
engines.** Mean shares across the fleet: `hot_section` 0.534, `clearance` 0.157, `erosion`
0.095, `fouling` 0.063, `lpt_wear` 0.044.

A constant predictor that always answers "hot section" scores **100 %**. So the workscope hit
rate K3 — *was the correct module called in advance* — is not a hard problem on this
benchmark, it is a degenerate one. There is no classification task, because there is no
variation to classify.

**Consequences, stated rather than worked around.**

1. **K3 is withdrawn**, and with it the attribution channel of K4 (availability) and K5
   (spares). Part B's availability argument rested on workscope pre-positioning shortening
   turnaround; if every engine needs hot-section work, an operator learns nothing from being
   told so, and there is no turnaround to shorten. What remains of K4/K5 is the conversion
   channel alone, which is K1 and already measured.
2. **This is a property of SynCFM56, and should be declared as one.** Every engine draws all
   five mechanisms with lognormal severity multipliers, and `hot_section` carries the largest
   rate, so the fleet is homogeneous in workscope by construction. Real fleets are not:
   module-specific failures, FOD-driven removals and build-standard variation produce workscope
   heterogeneity this generator does not model. **Add to the declared limitations.**
3. **F13's scientific finding is untouched.** That mechanism *shares* are partially recoverable
   from trajectory shape where the instantaneous state is certified unidentifiable is a
   statement about identifiability, it is measured, and it is in the report
   (§sec:f13-mech). What does not follow is the operational cash-out.
4. **The one share with clear operational meaning already had a better number.** The washable
   fraction is L4, at $R^2 = 0.86$ (§sec:f8-l4) — better than F13's fouling share at
   $0.45$–$0.51$, on essentially the same quantity with a simpler estimator. F13 did not
   improve on it.

**What survives as a question worth asking:** not *which* mechanism, but *how much* — the split
between washable and permanent loss, at a stated horizon. That is L4's question, L4 answered
it, and the only open extension is whether its $R^2 = 0.86$ holds as the horizon lengthens.
That is one afternoon, and it is the honest remainder of this section.

### Hypotheses (reduced)

- **H15.9 — the washable-fraction estimate degrades gracefully with horizon.** L4's
  $R^2 = 0.86$ is measured at cuts from 1 600 cycles with a trailing window. Re-slice by
  *remaining* life and report $R^2(H)$. Useful if it holds at a procurement horizon; a
  declared limitation if it collapses.
  *No pass/fail: a curve, with the operator reading their own lead time off the axis.*

~~- **H15.9 — the horizon curve is operationally useful.**~~

~~- **superseded — the horizon curve is operationally useful.** At a procurement horizon of 600
  cycles, the dominant mechanism is called correctly for a majority of engines, and the curve
  is monotone in remaining life (later is easier).
  *Refutes:* attribution only becomes reliable after the decision has to be made, which would
  make K4/K5's attribution channel worthless and leave Part B resting on the conversion channel
  alone. Worth knowing before the KPI layer is built on it.

- **H15.10 — the curve is mechanism-specific.** The horizon at which attribution becomes
  reliable differs by mechanism, tracking F13's split: `hot_section` and `clearance` early,
  the two plain-linear mechanisms never.
  *This is a measurement, not a pass/fail, and it is the deliverable an operator would actually
  use: which findings can be anticipated and which cannot.*

## 17. Adopted from the ACTIVE EHM proposal

An external proposal ("ACTIVE EHM: Dynamic Fingerprinting for Active Diagnosis") argued that
mechanisms indistinguishable to steady-state GPA may separate during repeatable transients.
The physics is sound — the ICM is a steady-state Jacobian, while transient response is
governed by rotor inertia, heat soak and volume dynamics, which depend on health parameters
differently. It is unrunnable here as written: `snapshots.parquet` holds exactly one row per
(engine, cycle) with two ACARS snapshots as columns, there is no intra-flight time axis, and
pyCycle is a steady-state deck. Building a transient generator would mean recovering time
constants this project itself wrote. Three things in it are worth taking anyway, and one
should be gated rather than dropped.

### A17.1 — Multi-hypothesis diagnosis, and the free experiment it unlocks

The proposal frames diagnosis over competing hypotheses — nominal/context change, gas-path
degradation, **sensor drift or bias**, actuator or variable geometry — rather than assuming
every anomaly is gas-path. That framing is better than this document's, which does assume it.

And the experiment is already paid for: **22 of the 100 fleet engines carry a slow additive
bias ramp on one sensor** (`conf/fault_catalog.yaml:78`), with `drift_channel` and
`drift_active` ground truth in the snapshots. The labels exist. Nothing needs generating.

The structure is the same one Part B is built on, which is why it belongs here:

- **Instantaneously unidentifiable.** The cockpit ICM has rank 3 and maps $\mathbb{R}^{10}
  \to \mathbb{R}^{3}$ surjectively, so *every* deviation vector — including a pure single-channel
  bias — has a health-state preimage. At a snapshot, a lying thermocouple and a real fault are
  perfectly confusable. This is not a modelling gap; it is the same rank argument as H2.
- **Temporally identifiable, plausibly.** Real degradation moves along mechanism trajectories
  (smooth, coordinated across channels, following the five catalogued shapes). A bias ramp
  moves along a single coordinate axis. Those are different curves in deviation space even
  where they pass through indistinguishable points.
- **Partially attempted already.** L7 found an augmented Kalman "tracks the drift, cannot
  un-corrupt the cockpit diagnosis" (§sec:f8-l7) — drift *estimation* works, *classification*
  was never tested.

This is the second independent instance of Part B's claim, on a different fault family, with
ground truth already on disk. It also sharpens the operational story: the question a line
engineer actually asks is not "which module" but **"is this an engine problem at all, or is my
instrument lying?"** — and a removal triggered by a drifting sensor is precisely a wasteful
removal, KPI K6.

> **H15.8 — instrument versus engine.** A sequence model over the deviation trajectory
> separates sensor drift from gas-path degradation by ≥ 20 points of balanced accuracy over
> the classical augmented-Kalman rule, on engines held out by ID.
> *Refutes:* drift and degradation are confusable over histories too, which would mean the
> project's entire diagnostic output is conditional on instrument health it cannot verify —
> a stronger and more uncomfortable finding than the confirmation.

### H15.8 — RUN, AND REFUTED (`data/processed/f15/h158_verdict.json`)

Grouped 5-fold CV over engine IDs, pooled out-of-fold scores, 14 cockpit-visible drift
positives against 78 clean engines, bootstrap intervals grouped by engine.

| family | AUC | 95 % CI | fraud check (invisible drift) |
|---|---|---|---|
| classical augmented Kalman | 0.614 | [0.430, 0.789] | 0.433 |
| sequence model (Bi-GRU) | 0.524 | [0.317, 0.731] | 0.377 |

**Neither family separates a drifting sensor from a real fault.** Both intervals contain 0.5;
the learned family is at chance.

**The fraud check is what makes the null credible.** The eight engines drifting on T25 or PS3
carry no information in the cockpit input by construction, and both families score them like
clean engines. A leaking pipeline would have scored those high too, so this is a null and not
a bug.

**Why.** L7 already showed the augmented Kalman *tracks* an EGT bias at Spearman 0.83. This
shows tracking is not attributing: because the cockpit ICM maps $\mathbb{R}^{10}$ onto
$\mathbb{R}^{3}$ surjectively, a genuine degradation also produces a nonzero apparent bias, so
bias magnitude carries no evidence about which of the two is happening. The surjectivity
argument, previously stated for a single snapshot, now holds empirically over whole histories.

> **The rule this sharpens into.** *Trajectory shape separates what has shape.* Gate one's
> sequence model won on `clearance` (bilinear break-in knee) and `fouling` (wash sawtooth), and
> lost on `erosion` and `lpt_wear` (featureless linear ramps). The fleet's sensor drift is a
> slow additive ramp with no discriminating event, so it lands with the ramps. Same rule, four
> instances, two families, two fault populations.

**Consequences, recorded not softened.** Part B is *not* a general structure — it is specific
to mechanisms whose temporal signature has features. The ambiguity-clock target loses
expected value for exactly the population §16 already flagged as its main risk: acute episodes
that ramp and then hold with no future event to read. K3 (workscope hit rate) stands, since
`hot_section`, `clearance` and `fouling` are the recoverable ones.

**Power caveat.** 14 positives. The classical interval [0.430, 0.789] does not exclude a
moderate effect, so the honest claim is *no detectable signal at this sample size*, not proof
of none. Point estimates are near chance and the learned family has none at all.

Report the drift-channel breakdown: a bias on EGT should be far harder than one on N2, because
EGT carries most of the degradation signal. If the model only succeeds on the easy channels,
say so.

### A17.2 — The safety boundary, adopted verbatim in spirit

The proposal's safety section is better written than anything currently in this document, and
this project needs it the moment it discusses observation recommendation — which it already
does, in F7's report-schedule design and F10's sensor-acquisition instrument. Add
`docs/safety-case-boundaries.md` stating what this work is and is not: offline engineering
support; no FADEC interface; no actuation commands; no new operational manoeuvres; no
modification of certified limits or protections; and **no presentation of a statistical
prediction as a certified conclusion**. Any "recommended observation" output is limited to
naming which already-authorised data would be most informative.

### A17.3 — The conceptual test, promoted to a pytest

The proposal's best engineering idea is a unit test of the *idea* rather than of a metric:
construct a case where two hypotheses are steady-state identical but temporally distinct,
verify GPA cannot separate them, verify the dynamic feature can, and verify the
information-gain module picks the discriminating observation. Adapted to this project's
timescale — chronic mechanisms rather than transients — that is exactly gate one's premise,
and it should live in `tests/` as a permanent regression, not only in a script. If a future
refactor silently destroys the temporal signal, a test should fail.

### A17.4 — Transients: gated, not dropped

Before any transient machinery is built, one cheap physics question decides whether it is
worth building at all, and the proposal never asks it: **does transient response actually open
the angle for the pair the wall is made of?** $\eta_{\mathrm{HPC}}$ and $\eta_{\mathrm{HPT}}$
are both hot-section-adjacent; their *dynamic* signatures may be as close as their steady ones.

**Gate T.** Bolt a low-order dynamic layer onto the existing steady model — one time constant
per spool from rotor inertia and torque imbalance, plus a heat-soak lag on EGT — and compute
the Fisher information of a step response through it, exactly as `identifiability.py` already
does for the steady case. Then measure the angle between the two signatures in the combined
(steady ⊕ dynamic) space.

- Angle opens materially → transients are a route through the wall, and the engineering
  investment is justified. Revisit the full proposal then.
- Angle stays near $1.3^{\circ}$ → the wall is dynamic too, the proposal is dead for the pair
  that matters, and that is a strong negative this project would own.

Days of work, no ML, and the same measure-before-building discipline that stopped the
nonlinear-curvature route in F10 before it consumed a milestone.

### Explicitly not adopted

| Proposed | Why not |
|---|---|
| Parallel `active_ehm/` package, 40 files, 7-command CLI | Fragments the codebase. Integrate into `src/ehmbrain/`; the convention here is Hydra configs + `scripts/*.py` + verdict JSONs, and it works |
| Learned expected-information-gain module | **F7** already designs observations by information gain (+79 % separability from a report-schedule change) and **F10** does it for sensors (HPC efficiency 45×) — both physics-derived and validated against truth, which a learned surrogate would not be |
| New uncertainty / abstention stack | Conformal RUL intervals (C4), the CRB certificate with split-conformal scale (F10), and F7's calibrated ambiguity sets already exist |
| MLflow, pydantic, new schema layer | Three dataset audits that were allowed to fail and did, plus the decision register and prereg tags, already provide stronger provenance than a tracking server |
| "Reduce ambiguous cases by ≥ 50 %" | An arbitrary target with no physics behind it. Thresholds here are tied to measured exploratory values and disclosed (H2 at +10 pp, H10.1 at $\rho \ge 0.6$) |
| Building the transient generator now | Blocked behind Gate T |

## 18. Order of execution (Part B)

Ordered by cost-to-information, cheapest decisive experiment first.

1. ~~Wait for F13 gate one.~~ **Done** (§10): G1a confirmed, G1b refuted as specified. The
   claim is headed by *trajectory shape*, not by *learning*, with learning credited per
   mechanism.
2. Freeze Part B as `prereg-v18`, separately from Part A.
3. **H15.8 — instrument vs engine.** First, because the labels already exist (22 drifted
   engines), it needs no new generation, and it is a second independent test of Part B's whole
   premise on a different fault family. If it fails, the rest is in doubt and that is worth
   knowing before building anything.
4. Attribution task: Bi-LSTM vs unidirectional vs GPA rule → H15.1, H15.2.
5. Certificate-gated hybrid → H15.3.
6. **Attribution horizon curve** (§16) — re-slice F13's cuts by remaining life → H15.9, H15.10.
   Cheapest remaining item and the input K4/K5 depend on.
7. KPI layer K1–K6 with the turnaround sweep → H15.4, H15.5.
8. `docs/safety-case-boundaries.md` (A17.2); conceptual test into `tests/` (A17.3).
9. **Gate T** (A17.4) — the transient angle question, physics only, no ML. Decides whether the
   external proposal's route is reopened or closed with a stated negative.
10. Report chapter, figures, A2 entry. Commit and push.
