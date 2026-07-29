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

F13 gate one measured the consequence directly, on held-out engines with truth replayed from
seeds: **four of five mechanisms recover at R² > 0.30 from the deviation trajectory alone, and
`hot_section` — the mechanism that loads the confusable pair — is the best recovered at
R² 0.786.** Instantaneously unidentifiable; over a history, readable.

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

**K3 — Workscope hit rate at horizon $H$ (new, and the load-bearing one).**
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

1. **F13 gate one G1b is still open.** If physics-designed shape features match the sequence
   model, the mechanism finding stands but stops being an *AI* finding, and Part B's KPIs
   should then be attributed to trajectory analysis, not to learning. The plan does not
   presume the answer.
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

## 16. Order of execution (Part B)

1. Wait for F13 gate one (G1b decides whether "sequence model" or "shape features" heads the
   claim).
2. Freeze Part B as `prereg-v18`, separately from Part A.
3. Attribution task: Bi-LSTM vs unidirectional vs GPA rule → H15.1, H15.2.
4. Certificate-gated hybrid → H15.3.
5. KPI layer K1–K6 with the turnaround sweep → H15.4, H15.5.
6. Report chapter, figures, A2 entry. Commit and push.
