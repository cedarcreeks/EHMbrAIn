# F14 — Replicating a published Bi-LSTM PHM result, and auditing it

**Status:** planned, NOT run. Implementation drafted at `scripts/f14_bilstm_replication.py`
(written, never executed). To freeze as `prereg-v17` before the first confirmatory run.

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

## 9. Order of execution

1. Freeze this document as `prereg-v17`, tag before running.
2. R1 (paper protocol, 5 seeds) → H14.1.
3. R2 (clean val split, 5 seeds) → H14.2, H14.4.
4. Traditional at cap 125 → H14.3.
5. Verdict JSON, figure, report section, A2 entry, bib entry.
6. Commit and push.
