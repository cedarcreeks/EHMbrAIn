# Pre-registration v12 — F-OPS: unscheduled→scheduled conversion per method

**Frozen:** 2026-07-06. **Tag:** prereg-v12. **Phase:** F-OPS (operational KPI bridging
F5/F11 results to F-ECON economics).

## Question

By how much does each approach (traditional stack vs AI suite) convert **unscheduled**
engine removals (run-to-failure, expensive, AOG) into **scheduled** removals (planned, with a
shop slot)? The economics chapter (F-ECON) previously *assumed* this conversion; F-OPS
*measures* it from the already-generated RUL predictions, and hands the measured number back
to F-ECON.

## Operational model (frozen)

An engine is inspected at life fraction `f`. Its true remaining useful life there is `R`
(cycles). The method predicts `R̂ = R + e`, where `e` is the signed RUL error already recorded
in `data/processed/f5/rul_errors.json` (`e = pred − true`, so **e > 0 = over-prediction**).
The operator books a removal slot a logistics horizon `L` cycles before predicted end-of-life.

- The engine reaches the booked slot **without failing first** iff the over-prediction does
  not exceed the horizon: `e ≤ L`. This is a **converted (scheduled)** removal.
- If `e > L` the engine fails before the slot → **unscheduled** removal (the bad outcome).
- If the method under-predicts grossly (`e < −W`, W = wasted-life tolerance) the engine is
  pulled far too early → a **wasteful early removal** (the prognosis analog of a false alarm).

Metrics per method, per fraction, per horizon:
- **gross conversion** = fraction with `e ≤ L` (unscheduled avoided).
- **wasteful-early rate** = fraction with `e < −W`.
- **net conversion** = fraction with `−W ≤ e ≤ L` (converted *and* not wastefully early). This
  is the false-alarm-adjusted metric and the primary comparison quantity.
- **baseline** (no monitoring) = 0 % converted (everything runs to failure).
- **floor ceiling** = conversion of an *unbiased* predictor whose error spread equals the F11
  aleatoric floor σ_f (1053/615/212 cy at 0.5/0.7/0.9): `Φ(L/σ_f)`. This is the physical
  ceiling — no method can reliably exceed it. Ties F-OPS to [[f11-prognostic-floor]].

Nominal operating point: **L = 400 cy, W = 800 cy**. Horizon swept L ∈ {200, 400, 800};
report sensitivity, do not cherry-pick.

## Hypotheses (frozen before the confirmatory pass)

- **H-OPS.1 — AI converts more, honestly.** The AI suite's **net** conversion exceeds the
  traditional stack's at the nominal (L=400, W=800) at **all three** life fractions.
  *CONFIRMED iff* `net_ai > net_trad` for f ∈ {0.5, 0.7, 0.9}.

- **H-OPS.2 — conversion is floor-capped; headroom concentrates late.** The gap between the
  physical ceiling and the AI's achieved gross conversion grows with life fraction — early in
  life the AI already sits near the irreducible ceiling, late in life a large reducible gap
  remains (consistent with F11's H11.2). *CONFIRMED iff* `(ceiling − gross_ai)` at f=0.9 is
  strictly larger than at f=0.5 (L=400).

- **H-OPS.3 — the honest downside is real, and the advantage survives it.** The AI's gross
  conversion is materially cut once wasteful early removals are subtracted (net accounting),
  yet the AI's net still beats the traditional net. *CONFIRMED iff* at f=0.5,
  `net_ai ≤ 0.6 · gross_ai` (downside real) **and** `net_ai > net_trad` (survives).

## Measured feasibility (exploratory, disclosed)

n = 20 test engines per fraction. Net conversion at L=400/W=800: trad 15/0/25 %, AI 35/50/45 %
(f = 0.5/0.7/0.9). Gross AI 65/70/45 %; floor ceiling 65/74/97 %; ceiling-gap 0/4/52 pts.
AI wasteful-early 30/20/0 % vs trad 5/5/5 %. All three hypotheses provisionally pass.

## Output

- `scripts/f_ops_conversion.py` → `data/processed/f_ops/conversion.json` (per method × fraction
  × L: gross, wasteful, net, BCa 95 % CI on net over engines; ceiling; verdicts).
- Figure `ops_conversion` (net conversion vs L, baseline/trad/AI + ceiling band), N4.
- Report section `sec:ops-conversion` in ch14 (economics), consumed by the F-ECON estimate;
  new row in the ch12 F-QA operational table.

## Honesty constraints

FA-adjusted (net) is the headline; gross also reported so the downside is visible. Horizon
swept, not chosen. Ceiling drawn explicitly so no claim exceeds the physical floor. n=20 → wide
CIs reported, not hidden. Sim-to-real caveat inherited (synthetic fleet). No cost figures here —
F-OPS is the *operational* KPI; money is F-ECON's job, fed by this measured rate.
