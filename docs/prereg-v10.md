# Pre-registration v10 — F8/L-RUL confirmatory (frozen 2026-07-06, tag prereg-v10)

Tests whether an ADVANCED classical prognostic narrows the H3 RUL gap that the operational
Theil-Sen linear extrapolation showed. Motivated by the user's scrutiny and the ch6 caveat:
the current traditional RUL is operational practice, not research state of the art.

Advanced method: SIMILARITY-BASED prognostics (the C-MAPSS-standard classical approach). The
health indicator HI(n) = smoothed takeoff-EGT degradation; a test engine's recent HI window
is matched against the run-to-failure HI curves of the TRAIN engines; the k best-aligned
matches' remaining lives give the RUL (inverse-distance weighted). Unlike Theil-Sen it follows
the nonlinear (accelerating) degradation shape.

Dataset: SynCFM56 v1.1 (frozen), test split. Predictions at 50/70/90 % life, RUL capped
12 000 cycles (same protocol as F5). Three methods compared on test RMSE:
- Theil-Sen (tuned F5 traditional config) -- the operational baseline.
- Similarity-based (advanced traditional), knobs {window, k} fixed to sensible values.
- AI GRU -- the F5 confirmatory numbers (1694/1042/858 at 50/70/90 %), disclosed as prior.

## Frozen decision rule (honest either way)

- **H-RUL.1 (advanced narrows the gap) CONFIRMED iff** similarity-based 90 %-life RMSE
  < Theil-Sen's (1981 cycles) -- the fancier classical method IS better than linear.
- **H-RUL.2 (AI still wins) CONFIRMED iff** the AI 90 %-life RMSE (858) < similarity-based's,
  AND Wilcoxon signed-rank (paired per engine, 90 %) p<0.05.

If H-RUL.2 fails (similarity matches or beats the AI), H3 weakens to 'AI = the best advanced
classical prognostic' -- an important, honestly-reported outcome.

Output: data/processed/f8/lrul_verdict.json
