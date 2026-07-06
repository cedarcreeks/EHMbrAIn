# Pre-registration v11 — F11 breakthrough confirmatory (frozen 2026-07-06, tag prereg-v11)

The second certificate: a ground-truth-validated IRREDUCIBILITY FLOOR for remaining-life
prediction. Diagnosis had its certificate (F10: what health can be known); prognosis gets its
own: how much RUL uncertainty is ALEATORIC (irreducible -- the engine's future is stochastic)
versus EPISTEMIC (reducible with better data/models). No fielded study decomposes and
validates this, because it needs ground truth plus counterfactual futures -- only a
synthetic-truth benchmark supplies both.

Method: the aleatoric floor at a life fraction = the spread (std) of TRUE remaining life among
engines with near-identical TRUE current health (kNN, k=10, on the standardized 10-dim health
state). Conditioning on the exact current state removes epistemic ignorance; the residual
spread is the irreducible future randomness (future wash schedule, future acute faults,
degradation rate not yet expressed). The epistemic gap = best achieved method RMSE minus the
floor.

Dataset: SynCFM56 v1.1 (frozen), all engines, fractions 50/70/90 % life. Best method per
fraction from {tuned Theil-Sen, similarity-based, AI GRU} (the L-RUL and F5 numbers).

Measured feasibility (exploratory, disclosed): floor 1053/615/212 cy at 50/70/90 %; AI RMSE
1694/1042/858.

## Frozen decision rule

- **H11.1 (irreducible floor is real and dominant early) CONFIRMED iff** the aleatoric floor
  at 50 % life is >= 60 % of the marginal true-RUL std -- i.e. most early-life RUL uncertainty
  is irreducible, no method can remove it.
- **H11.2 (reducible headroom concentrates late) CONFIRMED iff** the ratio
  (best-method RMSE / aleatoric floor) is strictly larger at 90 % than at 50 % -- the epistemic
  gap a better method could close is a late-life phenomenon.

Output: data/processed/f11/prognostic_floor.json
