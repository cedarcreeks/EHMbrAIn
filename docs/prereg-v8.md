# Pre-registration v8 — F8/L5 confirmatory (frozen 2026-07-05, tag prereg-v8)

Tests whether the AI RUL advantage (H3) is architecture-robust or specific to the GRU.
FAIRNESS is the point: all three sequence architectures get IDENTICAL treatment -- same
inputs, same data, same RUL cap, same epoch budget, same three seeds, same fixed sensible
hyperparameters (no per-architecture Optuna campaign; this is an equal-modest-effort
comparison, disclosed as such, not a claim about which architecture is best-tuned).

Dataset: SynCFM56 v1.1 (frozen), test split. Architectures: GRU (recurrent, the F5 model),
TCN (dilated causal 1-D convolutions), Transformer (self-attention encoder + mean pool).
Traditional baseline: the tuned F5 margin extrapolation (90 %-life test RMSE = 1981 cycles).

## Frozen decision rule

- **H5L.1 (architecture-robust) CONFIRMED iff** all three AI architectures beat the
  traditional baseline at 90 % life (mean-across-seeds test RMSE < 1981 cycles). This
  establishes the RUL advantage is a property of learning-from-sequences, not of one
  architecture. If some architecture fails, that is the honest caveat.
- Secondary (descriptive, not gated): which architecture has the lowest RMSE.

Output: data/processed/f8/arch_verdict.json
