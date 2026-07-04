"""C7 sim-to-real check: does the RUL method *ranking* survive on an external
simulator's data?

Benchmark: NASA C-MAPSS FD001 (the canonical turbofan run-to-failure set;
100 train units, 100 test units with hidden RUL truth). Scope note, disclosed:
the plan named N-CMAPSS DS02, whose multi-GB distribution is impractical for
the desktop-reproducibility norm; FD001 answers the same question --- ranking
transfer --- on the field's reference benchmark. Detection/isolation do not
transfer (FD001 has no event labels); this is an H3-shaped check only.

Methods, mirroring the SynCFM56 pair:
  traditional  per-unit health index (first PC of the 14 informative sensors,
               oriented to grow, Holt-smoothed) extrapolated linearly to the
               fleet-average failure level --- the classical margin-projection
               recipe adapted to C-MAPSS conventions.
  AI           the same GRU architecture family as F5, trained on windowed
               normalized sensors (train units only), RUL capped at 130
               (community standard for FD001).

Output: data/processed/f5/sim_to_real.json
Foreground (MPS). Usage: uv run python scripts/sim_to_real.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CM = REPO_ROOT / 'data' / 'external' / 'cmapss'
OUT = REPO_ROOT / 'data' / 'processed' / 'f5'

SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # informative set
RUL_CAP = 130.0
SEQ = 30


def load(split):
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(CM / f'{split}_FD001.txt', sep=r'\s+', header=None,
                     names=cols)
    return df


def health_index(df, mu, sd, w):
    """First-PC composite of normalized sensors, oriented to increase."""
    X = (df[[f's{i}' for i in SENSORS]].to_numpy(float) - mu) / sd
    return X @ w


def traditional_rul(df, mu, sd, w, fail_level, window=60):
    """Per-unit linear extrapolation of the smoothed health index."""
    preds = {}
    for uid, g in df.groupby('unit'):
        hi = health_index(g, mu, sd, w)
        # Holt-lite smoothing
        s = pd.Series(hi).ewm(alpha=0.15).mean().to_numpy()
        n = len(s)
        a = max(0, n - window)
        x = np.arange(a, n)
        y = s[a:]
        slope, intercept = np.polyfit(x, y, 1)
        if slope <= 1e-6:
            preds[uid] = RUL_CAP
            continue
        preds[uid] = float(np.clip((fail_level - s[-1]) / slope, 0, RUL_CAP))
    return preds


def main():
    import torch
    from ehmbrain.ai.models import RULNet, predict_torch, train_torch

    train = load('train')
    test = load('test')
    rul_true = pd.read_csv(CM / 'RUL_FD001.txt', header=None)[0].to_numpy(float)
    rul_true_capped = np.minimum(rul_true, RUL_CAP)

    Xtr_raw = train[[f's{i}' for i in SENSORS]].to_numpy(float)
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-9
    # PCA direction from train, oriented so HI rises toward failure
    Xn = (Xtr_raw - mu) / sd
    _, _, Vt = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)
    w = Vt[0]
    tails, heads = [], []
    for uid, g in train.groupby('unit'):
        hi = health_index(g, mu, sd, w)
        heads.append(hi[:10].mean())
        tails.append(hi[-5:].mean())
    if np.mean(tails) < np.mean(heads):
        w = -w
        tails = [-t for t in tails]
    fail_level = float(np.mean(tails))

    # --- traditional ---
    trad_pred = traditional_rul(test, mu, sd, w, fail_level)
    trad_err = np.array([trad_pred[u] for u in sorted(trad_pred)]) - rul_true_capped
    trad_rmse = float(np.sqrt(np.mean(trad_err ** 2)))

    # --- AI (GRU, 3 seeds) ---
    def unit_windows(df):
        X, y = [], []
        for uid, g in df.groupby('unit'):
            S = (g[[f's{i}' for i in SENSORS]].to_numpy(float) - mu) / sd
            n = len(S)
            for end in range(SEQ, n + 1, 2):
                X.append(S[end - SEQ:end])
                y.append(min(n - end, RUL_CAP))
        return np.array(X, np.float32), np.array(y, np.float32)

    Xtr, ytr = unit_windows(train)
    Xte = []
    for uid, g in test.groupby('unit'):
        S = (g[[f's{i}' for i in SENSORS]].to_numpy(float) - mu) / sd
        S = S[-SEQ:] if len(S) >= SEQ else np.pad(S, ((SEQ - len(S), 0), (0, 0)),
                                                  'edge')
        Xte.append(S)
    Xte = np.array(Xte, np.float32)

    cpu = torch.device('cpu')
    rmses = []
    for seed in (0, 1, 2):
        net = train_torch(RULNet(ch=len(SENSORS), hidden=64, layers=2),
                          Xtr, ytr, epochs=12, lr=1e-3, seed=seed)
        pred = np.clip(predict_torch(net, Xte, dev=cpu), 0, RUL_CAP)
        rmses.append(float(np.sqrt(np.mean((pred - rul_true_capped) ** 2))))
    ai_rmse = float(np.mean(rmses))

    result = {
        'benchmark': 'C-MAPSS FD001 (scope note: N-CMAPSS impractical for the '
                     'desktop norm; disclosed substitution)',
        'traditional_rmse_cycles': trad_rmse,
        'ai_rmse_cycles_mean': ai_rmse,
        'ai_rmse_cycles_per_seed': rmses,
        'ranking_preserved': bool(ai_rmse < trad_rmse),
        'syncfm56_ranking': 'AI < traditional (H3 confirmed)',
        'rul_cap': RUL_CAP, 'n_test_units': int(len(rul_true)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sim_to_real.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
