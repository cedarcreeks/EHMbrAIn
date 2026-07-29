"""F14 (prereg-v17): replication of Mukherjee et al., "Development of
Bidirectional-LSTM Model for Prognostic Health Monitoring (PHM) of NASA Turbofan
Engine", Springer LNICT 2026 (doi 10.1007/978-981-96-9650-5_44), inside this
project's infrastructure -- and an audit of whether its headline survives this
project's evidence standards.

WHAT THE PAPER REPORTS. C-MAPSS FD001. Bi-LSTM RMSE 14.121, MAE 10.149,
MSE 199.409, R2 0.885, beating 14 other models; the weakest, linear regression,
scores RMSE 22.914 / R2 0.696. Protocol: 12 sensors selected by |Pearson corr
with RUL| >= 0.5 (2,3,4,7,8,11,12,13,15,17,20,21), the cycle column dropped as
leakage, min-max scaling, a 30-cycle sliding window, piecewise-linear RUL capped
at 125, two stacked Bi-LSTM layers (64 then 128 units, recurrent dropout 0.3),
Dense 128, Dense 1, Adam lr 1e-3, MSE loss, batch 32, 25 epochs, with
ModelCheckpoint and ReduceLROnPlateau(patience=2, factor=0.01, min_lr=1e-5).

TWO THINGS THIS SCRIPT TESTS, because the paper's own text raises them:

1. Test-set model selection. Section 4.2 says ModelCheckpoint kept the weights
   at minimum *validation* loss, and the Fig. 7/8 captions label that curve
   "validation (test)". The reported 14.121 is therefore selected on the same
   100 units it is reported on. R1 reproduces this faithfully; R2 repeats it
   with a validation split carved out of the TRAIN units, which is the only
   protocol under which the number is an out-of-sample claim.

2. Single-run ranking. Fifteen models are ranked on gaps of 0.1-0.6 RMSE
   (bi_lstm 14.121, lstm 14.241, tcn 14.601, gru 14.724) with one run each, no
   seeds and no intervals. R3 runs 5 seeds and reports mean +- sd, which is what
   decides whether "Bi-LSTM is best" is a result or a coin flip.

3. What "traditional" means. The paper's weakest comparator is a linear
   regression on windowed sensors -- a weak ML baseline, not a fielded EHM
   prognostic. R4 compares against the classical health-index projection this
   project already fields on FD001 (scripts/sim_to_real.py), at the paper's cap.

DISCLOSED DEVIATIONS (PyTorch cannot express two Keras options exactly):
  - Keras recurrent_dropout=0.3 applies dropout inside the recurrence; torch's
    nn.LSTM dropout applies between stacked layers. The latter is used.
  - Table 3 lists ReLU as the Bi-LSTM activation (Keras allows swapping the cell
    activation); torch's fused nn.LSTM is tanh-only. The cell stays tanh.
  Both are recorded in the output so the replication gap is visible.

Output: data/processed/f14/bilstm_verdict.json
Usage: uv run python scripts/f14_bilstm_replication.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_to_real import load, health_index, SENSORS as TRAD_SENSORS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CM = REPO_ROOT / 'data' / 'external' / 'cmapss'
OUT = REPO_ROOT / 'data' / 'processed' / 'f14'

PAPER_SENSORS = [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]   # |corr| >= 0.5
RUL_CAP = 125.0          # the paper's piecewise-linear threshold
SEQ = 30                 # the paper's sliding window
EPOCHS = 25
BATCH = 32
SEEDS = (0, 1, 2, 3, 4)
PAPER = {'MAE': 10.149, 'MSE': 199.409, 'RMSE': 14.121, 'R2': 0.885}


# --------------------------------------------------------------------------

def bilstm(n_feat):
    """Two stacked Bi-LSTMs (64 then 128), Dense 128 ReLU, Dense 1 ReLU."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.LSTM(n_feat, 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.l2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                      nn.Linear(128, 1), nn.ReLU())

        def forward(self, x):
            h, _ = self.l1(x)
            h, _ = self.l2(self.drop(h))
            return self.head(h[:, -1]).squeeze(-1)

    return Net()


def train_paper(model, Xtr, ytr, Xva, yva, seed=0, verbose=False):
    """The paper's training settings, including ModelCheckpoint on the
    monitored split and ReduceLROnPlateau(patience=2, factor=0.01)."""
    import copy

    import torch
    import torch.nn as nn
    from ehmbrain.ai.models import device

    torch.manual_seed(seed)
    dev = device()
    model = model.to(dev)
    Xtr_t = torch.as_tensor(Xtr); ytr_t = torch.as_tensor(ytr)
    Xva_t = torch.as_tensor(Xva).to(dev); yva_t = torch.as_tensor(yva).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.01, patience=2, min_lr=1e-5)
    lossf = nn.MSELoss()
    best, best_state = np.inf, copy.deepcopy(model.state_dict())
    n = len(Xtr_t)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xva_t), yva_t))
        sched.step(vl)
        if vl < best:                      # ModelCheckpoint
            best, best_state = vl, copy.deepcopy(model.state_dict())
        if verbose:
            print(f'    epoch {ep + 1:2d}/{EPOCHS}  val MSE {vl:8.2f}  '
                  f'best {best:8.2f}', flush=True)
    model.load_state_dict(best_state)
    return model


def metrics(pred, true):
    e = np.asarray(pred, float) - np.asarray(true, float)
    mse = float(np.mean(e ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {'MAE': float(np.mean(np.abs(e))), 'MSE': mse,
            'RMSE': float(np.sqrt(mse)),
            'R2': float(1.0 - np.sum(e ** 2) / ss_tot)}


# --------------------------------------------------------------------------

def build_data():
    """Min-max scaled windows on the paper's 12 sensors, PWL RUL capped at 125."""
    train, test = load('train'), load('test')
    cols = [f's{i}' for i in PAPER_SENSORS]
    lo = train[cols].to_numpy(float).min(0)
    hi = train[cols].to_numpy(float).max(0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)

    def scale(g):
        return (g[cols].to_numpy(float) - lo) / rng

    def windows(df, units):
        X, y, u = [], [], []
        for uid, g in df[df.unit.isin(units)].groupby('unit'):
            S = scale(g)
            n = len(S)
            for end in range(SEQ, n + 1):
                X.append(S[end - SEQ:end])
                y.append(min(n - end, RUL_CAP))
                u.append(uid)
        return (np.array(X, np.float32), np.array(y, np.float32), np.array(u))

    # test: one window per unit, at its last observed cycle
    Xte = []
    for uid, g in test.groupby('unit'):
        S = scale(g)
        S = S[-SEQ:] if len(S) >= SEQ else np.pad(
            S, ((SEQ - len(S), 0), (0, 0)), 'edge')
        Xte.append(S)
    Xte = np.array(Xte, np.float32)
    yte = np.minimum(pd.read_csv(CM / 'RUL_FD001.txt', header=None)[0]
                     .to_numpy(float), RUL_CAP)
    return train, test, windows, Xte, yte


def traditional_at_cap(train, test, cap):
    """This project's fielded classical prognostic on FD001 (health-index
    projection, scripts/sim_to_real.py), evaluated at the paper's cap."""
    cols = [f's{i}' for i in TRAD_SENSORS]
    Xtr = train[cols].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xn = (Xtr - mu) / sd
    _, _, Vt = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)
    w = Vt[0]
    heads, tails = [], []
    for uid, g in train.groupby('unit'):
        hi = health_index(g, mu, sd, w)
        heads.append(hi[:10].mean()); tails.append(hi[-5:].mean())
    if np.mean(tails) < np.mean(heads):
        w = -w
        tails = [-t for t in tails]
    fail = float(np.mean(tails))
    preds = {}
    for uid, g in test.groupby('unit'):
        s = pd.Series(health_index(g, mu, sd, w)).ewm(alpha=0.15).mean().to_numpy()
        n = len(s)
        a = max(0, n - 60)
        slope = np.polyfit(np.arange(a, n), s[a:], 1)[0]
        preds[uid] = (cap if slope <= 1e-6
                      else float(np.clip((fail - s[-1]) / slope, 0, cap)))
    return np.array([preds[u] for u in sorted(preds)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, windows, Xte, yte = build_data()
    units = np.array(sorted(train.unit.unique()))
    rs = np.random.default_rng(0)
    va_units = set(rs.choice(units, size=20, replace=False).tolist())
    tr_units = [u for u in units if u not in va_units]

    Xall, yall, _ = windows(train, units)
    Xtr, ytr, _ = windows(train, tr_units)
    Xva, yva, _ = windows(train, sorted(va_units))
    print(f'windows: all-train {len(yall)}, train-only {len(ytr)}, '
          f'val {len(yva)}, test units {len(yte)}', flush=True)

    from ehmbrain.ai.models import predict_torch

    def run(protocol, seed):
        """R1 monitors the TEST set (the paper's protocol); R2 monitors a
        validation split held out of the training units."""
        if protocol == 'R1_paper_test_monitored':
            net = train_paper(bilstm(len(PAPER_SENSORS)), Xall, yall,
                              Xte, yte.astype(np.float32), seed=seed)
        else:
            net = train_paper(bilstm(len(PAPER_SENSORS)), Xtr, ytr,
                              Xva, yva, seed=seed)
        pred = np.clip(predict_torch(net, Xte), 0, RUL_CAP)
        return metrics(pred, yte), pred

    results = {}
    for protocol in ('R1_paper_test_monitored', 'R2_clean_val_split'):
        per_seed, preds = [], []
        for s in SEEDS:
            m, p = run(protocol, s)
            per_seed.append(m)
            preds.append(p)
            print(f'  {protocol}  seed {s}: RMSE {m["RMSE"]:.3f}  '
                  f'R2 {m["R2"]:.3f}', flush=True)
        agg = {k: {'mean': float(np.mean([m[k] for m in per_seed])),
                   'sd': float(np.std([m[k] for m in per_seed])),
                   'min': float(np.min([m[k] for m in per_seed])),
                   'max': float(np.max([m[k] for m in per_seed]))}
               for k in ('MAE', 'MSE', 'RMSE', 'R2')}
        # A run whose output ReLU dies predicts 0 everywhere, giving exactly
        # RMSE = sqrt(mean(y^2)). Pooling those with trained runs makes the mean
        # a mixture of two modes and reports neither, so both are separated.
        dead = float(np.sqrt(np.mean(yte ** 2)))
        ok = [m for m in per_seed if abs(m['RMSE'] - dead) > 1e-2]
        cond = ({k: {'mean': float(np.mean([m[k] for m in ok])),
                     'sd': float(np.std([m[k] for m in ok], ddof=1))
                     if len(ok) > 1 else None}
                 for k in ('MAE', 'MSE', 'RMSE', 'R2')} if ok else None)
        results[protocol] = {'per_seed': per_seed, 'aggregate': agg,
                             'n_seeds': len(SEEDS),
                             'collapsed_runs': len(per_seed) - len(ok),
                             'collapse_rmse': dead,
                             'aggregate_trained_only': cond}

    trad_pred = traditional_at_cap(train, test, RUL_CAP)
    trad = metrics(trad_pred, yte)

    r1, r2 = (results[p]['aggregate'] for p in
              ('R1_paper_test_monitored', 'R2_clean_val_split'))
    r1c, r2c = (results[p]['aggregate_trained_only'] for p in
                ('R1_paper_test_monitored', 'R2_clean_val_split'))
    n_col = sum(results[p]['collapsed_runs'] for p in results)
    verdict = {
        'paper': {'citation': ('Mukherjee, Hazra, Das, Datta (2026), Springer, '
                               'doi 10.1007/978-981-96-9650-5_44'),
                  'reported': PAPER,
                  'reported_weakest_baseline': {'model': 'linear regression',
                                                'RMSE': 22.914, 'R2': 0.696}},
        'protocol': {'sensors': PAPER_SENSORS, 'rul_cap': RUL_CAP,
                     'window': SEQ, 'epochs': EPOCHS, 'batch': BATCH,
                     'seeds': list(SEEDS),
                     'deviations': [
                         'torch nn.LSTM dropout is between stacked layers, not '
                         'the Keras recurrent_dropout inside the recurrence',
                         'torch fused nn.LSTM cell activation is tanh; Table 3 '
                         'of the paper lists ReLU']},
        'R1_paper_protocol': r1,
        'R2_clean_val_split': r2,
        'R1_trained_only': r1c,
        'R2_trained_only': r2c,
        'per_seed': {p: results[p]['per_seed'] for p in results},
        'traditional_health_index_projection': trad,
        'H14.5_trains_reliably': {
            'collapsed_runs': n_col, 'total_runs': 2 * len(SEEDS),
            'collapse_rmse': results['R2_clean_val_split']['collapse_rmse'],
            'confirmed': bool(n_col == 0),
            'mechanism': ('Table 3 puts ReLU on the single-unit output layer; '
                          'once it dies the gradient is zero, and Table 4 '
                          'ReduceLROnPlateau(factor=0.01, min_lr=1e-5) floors '
                          'the learning rate before it can recover'),
            'framework_caveat': ('collapse probability depends on init: Keras '
                                 'Dense defaults to glorot_uniform, torch '
                                 'Linear to kaiming_uniform(a=sqrt(5)). The '
                                 'mechanism is architecture-specified, the rate '
                                 'measured here is not transferable as-is')},
        'H14.1_replicates': {
            'paper_rmse': PAPER['RMSE'],
            'replicated_rmse_mean': r1['RMSE']['mean'],
            'replicated_rmse_range': [r1['RMSE']['min'], r1['RMSE']['max']],
            'confirmed': bool(r1['RMSE']['min'] <= PAPER['RMSE'] * 1.15),
            'note': 'the paper protocol reproduced faithfully, monitoring included'},
        'H14.2_survives_clean_selection': {
            'rmse_test_monitored': r1['RMSE']['mean'],
            'rmse_clean_val': r2['RMSE']['mean'],
            'penalty_cycles': r2['RMSE']['mean'] - r1['RMSE']['mean'],
            'confirmed': bool(r2['RMSE']['mean'] <= r1['RMSE']['mean'] * 1.05),
            'note': ('how much of the headline came from selecting the '
                     'checkpoint on the reported test set'),
            'frozen_criterion_invalid_under_collapse': (
                'the pre-registered comparison is between pooled means, which '
                'under a bimodal outcome mostly compares how many runs '
                'collapsed. The substantive figure is the trained-only pair '
                'below'),
            'trained_only': {
                'rmse_test_monitored': r1c['RMSE']['mean'] if r1c else None,
                'rmse_clean_val': r2c['RMSE']['mean'] if r2c else None,
                'penalty_cycles': (r2c['RMSE']['mean'] - r1c['RMSE']['mean']
                                   if r1c and r2c else None)}},
        'H14.3_beats_traditional': {
            'ai_rmse': r2['RMSE']['mean'], 'traditional_rmse': trad['RMSE'],
            'ratio': trad['RMSE'] / r2['RMSE']['mean'],
            'paper_ratio_vs_linreg': 22.914 / PAPER['RMSE'],
            'confirmed': bool(r2['RMSE']['mean'] < trad['RMSE']),
            'note': ('stated against a fielded classical prognostic, not '
                     'against a linear regression on windowed sensors'),
            'trained_only': {
                'ai_rmse': r2c['RMSE']['mean'] if r2c else None,
                'ratio_vs_fielded': (trad['RMSE'] / r2c['RMSE']['mean']
                                     if r2c else None),
                'ratio_vs_paper_linreg': (22.914 / r2c['RMSE']['mean']
                                          if r2c else None)},
            'baseline_ordering_note': (
                'the fielded health-index projection scores WORSE on FD001 '
                'than the linear regression the paper uses as its weakest '
                'comparator. It is the operationally realistic method, not the '
                'strongest classical one, so the ratio against the paper\'s own '
                'linear regression is the more demanding figure')},
        'H14.4_seed_stability': {
            'rmse_sd_cycles': r2['RMSE']['sd'],
            'paper_model_ranking_gaps': {'bi_lstm_vs_lstm': 14.241 - 14.121,
                                         'bi_lstm_vs_tcn': 14.601 - 14.121,
                                         'bi_lstm_vs_gru': 14.724 - 14.121},
            'gaps_within_1sd': None,   # filled below
            'note': ('the paper ranks 15 architectures on single runs; if the '
                     'seed spread exceeds the gaps, the ranking is noise')},
    }
    # use the trained-only sd: the pooled sd is inflated by collapsed runs and
    # would make the ranking look like noise for the wrong reason
    sd = r2c['RMSE']['sd'] if r2c and r2c['RMSE']['sd'] else r2['RMSE']['sd']
    verdict['H14.4_seed_stability']['rmse_sd_cycles'] = sd
    verdict['H14.4_seed_stability']['sd_basis'] = 'trained-only runs'
    verdict['H14.4_seed_stability']['gaps_within_1sd'] = {
        k: bool(abs(v) < sd)
        for k, v in verdict['H14.4_seed_stability']['paper_model_ranking_gaps'].items()}

    (OUT / 'bilstm_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    print(f"  paper reported            RMSE {PAPER['RMSE']:6.3f}   R2 {PAPER['R2']:.3f}")
    print(f"  R1 paper protocol         RMSE {r1['RMSE']['mean']:6.3f} "
          f"+-{r1['RMSE']['sd']:.3f}   R2 {r1['R2']['mean']:.3f}")
    print(f"  R2 clean val split        RMSE {r2['RMSE']['mean']:6.3f} "
          f"+-{r2['RMSE']['sd']:.3f}   R2 {r2['R2']['mean']:.3f}")
    print(f"  traditional (health idx)  RMSE {trad['RMSE']:6.3f}   R2 {trad['R2']:.3f}")
    print(f"\n  collapsed runs (output ReLU dead, predicts 0): {n_col}/{2 * len(SEEDS)}"
          f"  at RMSE {results['R2_clean_val_split']['collapse_rmse']:.3f}")
    if r1c:
        print(f"  R1 trained only           RMSE {r1c['RMSE']['mean']:6.3f} "
              f"+-{r1c['RMSE']['sd'] or 0:.3f}")
    if r2c:
        print(f"  R2 trained only           RMSE {r2c['RMSE']['mean']:6.3f} "
              f"+-{r2c['RMSE']['sd'] or 0:.3f}"
              f"   -> {trad['RMSE'] / r2c['RMSE']['mean']:.2f}x vs fielded, "
              f"{22.914 / r2c['RMSE']['mean']:.2f}x vs paper's linreg")
    print()
    for h in ('H14.1_replicates', 'H14.2_survives_clean_selection',
              'H14.3_beats_traditional'):
        print(f"  {h}: {verdict[h]['confirmed']}")
    print(f"  H14.4 seed sd {sd:.3f} cy vs paper ranking gaps "
          f"{verdict['H14.4_seed_stability']['gaps_within_1sd']}")


if __name__ == '__main__':
    main()
