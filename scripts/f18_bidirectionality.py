"""F18 / H15.2 (prereg-v21): does bidirectionality earn its keep, and where?

The line that started this thread was a published Bi-LSTM PHM result (L-EXT,
sec:f14-ext). F14 replicated its number and showed its fifteen-way architecture
ranking sits inside seed noise -- but never tested the one architectural claim
that has a physical argument behind it: that running the sequence backwards as
well as forwards helps.

THE PREDICTION, made before this run and falsifiable either way. A bidirectional
layer concatenates a forward and a backward pass, so what it buys depends
entirely on whether the backward direction carries information the forward one
lacks:

  ATTRIBUTION (which mechanism is consuming the engine). Decided at removal,
  when the WHOLE history is in hand. The backward pass is then legitimate
  smoothing, not forecasting, and it is genuinely informative: a wash sawtooth
  observed late re-labels ambiguous early fouling, and an end-of-life
  flow-capacity rise re-labels a mid-life efficiency drop as hot-section creep.
  Context flows from the end of the record to its middle. Expect a real gain.

  PROGNOSIS (remaining life). The backward pass runs over the input window,
  which is entirely past. Not leakage -- but there is no future context inside
  the window to exploit. Expect near-nothing.

  So: bidirectionality should help attribution materially and RUL barely. If the
  measured pattern is reversed, the reading of the mechanism is wrong.

The source paper's own numbers are consistent with the second half: bi_lstm
14.121 against lstm 14.241 is a 0.120 gap, and F14 measured the seed sd at 2.94.

DESIGN. Same harness, same data, same seeds as F17, with bidirectional as the
only variable. Ten seeds, paired on seed because seed effects are shared.
Attribution uses F13's mechanism-share task; prognosis uses the F5 RUL target on
the identical sequences, so the two tasks differ only in what is predicted.

Output: data/processed/f18/bidir_verdict.json
Usage: uv run python scripts/f18_bidirectionality.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                        # noqa: E402
from f13_gate1_mechanism import (MECHANISMS, CUT_MIN, SEQ_LEN,    # noqa: E402
                                 CUTS_PER_ENGINE, MIN_LOSS_C,
                                 mechanism_shares, r2_per_col)
from ehmbrain.datagen.fleet import load_icm                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f18'
SEEDS = tuple(range(10))
ALPHA = 0.05
RUL_CAP = 12000.0


def net(ch, n_out, bidirectional, hidden=128, layers=2):
    """One architecture, one switch. Hidden size is halved when bidirectional so
    both variants carry the same parameter budget -- otherwise the comparison
    would confound direction with capacity."""
    import torch.nn as nn
    h = hidden // 2 if bidirectional else hidden

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(ch, h, layers, batch_first=True, dropout=0.1,
                              bidirectional=bidirectional)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                      nn.Linear(32, n_out))

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1])

    return Net()


def build(catalog, H, ch, base, c, ids, rng):
    """Shared sequences; two targets so the tasks differ only in what is asked."""
    X, Ymech, Yrul = [], [], []
    mu, sd = c['norm']
    for eid in ids:
        life, shares, total = mechanism_shares(catalog, eid, H, ch, base)
        dev = c['dev'][eid]
        n = min(len(dev), life)
        if n <= CUT_MIN + 200:
            continue
        for t in rng.integers(CUT_MIN, n, size=CUTS_PER_ENGINE):
            t = int(t)
            if abs(total[t]) < MIN_LOSS_C:
                continue
            seg = (dev[:t] - mu) / sd
            idx = np.linspace(0, len(seg) - 1, SEQ_LEN)
            s = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, j])
                          for j in range(seg.shape[1])], axis=1)
            X.append(np.concatenate([s, np.full((SEQ_LEN, 1), t / 10000.0)],
                                    axis=1))
            Ymech.append(shares[t])
            Yrul.append(min(life - t, RUL_CAP) / 1000.0)
    return (np.asarray(X, np.float32), np.asarray(Ymech, np.float32),
            np.asarray(Yrul, np.float32)[:, None])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    H, ch, base = load_icm('takeoff_hot')
    c = fleet_cache()
    fleet = c['fleet']
    rng = np.random.default_rng(11)
    print('== building shared sequences ==', flush=True)
    Xtr, Mtr, Rtr = build(catalog, H, ch, base, c, split_ids(fleet, 'train'), rng)
    Xte, Mte, Rte = build(catalog, H, ch, base, c, split_ids(fleet, 'test'), rng)
    print(f'   train {len(Xtr)} / test {len(Xte)} cuts', flush=True)

    from scipy import stats
    from ehmbrain.ai.models import predict_torch, train_torch

    tasks = {'attribution': (Mtr, Mte, len(MECHANISMS)),
             'prognosis': (Rtr, Rte, 1)}
    res, per_seed = {}, {}
    for task, (Ytr, Yte, n_out) in tasks.items():
        ymean = Ytr.mean(axis=0)
        per_seed[task] = {}
        for bi in (False, True):
            key = 'bidirectional' if bi else 'unidirectional'
            per_seed[task][key] = {}
            for s in SEEDS:
                m = train_torch(net(Xtr.shape[2], n_out, bi), Xtr, Ytr,
                                epochs=122, lr=0.0057, bs=128, seed=s)
                p = predict_torch(m, Xte)
                if task == 'attribution':
                    sc = float(np.nanmean(r2_per_col(Yte, p, ymean)))
                else:                       # RUL: report RMSE in cycles
                    sc = float(np.sqrt(np.mean((p - Yte) ** 2)) * 1000.0)
                per_seed[task][key][s] = sc
                print(f'    {task:12s} {key:15s} seed {s}  {sc:+.3f}', flush=True)
            vals = list(per_seed[task][key].values())
            res.setdefault(task, {})[key] = {
                'mean': float(np.mean(vals)),
                'sd': float(np.std(vals, ddof=1)),
                'per_seed': per_seed[task][key]}

    verdict = {'design': {
        'seeds': list(SEEDS), 'seq_len': SEQ_LEN,
        'parameter_budget': ('hidden halved when bidirectional so both variants '
                             'carry the same parameter count -- otherwise the '
                             'comparison confounds direction with capacity'),
        'prediction': ('bidirectionality helps attribution (whole history in '
                       'hand at removal; backward pass is smoothing) and barely '
                       'helps prognosis (window is all past)')},
        'per_task': res}

    for task, better_is_low in (('attribution', False), ('prognosis', True)):
        u = [per_seed[task]['unidirectional'][s] for s in SEEDS]
        b = [per_seed[task]['bidirectional'][s] for s in SEEDS]
        alt = 'less' if better_is_low else 'greater'
        t = stats.ttest_rel(b, u, alternative=alt)
        delta = float(np.mean(b) - np.mean(u))
        verdict[f'H15.2_{task}'] = {
            'delta_bi_minus_uni': delta,
            'paired_t': float(t.statistic), 'p_one_sided': float(t.pvalue),
            'significant': bool(t.pvalue < ALPHA),
            'direction_tested': ('lower RMSE is better' if better_is_low
                                 else 'higher R2 is better')}

    a, p = verdict['H15.2_attribution'], verdict['H15.2_prognosis']
    verdict['H15.2_pattern_as_predicted'] = {
        'attribution_helps': a['significant'],
        'prognosis_barely': not p['significant'],
        'confirmed': bool(a['significant'] and not p['significant']),
        'criterion': ('bidirectionality must help attribution significantly and '
                      'NOT help prognosis significantly; the reverse pattern '
                      'falsifies the stated mechanism')}
    (OUT / 'bidir_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    for task in tasks:
        r = res[task]
        unit = 'R2' if task == 'attribution' else 'cy RMSE'
        print(f"  {task:12s} uni {r['unidirectional']['mean']:+.3f}"
              f" +-{r['unidirectional']['sd']:.3f}   "
              f"bi {r['bidirectional']['mean']:+.3f} +-{r['bidirectional']['sd']:.3f}"
              f"  [{unit}]  delta {verdict[f'H15.2_{task}']['delta_bi_minus_uni']:+.3f}"
              f"  p={verdict[f'H15.2_{task}']['p_one_sided']:.4f}")
    print(f"\n  H15.2 pattern as predicted: "
          f"{verdict['H15.2_pattern_as_predicted']['confirmed']}")


if __name__ == '__main__':
    main()
