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
import os
import subprocess
import sys
import time
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
ARMS = ('uni', 'bi_equal', 'bi_double')
ALPHA = 0.05
RUL_CAP = 12000.0


def net(ch, n_out, arm, hidden=128, layers=2):
    """One architecture, three arms, correct pooling.

    POOLING (the bug the first run had). For a unidirectional layer the state at
    the final step has seen the whole sequence, so o[:, -1] is right. For a
    BIDIRECTIONAL layer it is wrong: the backward pass starts at the last step,
    so its output at position -1 has seen exactly ONE sample. Taking o[:, -1]
    hands the head a complete forward summary concatenated with almost nothing,
    which cripples the variant by construction. The correct read is the forward
    state at the LAST step with the backward state at the FIRST step.

    ARMS separate direction from capacity, which the first run conflated:
      uni         h = hidden                 (128 units, reference)
      bi_equal    h = hidden/2 per direction (same parameter budget as uni)
      bi_double   h = hidden   per direction (~2x parameters, the comparison the
                                              source paper actually makes)
    """
    import torch.nn as nn
    import torch
    bi = arm != 'uni'
    h = hidden if arm in ('uni', 'bi_double') else hidden // 2
    out_dim = 2 * h if bi else h

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bi, self.h = bi, h
            self.gru = nn.GRU(ch, h, layers, batch_first=True, dropout=0.1,
                              bidirectional=bi)
            self.head = nn.Sequential(nn.Linear(out_dim, 32), nn.GELU(),
                                      nn.Linear(32, n_out))

        def forward(self, x):
            o, _ = self.gru(x)
            if self.bi:                      # forward @ last, backward @ first
                z = torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1)
            else:
                z = o[:, -1]
            return self.head(z)

    return Net()


# --------------------------------------------------------------------------
# PARALLELISM. Sixty independent trainings run in about three hours in series;
# MPS buys only 1.26x over a single CPU thread here because the model is tiny
# (twelve batches per epoch), so the GPU idles waiting for Python. Eight
# single-threaded CPU processes finish in roughly twenty-five minutes.
#
# Not multiprocessing: spawn hung re-importing this module (and with it torch,
# sklearn and optuna) in every child, and fork deadlocked because the parent
# already holds BLAS/torch worker threads that fork does not copy -- the classic
# macOS Accelerate-plus-fork hazard. Clean subprocesses avoid both, and as a
# bonus each shard is separately observable and restartable.
#
# Parent: build once, cache to npz, launch N shards, collect. Shard: load the
# cache, train its slice, write predictions. Seeds are explicit per job, so the
# result does not depend on how the work was divided.
# --------------------------------------------------------------------------


def _shard_main(shard, n_shards):
    import torch
    torch.set_num_threads(1)
    from ehmbrain.ai.models import predict_torch, train_torch
    z = np.load(OUT / 'cache.npz')
    Xtr, Mtr, Rtr, Xte = z['Xtr'], z['Mtr'], z['Rtr'], z['Xte']
    cpu = torch.device('cpu')
    jobs = [j for i, j in enumerate(_all_jobs()) if i % n_shards == shard]
    out = {}
    for k, (task, arm, seed) in enumerate(jobs, 1):
        Ytr = Mtr if task == 'attribution' else Rtr
        m = train_torch(net(Xtr.shape[2], Ytr.shape[1], arm), Xtr, Ytr,
                        epochs=122, lr=0.0057, bs=128, seed=seed, dev=cpu)
        out[f'{task}|{arm}|{seed}'] = predict_torch(m, Xte, dev=cpu)
        print(f'  shard {shard}: {k}/{len(jobs)}  {task} {arm} seed {seed}',
              flush=True)
    np.savez_compressed(OUT / f'preds_{shard}.npz', **out)


def _all_jobs():
    return [(t, a, s) for t in ('attribution', 'prognosis')
            for a in ARMS for s in SEEDS]


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

    tasks = {'attribution': (Mtr, Mte, len(MECHANISMS)),
             'prognosis': (Rtr, Rte, 1)}

    np.savez_compressed(OUT / 'cache.npz', Xtr=Xtr, Mtr=Mtr, Rtr=Rtr, Xte=Xte)
    jobs = _all_jobs()
    n_shards = min(8, max(1, (os.cpu_count() or 4) - 2))
    print(f'== {len(jobs)} trainings over {n_shards} shard processes ==', flush=True)
    t0 = time.time()
    procs = [subprocess.Popen(
        [sys.executable, '-u', str(Path(__file__).resolve()),
         '--shard', str(i), str(n_shards)]) for i in range(n_shards)]
    for pr in procs:
        pr.wait()
    bad = [i for i, pr in enumerate(procs) if pr.returncode != 0]
    if bad:
        raise SystemExit(f'shards failed: {bad}')
    print(f'   all shards done [{(time.time() - t0) / 60:.1f} min]', flush=True)

    preds = {}
    for i in range(n_shards):
        z = np.load(OUT / f'preds_{i}.npz')
        for k in z.files:
            t, a, sd = k.split('|')
            preds[(t, a, int(sd))] = z[k]

    res, per_seed = {}, {}
    for task, (Ytr, Yte, n_out) in tasks.items():
        ymean = Ytr.mean(axis=0)
        per_seed[task] = {}
        for arm in ARMS:
            per_seed[task][arm] = {}
            for s_ in SEEDS:
                p = preds[(task, arm, s_)]
                if task == 'attribution':
                    sc = float(np.nanmean(r2_per_col(Yte, p, ymean)))
                else:
                    sc = float(np.sqrt(np.mean((p - Yte) ** 2)) * 1000.0)
                per_seed[task][arm][s_] = sc
            vals = list(per_seed[task][arm].values())
            res.setdefault(task, {})[arm] = {
                'mean': float(np.mean(vals)),
                'sd': float(np.std(vals, ddof=1)),
                'per_seed': per_seed[task][arm]}

    verdict = {'design': {
        'seeds': list(SEEDS), 'seq_len': SEQ_LEN,
        'arms': {'uni': 'hidden 128, unidirectional (reference)',
                 'bi_equal': 'hidden 64 per direction -- same parameter budget',
                 'bi_double': 'hidden 128 per direction -- ~2x parameters, the '
                              'comparison the source paper actually makes'},
        'pooling_fix': ('the first run read o[:, -1] for the bidirectional arms. '
                        'For a backward pass that position has seen exactly one '
                        'sample, so half the representation handed to the head '
                        'was near-empty and the variant was crippled by '
                        'construction. Corrected to forward-at-last concatenated '
                        'with backward-at-first. The first run measured a broken '
                        'bidirectional layer, not bidirectionality'),
        'invalidated_first_run': {
            'attribution': {'uni': 0.241, 'bi': -0.008, 'delta': -0.249},
            'prognosis_cy': {'uni': 1441.0, 'bi': 1412.3, 'delta': -28.8,
                             'p': 0.1852},
            'status': 'void -- pooling bug, retained for the record'},
        'prediction': ('bidirectionality helps attribution (whole history in '
                       'hand at removal; backward pass is smoothing) and barely '
                       'helps prognosis (window is all past)')},
        'per_task': res}

    for task, better_is_low in (('attribution', False), ('prognosis', True)):
        alt = 'less' if better_is_low else 'greater'
        u = [per_seed[task]['uni'][s] for s in SEEDS]
        verdict[f'H15.2_{task}'] = {
            'direction_tested': ('lower RMSE is better' if better_is_low
                                 else 'higher R2 is better')}
        for arm in ('bi_equal', 'bi_double'):
            b = [per_seed[task][arm][s] for s in SEEDS]
            t = stats.ttest_rel(b, u, alternative=alt)
            verdict[f'H15.2_{task}'][arm] = {
                'delta_vs_uni': float(np.mean(b) - np.mean(u)),
                'paired_t': float(t.statistic),
                'p_one_sided': float(t.pvalue),
                'significant': bool(t.pvalue < ALPHA)}

    a, pr = verdict['H15.2_attribution'], verdict['H15.2_prognosis']
    helps_attr = any(a[k]['significant'] for k in ('bi_equal', 'bi_double'))
    helps_prog = any(pr[k]['significant'] for k in ('bi_equal', 'bi_double'))
    verdict['H15.2_pattern_as_predicted'] = {
        'attribution_helps': helps_attr, 'prognosis_barely': not helps_prog,
        'confirmed': bool(helps_attr and not helps_prog),
        'criterion': ('bidirectionality must help attribution significantly in '
                      'at least one arm and NOT help prognosis; the reverse '
                      'pattern falsifies the stated mechanism'),
        'capacity_vs_direction': ('if bi_double helps and bi_equal does not, the '
                                  'gain is capacity rather than direction -- '
                                  'which is precisely what the source paper\'s '
                                  'bi-versus-uni comparison cannot separate')}
    (OUT / 'bidir_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    for task in tasks:
        r, unit = res[task], ('R2' if task == 'attribution' else 'cy RMSE')
        print(f'  {task}  [{unit}]')
        for arm in ARMS:
            line = f"    {arm:10s} {r[arm]['mean']:+9.3f} +-{r[arm]['sd']:.3f}"
            if arm != 'uni':
                d = verdict[f'H15.2_{task}'][arm]
                line += (f"   delta {d['delta_vs_uni']:+8.3f}  "
                         f"p={d['p_one_sided']:.4f}"
                         f"{'  *' if d['significant'] else ''}")
            print(line)
    print(f"\n  H15.2 pattern as predicted: "
          f"{verdict['H15.2_pattern_as_predicted']['confirmed']}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--shard':
        OUT.mkdir(parents=True, exist_ok=True)
        _shard_main(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
