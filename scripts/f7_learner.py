"""F7 WP7.2-7.4: stacked-WLS MOPA baseline, u-aware sequence learner, the
drift-robustness experiment (the no-prior-art claim), and conformal isolation
sets.

Isolation protocol: oracle-timed per episode (as F5), window of W cycles
ending at onset+500. Two window lengths probe drift robustness: SHORT (300)
barely drifts; LONG (2000) spans chronic drift + possible washes — classical
constant-health stacking should degrade there, a temporal learner should not.

Methods:
  stacked-WLS   for each flight t in the window: rows [H_to(u_t); H_cr(u_t)]
                and measured deviation steps vs the pre-onset reference;
                regularized LSQ -> x_step; predict argmax|x_j| over the 6
                target params ('none' if below threshold). Classical MOPA
                adapted to opportunistic windows, constant-health assumption.
  GRU learner   sequence of per-flight [dz_cr(3), dev_to(1), dTs_to, N1_cr]
                (deviation channels referenced to the same pre-onset mean),
                trained on train-split episodes, 7-class output.
  conformal     adaptive prediction sets (APS) on the learner's softmax,
                calibrated on val episodes; coverage and set size reported
                per pair class (u-breakable vs fundamental).

Usage (foreground, MPS): uv run python scripts/f7_learner.py [mode]
  mode: 'val' (development numbers) or 'test' (confirmatory, run ONCE after
  prereg-v2 freeze).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                     # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ICMD = REPO_ROOT / 'data' / 'processed' / 'icm'
OUT = REPO_ROOT / 'data' / 'processed' / 'f7'

COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
TARGETS = ['fan.eta', 'lpc.eta', 'hpc.eta', 'hpt.eta', 'hpt.flow', 'lpt.eta']
CLASSES = ['none'] + TARGETS
FUNDAMENTAL = {('hpc.eta', 'hpt.eta')}
U_BREAKABLE = ('hpt.eta', 'hpt.flow')     # the pair the schedule can help
WINDOWS = {'short': 300, 'long': 2000}
DS = 10                                    # downsample inside window


def load_H():
    H = {}
    for pt in ('takeoff', 'takeoff_hot', 'cruise', 'cruise_lowpwr'):
        z = np.load(ICMD / f'icm_{pt}.npz', allow_pickle=True)
        ch = [str(c) for c in z['channels']]
        H[pt] = z['H'][[ch.index(c) for c in COCKPIT]]
    return H


DEV_HOLDOUT = 15   # last N train engines held out for development evaluation


def split_engines(fleet, split):
    tr = split_ids(fleet, 'train')
    if split == 'dev_fit':
        return tr[:-DEV_HOLDOUT]
    if split == 'dev':
        return tr[-DEV_HOLDOUT:]
    return split_ids(fleet, split)


def episode_table(fleet, split):
    rows = []
    for eid in split_engines(fleet, split):
        v = fleet[eid]
        eps = v['episodes']
        bounds = [int(o) for o, _ in eps] + [v['life']]
        for k, (onset, param) in enumerate(eps):
            t = min(bounds[k + 1] - 1, v['life'] - 1, int(onset) + 500)
            if t <= int(onset) + 100:
                continue
            rows.append({'engine': eid, 'onset': int(onset), 't': int(t),
                         'param': param})
    return rows


def _block_mean(A, ds):
    n = (len(A) // ds) * ds
    if n == 0:
        return A[:1] * 1.0
    return A[:n].reshape(-1, ds, A.shape[1]).mean(axis=1)


def window_data(c, ep, W):
    """Per-flight sequences for the window [t-W, t]: deviations referenced to
    the pre-onset mean, block-averaged (sqrt(DS) noise gain — what monitoring
    smoothing does; applied identically to both methods)."""
    eid, t, onset = ep['engine'], ep['t'], ep['onset']
    F = c['dev'][eid]                      # (n,4): dz_cr(3) + dev_to_EGT
    u = c['u'][eid]                        # (n,2): dTs_to, N1_cr
    a = max(0, t - W)
    pre = F[max(0, onset - 400):max(1, onset - 20)].mean(axis=0)
    seq_F = _block_mean(F[a:t] - pre, DS)
    seq_u = _block_mean(u[a:t], DS)
    return seq_F, seq_u


def per_block_xhat(c, H, seq_F, seq_u, lam=2.0):
    """Per-block regularized WLS projection x_hat_t (10,): the physics
    projection sequence the learner fuses (learned MOPA)."""
    out = []
    for i in range(len(seq_F)):
        dts, n1 = seq_u[i]
        w_to = float(np.clip(dts / 30.0, 0, 1))
        w_cr = float((n1 - 4666.0) / (4400.0 - 4666.0))
        Hto = H['takeoff'] * (1 - w_to) + H['takeoff_hot'] * w_to
        Hcr = H['cruise'] * (1 - w_cr) + H['cruise_lowpwr'] * w_cr
        Hs = np.vstack([Hto[2:3], Hcr])
        zs = np.concatenate([seq_F[i, 3:4] / c['b_to_egt'] * 100.0,
                             seq_F[i, :3]])
        A = Hs.T @ Hs + lam * np.eye(10)
        out.append(np.linalg.solve(A, Hs.T @ zs))
    return np.array(out)


def stacked_wls(c, H, ep, W, lam=2.0, thr=0.25):
    seq_F, seq_u = window_data(c, ep, W)
    rows_H, rows_z = [], []
    for i in range(len(seq_F)):
        dts, n1 = seq_u[i]
        w_to = float(np.clip(dts / 30.0, 0, 1))
        w_cr = float((n1 - 4666.0) / (4400.0 - 4666.0))
        Hto = H['takeoff'] * (1 - w_to) + H['takeoff_hot'] * w_to
        Hcr = H['cruise'] * (1 - w_cr) + H['cruise_lowpwr'] * w_cr
        # takeoff block: only EGT channel measured in our takeoff dev (ch 3)
        rows_H.append(Hto[2:3])            # EGT row
        rows_z.append(seq_F[i, 3:4] / c['b_to_egt'] * 100.0)  # K -> %
        rows_H.append(Hcr)
        rows_z.append(seq_F[i, :3])
    Hs = np.vstack(rows_H)
    zs = np.concatenate(rows_z)
    A = Hs.T @ Hs + lam * np.eye(10)
    x = np.linalg.solve(A, Hs.T @ zs)
    x6 = np.array([x[HEALTH_PARAMS.index(p)] for p in TARGETS])
    j = int(np.argmax(np.abs(x6)))
    return TARGETS[j] if abs(x6[j]) > thr else 'none'


def gru_dataset(c, fleet, split, W, rng, per_ep=6):
    X, y = [], []
    for eid in split_engines(fleet, split):
        v = fleet[eid]
        eps = v['episodes']
        bounds = [int(o) for o, _ in eps] + [v['life']]
        # positive samples around each episode's eval zone
        H = load_H()
        for k, (onset, param) in enumerate(eps):
            hi = min(bounds[k + 1] - 1, v['life'] - 1)
            lo = min(int(onset) + 300, hi - 1)
            for t in rng.integers(lo, hi, size=per_ep):
                ep = {'engine': eid, 'onset': int(onset), 't': int(t)}
                sF, sU = window_data(c, ep, W)
                xh = per_block_xhat(c, H, sF, sU)
                X.append(np.concatenate([xh, sU_norm(sU)], axis=1))
                y.append(CLASSES.index(param))
        # negatives before first onset
        first = bounds[0] if eps else v['life']
        if first - 300 > W + 600:
            for t in rng.integers(W + 600, first - 300, size=per_ep):
                ep = {'engine': eid, 'onset': int(t) - 500, 't': int(t)}
                sF, sU = window_data(c, ep, W)
                xh = per_block_xhat(c, H, sF, sU)
                X.append(np.concatenate([xh, sU_norm(sU)], axis=1))
                y.append(0)
    L = max(len(x) for x in X)
    Xp = np.zeros((len(X), L, X[0].shape[1]), np.float32)
    for i, x in enumerate(X):
        Xp[i, -len(x):] = x
    return Xp, np.array(y)


def sU_norm(sU):
    return np.column_stack([sU[:, 0] / 30.0, (sU[:, 1] - 4550.0) / 120.0])


def run_learner(c, fleet, W, eval_split, rng, seed=0):
    import torch
    import torch.nn as nn
    from ehmbrain.ai.models import train_torch, predict_torch

    class GRUCls(nn.Module):
        def __init__(self, ch=12, hidden=64, ncls=7):
            super().__init__()
            self.gru = nn.GRU(ch, hidden, 1, batch_first=True)
            self.head = nn.Linear(hidden, ncls)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h[:, -1])

    fit_split = 'dev_fit' if eval_split == 'dev' else 'train'
    Xtr, ytr = gru_dataset(c, fleet, fit_split, W, rng)
    # per-channel standardization (x_hat channels are tiny) + class weights
    mu_x = Xtr.reshape(-1, Xtr.shape[2]).mean(axis=0)
    sd_x = Xtr.reshape(-1, Xtr.shape[2]).std(axis=0) + 1e-9
    Xtr = (Xtr - mu_x) / sd_x
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    wts = counts.sum() / np.maximum(counts, 1.0)
    wts = wts / wts.mean()
    from ehmbrain.ai.models import device as _dev
    net = GRUCls()
    train_torch(net, Xtr, ytr.astype(np.int64), epochs=80, lr=2e-3, seed=seed,
                loss_fn=nn.CrossEntropyLoss(
                    weight=torch.tensor(wts, dtype=torch.float32).to(_dev())))
    cpu = torch.device('cpu')

    Hd = load_H()

    def predict_ep(ep):
        sF, sU = window_data(c, ep, W)
        xh = per_block_xhat(c, Hd, sF, sU)
        x = np.concatenate([xh, sU_norm(sU)], axis=1).astype(np.float32)
        x = (x - mu_x) / sd_x
        logits = predict_torch(net, x[None], dev=cpu)
        return logits[0]

    return predict_ep


def aps_set(probs, qhat):
    order = np.argsort(-probs)
    cum, out = 0.0, []
    for j in order:
        out.append(j)
        cum += probs[j]
        if cum >= qhat:
            break
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'val'
    OUT.mkdir(parents=True, exist_ok=True)
    c = fleet_cache()
    fleet = c['fleet']
    # add operating-condition + takeoff baseline caches
    import pandas as pd
    FLEET_DIR = REPO_ROOT / 'data' / 'processed' / 'fleet'
    snap = pd.read_parquet(FLEET_DIR / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'to_dTs_C', 'cr_N1_cmd'])
    c['u'] = {}
    for eid in fleet:
        e = snap[snap.engine_id == eid].sort_values('cycle')
        c['u'][eid] = e[['to_dTs_C', 'cr_N1_cmd']].to_numpy(float)
    z = np.load(ICMD / 'icm_takeoff.npz', allow_pickle=True)
    c['b_to_egt'] = json.loads(str(z['baseline']))['EGT_degK']

    H = load_H()
    rng = np.random.default_rng(7)
    eps_eval = episode_table(fleet, mode)
    eps_val = episode_table(fleet, 'val')

    import scipy  # noqa: F401  (env check)
    results = {}
    for wname, W in WINDOWS.items():
        wls_pred = [stacked_wls(c, H, ep, W) for ep in eps_eval]
        predict_ep = run_learner(c, fleet, W, mode, rng, seed=0)
        logits = [predict_ep(ep) for ep in eps_eval]
        probs = [np.exp(l - l.max()) / np.exp(l - l.max()).sum() for l in logits]
        gru_pred = [CLASSES[int(np.argmax(p))] for p in probs]

        # conformal calibration on val episodes
        val_logits = [predict_ep(ep) for ep in eps_val]
        val_probs = [np.exp(l - l.max()) / np.exp(l - l.max()).sum()
                     for l in val_logits]
        scores = []
        for p, ep in zip(val_probs, eps_val):
            yidx = CLASSES.index(ep['param'])
            order = np.argsort(-p)
            cum = 0.0
            for j in order:
                cum += p[j]
                if j == yidx:
                    break
            scores.append(cum)
        qhat = float(np.quantile(scores, min(1.0, 0.9 * (1 + 1 / max(1, len(scores))))))
        sets = [aps_set(p, qhat) for p in probs]

        def acc(preds):
            return float(np.mean([pr == ep['param']
                                  for pr, ep in zip(preds, eps_eval)]))

        cover = float(np.mean([CLASSES.index(ep['param']) in s_
                               for s_, ep in zip(sets, eps_eval)]))
        size_by = {}
        for s_, ep in zip(sets, eps_eval):
            key = ('u_breakable' if ep['param'] in U_BREAKABLE else
                   'fundamental' if ep['param'] == 'hpc.eta' else 'other')
            size_by.setdefault(key, []).append(len(s_))
        results[wname] = {
            'window_cycles': W,
            'stacked_wls_acc': acc(wls_pred),
            'gru_acc': acc(gru_pred),
            'conformal': {'qhat': qhat, 'coverage': cover,
                          'median_set_size': {k: float(np.median(v))
                                              for k, v in size_by.items()}},
            'n_episodes': len(eps_eval)}
        print(f'[{mode}] W={W}: WLS {results[wname]["stacked_wls_acc"]:.2f}  '
              f'GRU {results[wname]["gru_acc"]:.2f}  '
              f'cover {cover:.2f}  sizes {results[wname]["conformal"]["median_set_size"]}',
              flush=True)

    (OUT / f'learner_{mode}.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
