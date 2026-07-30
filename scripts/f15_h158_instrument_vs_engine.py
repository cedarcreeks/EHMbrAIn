"""F15/H15.8 (prereg-v18): instrument or engine?

Question. A sensor bias and a real gas-path fault are perfectly confusable at a
snapshot -- the cockpit ICM has rank 3 and maps R^10 -> R^3 surjectively, so
EVERY deviation vector, including a pure single-channel bias, has a health-state
preimage. That is the same rank argument that makes H2 unbreakable. The open
question is whether they stay confusable over a HISTORY: real degradation moves
along smooth mechanism trajectories coordinated across channels, while a bias
ramp moves along one coordinate axis. This is the second independent test of the
structure F13 gate one found for mechanism attribution.

L7 already established the classical half (prereg-v6): an augmented-state Kalman
TRACKS an EGT bias (Spearman 0.83 over 13 engines) but cannot un-corrupt the
diagnosis (phantom error 1.69 -> 1.64 %, ~3 %). Tracking was tested;
CLASSIFICATION -- is this an instrument problem at all? -- never was.

DESIGN NOTES, both forced by the data and both disclosed.

1. Only cockpit-VISIBLE drift can count as a positive. The fleet drifts four
   channels (fault_catalog.yaml:78) but COCKPIT is [N2_rpm, WF_kgps, EGT_degK],
   so T25_degK and PS3_bar biases are invisible by construction. Positives are
   the EGT and WF engines; the T25/PS3 engines are held out as a FRAUD CHECK --
   any detector that scores them like positives is hallucinating, since the
   information is not in the input. Same role G5 plays in docs/f12-proposal.md.

2. The frozen F5 splits put only 3 drifted engines in test, which is why L7's
   confirmatory pass was under-powered and said so. This task therefore uses
   grouped 5-fold cross-validation over engine IDs, pooling out-of-fold scores,
   so every positive engine is scored while held out. That deviates from the
   frozen-split convention used by F5/F7/F10 and is a different task from those;
   the deviation is recorded here rather than hidden.

Both families see the same 4 cockpit deviation channels used everywhere else.

Output: data/processed/f15/h158_verdict.json
Usage: uv run python scripts/f15_h158_instrument_vs_engine.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache                                   # noqa: E402
from f8_l7_drift import kalman                                    # noqa: E402
from ehmbrain.datagen.fleet import load_icm                       # noqa: E402
from ehmbrain.trad.pipeline import BaselineModel, COCKPIT         # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f15'
VISIBLE = ('EGT_degK', 'WF_kgps')      # drift channels the cockpit set can see
INVISIBLE = ('T25_degK', 'PS3_bar')    # drift channels it structurally cannot
STRIDE = 25                            # Kalman subsample (F10 uses 20 for Fisher)
SEQ_LEN = 256
FOLDS = 5
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def drift_labels():
    """Per engine: drifted?, which channel, and the onset cycle."""
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'drift_active',
                                    'drift_channel'])
    rows = {}
    for eid, g in snap.groupby('engine_id'):
        act = g.drift_active.to_numpy()
        ch = g.drift_channel.dropna()
        onset = int(g.cycle.to_numpy()[np.argmax(act)]) if act.any() else None
        rows[eid] = {'drift': bool(act.any()),
                     'channel': (str(ch.iloc[-1]) if len(ch) else None),
                     'onset': onset, 'life': int(g.cycle.max())}
    return rows


# --------------------------------------------------------------------------
# family A: classical -- augmented-state Kalman bias magnitude (L7 machinery)
# --------------------------------------------------------------------------

def classical_scores(labels):
    """Max |b_hat| over the trajectory, per candidate biased channel, taking the
    strongest as the engine's 'this is an instrument' score. No training, so no
    cross-validation needed -- every engine is scored the same way."""
    H, ch, base = load_icm('cruise')
    bm = BaselineModel()
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'cr_N1_cmd']
                           + [f'cr_{c}' for c in COCKPIT])
    out, traj = {}, {}
    for eid, g in snap.groupby('engine_id'):
        g = g.sort_values('cycle')
        n1 = g.cr_N1_cmd.to_numpy()
        meas = g[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
        dz = bm.deviations(meas, n1)[::STRIDE]
        Ha, Hb, w = bm.cruise(n1)[1]
        ws = w[::STRIDE]

        def H_at(i):
            return Ha * (1 - ws[i]) + Hb * ws[i]

        best, best_series = 0.0, None
        for j in range(len(COCKPIT)):
            xs = kalman(dz, H_at, augmented=True, bias_row=j)
            b = np.abs(xs[:, 10])
            # normalise by that channel's noise so channels are comparable
            m = float(np.nanmax(b))
            if m > best:
                best, best_series = m, b
        out[eid] = best
        traj[eid] = best_series
    return out, traj


# --------------------------------------------------------------------------
# family B: sequence model over the deviation trajectory, grouped 5-fold CV
# --------------------------------------------------------------------------

def sequences(c, ids, labels, cuts=12, rng=None):
    X, y, e = [], [], []
    mu, sd = c['norm']
    for eid in ids:
        dev = (c['dev'][eid] - mu) / sd
        n = len(dev)
        lab = 1.0 if labels[eid]['drift'] else 0.0
        for t in rng.integers(int(0.3 * n), n, size=cuts):
            seg = dev[:int(t)]
            idx = np.linspace(0, len(seg) - 1, SEQ_LEN)
            s = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, j])
                          for j in range(seg.shape[1])], axis=1)
            X.append(np.concatenate([s, np.full((SEQ_LEN, 1), t / 10000.0)],
                                    axis=1))
            y.append(lab)
            e.append(eid)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(e)


def net(ch, hidden=64):
    """Bidirectional GRU with the CORRECT read of a bidirectional output.

    The first version of this script took h[:, -1]. For the forward pass that is
    right -- the state at the last step has seen everything. For the backward
    pass it is wrong: a backward pass starts at the end, so at position -1 it has
    seen exactly one sample, and half the representation handed to the head was
    close to the raw last observation. Corrected here to forward-at-last
    concatenated with backward-at-first (same bug and fix as F18/H15.2).
    """
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = hidden
            self.gru = nn.GRU(ch, hidden, 2, batch_first=True, dropout=0.1,
                              bidirectional=True)
            self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.GELU(),
                                      nn.Linear(32, 1))

        def forward(self, x):
            o, _ = self.gru(x)
            z = torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1)
            return self.head(z).squeeze(-1)

    return Net()


def sequence_scores(c, labels, pos_ids, neg_ids):
    """Pooled out-of-fold engine scores from grouped 5-fold CV."""
    import torch.nn as nn
    from ehmbrain.ai.models import predict_torch, train_torch

    ids = sorted(pos_ids + neg_ids)
    rng = np.random.default_rng(5)
    order = rng.permutation(ids)
    folds = np.array_split(order, FOLDS)
    scores = {}
    for k, te_ids in enumerate(folds):
        tr_ids = [i for i in ids if i not in set(te_ids.tolist())]
        Xtr, ytr, _ = sequences(c, tr_ids, labels, cuts=12,
                                rng=np.random.default_rng(100 + k))
        Xte, yte, ete = sequences(c, te_ids.tolist(), labels, cuts=12,
                                  rng=np.random.default_rng(200 + k))
        preds = []
        for s in SEEDS:
            m = train_torch(net(Xtr.shape[2]), Xtr, ytr, epochs=40, lr=1e-3,
                            bs=64, seed=s, loss_fn=nn.BCEWithLogitsLoss())
            preds.append(predict_torch(m, Xte))
        p = np.mean(preds, axis=0)
        for eid in np.unique(ete):                 # engine score = mean of cuts
            scores[int(eid)] = float(np.mean(p[ete == eid]))
        print(f'  fold {k + 1}/{FOLDS} done ({len(te_ids)} engines)', flush=True)
    return scores


# --------------------------------------------------------------------------

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float('nan')
    gt = sum((p > neg).sum() for p in pos)
    tie = sum((p == neg).sum() for p in pos)
    return float((gt + 0.5 * tie) / (len(pos) * len(neg)))


def boot_auc(pos, neg, n=2000, seed=0):
    r = np.random.default_rng(seed)
    vals = [auc(r.choice(pos, len(pos), replace=True),
                r.choice(neg, len(neg), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = drift_labels()
    c = fleet_cache()

    pos = [e for e, v in labels.items() if v['drift'] and v['channel'] in VISIBLE]
    inv = [e for e, v in labels.items() if v['drift'] and v['channel'] in INVISIBLE]
    neg = [e for e, v in labels.items() if not v['drift']]
    print(f'visible-drift positives {len(pos)}  invisible {len(inv)}  '
          f'clean {len(neg)}', flush=True)

    print('== family A: augmented-state Kalman (L7) ==', flush=True)
    cls, _ = classical_scores(labels)
    print('== family B: Bi-GRU, grouped 5-fold CV ==', flush=True)
    seq = sequence_scores(c, labels, pos + inv, neg)

    res = {}
    for name, sc in (('classical_augmented_kalman', cls), ('sequence_model', seq)):
        p = [sc[e] for e in pos if e in sc]
        n_ = [sc[e] for e in neg if e in sc]
        i_ = [sc[e] for e in inv if e in sc]
        a = auc(p, n_)
        lo, hi = boot_auc(p, n_)
        res[name] = {
            'auc_visible_drift': a, 'auc_ci95': [lo, hi],
            'n_pos': len(p), 'n_neg': len(n_),
            'fraud_check_invisible_drift': {
                'auc_vs_clean': auc(i_, n_), 'n': len(i_),
                'note': ('T25/PS3 biases are not in the cockpit input; an AUC '
                         'materially above 0.5 here means the score is reading '
                         'something other than the drift')}}

    verdict = {
        'design': {
            'positives': 'cockpit-visible drift (EGT_degK, WF_kgps)',
            'held_out_fraud_check': 'invisible drift (T25_degK, PS3_bar)',
            'cv': f'grouped {FOLDS}-fold over engine IDs, pooled out-of-fold',
            'deviation_disclosed': ('frozen F5 splits put only 3 drifted engines '
                                    'in test, which under-powered L7; this task '
                                    'uses grouped CV instead'),
            'inputs': '4 cockpit deviation channels + age, identical both families',
            'seeds': list(SEEDS)},
        'per_family': res,
        'superseded_first_run': {
            'classical_auc': 0.614, 'sequence_auc': 0.524,
            'fraud_check': {'classical': 0.433, 'sequence': 0.377},
            'note': ('the first pass read h[:, -1] from a bidirectional GRU, so '
                     'the backward half of the representation had seen one '
                     'sample. The classical arm shares none of that code and its '
                     'number is unchanged; the sequence arm is re-run here with '
                     'the pooling corrected')},
        'H15.8_instrument_vs_engine': {
            'auc_sequence': res['sequence_model']['auc_visible_drift'],
            'auc_classical': res['classical_augmented_kalman']['auc_visible_drift'],
            'delta': (res['sequence_model']['auc_visible_drift']
                      - res['classical_augmented_kalman']['auc_visible_drift']),
            'confirmed': None,      # filled below
            'note': ('L7 showed the augmented Kalman TRACKS drift but cannot '
                     'un-corrupt the diagnosis; this asks whether either family '
                     'can CLASSIFY it')},
    }
    s_lo = res['sequence_model']['auc_ci95'][0]
    c_hi = res['classical_augmented_kalman']['auc_ci95'][1]
    verdict['H15.8_instrument_vs_engine']['confirmed'] = bool(s_lo > c_hi)
    verdict['H15.8_instrument_vs_engine']['criterion'] = (
        'sequence CI lower bound above classical CI upper bound (non-overlap)')

    (OUT / 'h158_verdict.json').write_text(json.dumps(verdict, indent=2))
    print()
    for k, v in res.items():
        print(f"  {k:28s} AUC {v['auc_visible_drift']:.3f} "
              f"[{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]   "
              f"fraud-check AUC {v['fraud_check_invisible_drift']['auc_vs_clean']:.3f}")
    print(f"\n  H15.8 confirmed: "
          f"{verdict['H15.8_instrument_vs_engine']['confirmed']}")


if __name__ == '__main__':
    main()
