"""F4: physics-informed hybrid RUL (stacking mechanism, contribution C3) and
the H4 data-efficiency experiment.

Hybrid = the same GRU, but its input gains the 10 Kalman-GPA tracked health
parameters (the traditional pipeline's physics-based state estimate) alongside
the 4 raw deviation channels. Pure vs hybrid trained at 10 %, 25 % and 100 %
of the training engines; both evaluated on the full test split at 50/70/90 %
of life. Hypothesis H4: the physics input matters most when data is scarce.

MUST run in the foreground: torch-MPS segfaults under backgrounded execution
(recorded platform finding). Kalman tracks are computed once in parallel
worker processes (norm N1) and cached.

Output: data/processed/ai/hybrid_metrics.json
Usage: uv run python scripts/run_hybrid.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ehmbrain.ai.data import load_fleet_features, normalization
from ehmbrain.ai.models import RULNet, predict_torch, train_torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'ai'
CACHE = OUT / 'kalman_tracks.npz'

RUL_DS = 20
RUL_SEQ = 64
RUL_CAP = 12.0
EVAL_FRACS = (0.5, 0.7, 0.9)
TRAIN_FRACS = (0.1, 0.25, 1.0)
SEEDS = (0, 1, 2)


def _kalman_worker(eid):
    import pandas as pd
    from ehmbrain.trad.pipeline import COCKPIT, BaselineModel, kalman_gpa
    e = pd.read_parquet(FLEET / 'snapshots.parquet',
                        filters=[('engine_id', '==', eid)]).sort_values('cycle')
    ev = pd.read_parquet(FLEET / 'events.parquet')
    washes = ev[(ev.engine_id == eid) & (ev.type == 'wash')].cycle.tolist()
    bm = BaselineModel()
    dz = bm.deviations(e[[f'cr_{c}' for c in COCKPIT]].to_numpy(float),
                       e.cr_N1_cmd.to_numpy())
    Ha, Hb, w = bm.cruise(e.cr_N1_cmd.to_numpy())[1]
    xs = kalman_gpa(dz, lambda i: Ha * (1 - w[i]) + Hb * w[i],
                    [0.07, 0.5, 0.23], q=2e-4, wash_cycles=washes)
    return eid, xs.astype(np.float32)


def kalman_tracks(fleet):
    if CACHE.exists():
        d = np.load(CACHE)
        return {int(k): d[k] for k in d.files}
    with ProcessPoolExecutor() as pool:
        tracks = dict(pool.map(_kalman_worker, list(fleet)))
    np.savez_compressed(CACHE, **{str(k): v for k, v in tracks.items()})
    return tracks


def sequences(fleet, tracks, mu, sd, eids, cuts_per_engine, rng, hybrid):
    X, y = [], []
    for eid in eids:
        v = fleet[eid]
        Fn = (v['F'] - mu) / sd
        feats = np.concatenate([Fn, tracks[eid]], axis=1) if hybrid else Fn
        feats = feats[::RUL_DS]
        cuts = rng.integers(RUL_SEQ * RUL_DS, v['life'], size=cuts_per_engine)
        for cut in cuts:
            i = int(cut) // RUL_DS
            if i < RUL_SEQ:
                continue
            X.append(feats[i - RUL_SEQ:i])
            y.append(min((v['life'] - int(cut)) / 1000.0, RUL_CAP))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def eval_sequences(fleet, tracks, mu, sd, eids, hybrid):
    rows = []
    for eid in eids:
        v = fleet[eid]
        Fn = (v['F'] - mu) / sd
        feats = np.concatenate([Fn, tracks[eid]], axis=1) if hybrid else Fn
        feats = feats[::RUL_DS]
        for f in EVAL_FRACS:
            cut = int(f * v['life'])
            i = cut // RUL_DS
            if i < RUL_SEQ:
                continue
            rows.append((f, feats[i - RUL_SEQ:i],
                         min((v['life'] - cut) / 1000.0, RUL_CAP)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading fleet + Kalman tracks...', flush=True)
    fleet = load_fleet_features()
    mu, sd = normalization(fleet)
    tracks = kalman_tracks(fleet)

    train_ids = sorted(e for e, v in fleet.items() if v['split'] == 'train')
    test_ids = sorted(e for e, v in fleet.items() if v['split'] == 'test')

    results = {}
    for hybrid in (False, True):
        name = 'hybrid' if hybrid else 'pure'
        ch = 14 if hybrid else 4
        test_rows = eval_sequences(fleet, tracks, mu, sd, test_ids, hybrid)
        for tf in TRAIN_FRACS:
            rmses = []
            for seed in SEEDS:
                rng = np.random.default_rng(100 + seed)
                sub = list(np.random.default_rng(seed).permutation(train_ids)
                           [:max(3, int(tf * len(train_ids)))])
                X, y = sequences(fleet, tracks, mu, sd, sub, 40, rng, hybrid)
                net = train_torch(RULNet(ch=ch), X, y, epochs=40, seed=seed)
                import torch
                cpu = torch.device('cpu')
                errs = []
                for f, seq, true in test_rows:
                    pred = float(predict_torch(net, seq[None], dev=cpu)[0])
                    errs.append((min(pred, RUL_CAP) - true) * 1000.0)
                rmses.append(float(np.sqrt(np.mean(np.square(errs)))))
            key = f'{name}@{int(tf * 100)}%'
            results[key] = {'rmse_cycles_mean': float(np.mean(rmses)),
                            'rmse_cycles_std': float(np.std(rmses)),
                            'n_train_engines': max(3, int(tf * len(train_ids))),
                            'seeds': list(SEEDS)}
            print(f'{key:12s} RMSE {np.mean(rmses):7.0f} ± {np.std(rmses):5.0f} '
                  f'({results[key]["n_train_engines"]} engines)', flush=True)

    (OUT / 'hybrid_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
