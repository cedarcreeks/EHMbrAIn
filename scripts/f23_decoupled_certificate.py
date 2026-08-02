"""F23 (prereg-v26): test the certificate against an estimator that does not
use the influence matrix.

WHY. F22 showed H10.1's rho = 0.70 cannot clear a physics-free null (median
0.242, p = 0.085). The reason is structural: the Cramer-Rao bound and the Kalman
estimator are both computed from the SAME influence matrix H, and both degrade
when a column of H is weak -- whether or not that column is the physically right
one. Shuffling H moves both, so the shuffle test is blunt, and with ten health
directions a rank test cannot resolve what is left.

THE FIX IS TO BREAK THE COUPLING BY DESIGN, NOT BY SHUFFLING. Replace the Kalman
with a LEARNED estimator: a sequence model mapping the deviation trajectory to
the ten health parameters, trained on train-split engines against ground truth.
It never sees H. So:

  * the certificate's CRB still comes from H (physics),
  * the estimator's per-direction error comes only from the data,
  * and permuting H changes the CRB while leaving the error untouched.

That makes the permutation an honest null: it asks whether THIS ordering of
certified precision matches the error ordering, against all orderings. And it
costs nothing to run 10 000 of them, because nothing is retrained -- which is why
this test is cheaper than the one it replaces.

WHAT A POSITIVE RESULT WOULD MEAN. If the CRB still ranks a learned estimator's
error, the agreement cannot be the shared matrix, because there is no shared
matrix. It would mean the certificate predicts what is recoverable from the data
at all -- which is what H10.1 always claimed and what F22 could not establish.

WHAT A NULL WOULD MEAN. That the certificate's ordering carries no information
about achievable accuracy once the coupling is removed, and C8's honesty claim
does not survive. Either outcome is reportable; neither requires the certificate
to be wrong about the physics it certifies.

Output: data/processed/f23/decoupled_verdict.json
Usage: uv run python scripts/f23_decoupled_certificate.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                        # noqa: E402
from f13_gate1_mechanism import MechNet, SEQ_LEN, r2_per_col      # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                       # noqa: E402
from ehmbrain.trad.identifiability import Certificate, COCKPIT    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f23'
SEEDS = tuple(range(10))
LATE = 0.85            # error is read over the last 15 % of life, as F10 does
CUTS = 8               # cuts per engine, late life only
N_PERM = 10000
ALPHA = 0.05


def build(c, fleet, ids, snap, rng):
    """Deviation trajectory -> the true 10-dim health state at the cut.

    The estimator sees only the four cockpit deviation channels every other
    experiment here uses. It is never shown H, which is the whole point.
    """
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    mu, sd = c['norm']
    X, Y, E = [], [], []
    for eid in ids:
        dev = c['dev'][eid]
        e = snap[snap.engine_id == eid].sort_values('cycle')
        theta = e[Xc].to_numpy()
        n = min(len(dev), len(theta))
        lo = int(LATE * n)
        if n - lo < 20:
            continue
        for t in rng.integers(lo, n, size=CUTS):
            t = int(t)
            seg = (dev[:t] - mu) / sd
            idx = np.linspace(0, len(seg) - 1, SEQ_LEN)
            s = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, j])
                          for j in range(seg.shape[1])], axis=1)
            X.append(np.concatenate([s, np.full((SEQ_LEN, 1), t / 10000.0)],
                                    axis=1))
            Y.append(theta[t])
            E.append(eid)
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(E))


def _shard_main(shard, n_shards):
    import torch
    torch.set_num_threads(1)
    from ehmbrain.ai.models import predict_torch, train_torch
    z = np.load(OUT / 'cache.npz')
    Xtr, Ytr, Xte = z['Xtr'], z['Ytr'], z['Xte']
    par = json.loads((REPO_ROOT / 'data' / 'processed' / 'f13' /
                      'gate1_verdict.json').read_text())['setup']['best_params_sequence']
    cpu = torch.device('cpu')
    out = {}
    for s in [s for i, s in enumerate(SEEDS) if i % n_shards == shard]:
        m = train_torch(MechNet(ch=Xtr.shape[2], hidden=par['hidden'],
                                layers=par['layers'], n_out=Ytr.shape[1]),
                        Xtr, Ytr, epochs=par['epochs'], lr=par['lr'],
                        bs=par['bs'], seed=s, dev=cpu)
        out[str(s)] = predict_torch(m, Xte, dev=cpu)
        print(f'  shard {shard}: seed {s} done', flush=True)
    np.savez_compressed(OUT / f'preds_{shard}.npz', **out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = fleet_cache()
    fleet = c['fleet']
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'cr_N1_cmd'] + Xc)
    rng = np.random.default_rng(11)
    print('== building (learned estimator sees no H) ==', flush=True)
    Xtr, Ytr, _ = build(c, fleet, split_ids(fleet, 'train'), snap, rng)
    Xte, Yte, Ete = build(c, fleet, split_ids(fleet, 'test'), snap, rng)
    print(f'   train {len(Ytr)} / test {len(Yte)} cuts', flush=True)
    np.savez_compressed(OUT / 'cache.npz', Xtr=Xtr, Ytr=Ytr, Xte=Xte)

    n_shards = int(os.environ.get('F23_SHARDS', 4))
    print(f'== {len(SEEDS)} trainings over {n_shards} shards ==', flush=True)
    t0 = time.time()
    procs = [subprocess.Popen(
        [sys.executable, '-u', str(Path(__file__).resolve()),
         '--shard', str(i), str(n_shards)]) for i in range(n_shards)]
    for p in procs:
        p.wait()
    if any(p.returncode for p in procs):
        raise SystemExit('shards failed')
    print(f'   done [{(time.time() - t0) / 60:.1f} min]', flush=True)

    preds = {}
    for i in range(n_shards):
        z = np.load(OUT / f'preds_{i}.npz')
        preds.update({int(k): z[k] for k in z.files})
    P = np.mean([preds[s] for s in SEEDS], axis=0)

    # per-direction absolute error of the LEARNED estimator, per engine
    err_by_engine = {}
    for eid in np.unique(Ete):
        m = Ete == eid
        err_by_engine[int(eid)] = np.abs(P[m] - Yte[m]).mean(0)
    # the certificate, from H -- unchanged from F10
    cert = Certificate(COCKPIT)
    sn = pd.read_parquet(FLEET / 'snapshots.parquet',
                         columns=['engine_id', 'cycle', 'cr_N1_cmd'])
    crb_by_engine = {}
    for eid in err_by_engine:
        g = sn[sn.engine_id == eid].sort_values('cycle')
        n1 = g.cr_N1_cmd.to_numpy()
        cr = cert.certify(n1[int(0.7 * len(n1)):])
        crb_by_engine[eid] = np.array([cr['std_pct'][p] for p in HEALTH_PARAMS])

    ids = sorted(err_by_engine)
    err = np.median([err_by_engine[e] for e in ids], axis=0)
    crb = np.median([crb_by_engine[e] for e in ids], axis=0)
    rho, p_param = spearmanr(crb, err)

    # PERMUTATION NULL, honest because the estimator never used H: permuting the
    # certified precisions changes the claim under test and nothing else.
    r = np.random.default_rng(0)
    null = np.array([spearmanr(r.permutation(crb), err).statistic
                     for _ in range(N_PERM)])
    p_emp = float((null >= rho).mean())
    # per-seed, to show the result is not one lucky model
    per_seed = []
    for s in SEEDS:
        Ps = preds[s]
        e_s = np.median([np.abs(Ps[Ete == e] - Yte[Ete == e]).mean(0)
                         for e in ids], axis=0)
        per_seed.append(float(spearmanr(crb, e_s).statistic))

    verdict = {
        'design': {
            'estimator': ('sequence model, deviation trajectory -> 10-dim health '
                          'state, trained on train-split engines against ground '
                          'truth. It never sees the influence matrix'),
            'why': ('F22 could not clear a physics-free null because the bound '
                    'and the Kalman share H. A learned estimator breaks that '
                    'coupling by design: permuting H changes the CRB and leaves '
                    'the error untouched, so the permutation is an honest null'),
            'n_seeds': len(SEEDS), 'n_permutations': N_PERM,
            'late_life_window': LATE},
        'crb_pct': crb.tolist(), 'learned_abs_err_pct': err.tolist(),
        'health_params': list(HEALTH_PARAMS),
        'rho': float(rho), 'p_parametric': float(p_param),
        'permutation_null': {'median': float(np.median(null)),
                             'p95': float(np.percentile(null, 95)),
                             'empirical_p': p_emp},
        'rho_per_seed': per_seed,
        'comparison': {'F10_kalman_rho': 0.697,
                       'F10_shuffle_null_median': 0.242,
                       'F10_shuffle_p': 0.085},
        'confirmed': bool(p_emp < ALPHA),
        'reading': ('confirmed: the certificate ranks the error of an estimator '
                    'that never used H, so the agreement cannot be the shared '
                    'matrix' if p_emp < ALPHA else
                    'not confirmed: with the coupling removed the certificate '
                    'ordering carries no demonstrable information about '
                    'achievable accuracy'),
    }
    (OUT / 'decoupled_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    print('  direction        CRB [%]   learned |err| [%]')
    for nm, a, b in zip(HEALTH_PARAMS, crb, err):
        print(f'  {nm:14s} {a:8.3f}   {b:8.3f}')
    print(f'\n  rho = {rho:+.3f}   permutation null median '
          f'{np.median(null):+.3f}, p95 {np.percentile(null, 95):.3f}')
    print(f'  empirical p = {p_emp:.4f}   (F10/Kalman was p = 0.085)')
    print(f'  per-seed rho: {[round(x, 2) for x in per_seed]}')
    print(f'  CONFIRMED: {verdict["confirmed"]}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--shard':
        OUT.mkdir(parents=True, exist_ok=True)
        _shard_main(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
