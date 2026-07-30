"""F19 / H15.11 (prereg-v22): does the certificate carry information, with the
channel count held fixed?

WHERE THIS COMES FROM. L-HYB (sec:f17-hybrid) refuted H15.3 because its control
fired: arm C (certificate) beat arm A (pure) by +0.191, but arm B (raw GPA
estimate, the mechanism H4 refuted) beat it by +0.139, so the gain was channel
count. A post-hoc C - B of +0.053 (t = 2.24, p = 0.026) was reported and
explicitly NOT claimed.

WHY THAT POST-HOC WAS ALSO CONFOUNDED. Arm C had 25 channels and arm B had 15.
So C - B carried exactly the same capacity confound that killed C - A, one level
down. Reporting it as evidence for the certificate would have repeated the error
the control was built to catch. This study removes the confound instead of
arguing around it.

AND ARM C CHANGED TWO THINGS AT ONCE. It masked the GPA estimate to the
certified-identifiable subspace AND appended the per-direction CRB as static
channels. Those are different claims: one says the certificate's *tagging* is
informative, the other says its *magnitudes* are. Four arms separate them, and
two of the four are matched on channel count:

  B    raw 10-dim GPA estimate                        15 ch   (reference)
  C1   estimate MASKED to the identifiable subspace   15 ch   <- same count as B
  C2   raw estimate + per-direction CRB channels      25 ch
  C    masked estimate + CRB channels                 25 ch   (L-HYB's arm C)

PRIMARY TEST: C1 vs B. Same channel count, same architecture, same seeds. The
only difference is whether the certificate is used to zero the directions it
says are unrecoverable. If C1 beats B there is no capacity explanation left --
the certificate's tags carry information the trajectory does not.

SECONDARY: C vs C2, also matched at 25 channels, asks the same question in the
presence of the CRB channels. C2 vs B is reported but remains count-confounded
and is not a test.

POWER. The L-HYB paired difference was +0.0529 with a paired sd of 0.0748, giving
t = 2.24 at ten seeds. Fifteen seeds put the same effect at t = 2.74, which is
why fifteen were chosen. Direction is taken from that post-hoc and disclosed.

Output: data/processed/f19/cert_isolated_verdict.json
Usage: uv run python scripts/f19_certificate_isolated.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                        # noqa: E402
from f13_gate1_mechanism import (MECHANISMS, CUT_MIN, SEQ_LEN,    # noqa: E402
                                 CUTS_PER_ENGINE, MIN_LOSS_C,
                                 MechNet, mechanism_shares, r2_per_col)
from f17_certificate_hybrid import physics_channels               # noqa: E402
from ehmbrain.datagen.fleet import load_icm                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f19'
SEEDS = tuple(range(15))
ALPHA = 0.05
ARMS = ('B_raw', 'C1_masked', 'C2_raw_plus_crb', 'C_masked_plus_crb')
# frozen: the pre-registered primary and secondary contrasts, both count-matched
CONTRASTS = (('C1_masked', 'B_raw', 'primary'),
             ('C_masked_plus_crb', 'C2_raw_plus_crb', 'secondary'))


def _all_jobs():
    return [(a, s) for a in ARMS for s in SEEDS]


def _shard_main(shard, n_shards):
    import torch
    torch.set_num_threads(1)
    from ehmbrain.ai.models import predict_torch, train_torch
    z = np.load(OUT / 'cache.npz')
    Y = z['Y']
    par = json.loads((REPO_ROOT / 'data' / 'processed' / 'f13' /
                      'gate1_verdict.json').read_text())['setup']['best_params_sequence']
    cpu = torch.device('cpu')
    jobs = [j for i, j in enumerate(_all_jobs()) if i % n_shards == shard]
    out = {}
    for k, (arm, seed) in enumerate(jobs, 1):
        Xtr, Xte = z[f'Xtr_{arm}'], z[f'Xte_{arm}']
        m = train_torch(MechNet(ch=Xtr.shape[2], hidden=par['hidden'],
                                layers=par['layers']),
                        Xtr, Y, epochs=par['epochs'], lr=par['lr'],
                        bs=par['bs'], seed=seed, dev=cpu)
        out[f'{arm}|{seed}'] = predict_torch(m, Xte, dev=cpu)
        print(f'  shard {shard}: {k}/{len(jobs)}  {arm} seed {seed}', flush=True)
    np.savez_compressed(OUT / f'preds_{shard}.npz', **out)


def build(catalog, H, ch, base, c, ids, phys, rng):
    """One set of cuts, four input encodings of the same physics."""
    X = {a: [] for a in ARMS}
    Y = []
    mu, sd = c['norm']
    for eid in ids:
        life, shares, total = mechanism_shares(catalog, eid, H, ch, base)
        dev = c['dev'][eid]
        p = phys[eid]
        n = min(len(dev), life, len(p['xhat']))
        if n <= CUT_MIN + 200:
            continue
        for t in rng.integers(CUT_MIN, n, size=CUTS_PER_ENGINE):
            t = int(t)
            if abs(total[t]) < MIN_LOSS_C:
                continue
            seg = (dev[:t] - mu) / sd
            idx = np.linspace(0, len(seg) - 1, SEQ_LEN)
            base_seq = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, j])
                                 for j in range(seg.shape[1])], axis=1)
            age = np.full((SEQ_LEN, 1), t / 10000.0)
            A = np.concatenate([base_seq, age], axis=1)

            xh = p['xhat'][:t]
            xh_r = np.stack([np.interp(idx, np.arange(len(xh)), xh[:, j])
                             for j in range(xh.shape[1])], axis=1)
            gated = xh_r * p['mask'][None, :]
            crb = np.tile(p['crb'][None, :], (SEQ_LEN, 1))

            X['B_raw'].append(np.concatenate([A, xh_r], axis=1))
            X['C1_masked'].append(np.concatenate([A, gated], axis=1))
            X['C2_raw_plus_crb'].append(np.concatenate([A, xh_r, crb], axis=1))
            X['C_masked_plus_crb'].append(np.concatenate([A, gated, crb], axis=1))
            Y.append(shares[t])
    return ({a: np.asarray(X[a], np.float32) for a in ARMS},
            np.asarray(Y, np.float32))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    H, ch, base = load_icm('takeoff_hot')
    c = fleet_cache()
    fleet = c['fleet']
    print('== physics channels: Kalman GPA + certificate per engine ==', flush=True)
    phys = physics_channels()
    rng = np.random.default_rng(11)
    print('== building sequences ==', flush=True)
    Xtr, Ytr = build(catalog, H, ch, base, c, split_ids(fleet, 'train'), phys, rng)
    Xte, Yte = build(catalog, H, ch, base, c, split_ids(fleet, 'test'), phys, rng)
    print('   train %d / test %d cuts; channels %s' %
          (len(Ytr), len(Yte), {a: Xtr[a].shape[2] for a in ARMS}), flush=True)

    np.savez_compressed(OUT / 'cache.npz', Y=Ytr,
                        **{f'Xtr_{a}': Xtr[a] for a in ARMS},
                        **{f'Xte_{a}': Xte[a] for a in ARMS})
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
            a, sd_ = k.split('|')
            preds[(a, int(sd_))] = z[k]

    from scipy import stats
    ymean = Ytr.mean(axis=0)
    per_seed, res = {}, {}
    for arm in ARMS:
        per_seed[arm] = {s: float(np.nanmean(r2_per_col(Yte, preds[(arm, s)], ymean)))
                         for s in SEEDS}
        v = list(per_seed[arm].values())
        res[arm] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
                    'n_channels': int(Xtr[arm].shape[2]),
                    'per_seed': per_seed[arm]}

    verdict = {
        'design': {
            'arms': {'B_raw': 'raw 10-dim Kalman GPA estimate (15 ch)',
                     'C1_masked': 'estimate masked to the certified-identifiable '
                                  'subspace (15 ch -- SAME count as B)',
                     'C2_raw_plus_crb': 'raw estimate + per-direction CRB (25 ch)',
                     'C_masked_plus_crb': 'masked estimate + CRB (25 ch, L-HYB arm C)'},
            'seeds': list(SEEDS), 'shared_hyperparameters': True,
            'why': ('L-HYB reported a post-hoc C - B of +0.053, but C had 25 '
                    'channels and B had 15, so it carried the same capacity '
                    'confound that killed C - A. The primary contrast here is '
                    'count-matched, so a win has no capacity explanation left'),
            'power': ('the L-HYB paired difference was +0.0529 with paired sd '
                      '0.0748 (t = 2.24 at n = 10); fifteen seeds put the same '
                      'effect at t = 2.74'),
            'direction_disclosed': ('one-sided, in the direction of the L-HYB '
                                    'post-hoc, which is exploratory evidence and '
                                    'is named as such')},
        'per_arm': res}

    for hi, lo, role in CONTRASTS:
        a = [per_seed[hi][s] for s in SEEDS]
        b = [per_seed[lo][s] for s in SEEDS]
        t = stats.ttest_rel(a, b, alternative='greater')
        verdict[f'{role}_{hi}_vs_{lo}'] = {
            'delta': float(np.mean(a) - np.mean(b)),
            'paired_t': float(t.statistic), 'p_one_sided': float(t.pvalue),
            'significant': bool(t.pvalue < ALPHA),
            'channels_matched': res[hi]['n_channels'] == res[lo]['n_channels']}
    # reported, not a test: count-confounded
    a = [per_seed['C2_raw_plus_crb'][s] for s in SEEDS]
    b = [per_seed['B_raw'][s] for s in SEEDS]
    verdict['reported_only_C2_vs_B'] = {
        'delta': float(np.mean(a) - np.mean(b)),
        'note': 'count-confounded (25 vs 15 channels); reported, not a test'}

    prim = verdict['primary_C1_masked_vs_B_raw']
    verdict['H15.11_certificate_tags_carry_information'] = {
        'confirmed': bool(prim['significant'] and prim['channels_matched']),
        'criterion': ('the count-matched primary contrast C1 vs B must reach '
                      'one-sided p < 0.05 across fifteen seeds. Channel count is '
                      'equal by construction, so a win cannot be capacity'),
        'lhyb_posthoc_disclosed': {'delta': 0.0529, 't': 2.24, 'p': 0.026,
                                   'status': 'exploratory, count-confounded'}}
    (OUT / 'cert_isolated_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    for arm in ARMS:
        r = res[arm]
        print(f"  {arm:20s} {r['n_channels']:3d} ch  {r['mean']:+.3f} +-{r['sd']:.3f}")
    print()
    for hi, lo, role in CONTRASTS:
        d = verdict[f'{role}_{hi}_vs_{lo}']
        print(f"  {role:9s} {hi} - {lo}: {d['delta']:+.3f}  t={d['paired_t']:.2f}  "
              f"p={d['p_one_sided']:.4f}  matched={d['channels_matched']}"
              f"{'  *' if d['significant'] else ''}")
    print(f"  reported only  C2 - B: "
          f"{verdict['reported_only_C2_vs_B']['delta']:+.3f} (count-confounded)")
    print(f"\n  H15.11 confirmed: "
          f"{verdict['H15.11_certificate_tags_carry_information']['confirmed']}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--shard':
        OUT.mkdir(parents=True, exist_ok=True)
        _shard_main(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
