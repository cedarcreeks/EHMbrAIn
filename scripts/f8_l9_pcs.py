"""F8/L9 (prereg-v5): validate the Physics-Consistency Score (contribution C5).

Three classifiers, PCS for each: COMPETENT (extended sensors, where faults are
separable), CONFUSED (cockpit, the v0 setting), CONTROL (cockpit, shuffled
labels). If PCS separates competent from control, the metric measures physical
reasoning and C5 is validated; if not, it is useless.

PCS = | cos( H_S^+ m_shap , e_true ) |. Foreground (SHAP). Output:
data/processed/f8/pcs_validation.json
Usage: uv run python scripts/f8_l9_pcs.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'
COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
EXTENDED = ['N2_rpm', 'WF_kgps', 'EGT_degK', 'P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK']
CLASSES = ['none'] + [f'acute_{p}' for p in
                      ('fan.eta', 'lpc.eta', 'hpc.eta', 'hpt.eta', 'hpt.flow', 'lpt.eta')]


def _ffill(a):
    """Forward-fill NaNs (dropout); backfill any leading NaNs."""
    a = a.copy()
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    if np.isnan(a[0]):
        m = np.isnan(a)
        a[m] = a[~m][0] if (~m).any() else 0.0
    return a


def dev_matrix(e, channels):
    """(n, len(channels)) percent deviations vs the N1-interpolated baseline,
    dropout NaNs forward-filled (as the rest of the AI pipeline does)."""
    _, cha, ba = load_icm('cruise')
    _, _, bb = load_icm('cruise_lowpwr')
    w = (e.cr_N1_cmd.to_numpy(float) - 4666.0) / (4400.0 - 4666.0)
    out = np.zeros((len(e), len(channels)))
    for j, c in enumerate(channels):
        base = ba[c] * (1 - w) + bb[c] * w
        meas = e[f'cr_{c}'].to_numpy(float)
        out[:, j] = _ffill((meas - base) / base * 100.0)
    return out


def diag_feats(F, t):
    pre = F[max(0, t - 800):max(1, t - 500)].mean(axis=0)
    post = F[max(0, t - 300):t].mean(axis=0)
    slope = (F[max(0, t - 150):t].mean(axis=0)
             - F[max(0, t - 300):max(1, t - 150)].mean(axis=0))
    return np.concatenate([post - pre, slope, post])


def build(fleet_df, index, events, channels, rng, shuffle=False):
    """Train HistGB, return (clf, Xtr, per-engine feature/label cache)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    nC = len(channels)
    cache = {}
    Xtr, ytr = [], []
    for rec in index:
        eid, split = rec['engine_id'], rec['split']
        e = fleet_df[fleet_df.engine_id == eid].sort_values('cycle').reset_index(drop=True)
        F = dev_matrix(e, channels)
        mu = F[:2000].mean(0); sd = F[:2000].std(0) + 1e-9
        Fn = (F - mu) / sd
        ev = events[events.engine_id == eid]
        eps = [(float(r.cycle), str(r.param)) for r in
               ev[ev.type == 'acute'].sort_values('cycle').itertuples()]
        cache[eid] = {'Fn': Fn, 'eps': eps, 'life': len(e), 'split': split}
        if split != 'train':
            continue
        bounds = [int(o) for o, _ in eps] + [len(e)]
        for k, (onset, param) in enumerate(eps):
            lo = min(int(onset) + 300, bounds[k + 1] - 2)
            for t in rng.integers(lo, bounds[k + 1], size=8) if bounds[k + 1] > lo else []:
                Xtr.append(diag_feats(Fn, int(t))); ytr.append(CLASSES.index(f'acute_{param}'))
        if eps and bounds[0] > 1200:
            for t in rng.integers(900, bounds[0] - 100, size=6):
                Xtr.append(diag_feats(Fn, int(t))); ytr.append(0)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    if shuffle:
        ytr = rng.permutation(ytr)
    clf = HistGradientBoostingClassifier(max_iter=400, max_depth=6, random_state=0,
                                         class_weight='balanced')
    clf.fit(Xtr, ytr)
    return clf, Xtr, cache, nC


def pcs_for(clf, Xtr, cache, channels, nC):
    import shap
    H, ch, _ = load_icm('cruise')
    Hp = np.linalg.pinv(H[[ch.index(c) for c in channels]])   # (10, nC)
    background = shap.utils.sample(Xtr, 80, random_state=0)
    expl = shap.PermutationExplainer(clf.predict_proba, background)
    rows = []
    for eid, c in cache.items():
        if c['split'] != 'test' or not c['eps']:
            continue
        bounds = [int(o) for o, _ in c['eps']] + [c['life']]
        for k, (onset, param) in enumerate(c['eps']):
            t = min(bounds[k + 1] - 1, c['life'] - 1, int(onset) + 500)
            if t <= int(onset) + 100:
                continue
            x = diag_feats(c['Fn'], t)
            pred = int(clf.predict(x[None])[0])
            phi = expl(x[None]).values[0][:, pred]
            m = phi[:nC]                                # step block, nC channels
            x_attr = Hp @ m
            j = HEALTH_PARAMS.index(param)
            denom = np.linalg.norm(x_attr)
            rows.append({'pcs': float(abs(x_attr[j]) / denom) if denom > 0 else 0.0,
                         'correct': CLASSES[pred] == f'acute_{param}'})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FLEET / 'fleet_index.json').read_text())['engines']
    events = pd.read_parquet(FLEET / 'events.parquet')
    cols = ['engine_id', 'cycle', 'split', 'cr_N1_cmd'] + [f'cr_{c}' for c in EXTENDED]
    df = pd.read_parquet(FLEET / 'snapshots.parquet', columns=cols)

    out = {}
    for name, chans, shuf in (('competent', EXTENDED, False),
                              ('confused', COCKPIT, False),
                              ('control', COCKPIT, True)):
        rng = np.random.default_rng(7)
        clf, Xtr, cache, nC = build(df, index, events, chans, rng, shuffle=shuf)
        rows = pcs_for(clf, Xtr, cache, chans, nC)
        pcs = np.array([r['pcs'] for r in rows])
        ok = np.array([r['correct'] for r in rows])
        out[name] = {'pcs_mean': float(pcs.mean()), 'pcs': pcs.tolist(),
                     'acc': float(ok.mean()),
                     'pcs_correct': float(pcs[ok].mean()) if ok.any() else None,
                     'pcs_incorrect': float(pcs[~ok].mean()) if (~ok).any() else None}
        print(f'{name:10s} acc={out[name]["acc"]:.2f}  PCS mean={out[name]["pcs_mean"]:.3f}',
              flush=True)

    pc, cc = np.array(out['competent']['pcs']), np.array(out['control']['pcs'])
    U, p = mannwhitneyu(pc, cc, alternative='greater')
    h91 = bool(out['competent']['pcs_mean'] > out['control']['pcs_mean'] and p < 0.05)
    comp = out['competent']
    h92 = bool(comp['pcs_correct'] is not None and comp['pcs_incorrect'] is not None
               and comp['pcs_correct'] > comp['pcs_incorrect'])
    verdict = {
        'H9.1_pcs_validity': {
            'pcs_competent': comp['pcs_mean'], 'pcs_confused': out['confused']['pcs_mean'],
            'pcs_control': out['control']['pcs_mean'],
            'mannwhitney_p': float(p), 'confirmed': h91},
        'H9.2_tracks_correctness': {
            'pcs_correct': comp['pcs_correct'], 'pcs_incorrect': comp['pcs_incorrect'],
            'confirmed': h92},
        'summary': {k: {'acc': out[k]['acc'], 'pcs_mean': out[k]['pcs_mean']} for k in out},
    }
    (OUT / 'pcs_validation.json').write_text(json.dumps(verdict, indent=2))
    print(f"H9.1 validity: competent {comp['pcs_mean']:.3f} vs control "
          f"{out['control']['pcs_mean']:.3f}, p={p:.4f} -> {h91}")
    print(f"H9.2 tracks correctness: {comp['pcs_correct']} vs {comp['pcs_incorrect']} -> {h92}")


if __name__ == '__main__':
    main()
