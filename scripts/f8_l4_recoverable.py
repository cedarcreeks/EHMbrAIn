"""F8/L4 (prereg-v7): estimate the recoverable (washable) fraction of an
engine's lost EGT margin from its deviation trajectory.

Ground truth is replayed deterministically from each engine's seed: the
per-mechanism health contributions, projected through the takeoff-hot EGT row,
give the share of margin loss due to FOULING (a wash restores it) versus
permanent wear. If a regressor can read that fraction off the measured
deviation trajectory, an operator can predict a wash's payoff before washing.

Foreground. Output: data/processed/f8/recoverable_verdict.json
Usage: uv run python scripts/f8_l4_recoverable.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from ehmbrain.datagen.fleet import generate_engine, load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.pipeline import COCKPIT, BaselineModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'
RECOVERABLE = 'fouling'          # the only wash-restored mechanism


def recoverable_series(catalog, eid, H, ch, base):
    """(life,) recoverable fraction of EGT-margin loss, replayed from seed."""
    seed = catalog['fleet']['seed']
    rng = np.random.default_rng(np.random.SeedSequence([seed, eid, 0]))
    eng = generate_engine(eid, catalog, H, ch, base, rng)
    contribs = eng['contributions']
    egt_row = ch.index('EGT_degK')
    Hegt = H[egt_row]                                  # (10,)
    x_total = eng['x']                                 # (life,10)
    margin_loss = x_total @ Hegt                        # EGT rise ~ margin loss
    foul = contribs.get(RECOVERABLE, np.zeros_like(x_total)) @ Hegt
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(np.abs(margin_loss) > 1e-6, foul / margin_loss, 0.0)
    return eng['life_cycles'], np.clip(frac, 0.0, 1.0)


def feats(dz, t, w=1500):
    """Trailing-window shape features of the cockpit deviations at cycle t."""
    a = max(0, t - w)
    seg = dz[a:t]
    if len(seg) < 20:
        return None
    lvl = seg[-100:].mean(axis=0)
    slope = seg[-100:].mean(axis=0) - seg[:100].mean(axis=0)
    half = len(seg) // 2
    curv = (seg[half:].mean(axis=0) - seg[:half].mean(axis=0)) - slope
    rough = np.abs(np.diff(seg[-300:], axis=0)).mean(axis=0)   # sawtooth roughness
    return np.concatenate([lvl, slope, curv, rough])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    H, ch, base = load_icm('takeoff_hot')
    index = json.loads((FLEET / 'fleet_index.json').read_text())['engines']
    bm = BaselineModel()
    cols = ['engine_id', 'cycle', 'split', 'cr_N1_cmd'] + [f'cr_{c}' for c in COCKPIT]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet', columns=cols)
    rng = np.random.default_rng(3)

    def ff(a):
        a = a.copy(); idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
        np.maximum.accumulate(idx, out=idx); a = a[idx]
        if np.isnan(a[0]):
            m = np.isnan(a); a[m] = a[~m][0] if (~m).any() else 0.0
        return a

    def samples(split, per):
        X, y = [], []
        for rec in index:
            if rec['split'] != split:
                continue
            eid = rec['engine_id']
            life, frac = recoverable_series(catalog, eid, H, ch, base)
            e = snap[snap.engine_id == eid].sort_values('cycle').reset_index(drop=True)
            meas = np.column_stack([ff(e[f'cr_{c}'].to_numpy(float)) for c in COCKPIT])
            dz = bm.deviations(meas, e.cr_N1_cmd.to_numpy())
            n = min(len(dz), life)
            for t in rng.integers(1600, n, size=per) if n > 1700 else []:
                f = feats(dz, int(t))
                if f is not None:
                    X.append(f); y.append(float(frac[int(t)]))
        return np.array(X), np.array(y)

    from sklearn.ensemble import HistGradientBoostingRegressor
    Xtr, ytr = samples('train', 30)
    Xte, yte = samples('test', 30)
    reg = HistGradientBoostingRegressor(max_iter=400, max_depth=6, random_state=0)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)

    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - ytr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rho, p = spearmanr(pred, yte)
    verdict = {
        'n_train': int(len(ytr)), 'n_test': int(len(yte)),
        'truth_mean': float(yte.mean()), 'truth_std': float(yte.std()),
        'test_r2': float(r2), 'spearman': float(rho), 'p_value': float(p),
        'baseline_r2': 0.0, 'pred': pred.tolist(), 'truth': yte.tolist(),
        'H4L.1_estimable': {'confirmed': bool(r2 > 0.30 and rho > 0.5 and p < 0.05)},
    }
    (OUT / 'recoverable_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"recoverable fraction: truth mean {yte.mean():.2f} std {yte.std():.2f}")
    print(f"H4L.1: test R2={r2:.3f}, Spearman={rho:.3f} (p={p:.4f}) -> "
          f"{verdict['H4L.1_estimable']['confirmed']}")


if __name__ == '__main__':
    main()
