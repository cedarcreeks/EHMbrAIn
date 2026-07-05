"""F8/L6 (prereg-v3): does a twin-residual physics feature help RUL under
data scarcity? Re-opens H4 on the nonlinear v2 fleet.

Twin-residual (mechanism M2): per snapshot the linear WLS residual
r = (I - H_ref M) Delta z, the part of the measurement the linear GPA cannot
explain. On v2 (nonlinear generator) this is informative; the RUL GRU input
grows from 4 to 7 channels. Pure vs hybrid at 10/25/100 % train data, seeds
0/1/2, single confirmatory pass. Verdict per docs/prereg-v3.md.

Foreground (MPS). Usage: uv run python scripts/f8_l6_hybrid.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from ehmbrain.ai.data import load_fleet_features, normalization
from ehmbrain.ai.models import RULNet, predict_torch, train_torch
from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.pipeline import COCKPIT

REPO_ROOT = Path(__file__).resolve().parents[1]
V2 = REPO_ROOT / 'data' / 'processed' / 'fleet_v2'
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'
EVAL_FRACS = (0.5, 0.7, 0.9)
RUL_CAP = 12.0     # kilocycles
FRACTIONS = (0.10, 0.25, 1.00)


def residual_projector():
    """(I - H_ref M) for the cockpit rows at cruise design N1, regularized WLS."""
    H, ch, _ = load_icm('cruise')
    rows = [ch.index(c) for c in COCKPIT]
    Hc = H[rows]                       # (3, 10)
    R_inv = np.diag(1.0 / np.array([0.07, 0.5, 0.23]) ** 2)
    P0_inv = np.eye(10) / 4.0
    M = np.linalg.solve(Hc.T @ R_inv @ Hc + P0_inv, Hc.T @ R_inv)   # (10,3)
    return np.eye(3) - Hc @ M          # (3,3)


def build(fleet, norm, proj, hybrid):
    mu, sd = norm
    data = {}
    for eid, v in fleet.items():
        F = (v['F'] - mu) / sd                      # (n,4)
        if hybrid:
            r = v['F'][:, :3] @ proj.T              # twin-residual (n,3), raw dz
            r = (r - r.mean(0)) / (r.std(0) + 1e-9)
            F = np.concatenate([F, r], axis=1)      # (n,7)
        data[eid] = F.astype(np.float32)
    return data


def sequences(data, fleet, ids, rng, seq=64, ds=20, per=40):
    X, y = [], []
    for eid in ids:
        v = fleet[eid]
        Fn = data[eid][::ds]
        for cut in rng.integers(seq * ds, v['life'], size=per):
            i = int(cut) // ds
            if i < seq:
                continue
            X.append(Fn[i - seq:i])
            y.append(min((v['life'] - int(cut)) / 1000.0, RUL_CAP))
    return np.array(X, np.float32), np.array(y, np.float32)


def eval_config(fleet, data, ch, frac, seed):
    import torch
    rng = np.random.default_rng(seed)
    train_ids = sorted(e for e, v in fleet.items() if v['split'] == 'train')
    k = max(3, int(len(train_ids) * frac))
    fit_ids = list(rng.permutation(train_ids)[:k])
    test_ids = sorted(e for e, v in fleet.items() if v['split'] == 'test')
    Xtr, ytr = sequences(data, fleet, fit_ids, rng)
    net = train_torch(RULNet(ch=ch, hidden=64, layers=2), Xtr, ytr,
                      epochs=30, lr=1e-3, seed=seed)
    cpu = torch.device('cpu')
    per_engine = {}
    for eid in test_ids:
        v = fleet[eid]
        Fn = data[eid][::20]
        errs = []
        for f in EVAL_FRACS:
            cut = int(f * v['life'])
            i = cut // 20
            if i < 64:
                continue
            pred = float(predict_torch(net, Fn[i - 64:i][None], dev=cpu)[0])
            errs.append(abs(min(pred, RUL_CAP) - min((v['life'] - cut) / 1000.0, RUL_CAP)) * 1000)
        if errs:
            per_engine[eid] = float(np.mean(errs))
    return per_engine


def main():
    fleet = load_fleet_features(fleet_dir=V2)
    norm = normalization(fleet)
    proj = residual_projector()
    data_pure = build(fleet, norm, proj, hybrid=False)
    data_hyb = build(fleet, norm, proj, hybrid=True)

    res = {'pure': {}, 'hybrid': {}}
    per_eng_10 = {'pure': [], 'hybrid': []}
    for frac in FRACTIONS:
        for name, data, ch in (('pure', data_pure, 4), ('hybrid', data_hyb, 7)):
            seeds = [eval_config(fleet, data, ch, frac, s) for s in (0, 1, 2)]
            eids = sorted(seeds[0])
            mean_err = {e: float(np.mean([s[e] for s in seeds])) for e in eids}
            rmse = float(np.sqrt(np.mean([v ** 2 for v in mean_err.values()])))
            res[name][f'{frac:.2f}'] = {'rmse_cycles': rmse}
            if abs(frac - 0.10) < 1e-6:
                per_eng_10[name] = mean_err
            print(f'{name:7s} {frac:.0%}: RMSE {rmse:.0f} cy', flush=True)

    eids = sorted(set(per_eng_10['pure']) & set(per_eng_10['hybrid']))
    p_val = float(stats.wilcoxon(
        [per_eng_10['pure'][e] for e in eids],
        [per_eng_10['hybrid'][e] for e in eids],
        alternative='greater').pvalue) if eids else 1.0

    def r(name, f):
        return res[name][f'{f:.2f}']['rmse_cycles']
    confirmed = (r('hybrid', 1.0) <= r('pure', 1.0)
                 and r('hybrid', 0.10) < r('pure', 0.10)
                 and r('hybrid', 0.25) < r('pure', 0.25)
                 and p_val < 0.05)
    verdict = {
        'hypothesis': 'H4-v2: twin-residual physics feature helps RUL under scarcity',
        'results': res,
        'wilcoxon_p_10pct_one_sided': p_val,
        'confirmed': bool(confirmed),
        'note': ('' if confirmed else
                 'physics injection does not meet the frozen bar; '
                 'partial pattern reported in the results block')}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'h4_v2_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: verdict[k] for k in
                      ('wilcoxon_p_10pct_one_sided', 'confirmed', 'note')}, indent=1))


if __name__ == '__main__':
    main()
