"""L-H2 / L-H2b (prereg-v9): does breaking the confusable-isolation wall
need REAL sensors, and can a VIRTUAL (model-predicted) sensor substitute?

WLS-GPA nearest-signature isolation on the confusable test episodes, three
sensor conditions: cockpit / cockpit+virtual / cockpit+real extended. The
virtual channels are the twin's prediction from the cockpit estimate
(H_extra . x_hat_cockpit) -- a function of the cockpit data, adding no rank.

Foreground. Output: data/processed/f8/wall_verdict.json
Usage: uv run python scripts/f8_lh2_wall.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'
COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
EXTRA = ['P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK']
EXTENDED = COCKPIT + EXTRA
TARGETS = ['fan.eta', 'lpc.eta', 'hpc.eta', 'hpt.eta', 'hpt.flow', 'lpt.eta']
CONFUSABLE = ('hpc.eta', 'hpt.eta', 'hpt.flow')
SIGMA = {'N2_rpm': 0.07, 'WF_kgps': 0.5, 'EGT_degK': 0.23,
         'P25_bar': 0.3, 'T25_degK': 0.2, 'PS3_bar': 0.3, 'T3_degK': 0.2}


def wls(H, R_diag, dz, lam=2.0):
    Rinv = np.diag(1.0 / np.asarray(R_diag) ** 2)
    A = H.T @ Rinv @ H + lam * np.eye(10)
    return np.linalg.solve(A, H.T @ Rinv @ dz)


def isolate(x_hat):
    x6 = np.array([x_hat[HEALTH_PARAMS.index(p)] for p in TARGETS])
    return TARGETS[int(np.argmax(np.abs(x6)))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Hc, ch, ba = load_icm('cruise')
    _, _, bb = load_icm('cruise_lowpwr')
    rows = {c: ch.index(c) for c in EXTENDED}
    index = json.loads((FLEET / 'fleet_index.json').read_text())['engines']
    test = {r['engine_id'] for r in index if r['split'] == 'test'}
    events = pd.read_parquet(FLEET / 'events.parquet')
    cols = ['engine_id', 'cycle', 'cr_N1_cmd'] + [f'cr_{c}' for c in EXTENDED]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet', columns=cols)

    # H rows and noise for each sensor set
    Hcock = Hc[[rows[c] for c in COCKPIT]]
    Hext = Hc[[rows[c] for c in EXTENDED]]
    Rc = [SIGMA[c] for c in COCKPIT]
    Re = [SIGMA[c] for c in EXTENDED]

    def dev(e, chans, w):
        out = []
        for c in chans:
            base = ba[c] * (1 - w) + bb[c] * w
            meas = e[f'cr_{c}'].to_numpy(float)
            out.append((meas - base) / base * 100.0)
        return np.column_stack(out)

    res = {'cockpit': [], 'virtual': [], 'real': []}
    for eid in sorted(test):
        e = snap[snap.engine_id == eid].sort_values('cycle').reset_index(drop=True)
        n = len(e)
        w = (e.cr_N1_cmd.to_numpy(float) - 4666.0) / (4400.0 - 4666.0)
        dz_ext = dev(e, EXTENDED, w)          # (n, 7)
        ac = events[(events.engine_id == eid) & (events.type == 'acute')].sort_values('cycle')
        eps = [(int(r.cycle), str(r.param)) for r in ac.itertuples()]
        bounds = [o for o, _ in eps] + [n]
        for k, (onset, param) in enumerate(eps):
            if param not in CONFUSABLE:
                continue
            t = min(bounds[k + 1] - 1, n - 1, onset + 500)
            if t <= onset + 100:
                continue
            pre = np.nanmean(dz_ext[max(0, onset - 300):max(1, onset - 20)], axis=0)
            post = np.nanmean(dz_ext[max(0, t - 300):t], axis=0)
            step = post - pre                  # (7,) extended deviations
            wm = float(np.nanmean(w[max(0, t - 300):t]))
            Hc_w = Hcock                        # cruise-ref H (angle geometry is cruise)
            # cockpit
            xc = wls(Hcock, Rc, step[:3])
            res['cockpit'].append(isolate(xc) == param)
            # real extended
            xr = wls(Hext, Re, step)
            res['real'].append(isolate(xr) == param)
            # virtual: extra channels predicted from cockpit estimate (H_extra . x_hat_cock)
            x_hat_c = xc
            virt_extra = Hc[[rows[c] for c in EXTRA]] @ x_hat_c   # (4,)
            step_v = np.concatenate([step[:3], virt_extra])
            xv = wls(Hext, Re, step_v)
            res['virtual'].append(isolate(xv) == param)

    def acc(a):
        return float(np.mean(a)) if a else 0.0

    def mcnemar(a, b):  # a=condition, b=cockpit; one-sided a>b
        a, b = np.array(a), np.array(b)
        n01 = int(np.sum(~b & a)); n10 = int(np.sum(b & ~a))
        if n01 + n10 == 0:
            return 1.0
        return float(binomtest(n01, n01 + n10, 0.5, alternative='greater').pvalue)

    ac_cock, ac_virt, ac_real = acc(res['cockpit']), acc(res['virtual']), acc(res['real'])
    p_real = mcnemar(res['real'], res['cockpit'])
    p_virt = mcnemar(res['virtual'], res['cockpit'])
    h1 = bool(ac_real - ac_cock >= 0.25 and p_real < 0.05)
    h2 = bool(p_virt > 0.05 and h1)
    verdict = {
        'n_confusable_episodes': len(res['cockpit']),
        'accuracy': {'cockpit': ac_cock, 'virtual': ac_virt, 'real': ac_real},
        'mcnemar_vs_cockpit': {'real_p': p_real, 'virtual_p': p_virt},
        'H-H2.1_real_breaks_wall': {'confirmed': h1},
        'H-H2.2_virtual_cannot': {'confirmed': h2},
    }
    (OUT / 'wall_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"confusable episodes: {len(res['cockpit'])}")
    print(f"accuracy  cockpit {ac_cock:.2f}  virtual {ac_virt:.2f}  real {ac_real:.2f}")
    print(f"H-H2.1 real breaks wall: +{(ac_real-ac_cock)*100:.0f}pp p={p_real:.3f} -> {h1}")
    print(f"H-H2.2 virtual cannot:   virtual p={p_virt:.3f} -> {h2}")


if __name__ == '__main__':
    main()
