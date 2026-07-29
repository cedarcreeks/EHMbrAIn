"""F17 / H15.3 (prereg-v20): the certificate-gated hybrid.

This project has refuted physics-into-learner injection twice. H4 stacked GPA
state ESTIMATES as features and lost; L6 fed twin RESIDUALS on the nonlinear
fleet and lost again. The stated diagnosis (sec:res-h4) is the design constraint
here:

    "smeared state estimates are noise-bearing, mutually correlated features
     whose information the learner already had. Physics injection needs an
     information channel the raw data lacks."

The F10 certificate is such a channel. Per engine and per direction it is
computed from influence-matrix geometry and the engine's ACTUAL flown N1
history -- from the experiment design, not from the measured values. A learner
reading the deviation trajectory cannot derive it, because it is not in the
trajectory. That is the whole claim being tested.

Task: mechanism-share attribution, the one place F13 gate one found real signal
(3 of 5 mechanisms above R2 0.30). Re-running RUL would just be H4 again.

THREE ARMS, so the certificate is isolated from "more channels help":

  A  pure          4 deviation channels + age            (F13's family B)
  B  H4-style      + the raw 10-dim Kalman GPA estimate  (the refuted mechanism,
                                                          included as a control)
  C  certificate   + GPA estimate PROJECTED onto the certified-identifiable
                    subspace, + the per-direction CRB vector as static channels

If C beats A but B does too, the result is "more features help" and the
certificate earns nothing. Only C > A with B <= A supports the claim.

All three arms share the architecture and hyperparameters F13 selected under its
symmetric budget, so the input channel is the only variable. Disclosed: those
parameters were tuned for arm A's 5 channels. If C shows promise, it must be
retuned per arm before anything is claimed.

Output: data/processed/f17/hybrid_verdict.json
Usage: uv run python scripts/f17_certificate_hybrid.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                        # noqa: E402
from f13_gate1_mechanism import (MECHANISMS, CUT_MIN, SEQ_LEN,    # noqa: E402
                                 CUTS_PER_ENGINE, MIN_LOSS_C,
                                 MechNet, mechanism_shares, r2_per_col)
from ehmbrain.datagen.fleet import load_icm                       # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                       # noqa: E402
from ehmbrain.trad.identifiability import Certificate, COCKPIT    # noqa: E402
from ehmbrain.trad.pipeline import BaselineModel, kalman_gpa      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f17'
R_DIAG = [0.07, 0.5, 0.23]
STRIDE = 25
IDENTIFIABLE_PCT = 0.7        # Certificate.certify() tag threshold
SEEDS = (0, 1, 2)
ARMS = ('A_pure', 'B_h4_style', 'C_certificate')


def physics_channels():
    """Per engine: the Kalman GPA state estimate over the (strided) trajectory,
    the per-direction CRB from the certificate, and the identifiable mask."""
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'cr_N1_cmd']
                           + [f'cr_{c}' for c in COCKPIT])
    bm = BaselineModel()
    cert = Certificate(COCKPIT)
    out = {}
    for eid, g in snap.groupby('engine_id'):
        g = g.sort_values('cycle')
        n1 = g.cr_N1_cmd.to_numpy()
        meas = g[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
        dz = bm.deviations(meas, n1)
        Ha, Hb, w = bm.cruise(n1)[1]
        xs = kalman_gpa(dz, lambda i: Ha * (1 - w[i]) + Hb * w[i], R_DIAG, q=2e-4)
        c = cert.certify(n1[::STRIDE])
        crb = np.array([c['std_pct'][p] for p in HEALTH_PARAMS])
        out[int(eid)] = {'xhat': xs,                       # (life, 10)
                         'crb': crb,                       # (10,)
                         'mask': (crb < IDENTIFIABLE_PCT).astype(float)}
    return out


def build(catalog, H, ch, base, c, ids, phys, rng):
    """Sequences for all three arms plus the shared target."""
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
        # standardise the GPA estimate on its own fleet scale, once
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
            B = np.concatenate([A, xh_r], axis=1)

            # C: only the certified-identifiable part of the estimate survives,
            # and the certificate itself rides along as static channels
            xh_gated = xh_r * p['mask'][None, :]
            crb_ch = np.tile(p['crb'][None, :], (SEQ_LEN, 1))
            C = np.concatenate([A, xh_gated, crb_ch], axis=1)

            X['A_pure'].append(A); X['B_h4_style'].append(B)
            X['C_certificate'].append(C)
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

    par = json.loads((REPO_ROOT / 'data' / 'processed' / 'f13' /
                      'gate1_verdict.json').read_text())['setup']['best_params_sequence']
    from ehmbrain.ai.models import predict_torch, train_torch
    ymean = Ytr.mean(axis=0)

    res = {}
    for arm in ARMS:
        preds = []
        for s in SEEDS:
            net = train_torch(MechNet(ch=Xtr[arm].shape[2], hidden=par['hidden'],
                                      layers=par['layers']),
                              Xtr[arm], Ytr, epochs=par['epochs'], lr=par['lr'],
                              bs=par['bs'], seed=s)
            preds.append(predict_torch(net, Xte[arm]))
        P = np.mean(preds, axis=0)
        r2 = r2_per_col(Yte, P, ymean)
        res[arm] = {'r2_per_mechanism': dict(zip(MECHANISMS, r2)),
                    'r2_mean': float(np.nanmean(r2)),
                    'n_channels': int(Xtr[arm].shape[2])}
        print(f'  {arm:16s} channels {res[arm]["n_channels"]:3d}  '
              f'mean R2 {res[arm]["r2_mean"]:+.3f}', flush=True)

    a, b, cc = (res[x]['r2_mean'] for x in ARMS)
    verdict = {
        'design': {
            'task': 'mechanism-share attribution (F13 gate one task)',
            'arms': {'A_pure': '4 deviation channels + age',
                     'B_h4_style': '+ raw 10-dim Kalman GPA estimate',
                     'C_certificate': '+ identifiable-subspace-gated estimate '
                                      '+ per-direction CRB as static channels'},
            'shared_hyperparameters': par,
            'disclosed': ('hyperparameters were selected by F13 for arm A\'s 5 '
                          'channels; arms B and C inherit them so the input is '
                          'the only variable. If C wins, retune per arm before '
                          'claiming anything'),
            'seeds': list(SEEDS)},
        'per_arm': res,
        'H15.3_certificate_channel_helps': {
            'delta_C_minus_A': cc - a, 'delta_B_minus_A': b - a,
            'confirmed': bool(cc > a + 0.02 and b <= a + 0.02),
            'criterion': ('C beats A by >0.02 mean R2 AND B does not -- '
                          'otherwise the gain is "more features", not the '
                          'certificate'),
            'note': ('H4 and L6 both refuted physics injection by feeding '
                     'smeared estimates; the certificate is information the '
                     'trajectory cannot contain, which is the distinction '
                     'being tested')},
    }
    (OUT / 'hybrid_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"\n  C-A {cc - a:+.3f}   B-A {b - a:+.3f}   "
          f"H15.3 confirmed: {verdict['H15.3_certificate_channel_helps']['confirmed']}")


if __name__ == '__main__':
    main()
