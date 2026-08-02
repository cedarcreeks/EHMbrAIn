"""F24 (prereg-v27): calibrate the CRB's absolute scale, then test magnitude with
a statistic that is not capped at ten directions.

WHERE THIS COMES FROM. H10.1's rank test is stuck: with ten health directions,
rho would have to reach about 0.55 to clear significance, and F23 measured 0.455
with the coupling removed (sec:f23-decoupled). More engines do not help, because
the ranking is over directions and there are ten of them. The way out is a
different statistic, and sec:f21-port names the prerequisite: the CRB's absolute
scale must be trustworthy first, and H10.2 showed it is not -- the bound is a
lower bound for UNBIASED estimators while any regularized estimator is biased, so
the CRB has the right shape and the wrong magnitude.

PART 1 -- FIX THE SCALE. Fit, on VALIDATION engines only,

    log |err| = a + b * log CRB

and read both parameters. `a` is the missing scale: exp(a) is how many times the
bound under-states achievable error. `b` says whether the distortion is a pure
scale (b = 1, the bound is right up to a constant) or a shape change (b != 1).
One or two parameters, fitted on data never used for the test.

PART 2 -- MEASURE MAGNITUDE WITH POWER. The certificate is per ENGINE: it
accumulates Fisher information over that engine's own flown conditions, so the
CRB differs between engines that flew differently. That gives variation WITHIN a
direction, across engines -- which is the axis the ten-direction ceiling does not
touch.

  between-direction  what F10 did. n = 10. Reported for continuity.
  WITHIN-direction   remove each direction's mean from both log CRB and log|err|,
                     then correlate. This asks whether an engine whose history
                     earned it a tighter bound on parameter j actually achieves a
                     smaller error on j than an engine whose history did not.
                     n = engines x directions, with direction effects gone.

The within-direction test is the powered one, and it is immune to the confound
that killed the ranking: direction identity is differenced out, so nothing can
ride on which parameters happen to be hard.

GATE FIRST. If the CRB barely varies between engines, the within-direction test
has no power either and must not be reported as evidence. The script computes the
within-direction coefficient of variation of the CRB before anything else and
says so plainly.

DECOUPLING IS PRESERVED. Errors come from F23's LEARNED estimator, which never
sees the influence matrix. Re-introducing the Kalman here would re-introduce the
coupling this whole line exists to remove.

Output: data/processed/f24/crb_scale_verdict.json
Usage: uv run python scripts/f24_crb_scale_magnitude.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                        # noqa: E402
from f23_decoupled_certificate import build, SEEDS                # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                       # noqa: E402
from ehmbrain.trad.identifiability import Certificate, COCKPIT    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
F23 = REPO_ROOT / 'data' / 'processed' / 'f23'
OUT = REPO_ROOT / 'data' / 'processed' / 'f24'
MIN_CV = 0.05          # below this the CRB does not vary enough between engines


def per_engine_tables(split, snap, sn, c, fleet, cert, preds_ready=None):
    """(engine x direction) CRB and learned-estimator |error|, in per cent."""
    rng = np.random.default_rng(11)
    X, Y, E = build(c, fleet, split_ids(fleet, split), snap, rng)
    if preds_ready is None:
        import torch
        from ehmbrain.ai.models import predict_torch, train_torch
        from f13_gate1_mechanism import MechNet
        par = json.loads((REPO_ROOT / 'data' / 'processed' / 'f13' /
                          'gate1_verdict.json').read_text())['setup']['best_params_sequence']
        z = np.load(F23 / 'cache.npz')
        cpu = torch.device('cpu')
        ps = []
        for s in SEEDS[:3]:            # 3 seeds is enough to average the val fit
            m = train_torch(MechNet(ch=z['Xtr'].shape[2], hidden=par['hidden'],
                                    layers=par['layers'], n_out=z['Ytr'].shape[1]),
                            z['Xtr'], z['Ytr'], epochs=par['epochs'],
                            lr=par['lr'], bs=par['bs'], seed=s, dev=cpu)
            ps.append(predict_torch(m, X, dev=cpu))
        P = np.mean(ps, axis=0)
    else:
        P = preds_ready
    rows = []
    for eid in np.unique(E):
        m = E == eid
        err = np.abs(P[m] - Y[m]).mean(0)
        g = sn[sn.engine_id == eid].sort_values('cycle')
        n1 = g.cr_N1_cmd.to_numpy()
        cr = cert.certify(n1[int(0.7 * len(n1)):])
        for j, p in enumerate(HEALTH_PARAMS):
            rows.append({'engine': int(eid), 'dir': p, 'j': j,
                         'crb': cr['std_pct'][p], 'err': float(err[j])})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = fleet_cache()
    fleet = c['fleet']
    cert = Certificate(COCKPIT)
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle', 'cr_N1_cmd'] + Xc)
    sn = snap[['engine_id', 'cycle', 'cr_N1_cmd']]

    print('== test split: reuse F23 predictions (learned estimator, no H) ==',
          flush=True)
    z = np.load(F23 / 'preds_0.npz')
    preds = {}
    for i in range(4):
        zz = np.load(F23 / f'preds_{i}.npz')
        preds.update({int(k): zz[k] for k in zz.files})
    P_test = np.mean([preds[s] for s in SEEDS], axis=0)
    te = per_engine_tables('test', snap, sn, c, fleet, cert, preds_ready=P_test)

    print('== validation split: fit the scale (never used for the test) ==',
          flush=True)
    va = per_engine_tables('val', snap, sn, c, fleet, cert)

    # ---- GATE: does the CRB vary between engines at all? -------------------
    cv = te.groupby('dir').apply(
        lambda d: d.crb.std() / max(d.crb.mean(), 1e-12), include_groups=False)
    cv_med = float(cv.median())
    gate_ok = cv_med >= MIN_CV
    print(f'  within-direction CV of the CRB: median {cv_med:.4f} '
          f'-> {"PASS" if gate_ok else "FAIL"}', flush=True)

    # ---- PART 1: the scale, fitted on validation ---------------------------
    lv_c, lv_e = np.log(va.crb.values), np.log(va.err.values)
    b, a = np.polyfit(lv_c, lv_e, 1)
    scale = float(np.exp(a))
    print(f'  calibration on val: log|err| = {a:.3f} + {b:.3f} log CRB', flush=True)
    print(f'    -> the bound understates achievable error by ~{1 / scale:.1f}x '
          f'at CRB = 1 %, exponent {b:.3f}', flush=True)

    # apply to test and score the calibrated prediction
    pred = np.exp(a + b * np.log(te.crb.values))
    resid = np.log(te.err.values) - np.log(pred)
    within_factor = float(np.exp(np.percentile(np.abs(resid), 90)))

    # ---- PART 2: magnitude, between and within direction --------------------
    r_between, p_between = stats.spearmanr(
        te.groupby('dir').crb.median(), te.groupby('dir').err.median())
    d = te.copy()
    d['lc'] = np.log(d.crb) - d.groupby('dir').crb.transform(lambda x: np.log(x).mean())
    d['le'] = np.log(d.err) - d.groupby('dir').err.transform(lambda x: np.log(x).mean())
    r_within, p_within = stats.pearsonr(d['lc'], d['le'])
    rs_within, ps_within = stats.spearmanr(d['lc'], d['le'])

    verdict = {
        'design': {
            'estimator': 'F23 learned estimator; never sees the influence matrix',
            'why_within': ('the certificate is per engine, so engines that flew '
                           'differently get different bounds. Differencing out '
                           'each direction removes the ten-direction ceiling and '
                           'the confound that direction identity carries'),
            'calibration_split': 'val engines only, never used in the test'},
        'gate_crb_varies_between_engines': {
            'within_direction_cv_median': cv_med, 'threshold': MIN_CV,
            'passed': bool(gate_ok),
            'note': ('if the bound barely differs between engines the '
                     'within-direction test has no power and is not evidence')},
        'part1_scale': {
            'log_intercept': float(a), 'log_slope': float(b),
            'scale_at_crb_1pct': scale,
            'understatement_factor': float(1 / scale),
            'p90_residual_factor': within_factor,
            'reading': ('exp(a) is how far the bound sits below achievable error; '
                        'slope 1 would mean a pure scale error, so a slope away '
                        'from 1 means the bound is also the wrong shape in '
                        'magnitude, not only in offset')},
        'part2_magnitude': {
            'between_direction': {'rho': float(r_between), 'p': float(p_between),
                                  'n': 10,
                                  'note': 'what F10 did; capped at ten points'},
            'within_direction': {'pearson_r': float(r_within),
                                 'p': float(p_within),
                                 'spearman_rho': float(rs_within),
                                 'spearman_p': float(ps_within),
                                 'n': int(len(d)),
                                 'note': ('direction effects differenced out, so '
                                          'this asks whether a tighter bound '
                                          'earned by an engine\'s own history '
                                          'predicts a smaller error for that '
                                          'engine')}},
        'confirmed': bool(gate_ok and p_within < 0.05 and r_within > 0),
    }
    (OUT / 'crb_scale_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    print(f'  between-direction (F10 statistic): rho={r_between:+.3f} '
          f'p={p_between:.4f}  n=10')
    print(f'  WITHIN-direction:                  r   ={r_within:+.3f} '
          f'p={p_within:.2e}  n={len(d)}')
    print(f'                                     rho ={rs_within:+.3f} '
          f'p={ps_within:.2e}')
    print(f'  calibrated band: 90 % of test errors within '
          f'{within_factor:.2f}x of prediction')
    print(f'  CONFIRMED: {verdict["confirmed"]}')


if __name__ == '__main__':
    main()
