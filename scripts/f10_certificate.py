"""F10 confirmatory (prereg-v4): validate the identifiability certificate
against ground truth.

H10.1 honesty  : Spearman(CRB predicted per-direction std, actual KF error) >= 0.6, p<0.05
H10.2 coverage : true 10-dim health inside the 90% CRB ellipsoid at 86-94% of test engines
H10.3 acquisition: extended sensors shrink CRB in the unobservable efficiencies >= 2x,
                   and predict which directions each addition rescues.

Foreground. Output: data/processed/f10/verdicts_f10.json
Usage: uv run python scripts/f10_certificate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.identifiability import Certificate, COCKPIT, EXTENDED
from ehmbrain.trad.pipeline import BaselineModel, kalman_gpa

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f10'
R_DIAG = [0.07, 0.5, 0.23]
EFF = ['fan.eta', 'lpc.eta', 'hpc.eta']       # the unobservable efficiencies


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FLEET / 'fleet_index.json').read_text())['engines']
    test_ids = [r['engine_id'] for r in index if r['split'] == 'test']
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    cols = (['engine_id', 'cycle', 'cr_N1_cmd']
            + [f'cr_{c}' for c in COCKPIT] + Xc)
    snap = pd.read_parquet(FLEET / 'snapshots.parquet', columns=cols)

    bm = BaselineModel()
    cert = Certificate(COCKPIT)
    cert_ext = Certificate(EXTENDED)

    crb_pred = {p: [] for p in HEALTH_PARAMS}
    kf_err = {p: [] for p in HEALTH_PARAMS}
    crb_ext = {p: [] for p in HEALTH_PARAMS}
    covered = []
    for eid in test_ids:
        e = snap[snap.engine_id == eid].sort_values('cycle').reset_index(drop=True)
        n = len(e)
        n1 = e.cr_N1_cmd.to_numpy()
        meas = e[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
        dz = bm.deviations(meas, n1)
        Ha, Hb, w = bm.cruise(n1)[1]
        xs = kalman_gpa(dz, lambda i: Ha * (1 - w[i]) + Hb * w[i], R_DIAG, q=2e-4)

        hist = n1[int(0.7 * n):]
        c = cert.certify(hist)
        ce = cert_ext.certify(hist)
        xt = e[Xc].to_numpy()
        err = np.abs(xs[int(0.85 * n):] - xt[int(0.85 * n):]).mean(axis=0)
        for j, p in enumerate(HEALTH_PARAMS):
            crb_pred[p].append(c['std_pct'][p])
            crb_ext[p].append(ce['std_pct'][p])
            kf_err[p].append(err[j])
        # H10.2: true late-life x in the 90% ellipsoid around the KF estimate
        x_est = xs[int(0.9 * n):].mean(axis=0)
        x_true = xt[int(0.9 * n):].mean(axis=0)
        ok, _ = cert.in_region(x_true, x_est, c['cov'], level=0.90)
        covered.append(ok)

    cp = [np.median(crb_pred[p]) for p in HEALTH_PARAMS]
    ke = [np.median(kf_err[p]) for p in HEALTH_PARAMS]
    rho, pv = spearmanr(cp, ke)
    coverage = float(np.mean(covered))

    # H10.3: acquisition value on the unobservable efficiencies
    shrink = {p: float(np.median(crb_pred[p]) / max(np.median(crb_ext[p]), 1e-9))
              for p in EFF}
    med_shrink = float(np.median(list(shrink.values())))
    per_dir_full = {p: {'cockpit': round(np.median(crb_pred[p]), 3),
                        'extended': round(np.median(crb_ext[p]), 3),
                        'kf_err': round(np.median(kf_err[p]), 3)}
                    for p in HEALTH_PARAMS}

    verdict = {
        'H10.1_honesty': {
            'spearman_rho': float(rho), 'p_value': float(pv),
            'per_direction': {p: {'crb_pred': round(np.median(crb_pred[p]), 3),
                                  'kf_actual_err': round(np.median(kf_err[p]), 3)}
                              for p in HEALTH_PARAMS},
            'confirmed': bool(rho >= 0.6 and pv < 0.05)},
        'H10.2_coverage': {
            'empirical_coverage': coverage, 'nominal': 0.90,
            'confirmed': bool(0.86 <= coverage <= 0.94)},
        'H10.3_acquisition': {
            'cockpit_vs_extended_crb_shrink_efficiencies': shrink,
            'median_shrink': med_shrink,
            'confirmed': bool(med_shrink >= 2.0)},
        'per_direction_full': per_dir_full,
    }
    (OUT / 'verdicts_f10.json').write_text(json.dumps(verdict, indent=2))
    print(f"H10.1 honesty:   rho={rho:.3f} p={pv:.4f}  -> {verdict['H10.1_honesty']['confirmed']}")
    print(f"H10.2 coverage:  {coverage:.2f} (nominal 0.90)  -> {verdict['H10.2_coverage']['confirmed']}")
    print(f"H10.3 acquisition: efficiencies shrink {med_shrink:.1f}x  -> {verdict['H10.3_acquisition']['confirmed']}")
    print('  per-efficiency shrink:', {k: round(v, 1) for k, v in shrink.items()})


if __name__ == '__main__':
    main()
