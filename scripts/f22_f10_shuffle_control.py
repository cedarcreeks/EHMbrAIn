"""F22 (prereg-v25): the control F10 never had.

WHY. F10's headline (H10.1, contribution C8) is that the identifiability
certificate is HONEST: the Cramer-Rao bound it computes per health direction
ranks the Kalman estimator's actual per-direction error, Spearman rho = 0.70
against ground truth. That has been the project's most-promoted result.

The port of F10 to N-CMAPSS (F21) failed a control that F10 itself never ran.
Both the bound and the estimator are computed from the SAME influence matrix, and
column-shuffling that matrix -- which destroys the physics and keeps the coupling
-- reproduced the correlation almost exactly (rho 0.830 against a real 0.842).
The agreement there measured the shared matrix, not the certificate.

The same coupling exists here: cert.certify() builds Fisher information from the
pyCycle H, and kalman_gpa estimates the state using the same pyCycle H. So the
same control must be run on our own result, and it is cheap.

MECHANISM BEING TESTED. Both the CRB for direction j and the estimator's error in
direction j depend on the R^-1-weighted norm of column j of H. A direction with a
weak column gets a large bound AND a large error whether or not that column is
the physically correct one. Shuffling columns preserves the multiset of column
norms, so if that is all the correlation reflects, the shuffled control ranks
just as well.

READ THE OUTCOME AS:
  control rho near zero  -> H10.1 measures the certificate. C8 stands.
  control rho near real  -> H10.1 measures the coupling. C8's headline does not
                            survive, and this refutes the project's own
                            most-promoted claim.

Output: data/processed/f22/f10_shuffle_control.json
Usage: uv run python scripts/f22_f10_shuffle_control.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ehmbrain.datagen.fleet import load_icm                       # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                       # noqa: E402
from ehmbrain.trad.identifiability import (COCKPIT, SIGMA_PCT,    # noqa: E402
                                           PRIOR_STD_PCT)
from ehmbrain.trad.pipeline import BaselineModel, kalman_gpa      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f22'
R_DIAG = [0.07, 0.5, 0.23]
N_DRAWS = 200


def rho_for(H_a, H_b, snap, test_ids, bm):
    """Spearman(CRB, actual Kalman error) with a given influence matrix pair.

    H_a/H_b are the two cruise-grid blocks, exactly as Certificate uses them, so
    passing the shuffled versions changes nothing else about the pipeline.
    """
    Rinv = np.diag([1.0 / SIGMA_PCT[s] ** 2 for s in COCKPIT])
    P0inv = np.eye(len(HEALTH_PARAMS)) / PRIOR_STD_PCT ** 2
    crb, err = [], []
    for eid in test_ids:
        e = snap[snap.engine_id == eid].sort_values('cycle').reset_index(drop=True)
        n = len(e)
        n1 = e.cr_N1_cmd.to_numpy()
        meas = e[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
        dz = bm.deviations(meas, n1)
        _, _, w = bm.cruise(n1)[1]

        def H_at(i):
            return H_a * (1 - w[i]) + H_b * w[i]

        xs = kalman_gpa(dz, H_at, R_DIAG, q=2e-4)
        # certificate: Fisher over the late-life flown conditions
        F = P0inv.copy()
        wl = ((n1[int(0.7 * n):] - 4666.0) / (4400.0 - 4666.0))[::20]
        for wi in wl:
            Hi = H_a * (1 - wi) + H_b * wi
            F += Hi.T @ Rinv @ Hi
        crb.append(np.sqrt(np.diag(np.linalg.inv(F))))
        xt = e[[f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]].to_numpy()
        err.append(np.abs(xs[int(0.85 * n):] - xt[int(0.85 * n):]).mean(axis=0))
    crb, err = np.asarray(crb), np.asarray(err)
    return spearmanr(np.median(crb, 0), np.median(err, 0)).statistic


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FLEET / 'fleet_index.json').read_text())['engines']
    test_ids = [r['engine_id'] for r in index if r['split'] == 'test']
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=(['engine_id', 'cycle', 'cr_N1_cmd']
                                    + [f'cr_{c}' for c in COCKPIT] + Xc))
    bm = BaselineModel()
    Ha_full, ch, _ = load_icm('cruise')
    Hb_full, _, _ = load_icm('cruise_lowpwr')
    rows = [ch.index(s) for s in COCKPIT]
    Ha, Hb = Ha_full[rows], Hb_full[rows]

    print('== real influence matrix ==', flush=True)
    rho_real = rho_for(Ha, Hb, snap, test_ids, bm)
    print(f'  rho = {rho_real:+.3f}   (F10 published 0.70)', flush=True)

    print(f'== control: {N_DRAWS} column shuffles ==', flush=True)
    rng = np.random.default_rng(0)
    ctrl = []
    for d in range(N_DRAWS):
        perm = rng.permutation(Ha.shape[1])
        r = rho_for(Ha[:, perm], Hb[:, perm], snap, test_ids, bm)
        if np.isfinite(r):
            ctrl.append(float(r))
        if (d + 1) % 50 == 0:
            print(f'  {d + 1}/{N_DRAWS}  median so far {np.median(ctrl):+.3f}',
                  flush=True)
    ctrl = np.asarray(ctrl)
    # one-sided: how often does a physics-free matrix rank at least as well?
    p_emp = float((ctrl >= rho_real).mean())

    verdict = {
        'question': ('does H10.1 measure the certificate, or the fact that the '
                     'bound and the estimator share one influence matrix?'),
        'rho_real': float(rho_real),
        'rho_published_F10': 0.70,
        'control': {'n_draws': int(len(ctrl)),
                    'median': float(np.median(ctrl)),
                    'p05': float(np.percentile(ctrl, 5)),
                    'p95': float(np.percentile(ctrl, 95)),
                    'empirical_p': p_emp,
                    'what_is_shuffled': ('columns of the influence matrix, in '
                                         'both cruise blocks, identically -- the '
                                         'physics is destroyed, the coupling and '
                                         'the multiset of column norms are kept')},
        'verdict': ('certificate' if p_emp < 0.05 else 'coupling'),
        'confirmed': bool(p_emp < 0.05),
        'consequence': ('if the control ranks as well as the real matrix, H10.1 '
                        'and contribution C8 measure the shared matrix rather '
                        'than the certificate, and the project\'s most-promoted '
                        'claim does not survive as stated'),
        'note_ncmapss': ('the same control on the N-CMAPSS port gave control '
                         'rho 0.830 against a real 0.842, which is why this was '
                         'run here'),
    }
    (OUT / 'f10_shuffle_control.json').write_text(json.dumps(verdict, indent=2))

    print()
    print(f'  real          rho = {rho_real:+.3f}')
    print(f'  shuffled      median {np.median(ctrl):+.3f}   '
          f'[p05 {np.percentile(ctrl, 5):+.3f}, p95 {np.percentile(ctrl, 95):+.3f}]')
    print(f'  empirical p (control >= real) = {p_emp:.4f}')
    print(f'  VERDICT: H10.1 measures the {verdict["verdict"].upper()}')


if __name__ == '__main__':
    main()
