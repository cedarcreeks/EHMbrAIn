"""Influence Coefficient Matrix (ICM) generation and analysis (WP1.4).

The ICM is the local Jacobian of the measured gas-path parameters with respect
to the component health parameters, computed by central finite differences on
the pyCycle model at a fixed operating point and constant N1 (the CFM56 rating
parameter). Entries are expressed in % measurement change per +1 % health
parameter change.

Health parameter order (matches conf/fault definitions and the GPA chapters):
    [fan.eta, fan.flow, lpc.eta, lpc.flow, hpc.eta, hpc.flow,
     hpt.eta, hpt.flow, lpt.eta, lpt.flow]
"""

import itertools

import numpy as np

from .cycle import TURBO_SCALARS, build_study_problem, set_health, snapshot

HEALTH_PARAMS = [f'{comp}.{kind}' for comp in TURBO_SCALARS for kind in ('eta', 'flow')]

# Measured channels for the ICM rows. N1 is excluded: it is the power-setting
# parameter held constant, so its deviation is identically zero.
COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
EXTENDED = COCKPIT + ['P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK']


def _icm_column(args):
    """Worker: one health parameter's central-difference column (norm N1:
    each process builds its own Problem; OpenMDAO objects don't cross
    process boundaries)."""
    param, op, step, channels = args
    prob = build_study_problem(**op)
    prob.set_solver_print(level=-1)
    prob.run_model()
    base = snapshot(prob, 'OD')
    z0 = np.array([base[c] for c in channels])
    zs = {}
    for sign in (+1, -1):
        set_health(prob, {param: sign * step})
        prob.run_model()
        s = snapshot(prob, 'OD')
        zs[sign] = np.array([s[c] for c in channels])
    col = (zs[+1] - zs[-1]) / z0 * 100.0 / (2 * step * 100.0)
    return param, col, base


def compute_icm(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4666.0, step=0.005,
                channels=EXTENDED, guesses=None, verbose=False, n_workers=None,
                overrides=None):
    """Central-difference ICM at one operating point.

    Returns (H, baseline) where H[i, j] = % change of channels[i] per +1 % of
    HEALTH_PARAMS[j], and baseline is the healthy snapshot dict.
    n_workers > 1 computes the columns in parallel processes (norm N1).
    `overrides` perturbs the design/calibration inputs (see build_study_problem).
    """
    op = dict(mn=mn, alt_ft=alt_ft, dTs=dTs, n1_rpm=n1_rpm, guesses=guesses,
              overrides=overrides)
    H = np.zeros((len(channels), len(HEALTH_PARAMS)))

    if n_workers and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        base = None
        jobs = [(p, op, step, channels) for p in HEALTH_PARAMS]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for param, col, b in pool.map(_icm_column, jobs):
                H[:, HEALTH_PARAMS.index(param)] = col
                base = b
                if verbose:
                    print(f'  {param:10s} done', flush=True)
        return H, base

    prob = build_study_problem(**op)
    prob.set_solver_print(level=-1)
    prob.run_model()
    base = snapshot(prob, 'OD')
    z0 = np.array([base[c] for c in channels])
    for j, param in enumerate(HEALTH_PARAMS):
        zs = {}
        for sign in (+1, -1):
            set_health(prob, {param: sign * step})
            prob.run_model()
            s = snapshot(prob, 'OD')
            zs[sign] = np.array([s[c] for c in channels])
        set_health(prob, {})  # back to healthy (also re-warms the solution)
        prob.run_model()
        H[:, j] = (zs[+1] - zs[-1]) / z0 * 100.0 / (2 * step * 100.0)
        if verbose:
            print(f'  {param:10s} done')
    return H, base


def signature_angles(H):
    """Pairwise angles (deg) between fault signatures (columns of H)."""
    n = H.shape[1]
    angles = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        u, v = H[:, i], H[:, j]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        cosang = np.clip(abs(u @ v) / denom, 0.0, 1.0)  # sign-agnostic
        angles[i, j] = angles[j, i] = np.degrees(np.arccos(cosang))
    return angles


def svd_report(H):
    """Rank, singular values and condition number of the ICM."""
    s = np.linalg.svd(H, compute_uv=False)
    tol = s.max() * max(H.shape) * np.finfo(float).eps
    return {
        'singular_values': s.tolist(),
        'rank': int((s > tol).sum()),
        'condition_number': float(s.max() / s[s > tol].min()),
    }


def confusable_pairs(H, params=HEALTH_PARAMS, threshold_deg=15.0):
    """Fault pairs whose signatures are closer than `threshold_deg` (H2 subset)."""
    ang = signature_angles(H)
    out = []
    for i, j in itertools.combinations(range(len(params)), 2):
        if ang[i, j] < threshold_deg:
            out.append((params[i], params[j], float(ang[i, j])))
    return sorted(out, key=lambda t: t[2])
