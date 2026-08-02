"""F20 (prereg-v23): estimate an influence coefficient matrix from N-CMAPSS,
then measure the confusable-pair angle on an engine model this project did not
build.

WHY. Every geometric claim in this document rests on one object: the influence
coefficient matrix of our own pyCycle twin, from which the 1.3-degree confusable
pair, the rank-3 cockpit limit and the identifiability certificate all follow.
It has never been checked against a model built by anyone else. N-CMAPSS is a
different engine, simulated by different people with a different code (C-MAPSS,
Frederick/DeCastro/Litt 2007), and -- in the repository distribution -- it ships
the TRUE health parameters alongside the measurements. So the geometry can be
measured there and compared.

THE OBSTACLE AND THE ROUTE. No Jacobian is published for C-MAPSS; the literature
treats it as a black box. But an ICM is a Jacobian, and every N-CMAPSS sample
pairs a health state theta with measurements x_s at a known operating condition
w. Regressing sensor deviations on theta at matched w estimates the matrix from
data. That is the ordinary way to obtain a Jacobian from a closed model.

THE IDENTIFIABILITY GATE, which comes first. Degradation is coordinated within a
unit, so theta columns correlate and a single subset may not separate all of
them. The seven failure modes are spread ACROSS subsets (DS01 affects HPT
efficiency only, DS05 HPC, DS06 LPC+HPC, DS08a all five), so pooling is what buys
the variation. This script computes the condition number and effective rank of
the theta design matrix BEFORE fitting anything, and refuses to report an angle
if the fit is not identifiable. A failure there is the finding, not a setback.

DISCLOSED SCOPE. This runs outside the reproducible pipeline: it needs a 14.68 GB
download that `make all` must never require, exactly as the FD001 substitution is
disclosed in sec:sim-to-real.

Output: data/processed/f20/ncmapss_icm.json
Usage: uv run python scripts/f20_ncmapss_icm.py [--files DS01,DS05,...]
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NC = REPO_ROOT / 'data' / 'external' / 'ncmapss' / 'h5'
OUT = REPO_ROOT / 'data' / 'processed' / 'f20'

# Our cockpit set is {N2, WF, EGT}. The N-CMAPSS analogues, by station:
#   N2  -> Nc   (physical core speed)
#   WF  -> Wf   (fuel flow)
#   EGT -> T48  (total temperature at HPT outlet -- the same station our EGT proxies)
COCKPIT_MAP = {'N2_rpm': 'Nc', 'WF_kgps': 'Wf', 'EGT_degK': 'T48'}
# The pair the wall is made of, in N-CMAPSS naming (efficiency modifiers).
PAIR_CANDIDATES = (('HPC_eff_mod', 'HPT_eff_mod'),
                   ('HPC_eff', 'HPT_eff'), ('hpc_eff_mod', 'hpt_eff_mod'))
COND_MAX = 1e4          # refuse to fit past this; theta columns too collinear
MIN_EFF_RANK_FRAC = 0.8  # effective rank must reach this fraction of n_theta


def list_files():
    return sorted(NC.rglob('*.h5'))


def load_one(path, stride=100):
    """Return (w, x_s, theta, names) for one file, subsampled.

    Subsampling is by row: N-CMAPSS is 1 Hz over whole flights, so consecutive
    rows are nearly identical and add cost without information.
    """
    import h5py
    try:
        fh = h5py.File(path, 'r')
    except OSError as e:                    # truncated/corrupt file
        return None, None, None, {'error': str(e)[:120]}
    with fh as f:
        def get(*cands):
            for c in cands:
                if c in f:
                    return f[c]
            return None

        def names(ds):
            """Variable names ship as byte strings; decode rather than str()."""
            if ds is None:
                return None
            out = []
            for x in np.atleast_1d(np.array(ds)).ravel():
                out.append(x.decode() if isinstance(x, (bytes, np.bytes_))
                           else str(x))
            return out

        w_var = names(get('W_var'))
        xs_var = names(get('X_s_var'))
        t_var = names(get('T_var'))
        if t_var is None:
            return None, None, None, None      # Challenge subset: no theta
        parts = []
        for grp in ('dev', 'test'):
            W = get(f'W_{grp}')
            X = get(f'X_s_{grp}')
            T = get(f'T_{grp}')
            if W is None or X is None or T is None:
                continue
            parts.append((np.array(W)[::stride], np.array(X)[::stride],
                          np.array(T)[::stride]))
        if not parts:
            return None, None, None, None
        w = np.concatenate([p[0] for p in parts])
        xs = np.concatenate([p[1] for p in parts])
        th = np.concatenate([p[2] for p in parts])
    return w, xs, th, {'w': w_var, 'xs': xs_var, 'theta': t_var}


def condition_report(theta):
    """Is theta varied enough to identify separate columns? Gate before fitting."""
    tc = theta - theta.mean(0)
    sd = tc.std(0)
    keep = sd > 1e-12
    tc = tc[:, keep] / sd[keep]
    s = np.linalg.svd(tc, compute_uv=False)
    s = s / s.max()
    # effective rank: singular values above 1 % of the largest
    eff = int((s > 0.01).sum())
    return {'cond': float(s[0] / max(s[-1], 1e-300)),
            'singular_spectrum': s.tolist(),
            'effective_rank': eff,
            'n_theta_varying': int(keep.sum()),
            'constant_columns': int((~keep).sum())}


def estimate_icm(w, xs, theta, n_bins=8):
    """Local linear fit of sensor deviation on theta, at matched conditions.

    Operating point is binned on the scenario descriptors so that within a bin
    the engine is at a comparable condition and the residual variation in x_s is
    attributable to theta. Sensors are expressed as per-cent deviation from the
    bin mean, matching this project's ICM convention (rows in % per unit health).
    """
    key = np.zeros(len(w), dtype=np.int64)
    for j in range(w.shape[1]):
        q = np.quantile(w[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        key = key * n_bins + np.digitize(w[:, j], q)
    Hs, weights = [], []
    for k in np.unique(key):
        m = key == k
        if m.sum() < 50:
            continue
        T = theta[m] - theta[m].mean(0)
        base = xs[m].mean(0)
        ok = np.abs(base) > 1e-9
        Y = np.zeros_like(xs[m])
        Y[:, ok] = (xs[m][:, ok] - base[ok]) / base[ok] * 100.0
        # ridge-free least squares; conditioning was gated upstream
        H, *_ = np.linalg.lstsq(T, Y, rcond=None)      # (n_theta, n_sensor)
        Hs.append(H.T)                                  # -> (n_sensor, n_theta)
        weights.append(m.sum())
    if not Hs:
        return None, 0
    W = np.asarray(weights, float)
    return np.average(np.asarray(Hs), axis=0, weights=W), len(Hs)


def angle_deg(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = list_files()
    if not files:
        raise SystemExit(f'no .h5 under {NC}; download first')
    print(f'{len(files)} files under {NC}', flush=True)

    Ws, Xs, Ts, names = [], [], [], None
    used, skipped = [], []
    for f in files:
        w, xs, th, nm = load_one(f)
        if w is None:
            why = (nm or {}).get('error', 'no T group (Challenge subset)')
            skipped.append({'file': f.name, 'reason': why})
            print(f'  SKIP {f.name}: {why}', flush=True)
            continue
        Ws.append(w); Xs.append(xs); Ts.append(th)
        names = names or nm
        used.append(f.name)
        print(f'  {f.name}: {len(w)} rows, theta {th.shape[1]}', flush=True)
    if not used:
        raise SystemExit('no file carried a T group')

    w = np.concatenate(Ws); xs = np.concatenate(Xs); th = np.concatenate(Ts)
    print(f'pooled: {len(w)} rows, {th.shape[1]} health params, '
          f'{xs.shape[1]} sensors', flush=True)

    # ---- GATE: is theta identifiable at all? -------------------------------
    cond = condition_report(th)
    ok = (cond['cond'] < COND_MAX and
          cond['effective_rank'] >= MIN_EFF_RANK_FRAC * cond['n_theta_varying'])
    print(f"condition {cond['cond']:.1f}, effective rank "
          f"{cond['effective_rank']}/{cond['n_theta_varying']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    verdict = {'files_used': used, 'files_skipped': skipped, 'names': names, 'n_rows': int(len(w)),
               'identifiability_gate': {**cond, 'cond_max': COND_MAX,
                                        'passed': bool(ok)},
               'scope': ('one-off study outside the reproducible pipeline; '
                         'needs a 14.68 GB download that make all must not require')}

    if not ok:
        verdict['conclusion'] = (
            'theta columns are too collinear to identify separate influence '
            'coefficients from this pooling. That is the finding: without '
            'independent variation per health parameter the matrix cannot be '
            'recovered from data, and no angle is reported.')
        (OUT / 'ncmapss_icm.json').write_text(json.dumps(verdict, indent=2))
        print('\n  GATE FAILED -- no ICM estimated, no angle reported')
        return

    H, n_bins_used = estimate_icm(w, xs, th)
    verdict['icm'] = {'shape': list(H.shape), 'bins_used': n_bins_used,
                      'matrix': H.tolist()}

    # ---- the comparison this exists for -----------------------------------
    tnames = names['theta']
    pair = next((p for p in PAIR_CANDIDATES
                 if p[0] in tnames and p[1] in tnames), None)
    if pair is None:
        verdict['angle'] = {'error': 'confusable pair not found in T names',
                            'theta_names': tnames}
    else:
        i, j = tnames.index(pair[0]), tnames.index(pair[1])
        xsn = names['xs']
        rows_ck = [xsn.index(v) for v in COCKPIT_MAP.values() if v in xsn]
        verdict['angle'] = {
            'pair': list(pair),
            'cockpit_analogue_sensors': [v for v in COCKPIT_MAP.values() if v in xsn],
            'angle_cockpit_deg': angle_deg(H[rows_ck, i], H[rows_ck, j]),
            'angle_all14_deg': angle_deg(H[:, i], H[:, j]),
            'ours_cockpit_deg': 1.3,
            'ours_extended_deg': 26.7,
            'note': ('our cockpit set is 3 channels and N-CMAPSS gives 14, so the '
                     'cockpit-analogue angle is the like-for-like comparison and '
                     'the 14-channel angle is the analogue of our extended set')}
        a = verdict['angle']
        print(f"\n  confusable pair {pair[0]} vs {pair[1]}")
        print(f"    cockpit analogue ({len(rows_ck)} ch): "
              f"{a['angle_cockpit_deg']:.2f} deg   (ours: 1.3)")
        print(f"    all 14 channels:  {a['angle_all14_deg']:.2f} deg   (ours ext: 26.7)")
    (OUT / 'ncmapss_icm.json').write_text(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
