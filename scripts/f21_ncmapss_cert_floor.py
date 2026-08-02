"""F21 (prereg-v24): port the identifiability certificate (F10) and the
prognostic floor (F11) to N-CMAPSS truth.

WHY THIS IS DIFFERENT FROM F10/F11 AS PUBLISHED. Both were validated against the
ground truth of the generator that produced the data -- our own. The honest
objection has always been that a certificate checked against its own simulator's
truth proves internal consistency, not honesty about engines. N-CMAPSS breaks
that loop: the truth is NASA's, the engine is not ours, and the influence matrix
is estimated from the data rather than taken from the model that made it
(sec:f20-ncmapss).

WHAT IS PORTED

F11, the prognostic floor, ports directly and needs no influence matrix at all.
Condition on each unit's TRUE health state, find its nearest neighbours in health
space, and read the spread of their true remaining lives. That spread is what no
present measurement can remove -- the aleatoric floor. Everything it needs
(theta, RUL) ships with the dataset.

F10, the certificate, needs three things N-CMAPSS does not hand over: an
influence matrix (estimated in F20), a sensor noise covariance (estimated here
from within-flight high-frequency residuals at near-constant conditions), and an
estimator to be honest ABOUT (a weighted least-squares gas-path solve using the
same estimated matrix). The certificate is then the Cramer-Rao bound from Fisher
information accumulated over the unit's actually flown conditions, and the test
is whether it ranks the estimator's true per-direction error.

DISCLOSED, because it bounds the claim: the bound and the estimator share the
same ESTIMATED influence matrix, so an error in it moves both. That is the same
coupling F10 has in this project (both use the pyCycle matrix), so the port is
not worse on that axis -- and it is strictly better on the axis that mattered,
because the truth is external.

Scope: outside the reproducible pipeline; needs the 14.68 GB download.

Output: data/processed/f21/ncmapss_cert_floor.json
Usage: uv run python scripts/f21_ncmapss_cert_floor.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f20_ncmapss_icm import (NC, angle_deg, condition_report,  # noqa: E402
                             estimate_icm, list_files)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f21'
FRACS = (0.5, 0.7, 0.9)         # life fractions, as F11
KNN = 10                        # neighbours in health space, as F11
PRIOR_STD = 2.0                 # diffuse prior on each modifier, as F10
LATE = 0.9                      # "late life" cut for the certificate test


def load_units(stride=20, max_files=None):
    """Per-unit arrays. Units are keyed (file, unit) so ids never collide."""
    import h5py
    units, names = {}, None
    for path in list_files()[:max_files]:
        try:
            fh = h5py.File(path, 'r')
        except OSError:
            continue
        with fh as f:
            if 'T_var' not in f:
                continue

            def dec(k):
                return [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                        for x in np.atleast_1d(np.array(f[k])).ravel()]
            names = names or {'w': dec('W_var'), 'xs': dec('X_s_var'),
                              'theta': dec('T_var'), 'a': dec('A_var')}
            for grp in ('dev', 'test'):
                if f'A_{grp}' not in f:
                    continue
                A = np.array(f[f'A_{grp}'])[::stride]
                W = np.array(f[f'W_{grp}'])[::stride]
                X = np.array(f[f'X_s_{grp}'])[::stride]
                T = np.array(f[f'T_{grp}'])[::stride]
                Y = np.array(f[f'Y_{grp}'])[::stride].ravel()
                for u in np.unique(A[:, 0]):
                    m = A[:, 0] == u
                    units[(path.name, float(u))] = {
                        'w': W[m], 'xs': X[m], 'theta': T[m],
                        'rul': Y[m], 'cycle': A[m, 1]}
        print(f'  {path.name}: {len(units)} units so far', flush=True)
    return units, names


# --------------------------------------------------------------------------
# F11: the prognostic floor, ported directly
# --------------------------------------------------------------------------

def prognostic_floor(units):
    """At each life fraction: condition on TRUE health, read the spread of true
    remaining life among nearest neighbours. That spread is irreducible."""
    out = {}
    for frac in FRACS:
        TH, RUL = [], []
        for u in units.values():
            n = len(u['rul'])
            i = int(frac * n)
            if i >= n:
                continue
            TH.append(u['theta'][i])
            RUL.append(u['rul'][i])
        TH, RUL = np.asarray(TH), np.asarray(RUL)
        if len(TH) < KNN + 2:
            continue
        Z = (TH - TH.mean(0)) / (TH.std(0) + 1e-12)
        floors = []
        for i in range(len(Z)):
            d = np.linalg.norm(Z - Z[i], axis=1)
            nb = np.argsort(d)[1:KNN + 1]
            floors.append(np.std(RUL[nb]))
        out[str(frac)] = {
            'aleatoric_floor': float(np.median(floors)),
            'marginal_std': float(np.std(RUL)),
            'irreducible_share': float(np.median(floors) / (np.std(RUL) + 1e-12)),
            'n_units': int(len(TH))}
    return out


# --------------------------------------------------------------------------
# F10: the certificate
# --------------------------------------------------------------------------

def estimate_noise(units, n_bins=6):
    """Sensor noise sigma [%] from high-frequency residuals.

    Within one unit the health state moves over thousands of cycles, so
    consecutive samples at a comparable condition differ almost entirely by
    measurement noise. The lag-1 difference therefore estimates sqrt(2)*sigma.
    """
    per_sensor = []
    for u in list(units.values())[:40]:
        xs = u['xs']
        if len(xs) < 50:
            continue
        d = np.diff(xs, axis=0)
        base = np.abs(xs).mean(0) + 1e-12
        per_sensor.append(np.median(np.abs(d), axis=0) / base * 100.0 / np.sqrt(2)
                          * 1.4826)
    return np.median(np.asarray(per_sensor), axis=0)


def certificate_and_error(units, H, sigma, names):
    """CRB per health direction against the actual WLS error, per unit."""
    Rinv = np.diag(1.0 / np.maximum(sigma, 1e-6) ** 2)
    P0inv = np.eye(H.shape[1]) / PRIOR_STD ** 2
    crbs, errs, keys = [], [], []
    for k, u in units.items():
        n = len(u['rul'])
        if n < 40:
            continue
        # Fisher accumulated over the conditions this unit actually flew.
        # n_eff is the number of distinct FLIGHT CYCLES, not of samples:
        # consecutive 1 Hz samples within a flight are strongly correlated and
        # counting them as independent collapses the bound to zero. A cycle is
        # the natural independent observation opportunity, matching how this
        # project accumulates Fisher over per-flight reports (sec:bt-honesty).
        n_eff = int(len(np.unique(u['cycle'])))
        F = P0inv + n_eff * (H.T @ Rinv @ H)
        S = np.linalg.inv(F)
        crbs.append(np.sqrt(np.diag(S)))
        # WLS estimate of theta at late life from percent deviations
        i0, i1 = int(LATE * n), n
        healthy = u['xs'][:max(5, n // 20)].mean(0)
        ok = np.abs(healthy) > 1e-9
        dz = np.zeros((i1 - i0, H.shape[0]))
        dz[:, ok] = ((u['xs'][i0:i1][:, ok] - healthy[ok]) / healthy[ok]) * 100.0
        M = np.linalg.solve(H.T @ Rinv @ H + P0inv, H.T @ Rinv)
        xhat = (M @ dz.T).T.mean(0)
        xtrue = u['theta'][i0:i1].mean(0)
        errs.append(np.abs(xhat - xtrue))
        keys.append(f'{k[0]}|{k[1]:.0f}')
    return np.asarray(crbs), np.asarray(errs), keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('== loading units ==', flush=True)
    units, names = load_units()
    print(f'{len(units)} units, {names["theta"]}', flush=True)

    print('== F11: prognostic floor (needs no influence matrix) ==', flush=True)
    floor = prognostic_floor(units)
    for f, v in floor.items():
        print(f"  {float(f):.0%} life: floor {v['aleatoric_floor']:7.1f} cy   "
              f"marginal {v['marginal_std']:7.1f}   "
              f"irreducible {v['irreducible_share']:.3f}", flush=True)

    print('== F10: certificate ==', flush=True)
    W = np.concatenate([u['w'] for u in units.values()])
    XS = np.concatenate([u['xs'] for u in units.values()])
    TH = np.concatenate([u['theta'] for u in units.values()])
    gate = condition_report(TH)
    H, nb = estimate_icm(W, XS, TH)
    sigma = estimate_noise(units)
    print(f'  ICM {H.shape} from {nb} bins; gate cond {gate["cond"]:.1f}, '
          f'rank {gate["effective_rank"]}/{gate["n_theta_varying"]}', flush=True)
    print('  sigma [%]: ' + '  '.join(f'{s:.3f}' for s in sigma), flush=True)

    crb, err, keys = certificate_and_error(units, H, sigma, names)

    # CONTROL. The bound and the estimator share the same estimated H, so an
    # error in H moves both and could manufacture agreement. Repeat with H
    # column-shuffled: the physics is destroyed but the coupling is identical.
    # If the shuffled control also ranks, the result is the coupling, not the
    # certificate. Same role the fraud check plays in sec:f15-instrument.
    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(20):
        Hs = H[:, rng.permutation(H.shape[1])]
        c2, e2, _ = certificate_and_error(units, Hs, sigma, names)
        r = spearmanr(c2.mean(0), e2.mean(0)).statistic
        if np.isfinite(r):
            ctrl.append(float(r))
    print(f'  control (column-shuffled H, {len(ctrl)} draws): '
          f'rho median {np.median(ctrl):+.3f}, '
          f'95th pct {np.percentile(np.abs(ctrl), 95):.3f}', flush=True)
    # the F10 test: does the bound RANK the actual per-direction error?
    rho, p = spearmanr(crb.mean(0), err.mean(0))
    per_unit = [spearmanr(crb[i], err[i]).statistic for i in range(len(crb))]

    verdict = {
        'scope': ('outside the reproducible pipeline; needs the 14.68 GB '
                  'N-CMAPSS download'),
        'n_units': len(units), 'theta_names': names['theta'],
        'F11_prognostic_floor': floor,
        'F10_certificate': {
            'icm_shape': list(H.shape), 'bins': nb,
            'identifiability_gate': {'cond': gate['cond'],
                                     'effective_rank': gate['effective_rank'],
                                     'n_theta_varying': gate['n_theta_varying']},
            'sigma_pct': sigma.tolist(),
            'n_eff_basis': 'distinct flight cycles per unit, not 1 Hz samples',
            'crb_pct_mean': crb.mean(0).tolist(),
            'wls_abs_err_mean': err.mean(0).tolist(),
            'spearman_across_directions': {'rho': float(rho), 'p': float(p)},
            'spearman_per_unit_median': float(np.nanmedian(per_unit)),
            'shuffled_H_control': {
                'rho_median': float(np.median(ctrl)) if ctrl else None,
                'abs_rho_p95': float(np.percentile(np.abs(ctrl), 95)) if ctrl else None,
                'n_draws': len(ctrl),
                'purpose': ('the bound and the estimator share the same estimated '
                            'H, so column-shuffling H keeps the coupling and '
                            'destroys the physics; a high control rho would mean '
                            'the agreement is the coupling, not the certificate')},
            'ours_F10_rho': 0.70,
            'confirmed': bool(rho >= 0.6 and p < 0.05)},
        'disclosed': ('the bound and the estimator share the same ESTIMATED '
                      'influence matrix, so an error in it moves both -- the '
                      'same coupling F10 has in this project, where both use the '
                      'pyCycle matrix. What is strictly better here is that the '
                      'TRUTH is external')}
    (OUT / 'ncmapss_cert_floor.json').write_text(json.dumps(verdict, indent=2))

    print()
    print('  direction        CRB [%]   |err| [%]   ratio')
    for nm, c, e in zip(names['theta'], crb.mean(0), err.mean(0)):
        print(f'  {nm:16s} {c:8.4f}  {e:8.4f}   {e / max(c, 1e-12):6.2f}')
    print(f"\n  Spearman(CRB, actual error) across directions: rho={rho:.3f} "
          f"(p={p:.4f})   ours: 0.70")
    print(f"  median per-unit rho: {np.nanmedian(per_unit):.3f}")
    print(f"  F10 ported and confirmed: "
          f"{verdict['F10_certificate']['confirmed']}")


if __name__ == '__main__':
    main()
