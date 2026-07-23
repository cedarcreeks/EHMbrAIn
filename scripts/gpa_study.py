"""Gas path analysis of the CFM56-7B26, end to end (report chapter "GPA in practice").

The classical GPA performance study for this engine: inject known health
deviations into the *nonlinear* twin, invert the measurements with the linear
influence matrix exactly as a fielded GPA system would, and measure what comes
back. Everything a GPA practitioner asks of an engine before trusting a
diagnosis:

  A  fault signature atlas          -- the response pattern of every fault
  B  noise-free recovery            -- what the inversion returns with perfect
                                       sensors (isolates the linearization and
                                       the regularization bias)
  C  recovery under sensor noise    -- Monte Carlo: bias, spread, isolation rate
  D  minimum detectable fault       -- the magnitude at which each fault becomes
                                       identifiable, per sensor set
  E  realistic multi-fault case     -- a real fleet engine's end-of-life state
  F  Monte-Carlo vs Cramer-Rao      -- consistency of the empirical spread with
                                       the theoretical bound of the certificate

Forward model: the L1 neural twin (nonlinear, 13-19x more faithful than the
linearization). Inversion: regularized WLS-GPA with the F1 cruise ICM and the
catalog's sensor noise -- the estimator of the traditional pipeline, at its
registered defaults.

Output: data/processed/gpa/gpa_study.json
Usage:  uv run python scripts/gpa_study.py
"""

import json
from pathlib import Path

import numpy as np
import yaml

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import COCKPIT, EXTENDED, HEALTH_PARAMS
from ehmbrain.perf.surrogate import SurrogateEmitter
from ehmbrain.trad.pipeline import isolate_step, wls_gpa

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'data' / 'processed' / 'gpa'
FLEET_DIR = REPO_ROOT / 'data' / 'processed' / 'fleet'

N1_CRUISE = 4666.0          # design N1: the cruise snapshot condition
PRIOR_SIGMA, LAM = 2.0, 1.0  # registered [T] defaults of the traditional WLS
RNG = np.random.default_rng(20260717)

# Physically expected direction of each health parameter under deterioration
# (report table 2.2): efficiencies fall; compressor flow capacity falls with
# fouling, turbine flow capacity opens with erosion.
FAULT_SIGN = {'fan.eta': -1, 'fan.flow': -1, 'lpc.eta': -1, 'lpc.flow': -1,
              'hpc.eta': -1, 'hpc.flow': -1, 'hpt.eta': -1, 'hpt.flow': +1,
              'lpt.eta': -1, 'lpt.flow': +1}


def sensor_sigma_pct(baseline):
    """Per-channel noise sigma expressed in % of the healthy cruise value,
    read from the versioned fault catalog (norm N4: no hardcoded noise)."""
    cat = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    sens = cat['sensors'] if 'sensors' in cat else cat['sensor_model']['channels']
    out = {}
    for ch, spec in sens.items():
        if ch not in baseline:
            continue
        if 'sigma_pct' in spec:
            out[ch] = float(spec['sigma_pct'])
        else:
            out[ch] = float(100.0 * spec['sigma'] / abs(baseline[ch]))
    return out


class Bench:
    """Forward (nonlinear twin) + inverse (linear WLS-GPA) at cruise."""

    def __init__(self):
        H_ext, channels, baseline = load_icm('cruise')
        self.channels = channels
        self.baseline = baseline
        self.H = {'extended': np.array([H_ext[channels.index(c)] for c in EXTENDED]),
                  'cockpit': np.array([H_ext[channels.index(c)] for c in COCKPIT])}
        self.sig_pct = sensor_sigma_pct(baseline)
        self.R = {k: np.array([self.sig_pct[c] for c in (COCKPIT if k == 'cockpit'
                                                         else EXTENDED)])
                  for k in ('cockpit', 'extended')}
        self.emit = SurrogateEmitter.cached()
        self.z0 = {c: float(v[0]) for c, v in self._raw(np.zeros((1, 10))).items()}

    def _raw(self, X):
        from ehmbrain.perf.surrogate import SURR_CHANNELS
        vals = self.emit.predict(X, np.full(len(X), N1_CRUISE), 'cruise')
        return {c: vals[:, SURR_CHANNELS.index(c)] for c in EXTENDED}

    def measure(self, X, sensor_set):
        """True (noise-free) % deviations of the sensor set for health rows X,
        from the NONLINEAR twin."""
        raw = self._raw(X)
        chans = COCKPIT if sensor_set == 'cockpit' else EXTENDED
        return np.column_stack([100.0 * (raw[c] - self.z0[c]) / self.z0[c]
                                for c in chans])

    def invert(self, dz, sensor_set):
        return np.array([wls_gpa(d, self.H[sensor_set], self.R[sensor_set],
                                 prior_sigma=PRIOR_SIGMA, lam=LAM) for d in dz])

    def crb_std(self, sensor_set):
        """Cramer-Rao per-parameter std for a single snapshot (the certificate's
        instrument, ch. 13), for cross-checking the Monte-Carlo spread."""
        H, R = self.H[sensor_set], self.R[sensor_set]
        F = H.T @ np.diag(1.0 / R ** 2) @ H + np.eye(10) / PRIOR_SIGMA ** 2
        return np.sqrt(np.diag(np.linalg.inv(F)))


def smearing_index(x_hat, j):
    """Fraction of the estimated magnitude that lands on healthy components."""
    tot = np.abs(x_hat).sum(axis=-1)
    return np.where(tot > 0, 1.0 - np.abs(x_hat[..., j]) / np.maximum(tot, 1e-12), np.nan)


def experiment_recovery(b, magnitude=1.0, n_mc=500):
    """B + C: single-fault recovery, noise-free and under sensor noise."""
    out = {}
    for j, p in enumerate(HEALTH_PARAMS):
        x = np.zeros((1, 10))
        x[0, j] = FAULT_SIGN[p] * magnitude
        entry = {'true_value_pct': float(x[0, j])}
        for ss in ('cockpit', 'extended'):
            dz = b.measure(x, ss)
            clean = b.invert(dz, ss)[0]
            noise = RNG.normal(0.0, b.R[ss], size=(n_mc, len(b.R[ss])))
            dz_noisy = dz + noise
            noisy = b.invert(dz_noisy, ss)
            top = np.argmax(np.abs(noisy), axis=1)
            # Two isolation rules: the classical nearest-signature match in
            # MEASUREMENT space (what the traditional pipeline fields) and the
            # naive largest-estimated-deviation rule on the inverted vector.
            named = [isolate_step(v, b.H[ss]) for v in dz_noisy]
            entry[ss] = {
                'estimate_noisefree_pct': float(clean[j]),
                'recovered_frac_noisefree': float(clean[j] / x[0, j]),
                'smearing_noisefree': float(smearing_index(clean, j)),
                'estimate_mean_pct': float(noisy[:, j].mean()),
                'estimate_std_pct': float(noisy[:, j].std()),
                # bias/variance split: the regularized estimator trades a large
                # shrinkage bias for a small variance, so its spread alone is
                # NOT comparable with the Cramer-Rao bound (which bounds
                # unbiased estimators) -- the same lesson H10.2 learned.
                'bias_pct': float(noisy[:, j].mean() - x[0, j]),
                'rmse_pct': float(np.sqrt(((noisy[:, j] - x[0, j]) ** 2).mean())),
                'smearing_mean': float(np.nanmean(smearing_index(noisy, j))),
                'isolation_rate_signature': float(np.mean([n == p for n in named])),
                'isolation_rate_argmax': float((top == j).mean()),
                'most_common_named_wrong': (
                    max({n for n in named if n != p},
                        key=lambda w: named.count(w), default=None)),
                'most_common_wrong': (HEALTH_PARAMS[int(np.bincount(
                    top[top != j], minlength=10).argmax())]
                    if (top != j).any() else None),
                'crb_std_pct': float(b.crb_std(ss)[j]),
            }
        out[p] = entry
    return out


def experiment_min_detectable(b, grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                              n_mc=200, target=0.9):
    """D: smallest injected magnitude the classical nearest-signature rule
    names correctly in >= `target` of noisy draws."""
    out = {}
    for j, p in enumerate(HEALTH_PARAMS):
        curves = {}
        for ss in ('cockpit', 'extended'):
            rates, found = [], None
            for m in grid:
                x = np.zeros((1, 10))
                x[0, j] = FAULT_SIGN[p] * m
                dz = b.measure(x, ss)
                noise = RNG.normal(0.0, b.R[ss], size=(n_mc, len(b.R[ss])))
                named = [isolate_step(v, b.H[ss]) for v in dz + noise]
                r = float(np.mean([n == p for n in named]))
                rates.append(r)
                if found is None and r >= target:
                    found = m
            curves[ss] = {'grid_pct': list(grid), 'isolation_rate': rates,
                          'min_detectable_pct': found}
        out[p] = curves
    return out


def experiment_multifault(b, n_mc=200):
    """E: a real fleet engine's end-of-life health state, diagnosed."""
    import pandas as pd
    cols = ['engine_id', 'cycle'] + [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    df = pd.read_parquet(FLEET_DIR / 'snapshots.parquet', columns=cols)
    eid = int(df.engine_id.max())
    e = df[df.engine_id == eid]
    row = e.iloc[int(0.97 * len(e))]           # late life, still on wing
    x = np.array([[row[f'x_{p.replace(".", "_")}'] for p in HEALTH_PARAMS]])
    out = {'engine_id': eid, 'cycle': int(row.cycle),
           'true_pct': dict(zip(HEALTH_PARAMS, x[0].tolist()))}
    for ss in ('cockpit', 'extended'):
        dz = b.measure(x, ss)
        noise = RNG.normal(0.0, b.R[ss], size=(n_mc, len(b.R[ss])))
        noisy = b.invert(dz + noise, ss)
        out[ss] = {
            'estimate_mean_pct': dict(zip(HEALTH_PARAMS, noisy.mean(0).tolist())),
            'estimate_std_pct': dict(zip(HEALTH_PARAMS, noisy.std(0).tolist())),
            'rmse_pct': float(np.sqrt(((noisy - x) ** 2).mean())),
            'deviations_pct': dict(zip(COCKPIT if ss == 'cockpit' else EXTENDED,
                                       dz[0].tolist())),
        }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b = Bench()
    print('sensor noise [% of healthy cruise value]:',
          {k: round(v, 3) for k, v in b.sig_pct.items()})

    print('A/B/C: signature atlas + single-fault recovery...', flush=True)
    recovery = experiment_recovery(b)
    print('D: minimum detectable fault...', flush=True)
    mindet = experiment_min_detectable(b)
    print('E: realistic multi-fault case...', flush=True)
    multi = experiment_multifault(b)

    H_ext = b.H['extended']
    report = {
        'setup': {
            'condition': f'cruise, M0.78/FL350, N1 = {N1_CRUISE:.0f} rpm',
            'forward_model': 'L1 neural twin (nonlinear)',
            'inverse_model': f'regularized WLS-GPA, prior_sigma={PRIOR_SIGMA}, lam={LAM}',
            'sensor_sigma_pct': b.sig_pct,
            'health_params': HEALTH_PARAMS,
            'channels_cockpit': COCKPIT, 'channels_extended': EXTENDED,
        },
        'signature_atlas': {p: dict(zip(EXTENDED, H_ext[:, j].tolist()))
                            for j, p in enumerate(HEALTH_PARAMS)},
        'crb_std_pct': {ss: dict(zip(HEALTH_PARAMS, b.crb_std(ss).tolist()))
                        for ss in ('cockpit', 'extended')},
        'recovery': recovery,
        'min_detectable': mindet,
        'multifault': multi,
    }
    (OUT_DIR / 'gpa_study.json').write_text(json.dumps(report, indent=2))

    print('\n== single-fault recovery at 1 % (cockpit / extended) ==')
    for p, e in recovery.items():
        print(f"  {p:9s} recovered {100 * e['cockpit']['recovered_frac_noisefree']:5.1f}%"
              f" / {100 * e['extended']['recovered_frac_noisefree']:5.1f}%   "
              f"smearing {e['cockpit']['smearing_mean']:.2f} / "
              f"{e['extended']['smearing_mean']:.2f}   "
              f"isolation {e['cockpit']['isolation_rate_signature']:.2f} / "
              f"{e['extended']['isolation_rate_signature']:.2f}")
    print('\n== minimum detectable magnitude [%] ==')
    for p, c in mindet.items():
        print(f"  {p:9s} cockpit {c['cockpit']['min_detectable_pct']}   "
              f"extended {c['extended']['min_detectable_pct']}")
    print(f"\nmulti-fault RMSE: cockpit {multi['cockpit']['rmse_pct']:.3f} %, "
          f"extended {multi['extended']['rmse_pct']:.3f} %")
    print(f'-> {OUT_DIR / "gpa_study.json"}')


if __name__ == '__main__':
    main()
