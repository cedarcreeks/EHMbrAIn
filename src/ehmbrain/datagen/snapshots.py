"""ACARS-style snapshot generator (WP2.3): two reports per flight (takeoff +
cruise), the way airline EHM data actually arrives.

Physics: the linearization built in F1 —
    z(u, x) = baseline(u) + H(u) · x / 100 * baseline(u)
with the ICM H and healthy baseline interpolated between the grid points that
bracket the sampled operating condition (takeoff: in Delta-T_ISA; cruise: in
commanded N1). Its validity is checked by the WP2.4 nonlinearity audit against
full pyCycle solves.

Truth columns carry the ground-truth health state, EGT margin, RUL and labels;
measured columns carry what the airline sees (noise, quantization, drift,
missing snapshots).
"""

import numpy as np
import pandas as pd

from .fleet import egt_margin_series, load_icm
from .sensors import drift_bias, dropout_mask, measure_channel
from .trajectories import PARAM_INDEX

CHANNELS = ['N1_rpm', 'N2_rpm', 'WF_kgps', 'EGT_degK',
            'P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK']

# Mechanism -> component whose damage dominates it (for isolation labels).
MECH_COMPONENT = {'fouling': 'hpc', 'erosion': 'hpc', 'clearance': 'hpt',
                  'hot_section': 'hpt', 'lpt_wear': 'lpt', 'fod': 'varies'}


class IcmInterpolator:
    """Linear interpolation of (H, baseline) between two ICM grid points."""

    def __init__(self, point_a, point_b, coord_a, coord_b):
        self.Ha, self.cha, self.ba = load_icm(point_a)
        self.Hb, _, self.bb = load_icm(point_b)
        self.coord_a, self.coord_b = coord_a, coord_b

    def at(self, coord):
        w = (np.asarray(coord, dtype=float) - self.coord_a) / (self.coord_b - self.coord_a)
        return w  # weights; combine per-sample in the caller (vectorized)

    def baseline_matrix(self, w):
        """(n, channels) healthy baseline for weight vector w."""
        ba = np.array([self.ba[c] for c in CHANNELS])
        bb = np.array([self.bb[c] for c in CHANNELS])
        return ba[None, :] * (1 - w[:, None]) + bb[None, :] * w[:, None]

    def deviations(self, w, x):
        """(n, channels) percentage deviations H(w)·x for each cycle.

        N1 is the held power-setting parameter: it is not an ICM channel and
        its deviation is identically zero.
        """
        dev = np.zeros((x.shape[0], len(CHANNELS)))
        for j, c in enumerate(CHANNELS):
            if c not in self.cha:
                continue
            row = self.cha.index(c)
            dev[:, j] = ((x @ self.Ha[row]) * (1 - w)
                         + (x @ self.Hb[row]) * w)
        return dev


def sample_conditions(n_cycles, rng):
    """Per-flight operating-condition scatter (v1: reference conditions with
    ambient/power variation; richer mission mix is a queued extension)."""
    return {
        'to_dTs_C': np.clip(rng.normal(10.0, 8.0, n_cycles), -20.0, 35.0),
        'cr_dTs_C': np.clip(rng.normal(0.0, 6.0, n_cycles), -15.0, 20.0),
        'cr_N1_cmd': rng.uniform(4450.0, 4666.0, n_cycles),
    }


def dominant_label(x_row, contributions_row, threshold_pct=0.5):
    """Snapshot label: 'healthy' below threshold, else the mechanism with the
    largest absolute efficiency contribution."""
    if np.max(np.abs(x_row)) < threshold_pct:
        return 'healthy'
    best, best_mag = 'healthy', 0.0
    for mech, xm in contributions_row.items():
        mag = float(np.max(np.abs(xm)))
        if mag > best_mag:
            best, best_mag = mech, mag
    return best


def engine_snapshots(engine, contributions, catalog, rng):
    """Full snapshot table for one engine (truth + measured), as a DataFrame."""
    x = engine['x']                      # (life, 10), percent
    life = engine['life_cycles']
    cfg = engine['config']
    cond = sample_conditions(life, rng)

    to_interp = IcmInterpolator('takeoff', 'takeoff_hot', 0.0, 30.0)
    cr_interp = IcmInterpolator('cruise', 'cruise_lowpwr', 4666.0, 4400.0)

    df = pd.DataFrame({'engine_id': engine['engine_id'],
                       'cycle': np.arange(life, dtype=np.int32)})
    for k, v in cond.items():
        df[k] = v.astype(np.float32)

    for prefix, interp, coord in (('to', to_interp, cond['to_dTs_C']),
                                  ('cr', cr_interp, cond['cr_N1_cmd'])):
        w = interp.at(coord)
        base = interp.baseline_matrix(w)
        dev = interp.deviations(w, x)
        true_z = base * (1.0 + dev / 100.0)
        for j, ch in enumerate(CHANNELS):
            bias = drift_bias(life, cfg.drifts, ch)
            spec = catalog['sensors'][ch]
            df[f'{prefix}_{ch}'] = measure_channel(true_z[:, j] + bias,
                                                   spec, rng).astype(np.float32)
            df[f'{prefix}_{ch}_true'] = true_z[:, j].astype(np.float32)
        lost = dropout_mask(life, catalog['sensors']['dropout'], rng)
        for j, ch in enumerate(CHANNELS):
            df.loc[lost, f'{prefix}_{ch}'] = np.nan
        df[f'{prefix}_lost'] = lost

    # Ground truth
    for name, idx in PARAM_INDEX.items():
        df[f'x_{name.replace(".", "_")}'] = x[:, idx].astype(np.float32)
    df['egtm_C'] = engine['egtm_C'].astype(np.float32)
    df['rul'] = (life - 1 - df['cycle']).astype(np.int32)
    df['label'] = [dominant_label(x[i], {m: c[i] for m, c in contributions.items()})
                   for i in range(life)]
    drift_ch = next(iter(cfg.drifts), '')
    df['drift_channel'] = drift_ch
    if drift_ch:
        onset = cfg.drifts[drift_ch][0]
        df['drift_active'] = df['cycle'] >= onset
    else:
        df['drift_active'] = False
    return df
