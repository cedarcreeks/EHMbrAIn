"""Sensor/measurement model (WP2.3): noise, quantization, drift bias and
missing data, per the contract in conf/fault_catalog.yaml.

Truth in, "what the airline actually sees" out. Vectorized per channel.
"""

import numpy as np


def drift_bias(n_cycles, drifts, channel):
    """Additive bias series for one channel: zero before onset, linear ramp after."""
    if channel not in drifts:
        return np.zeros(n_cycles)
    onset, rate = drifts[channel]
    cycles = np.arange(n_cycles, dtype=float)
    return np.maximum(cycles - onset, 0.0) * rate


def measure_channel(true_vals, spec, rng):
    """Apply noise + quantization to one channel's true series."""
    x = np.asarray(true_vals, dtype=float)
    if 'sigma_pct' in spec:
        noisy = x * (1.0 + rng.normal(0.0, spec['sigma_pct'] / 100.0, size=x.shape))
    else:
        noisy = x + rng.normal(0.0, spec['sigma'], size=x.shape)
    q = spec.get('quant')
    return np.round(noisy / q) * q if q else noisy


def dropout_mask(n_cycles, spec, rng):
    """True where the snapshot is LOST (isolated MCAR + ACARS outage bursts)."""
    lost = rng.uniform(size=n_cycles) < spec['mcar_frac']
    n_bursts = rng.poisson(spec['burst_prob_per_kcycle'] * n_cycles / 1000.0)
    lo, hi = spec['burst_len_cycles']
    for _ in range(n_bursts):
        start = int(rng.uniform(0, n_cycles))
        lost[start:start + int(rng.uniform(lo, hi))] = True
    return lost
