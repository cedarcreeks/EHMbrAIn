"""Fleet assembly (WP2.2): engine sampling, end-of-life determination via the
EGT margin, and leakage-free splits.

End of life: the engine comes off the wing when its EGT margin at the hot-day
takeoff reference condition reaches zero. The margin trajectory is computed by
projecting the health vector through the takeoff-hot ICM's EGT row (the same
linearization the snapshot generator uses):
    dEGT_K(n) = H_egt [%/%] . x(n) [%] * EGT_baseline_K / 100
    EGTM(n)   = EGTM_new - dEGT_K(n)
"""

import json
from pathlib import Path

import numpy as np

from .trajectories import health_series, sample_engine_config

REPO_ROOT = Path(__file__).resolve().parents[2].parent
ICM_DIR = REPO_ROOT / 'data' / 'processed' / 'icm'


def load_icm(point='takeoff_hot'):
    """ICM matrix, channel list and healthy baseline snapshot for one point."""
    d = np.load(ICM_DIR / f'icm_{point}.npz', allow_pickle=True)
    return (d['H'], [str(c) for c in d['channels']],
            json.loads(str(d['baseline'])))


def egt_margin_series(x, H, channels, baseline, egtm_new_C):
    """EGTM(n) in Kelvin-equivalent C over the trajectory (vectorized)."""
    row = channels.index('EGT_degK')
    degt_K = x @ H[row] * baseline['EGT_degK'] / 100.0
    return egtm_new_C - degt_K


def generate_engine(engine_id, catalog, H, channels, baseline, rng):
    """One engine, truncated at its end of life.

    Returns dict with config, truth trajectory, events, EGTM series and life.
    """
    max_cycles = catalog['fleet']['max_cycles']
    cfg = sample_engine_config(engine_id, catalog, max_cycles, rng)
    x, contributions, events = health_series(cfg, catalog, max_cycles)
    egtm = egt_margin_series(x, H, channels, baseline, cfg.egtm_new_C)

    below = np.nonzero(egtm <= 0.0)[0]
    life = int(below[0]) + 1 if len(below) else max_cycles
    x = x[:life]
    egtm = egtm[:life]
    events = [e for e in events if e['cycle'] < life]
    contributions = {m: c[:life] for m, c in contributions.items()}

    return {'engine_id': engine_id, 'config': cfg, 'x': x, 'egtm_C': egtm,
            'events': events, 'life_cycles': life,
            'censored': bool(life == max_cycles),
            'contributions': contributions}


def assign_splits(engine_ids, split_spec, rng):
    """Leakage-free split by engine: every id in exactly one partition."""
    ids = list(engine_ids)
    rng.shuffle(ids)
    n_train, n_val = split_spec['train'], split_spec['val']
    out = {}
    for i, eid in enumerate(ids):
        out[eid] = ('train' if i < n_train
                    else 'val' if i < n_train + n_val else 'test')
    return out
