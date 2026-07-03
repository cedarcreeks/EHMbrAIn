"""WP2.1/WP2.2: health trajectories, wash sawtooth, FOD steps, fleet sampling."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ehmbrain.datagen.fleet import assign_splits
from ehmbrain.datagen.trajectories import (PARAM_INDEX, health_series,
                                           sample_engine_config)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def catalog():
    return yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())


@pytest.fixture()
def engine(catalog):
    rng = np.random.default_rng(42)
    cfg = sample_engine_config(0, catalog, 20000, rng)
    x, contributions, events = health_series(cfg, catalog, 20000)
    return cfg, x, contributions, events


def test_shapes_and_signs(engine):
    _, x, contributions, _ = engine
    assert x.shape == (20000, 10)
    # All efficiency deviations must be <= 0 at end of life (degradation).
    for comp in ('fan', 'lpc', 'hpc', 'hpt', 'lpt'):
        assert x[-1, PARAM_INDEX[f'{comp}.eta']] < 0
    # Hot-section flow capacity increases (nozzle-area opening).
    assert x[-1, PARAM_INDEX['hpt.flow']] > 0
    # Compressor fouling reduces flow capacity.
    assert x[-1, PARAM_INDEX['hpc.flow']] < 0


def test_wash_sawtooth(engine):
    cfg, _, contributions, _ = engine
    fouling = contributions['fouling'][:, PARAM_INDEX['hpc.eta']]
    for wc, r in zip(cfg.wash_cycles, cfg.wash_recovery):
        i = int(np.ceil(wc))          # wash applies at the first cycle >= wc
        if i + 1 >= len(fouling):
            break
        # Wash improves (moves toward zero) the fouling contribution by ~r.
        assert abs(fouling[i]) < abs(fouling[i - 1])
        np.testing.assert_allclose(fouling[i], fouling[i - 1] * (1 - r), rtol=0.05)
    # Regrowth after a wash heads back toward the same asymptote: the level
    # just before the second wash exceeds the level just after the first.
    if len(cfg.wash_cycles) >= 2:
        i1, i2 = int(np.ceil(cfg.wash_cycles[0])), int(np.ceil(cfg.wash_cycles[1]))
        assert abs(fouling[i2 - 1]) > abs(fouling[i1])


def test_fod_steps(engine):
    cfg, _, contributions, _ = engine
    fod = contributions['fod']
    for wc, comp, step in zip(cfg.fod_cycles, cfg.fod_component, cfg.fod_step_pct):
        i = int(np.searchsorted(np.arange(20000.0), wc))
        if i >= 20000 or i == 0:
            continue
        jump = fod[i, PARAM_INDEX[f'{comp}.eta']] - fod[i - 1, PARAM_INDEX[f'{comp}.eta']]
        np.testing.assert_allclose(jump, step, rtol=1e-9)


def test_reproducibility(catalog):
    a = sample_engine_config(7, catalog, 10000, np.random.default_rng(7))
    b = sample_engine_config(7, catalog, 10000, np.random.default_rng(7))
    assert np.array_equal(a.wash_cycles, b.wash_cycles)
    assert a.multipliers == b.multipliers


def test_splits_leakage_free(catalog):
    rng = np.random.default_rng(0)
    splits = assign_splits(range(100), catalog['fleet']['split'], rng)
    assert len(splits) == 100
    counts = {s: sum(1 for v in splits.values() if v == s)
              for s in ('train', 'val', 'test')}
    assert counts == {'train': 70, 'val': 10, 'test': 20}
