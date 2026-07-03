"""WP2.3: sensor model and ACARS snapshot generator."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ehmbrain.datagen.fleet import generate_engine, load_icm
from ehmbrain.datagen.sensors import dropout_mask, drift_bias, measure_channel
from ehmbrain.datagen.snapshots import engine_snapshots

REPO_ROOT = Path(__file__).resolve().parents[1]
ICM_DIR = REPO_ROOT / 'data' / 'processed' / 'icm'


@pytest.fixture(scope='module')
def catalog():
    return yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())


def test_measure_channel_stats():
    rng = np.random.default_rng(1)
    true = np.full(20000, 1000.0)
    m = measure_channel(true, {'sigma': 2.5, 'quant': 1.0}, rng)
    assert abs(np.std(m) - 2.5) < 0.15          # noise magnitude
    assert np.allclose(m, np.round(m))           # quantization grid
    m2 = measure_channel(true, {'sigma_pct': 0.5, 'quant': None}, rng)
    assert abs(np.std(m2) - 5.0) < 0.3           # 0.5 % of 1000


def test_drift_bias_ramp():
    b = drift_bias(1000, {'EGT_degK': (400.0, 0.01)}, 'EGT_degK')
    assert np.all(b[:401] == 0.0)
    assert abs(b[-1] - (999 - 400) * 0.01) < 1e-9
    assert np.all(drift_bias(1000, {}, 'EGT_degK') == 0.0)


def test_dropout_fraction(catalog):
    rng = np.random.default_rng(2)
    lost = dropout_mask(50000, catalog['sensors']['dropout'], rng)
    assert 0.02 < lost.mean() < 0.12             # MCAR + bursts


@pytest.mark.skipif(not ICM_DIR.exists(), reason='ICM artifacts not generated')
def test_engine_snapshots_consistency(catalog):
    H, ch, base = load_icm('takeoff_hot')
    rng = np.random.default_rng(3)
    engine = generate_engine(0, catalog, H, ch, base, rng)
    df = engine_snapshots(engine, engine['contributions'], catalog,
                          np.random.default_rng(4))

    assert len(df) == engine['life_cycles']
    assert int(df.rul.iloc[-1]) == 0
    assert int(df.rul.iloc[0]) == engine['life_cycles'] - 1
    # True channels carry no NaN (dropout affects measured only).
    assert not df.filter(like='_true').isna().any().any()
    # Degradation raises the true cruise EGT over life (beyond ambient scatter).
    egt = df.cr_EGT_degK_true.to_numpy()
    assert egt[-100:].mean() - egt[:100].mean() > 20.0
    # EGT margin decays to ~zero at end of life.
    assert df.egtm_C.iloc[0] > 50.0
    assert df.egtm_C.iloc[-1] <= 0.5
    # N1 measured stays near its baseline (setting, not a health channel).
    assert abs(df.to_N1_rpm.dropna().std()) < 20.0