"""F3 primitives: smoothing, detectors, WLS, Kalman, Theil-Sen."""

import numpy as np

from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.pipeline import (cusum, ewma, gap_alert, holt_smooth,
                                    kalman_gpa, theil_sen_rul, wls_gpa)


def test_holt_tracks_trend_and_innovations():
    y = 0.001 * np.arange(5000) + np.random.default_rng(0).normal(0, 0.2, 5000)
    level, trend, innov = holt_smooth(y)
    assert abs(np.nanmean(trend[2000:]) - 0.001) < 5e-4   # slope recovered on average
    assert abs(np.nanmean(innov[1000:])) < 0.02   # innovations centered


def test_cusum_catches_step_not_trend():
    rng = np.random.default_rng(1)
    trend = 0.0002 * np.arange(8000) + rng.normal(0, 0.2, 8000)
    _, _, innov = holt_smooth(trend)
    assert cusum(innov, drift_k=0.75, h=8.0) is None   # chronic: silent
    step = trend.copy()
    step[4000:] += 1.5
    _, _, innov2 = holt_smooth(step)
    d = cusum(innov2, drift_k=0.75, h=8.0)
    assert d is not None and 4000 <= d < 4200          # step: fast alarm


def test_gap_alert_catches_ramp():
    rng = np.random.default_rng(2)
    y = 0.0002 * np.arange(10000) + rng.normal(0, 0.2, 10000)
    assert gap_alert(y) is None
    ramp = y.copy()
    ramp[6000:6300] += np.linspace(0, 1.0, 300)
    ramp[6300:] += 1.0
    d = gap_alert(ramp)
    assert d is not None and 6000 <= d <= 6500


def test_wls_recovers_strong_fault():
    rng = np.random.default_rng(3)
    H = rng.normal(0, 0.5, (3, 10))
    x_true = np.zeros(10)
    x_true[6] = -1.5
    dz = H @ x_true
    x_hat = wls_gpa(dz, H, [0.07, 0.5, 0.23], lam=0.1)
    assert np.argmax(np.abs(x_hat)) == 6           # right dominant parameter


def test_kalman_converges_on_static_fault():
    rng = np.random.default_rng(4)
    H = rng.normal(0, 0.5, (3, 10))
    x_true = np.zeros(10)
    x_true[4] = -2.0
    R = np.array([0.07, 0.5, 0.23])
    dz = (H @ x_true)[None, :] + rng.normal(0, R, (4000, 3))
    xs = kalman_gpa(dz, lambda i: H, R, q=1e-5)
    resid = xs[-1] - x_true
    # Underdetermined (3 obs, 10 states): the estimate matches in observation
    # space even if not in state space.
    assert np.linalg.norm(H @ resid) < 0.1


def test_theil_sen_linear_decay():
    y = 80.0 - 0.01 * np.arange(6000)
    rul = theil_sen_rul(y)
    assert rul is not None
    assert abs(rul - (80.0 - 0.01 * 5999) / 0.01) < 300
