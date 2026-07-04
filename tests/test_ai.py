"""F4 primitives: feature builder, window machinery, models, conformal logic."""

import numpy as np
import pytest
import torch

from ehmbrain.ai.data import _ffill, windows_from
from ehmbrain.ai.models import (RULNet, WindowAE, ae_scores, device,
                                predict_torch, train_torch)


def test_ffill():
    a = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    out = _ffill(a)
    np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 1.0, 4.0, 4.0])  # leading NaN back-filled


def test_windows_shape_and_alignment():
    F = np.arange(400, dtype=float).reshape(100, 4)
    ends, w = windows_from(F, w=10, stride=5)
    assert w.shape == (len(ends), 10, 4)
    # window i ends at ends[i] (exclusive): last row is F[ends[i]-1]
    np.testing.assert_array_equal(w[0][-1], F[ends[0] - 1])


def test_device_picks_accelerator_on_mac():
    d = device()
    if torch.backends.mps.is_available():
        assert d.type == 'mps'


def test_ae_separates_anomalies():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, (800, 20, 4)).astype(np.float32)
    ae = train_torch(WindowAE(20), healthy, epochs=15, seed=1)
    s_h = ae_scores(ae, healthy[:100])
    anomalous = healthy[:100] + 3.0        # constant offset the AE never saw
    s_a = ae_scores(ae, anomalous)
    assert np.median(s_a) > 3 * np.median(s_h)


def test_rulnet_learns_countdown():
    rng = np.random.default_rng(5)
    # RUL encoded linearly in channel 0's level
    n, seq = 600, 16
    rul = rng.uniform(0, 10, n).astype(np.float32)
    X = np.zeros((n, seq, 4), dtype=np.float32)
    X[:, :, 0] = rul[:, None] / 10.0
    X += rng.normal(0, 0.05, X.shape).astype(np.float32)
    net = train_torch(RULNet(hidden=32, layers=1), X, rul, epochs=60, seed=2)
    pred = predict_torch(net, X)
    rmse = float(np.sqrt(((pred - rul) ** 2).mean()))
    assert rmse < 1.0
