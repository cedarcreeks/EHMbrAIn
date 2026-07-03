"""Corrected-parameter machinery: ISA, theta/delta and exponent fitting."""

import numpy as np

from ehmbrain.common.corrections import (correct, fit_correction_exponents,
                                         isa_static, theta_delta)


def test_isa_sea_level():
    T, P = isa_static(0.0)
    assert abs(T - 288.15) < 1e-9
    assert abs(P - 101.325) < 1e-9


def test_isa_tropopause_and_cruise():
    T35, P35 = isa_static(35000.0)
    assert abs(T35 - 218.81) < 0.1          # 288.15 - 0.0019812*35000
    assert abs(P35 - 23.84) < 0.15          # standard tables ~23.84 kPa
    T40, _ = isa_static(40000.0)
    assert abs(T40 - 216.65) < 1e-9         # stratosphere is isothermal


def test_theta_delta_cruise():
    th, de = theta_delta(35000.0, 0.78)
    # Ram rise: Tt = 218.81*(1+0.2*0.78^2) ~ 245.4 K -> theta ~ 0.8516
    assert abs(th - 0.8516) < 0.002
    assert 0.33 < de < 0.36


def test_exponent_fit_recovers_truth():
    rng = np.random.default_rng(7)
    n = 400
    theta = rng.uniform(0.85, 1.1, n)
    delta = rng.uniform(0.3, 1.0, n)
    n1 = rng.uniform(4000, 5100, n)
    n1c = n1 / np.sqrt(theta)
    a_true, b_true = 0.62, 1.0
    y = 1e-3 * (n1c / 4500) ** 2.5 * theta ** a_true * delta ** b_true
    a, b, resid = fit_correction_exponents(theta, delta, n1, y)
    assert abs(a - a_true) < 0.01
    assert abs(b - b_true) < 0.01
    assert resid < 0.1
    y_corr = correct(y, theta, delta, a, b)
    # After correction the ambient dependence is gone: y_corr is a pure
    # function of N1c (spread at fixed N1c ~ 0).
    order = np.argsort(n1c)
    spread = np.std(np.diff(np.log(y_corr[order]))[np.diff(n1c[order]) < 5])
    assert spread < 0.01
