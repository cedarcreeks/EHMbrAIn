"""Corrected-parameter machinery (SAE-style theta/delta normalization).

theta = Tt2 / 288.15 K, delta = Pt2 / 101.325 kPa, computed from the flight
condition (ISA atmosphere + adiabatic ram rise). Corrected parameters follow
    Y_corr = Y / (theta**a * delta**b),
with exponents (a, b) FITTED on the healthy-engine deck (docs/f1-model-spec.md:
exponents are derived from our own model, not copied from generic tables).
"""

import numpy as np

T0_K = 288.15
P0_KPA = 101.325
LAPSE_K_PER_FT = 0.0019812      # 6.5 K/km
TROPOPAUSE_FT = 36089.0
T_STRAT_K = 216.65


def isa_static(alt_ft, dTs_C=0.0):
    """ISA static temperature [K] and pressure [kPa] at altitude (+ offset)."""
    alt_ft = np.asarray(alt_ft, dtype=float)
    T_isa = np.where(alt_ft < TROPOPAUSE_FT, T0_K - LAPSE_K_PER_FT * alt_ft, T_STRAT_K)
    P = np.where(alt_ft < TROPOPAUSE_FT,
                 P0_KPA * (T_isa / T0_K) ** 5.25588,
                 P0_KPA * 0.22336 * np.exp(-(alt_ft - TROPOPAUSE_FT) / 20805.8))
    return T_isa + dTs_C, P


def theta_delta(alt_ft, mn, dTs_C=0.0, ram_recovery=0.999):
    """theta and delta at station 2 (fan face) for a flight condition."""
    Ts, Ps = isa_static(alt_ft, dTs_C)
    mn = np.asarray(mn, dtype=float)
    ram_t = 1.0 + 0.2 * mn ** 2
    Tt2 = Ts * ram_t
    Pt2 = Ps * ram_t ** 3.5 * ram_recovery
    return Tt2 / T0_K, Pt2 / P0_KPA


def fit_correction_exponents(theta, delta, n1_rpm, y, poly_deg=3):
    """Fit (a, b) in log y = s(log N1c) + a log theta + b log delta.

    s() is a polynomial in the log corrected speed; the exponents absorb the
    residual ambient dependence. Returns (a, b, residual_std_pct).
    """
    theta, delta, y = map(np.asarray, (theta, delta, y))
    n1c = np.asarray(n1_rpm) / np.sqrt(theta)
    x = np.log(n1c)
    A = np.column_stack([x ** k for k in range(poly_deg + 1)]
                        + [np.log(theta), np.log(delta)])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    a, b = coef[-2], coef[-1]
    resid = np.log(y) - A @ coef
    return float(a), float(b), float(np.std(resid) * 100.0)


def correct(y, theta, delta, a, b):
    """Apply the fitted correction: Y_corr = Y / (theta^a * delta^b)."""
    return np.asarray(y) / (np.asarray(theta) ** a * np.asarray(delta) ** b)
