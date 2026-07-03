"""Traditional EHM pipeline (phase F3): the strong classical baseline.

Modules, mirroring industrial practice (SAGE/ADEM-style):
  deviations   corrected measured channels vs. the published healthy baseline
  smoothing    Holt double exponential (level + trend) per engine/channel
  detection    threshold with k-of-n persistence (gradual) + CUSUM (steps)
  WLS GPA      snapshot health estimate, regularized (report eq. 2.5)
  Kalman GPA   random-walk health tracking (report eq. 2.6), wash-aware
  isolation    nearest-signature rule on the detected step (expert rule)
  RUL          Theil-Sen extrapolation of the tracked EGT margin to zero

Cockpit-only measurement set (N2, WF, EGT deviations; N1 deviation is zero by
construction). All deviations in percent of baseline, matching the ICM units.
"""

import numpy as np

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS

COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']


# ----------------------------------------------------------------------------
# Deviations and smoothing
# ----------------------------------------------------------------------------

class BaselineModel:
    """Healthy-engine baseline + ICM at the two snapshot families, linearly
    interpolated exactly like the published F1 artifacts allow (this is the
    model an OEM would ship; both EHM pipelines may use it)."""

    def __init__(self):
        self.Hc_a, self.ch, self.bc_a = load_icm('cruise')
        self.Hc_b, _, self.bc_b = load_icm('cruise_lowpwr')
        self.rows = [self.ch.index(c) for c in COCKPIT]

    def cruise(self, n1_cmd):
        """(baseline, H) for cruise snapshots at commanded N1 (vectorized)."""
        w = (np.asarray(n1_cmd, float) - 4666.0) / (4400.0 - 4666.0)
        base = np.array([[self.bc_a[c] for c in COCKPIT]])
        base_b = np.array([[self.bc_b[c] for c in COCKPIT]])
        b = base * (1 - w[:, None]) + base_b * w[:, None]
        Ha, Hb = self.Hc_a[self.rows], self.Hc_b[self.rows]
        return b, (Ha, Hb, w)

    def deviations(self, measured, n1_cmd):
        """% deviations of measured cockpit channels vs. baseline."""
        b, _ = self.cruise(n1_cmd)
        return (measured - b) / b * 100.0


def holt_smooth(y, alpha=0.15, beta=0.05):
    """Holt double exponential smoothing; NaNs carried through (no update).

    Returns (level, trend, innovations): the innovation is the one-step
    forecast error v - (l + t) BEFORE the update — the right series to run
    step detectors on, because the slow chronic trend lives in (l, t) and is
    thereby removed.
    """
    y = np.asarray(y, float)
    level = np.full_like(y, np.nan)
    trend = np.zeros_like(y)
    innov = np.full_like(y, np.nan)
    l, t = None, 0.0
    for i, v in enumerate(y):
        if np.isfinite(v):
            if l is None:
                l = v
            else:
                innov[i] = v - (l + t)
                l_new = alpha * v + (1 - alpha) * (l + t)
                t = beta * (l_new - l) + (1 - beta) * t
                l = l_new
        if l is not None:
            level[i] = l
            trend[i] = t
    return level, trend, innov


def ewma(y, alpha):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    m = None
    for i, v in enumerate(y):
        if np.isfinite(v):
            m = v if m is None else alpha * v + (1 - alpha) * m
        if m is not None:
            out[i] = m
    return out


def gap_alert(dev, alpha_fast=0.1, alpha_slow=0.005, warmup=2500,
              nsig=5.0, k=10, n=14, positive_only=True):
    """Dual-EWMA gap detector: fast minus slow EWMA of a deviation series.

    An acute ramp opens a gap of roughly the fault magnitude within the ramp
    duration; the chronic trend contributes only a small constant offset and
    washes pull the gap negative (hence one-sided for EGT). The gap noise
    scale is estimated on the warmup segment. Returns first alarm index."""
    fast, slow = ewma(dev, alpha_fast), ewma(dev, alpha_slow)
    gap = fast - slow
    base = gap[200:warmup]
    med = np.nanmedian(base)
    mad = np.nanmedian(np.abs(base - med)) or 1e-9
    z = (gap - med) / (1.4826 * mad)
    exceed = z > nsig if positive_only else np.abs(z) > nsig
    exceed[:warmup] = False
    counts = np.convolve(exceed.astype(int), np.ones(n, int), 'full')[:len(z)]
    hits = np.nonzero(counts >= k)[0]
    return int(hits[0]) if len(hits) else None


def trend_alert(trend, warmup=3000, nsig=6.0, k=15, n=20, positive_only=True):
    """Alarm when the Holt TREND departs from its chronic statistics: catches
    acute ramps (a sustained slope change) that innovation-CUSUM misses.
    Chronic trend stats (median, MAD) come from the warmup segment; washes
    pull the EGT trend negative, so the EGT detector is one-sided positive.
    Returns first alarm index after warmup, or None."""
    t = np.asarray(trend, float)
    base = t[100:warmup]
    med = np.nanmedian(base)
    mad = np.nanmedian(np.abs(base - med)) or 1e-9
    dev = (t - med) / (1.4826 * mad)
    exceed = dev > nsig if positive_only else np.abs(dev) > nsig
    exceed[:warmup] = False
    counts = np.convolve(exceed.astype(int), np.ones(n, int), 'full')[:len(t)]
    hits = np.nonzero(counts >= k)[0]
    return int(hits[0]) if len(hits) else None


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------

def persistence_alert(smoothed, threshold, k=5, n=7):
    """First index where >=k of the last n smoothed values exceed threshold."""
    exceed = np.abs(smoothed) > threshold
    counts = np.convolve(exceed.astype(int), np.ones(n, int), 'full')[:len(exceed)]
    hits = np.nonzero(counts >= k)[0]
    return int(hits[0]) if len(hits) else None


def cusum(dev, drift_k=0.5, h=5.0):
    """Two-sided CUSUM on a deviation series (units of its own std).
    Returns first alarm index or None."""
    x = np.asarray(dev, float)
    x = np.where(np.isfinite(x), x, 0.0)
    s = np.nanstd(x[:200]) or 1.0
    z = x / s
    gp = gm = 0.0
    for i, v in enumerate(z):
        gp = max(0.0, gp + v - drift_k)
        gm = max(0.0, gm - v - drift_k)
        if gp > h or gm > h:
            return i
    return None


# ----------------------------------------------------------------------------
# GPA estimators
# ----------------------------------------------------------------------------

def wls_gpa(dz, H, R_diag, prior_sigma=2.0, lam=1.0):
    """Regularized weighted-least-squares snapshot estimate (report eq. 2.5)."""
    R_inv = np.diag(1.0 / np.asarray(R_diag) ** 2)
    P0_inv = np.eye(len(HEALTH_PARAMS)) / prior_sigma ** 2
    A = H.T @ R_inv @ H + lam * P0_inv
    return np.linalg.solve(A, H.T @ R_inv @ dz)


def kalman_gpa(dz_series, H_series, R_diag, q=1e-4, wash_cycles=(),
               wash_reset_frac=0.5):
    """Random-walk Kalman tracking of the 10 health parameters.

    dz_series: (n, 3) cockpit deviations [%] (NaN rows are skipped);
    H_series:  callable i -> (3, 10) ICM at snapshot i;
    q:         process-noise variance per cycle [%^2];
    washes:    at logged wash cycles the recoverable (compressor-flow/eta)
               states move toward zero by `wash_reset_frac`, with covariance
               reopened - the maintenance log is available to the EHM system.
    Returns (n, 10) posterior means.
    """
    n = len(dz_series)
    nx = len(HEALTH_PARAMS)
    R = np.diag(np.asarray(R_diag, float) ** 2)
    x = np.zeros(nx)
    P = np.eye(nx) * 1.0
    Q = np.eye(nx) * q
    washes = set(int(np.ceil(c)) for c in wash_cycles)
    recoverable = [i for i, p in enumerate(HEALTH_PARAMS)
                   if p.split('.')[0] in ('fan', 'lpc', 'hpc')]
    out = np.zeros((n, nx))
    for i in range(n):
        P = P + Q
        if i in washes:
            x[recoverable] *= (1.0 - wash_reset_frac)
            P[np.ix_(recoverable, recoverable)] += np.eye(len(recoverable)) * 0.25
        z = dz_series[i]
        if np.all(np.isfinite(z)):
            H = H_series(i)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.solve(S, np.eye(len(z)))
            x = x + K @ (z - H @ x)
            P = (np.eye(nx) - K @ H) @ P
        out[i] = x
    return out


# ----------------------------------------------------------------------------
# Isolation and RUL
# ----------------------------------------------------------------------------

def isolate_step(step_dz, H, min_norm=0.15):
    """Expert rule: match a detected deviation step against the single-
    parameter fault signatures (columns of H); cosine similarity, sign-aware.
    Returns HEALTH_PARAMS name or 'none'."""
    v = np.asarray(step_dz, float)
    if not np.all(np.isfinite(v)) or np.linalg.norm(v) < min_norm:
        return 'none'
    sims = []
    for j in range(H.shape[1]):
        col = H[:, j]
        for sign in (+1.0, -1.0):     # faults can move a parameter either way
            sims.append((float(v @ (sign * col))
                         / (np.linalg.norm(v) * np.linalg.norm(col)), j))
    best, j = max(sims)
    return HEALTH_PARAMS[j] if best > 0.6 else 'none'


def theil_sen_rul(egtm_est, window=1500, stride=25):
    """Robust slope of the recent EGT-margin estimate -> cycles to zero.
    Returns predicted RUL (cycles) or None if the trend is non-decreasing."""
    y = np.asarray(egtm_est, float)
    n = len(y)
    a = max(0, n - window)
    idx = np.arange(a, n, max(1, stride))
    yy = y[idx]
    ok = np.isfinite(yy)
    idx, yy = idx[ok], yy[ok]
    if len(yy) < 10:
        return None
    slopes = [(yy[j] - yy[i]) / (idx[j] - idx[i])
              for i in range(len(yy)) for j in range(i + 1, len(yy))]
    slope = float(np.median(slopes))
    if slope >= -1e-6:
        return None
    return float(-y[~np.isnan(y)][-1] / slope) if np.isfinite(y[~np.isnan(y)][-1]) else None