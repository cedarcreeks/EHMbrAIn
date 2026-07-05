"""F10: the identifiability certificate.

Given an engine's actual operating-condition history and a sensor set, the
accumulated Fisher information through the (calibrated) twin bounds what
gas-path diagnosis can and cannot know:

    F = P0^-1 + sum_t H(u_t)^T R^-1 H(u_t)          (information)
    Sigma = F^-1                                     (Cramer-Rao posterior cov)

Per-direction posterior std sqrt(diag(Sigma)) is the certificate: small =
identifiable, large = confounded/unobservable. The off-diagonals expose the
smearing structure (which parameters are jointly confounded). Validated against
ground truth in scripts/f10_certificate.py.

Linear H suffices: the confusable-pair curvature test (F10 feasibility) showed
the nonlinear correction is far below the noise floor, so the first-order
Fisher information is the right instrument here.
"""

import numpy as np

from ..datagen.fleet import load_icm
from ..perf.icm import HEALTH_PARAMS

# Cockpit and extended measurement sets with their noise sigmas [% of reading].
COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
EXTENDED = COCKPIT + ['P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK']
SIGMA_PCT = {'N2_rpm': 0.07, 'WF_kgps': 0.5, 'EGT_degK': 0.23,
             'P25_bar': 0.3, 'T25_degK': 0.2, 'PS3_bar': 0.3, 'T3_degK': 0.2}
PRIOR_STD_PCT = 2.0        # diffuse prior on each health parameter


def _H_rows(sensors, point_a='cruise', point_b='cruise_lowpwr'):
    """(Ha, Hb) blocks for the chosen sensors at the two cruise grid points."""
    Ha, ch, _ = load_icm(point_a)
    Hb, _, _ = load_icm(point_b)
    rows = [ch.index(s) for s in sensors]
    return Ha[rows], Hb[rows]


class Certificate:
    """Accumulated-Fisher identifiability certificate for one engine."""

    def __init__(self, sensors=COCKPIT):
        self.sensors = sensors
        self.Ha, self.Hb = _H_rows(sensors)
        self.Rinv = np.diag([1.0 / SIGMA_PCT[s] ** 2 for s in sensors])
        self.P0inv = np.eye(len(HEALTH_PARAMS)) / PRIOR_STD_PCT ** 2

    def H_at(self, n1_cmd):
        w = (np.asarray(n1_cmd, float) - 4666.0) / (4400.0 - 4666.0)
        return self.Ha * (1 - w) + self.Hb * w

    def fisher(self, n1_history, stride=20):
        """Accumulated Fisher information over a commanded-N1 history."""
        F = self.P0inv.copy()
        for n1 in np.asarray(n1_history)[::stride]:
            H = self.H_at(n1)
            F += H.T @ self.Rinv @ H
        return F

    def certify(self, n1_history, stride=20):
        """Returns per-direction std [%], the posterior covariance, and tags."""
        F = self.fisher(n1_history, stride)
        Sigma = np.linalg.inv(F)
        std = np.sqrt(np.diag(Sigma))
        tags = {}
        for j, p in enumerate(HEALTH_PARAMS):
            tags[p] = ('identifiable' if std[j] < 0.7
                       else 'confounded' if std[j] < 1.5 else 'unobservable')
        return {'std_pct': dict(zip(HEALTH_PARAMS, std.tolist())),
                'cov': Sigma, 'tags': tags}

    def in_region(self, x_true, x_est, Sigma, level=0.90):
        """Is x_true inside the level-coverage ellipsoid around x_est?
        (x-mu)^T Sigma^-1 (x-mu) <= chi2_{k}(level)."""
        from scipy.stats import chi2
        d = np.asarray(x_true) - np.asarray(x_est)
        maha2 = float(d @ np.linalg.solve(Sigma, d))
        return maha2 <= chi2.ppf(level, len(HEALTH_PARAMS)), maha2
