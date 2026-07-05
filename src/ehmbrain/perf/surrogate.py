"""Differentiable neural surrogate of the twin (F8/L1), and its inference-time
emitter used by the nonlinear fleet generator (F8/L2).

Architecture: linearization + learned residual. The linear part reuses the F1
ICM (baseline * (1 + H x /100)); a 3-layer GELU MLP corrects the residual the
linearization misses. Trained per snapshot family (cruise: input u = commanded
N1; takeoff: u = dTs in C at rated N1). See scripts/f8_surrogate.py.

The emitter runs CPU-only inference (small MLP), safe inside the datagen
multiprocessing workers where MPS is forbidden. Checkpoints are cached per
process.
"""

from pathlib import Path

import numpy as np

from .icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[3]
F8 = REPO_ROOT / 'data' / 'processed' / 'f8'

# The 8 channels the surrogate predicts (note: includes Fn, excludes N1 which
# is the held power-setting input; the snapshot generator supplies N1 itself).
SURR_CHANNELS = ['N2_rpm', 'WF_kgps', 'EGT_degK', 'P25_bar', 'T25_degK',
                 'PS3_bar', 'T3_degK', 'Fn_lbf']


def _make_mlp(width=256, d_in=11, d_out=8):
    """Rebuilds the trained Surrogate architecture (self.net Sequential), so
    the checkpoint's 'net.*' keys load directly."""
    import torch.nn as nn

    class Surrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, width), nn.GELU(),
                nn.Linear(width, width), nn.GELU(),
                nn.Linear(width, width), nn.GELU(),
                nn.Linear(width, d_out))

        def forward(self, x):
            return self.net(x)

    return Surrogate()


def linear_prediction(X, u, family='cruise'):
    """F1 linearization baseline*(1 + H x /100) over the surrogate channels.

    X: (n, 10) health deviations [%]; u: (n,) operating condition (N1 for
    cruise, dTs C for takeoff). Fn carries baseline only (no ICM row).
    """
    from ..datagen.fleet import load_icm
    if family == 'cruise':
        Ha, cha, ba = load_icm('cruise')
        Hb, _, bb = load_icm('cruise_lowpwr')
        w = (u - 4666.0) / (4400.0 - 4666.0)
    else:
        Ha, cha, ba = load_icm('takeoff')
        Hb, _, bb = load_icm('takeoff_hot')
        w = u / 30.0
    out = np.zeros((len(X), len(SURR_CHANNELS)), np.float32)
    for j, chn in enumerate(SURR_CHANNELS):
        base = ba.get(chn, 0.0) * (1 - w) + bb.get(chn, 0.0) * w
        if chn in cha:
            r = cha.index(chn)
            dev = (X @ Ha[r]) * (1 - w) + (X @ Hb[r]) * w
            out[:, j] = base * (1 + dev / 100.0)
        else:
            out[:, j] = base
    return out


class SurrogateEmitter:
    """Loads both family checkpoints once and predicts channel values."""

    _cache = {}

    def __init__(self):
        import torch
        self.torch = torch
        self.nets, self.meta = {}, {}
        for family, suffix in (('cruise', ''), ('takeoff', '_takeoff')):
            ckpt = F8 / f'surrogate{suffix}.pt'
            d = torch.load(ckpt, map_location='cpu', weights_only=False)
            net = _make_mlp()
            net.load_state_dict(d['state_dict'])
            net.eval()
            self.nets[family] = net
            self.meta[family] = d

    @classmethod
    def cached(cls):
        if 'inst' not in cls._cache:
            cls._cache['inst'] = cls()
        return cls._cache['inst']

    def predict(self, x_pct, u, family):
        """(n, 8) surrogate channel values for health x_pct and condition u."""
        import numpy as np
        torch = self.torch
        d = self.meta[family]
        u_mu, u_sd = d['u_norm']
        y_mu, y_sd = d['y_mu'], d['y_sd']
        bypass = d['bypass']
        un = ((np.asarray(u, np.float32) - u_mu) / u_sd)[:, None]
        Xin = np.concatenate([np.asarray(x_pct, np.float32), un], axis=1)
        with torch.no_grad():
            corr = self.nets[family](torch.from_numpy(Xin.astype(np.float32))).numpy()
        corr = corr * y_sd + y_mu
        corr[:, bypass] = 0.0
        return linear_prediction(np.asarray(x_pct, np.float32),
                                 np.asarray(u, np.float32), family) + corr

    def channel(self, x_pct, u, family, channel_name):
        """Single named channel (n,)."""
        return self.predict(x_pct, u, family)[:, SURR_CHANNELS.index(channel_name)]
