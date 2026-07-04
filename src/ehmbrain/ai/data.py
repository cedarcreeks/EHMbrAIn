"""Shared feature builder for the AI suite (phase F4).

Fairness by construction: the AI consumes exactly the same corrected-space
deviations against the same published baseline as the traditional pipeline
(ehmbrain.trad.pipeline.BaselineModel) — cockpit set for the base case.
Per-cycle feature vector (4 channels):
    [dN2 %, dWF %, dEGT % (cruise), dEGT_takeoff K]
NaNs (lost snapshots) are forward-filled, as a line-replaceable monitoring
unit would hold the last valid report.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.trad.pipeline import COCKPIT, BaselineModel

REPO_ROOT = Path(__file__).resolve().parents[3]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'

N_CH = 4


def _ffill(a):
    """Forward-fill; leading NaNs (first snapshot lost) are back-filled with
    the first finite value so no NaN survives into the feature matrix."""
    a = np.asarray(a, float)
    out = a.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    finite = np.isfinite(out)
    if finite.any() and not finite[0]:
        out[:np.argmax(finite)] = out[finite][0]
    return out


def takeoff_egt_baseline(dts_c):
    _, _, b0 = load_icm('takeoff')
    _, _, b30 = load_icm('takeoff_hot')
    w = np.asarray(dts_c, float) / 30.0
    return b0['EGT_degK'] * (1 - w) + b30['EGT_degK'] * w


def engine_features(e, bm):
    """(n, 4) feature matrix for one engine's snapshot frame (sorted by cycle)."""
    measured = e[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
    dz = bm.deviations(measured, e.cr_N1_cmd.to_numpy())
    to_dev = e.to_EGT_degK.to_numpy(float) - takeoff_egt_baseline(e.to_dTs_C.to_numpy())
    F = np.column_stack([dz, to_dev])
    return np.column_stack([_ffill(F[:, j]) for j in range(F.shape[1])])


def load_fleet_features():
    """Per-engine dict: features, split, acute info, life. Cached in-process."""
    index = json.loads((FLEET / 'fleet_index.json').read_text())
    events = pd.read_parquet(FLEET / 'events.parquet')
    bm = BaselineModel()
    out = {}
    snap_cols = (['engine_id', 'cycle', 'split', 'cr_N1_cmd', 'to_dTs_C',
                  'to_EGT_degK'] + [f'cr_{c}' for c in COCKPIT])
    df = pd.read_parquet(FLEET / 'snapshots.parquet', columns=snap_cols)
    for rec in index['engines']:
        eid = rec['engine_id']
        e = df[df.engine_id == eid].sort_values('cycle').reset_index(drop=True)
        ev = events[events.engine_id == eid]
        ac = ev[ev.type == 'acute'].sort_values('cycle')
        episodes = [(float(r.cycle), str(r.param)) for r in ac.itertuples()]
        out[eid] = {
            'F': engine_features(e, bm),
            'split': e.split.iloc[0],
            'life': len(e),
            'episodes': episodes,
            'acute_param': episodes[0][1] if episodes else None,
            'acute_onset': episodes[0][0] if episodes else None,
        }
    return out


def normalization(fleet, splits=('train',)):
    """Global per-channel mean/std from the chosen splits (train only)."""
    Fs = np.concatenate([v['F'] for v in fleet.values() if v['split'] in splits])
    return Fs.mean(axis=0), Fs.std(axis=0) + 1e-9


def windows_from(F, w, stride, lo=0, hi=None):
    """End indices + stacked windows (k, w, ch) between lo and hi."""
    hi = hi if hi is not None else len(F)
    ends = np.arange(lo + w, hi, stride)
    if not len(ends):
        return ends, np.empty((0, w, F.shape[1]))
    return ends, np.stack([F[e - w:e] for e in ends])
