"""WP1.3 closure: corrected-parameter exponents + corrected-space baseline.

Fits the correction exponents (a, b) per channel on the healthy deck (the
exponents come from our own model, not from generic tables), then rebuilds the
baseline interpolator in corrected space --- Y_corr versus (N1c, MN) --- and
re-runs the leave-one-out audit there. This is the promised fix for the
raw-space LOO tails: theta/delta collapse the altitude and day-temperature
dependence.

Output: data/processed/decks/corrected_baseline.json
Usage: uv run python scripts/make_corrected_baseline.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.common.corrections import correct, fit_correction_exponents, theta_delta

REPO_ROOT = Path(__file__).resolve().parents[1]
DECK = REPO_ROOT / 'data' / 'processed' / 'decks' / 'baseline_deck.parquet'
OUT = REPO_ROOT / 'data' / 'processed' / 'decks' / 'corrected_baseline.json'

CHANNELS = {          # channel -> textbook expectation for (a, b), for the report
    'WF_kgps': 'a~0.5-0.7, b~1 (fuel flow)',
    'N2_rpm': 'a~0.5, b~0 (corrected speed)',
    'EGT_degK': 'a~0.9-1.0, b~0 (temperature ratio)',
    'Fn_lbf': 'a~0, b~1 (thrust follows delta)',
}


def _loo_channel(args):
    from scipy.interpolate import LinearNDInterpolator
    Xn, y = args
    errs = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        yi = LinearNDInterpolator(Xn[mask], y[mask])(Xn[i])
        if yi is not None and np.isfinite(yi):
            errs.append(abs(float(yi) - y[i]) / abs(y[i]) * 100)
    return {'n_interior': len(errs), 'median_pct': float(np.median(errs)),
            'p95_pct': float(np.percentile(errs, 95)), 'max_pct': float(np.max(errs))}


def main():
    df = pd.read_parquet(DECK)
    theta, delta = theta_delta(df['alt_ft'], df['MN'], df['dTs_C'])
    n1c = df['N1_rpm'] / np.sqrt(theta)

    result = {'exponents': {}, 'loo_corrected': {}}
    jobs = []
    X = np.column_stack([n1c, df['MN']])
    Xn = X / (X.max(axis=0) - X.min(axis=0))
    for ch in CHANNELS:
        a, b, resid = fit_correction_exponents(theta, delta, df['N1_rpm'], df[ch])
        result['exponents'][ch] = {'a_theta': round(a, 4), 'b_delta': round(b, 4),
                                   'fit_residual_pct': round(resid, 3),
                                   'expected': CHANNELS[ch]}
        y_corr = correct(df[ch], theta, delta, a, b)
        jobs.append((Xn, np.asarray(y_corr)))

    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:   # norm N1
        for ch, loo in zip(CHANNELS, pool.map(_loo_channel, jobs)):
            result['loo_corrected'][ch] = loo

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
