"""WP1.3: sweep the healthy-engine model over the operating envelope and write
the baseline decks.

Output: data/processed/decks/baseline_deck.parquet with one row per converged
(alt, MN, dTs, PC) point carrying every sensor channel at the part-power point
plus the local maximum thrust, and a leave-one-out interpolation error report.

Norm N1: the (alt, MN) blocks are independent and run in parallel processes,
each with its own Problem; the warm-start continuation (dTs, then descending
PC) stays inside a block. Norm N2: recovery from a failed solve is always a
cold rebuild. Runtime: a few minutes on all cores.
Usage: uv run python scripts/make_decks.py
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.perf.cycle import MPCFM56, build_deck_problem, snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'data' / 'processed' / 'decks'

# (alt_ft, [Mach]) — realistic flight-envelope blocks (no M0.8 at sea level).
ENVELOPE = [
    (0.0, [0.001, 0.2, 0.4]),
    (10000.0, [0.2, 0.4, 0.6]),
    (20000.0, [0.4, 0.6, 0.7]),
    (30000.0, [0.6, 0.7, 0.78]),
    (35000.0, [0.65, 0.72, 0.78, 0.82]),
    (39000.0, [0.72, 0.78, 0.82]),
]
DTS_LIST = [0.0, 15.0, 30.0]        # delta-ISA, degC steps expressed in degR below
PC_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def converged(prob):
    fn = prob.get_val('OD_pt.perf.Fn', units='lbf')[0]
    fn_max = prob.get_val('OD_max.perf.Fn', units='lbf')[0]
    pc = prob.get_val('OD_pt.PC')[0]
    return abs(fn - pc * fn_max) < 0.01 * max(abs(pc * fn_max), 1.0)


def fresh_problem(alt):
    """Cold problem with altitude-appropriate guesses. A failed Newton attempt
    corrupts the point's internal states beyond repair (seen with the idle
    anchor), so recovery is always a rebuild, never a re-seed."""
    prob = build_deck_problem()
    prob.set_solver_print(level=-1)
    if alt > 15000.0:
        for pt in ('OD_max', 'OD_pt'):
            prob[f'{pt}.balance.FAR'] = 0.025
            prob[f'{pt}.balance.W'] = 320.0
            prob[f'{pt}.balance.BPR'] = 5.3
            prob[f'{pt}.balance.lp_Nmech'] = 4700.0
            prob[f'{pt}.balance.hp_Nmech'] = 14000.0
    return prob


def _loo_channel(args):
    """Worker: leave-one-out error stats for one output channel."""
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


def sweep_block(args):
    """Worker: full (alt, MN) block — all dTs and PC settings, warm within."""
    alt, mn = args
    prob = fresh_problem(alt)
    rows, failures = [], []
    for dts_c in DTS_LIST:
        dts_r = dts_c * 1.8

        def set_block(p):
            for pt in ('OD_max', 'OD_pt'):
                p.set_val(f'{pt}.fc.MN', mn)
                p.set_val(f'{pt}.fc.alt', alt, units='ft')
                p.set_val(f'{pt}.fc.dTs', dts_r, units='degR')

        def solve_pc(p, pc):
            p.set_val('OD_pt.PC', pc)
            try:
                p.run_model()
                return converged(p)
            except Exception:
                return False

        set_block(prob)
        if not solve_pc(prob, PC_LIST[0]):
            prob = fresh_problem(alt)
            set_block(prob)
            if not solve_pc(prob, PC_LIST[0]):
                failures.append({'alt': alt, 'MN': mn, 'dTs_C': dts_c, 'PC': 'block'})
                prob = fresh_problem(alt)
                continue

        for pc in PC_LIST:
            if not solve_pc(prob, pc):
                failures.append({'alt': alt, 'MN': mn, 'dTs_C': dts_c, 'PC': pc})
                prob = fresh_problem(alt)
                break
            row = {'alt_ft': alt, 'MN': mn, 'dTs_C': dts_c, 'PC': pc,
                   'Fn_max_lbf': float(prob.get_val('OD_max.perf.Fn', units='lbf')[0])}
            row.update(snapshot(prob, 'OD_pt'))
            rows.append(row)
    return alt, mn, rows, failures


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [(alt, mn) for alt, mns in ENVELOPE for mn in mns]
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    rows, failures = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for alt, mn, brows, bfails in pool.map(sweep_block, blocks):
            rows.extend(brows)
            failures.extend(bfails)
            print(f'alt {alt:7.0f}  MN {mn:5.3f}  rows {len(rows)}  '
                  f'fails {len(failures)}  t {time.time()-t0:6.0f}s', flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / 'baseline_deck.parquet', index=False)

    # Leave-one-out linear-interpolation error over the 4-D scattered grid,
    # per output channel, evaluated only on interior points. One worker per
    # channel (norm N1).
    X = df[['alt_ft', 'MN', 'dTs_C', 'PC']].to_numpy(float)
    Xn = X / (X.max(axis=0) - X.min(axis=0))
    channels = ['WF_kgps', 'N1_rpm', 'N2_rpm', 'EGT_degK', 'Fn_lbf']
    jobs = [(Xn, df[ch].to_numpy(float)) for ch in channels]
    with ProcessPoolExecutor(max_workers=len(channels)) as pool:
        loo = dict(zip(channels, pool.map(_loo_channel, jobs)))

    report = {'n_rows': len(df), 'n_failures': len(failures), 'failures': failures,
              'loo_error': loo, 'runtime_s': time.time() - t0}
    (OUT_DIR / 'deck_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['loo_error'], indent=2))
    print(f'{len(df)} rows, {len(failures)} failures -> {OUT_DIR}')


if __name__ == '__main__':
    main()
