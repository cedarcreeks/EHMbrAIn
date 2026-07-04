"""F8/L1: training data for the differentiable neural surrogate of the twin.

Samples (health state x, commanded N1) at the cruise snapshot family and
solves the FULL pyCycle Study model for each — the ground truth the surrogate
must reproduce. Parallel workers (norm N1), cold rebuild on failure (norm N2).

x sampling: per-parameter U(-3 %, +1 %) for eta / U(-3 %, +3 %) for flow,
covering the fleet's realistic envelope with margin; 15 % healthy-ish draws
(x ~ N(0, 0.3)) so the surrogate is accurate near zero too.

Two snapshot families (matching the fleet's report schedule):
  cruise   x + N1 in [4400, 4666] rpm (dTs = 0)
  takeoff  x + dTs in [-20, 35] C at rated N1 4813 rpm (mn 0.001, SL)

Output: data/processed/f8/surrogate_data[_takeoff].parquet
Usage: uv run python scripts/f8_surrogate_data.py [n_samples] [family]
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.perf.cycle import build_study_problem, set_health, snapshot
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'

CHANNELS = ['N2_rpm', 'WF_kgps', 'EGT_degK', 'P25_bar', 'T25_degK',
            'PS3_bar', 'T3_degK', 'Fn_lbf']


def sample_x(rng):
    if rng.uniform() < 0.15:
        x = rng.normal(0.0, 0.3, 10)
    else:
        x = np.empty(10)
        for i, p in enumerate(HEALTH_PARAMS):
            x[i] = (rng.uniform(-3.0, 1.0) if p.endswith('.eta')
                    else rng.uniform(-3.0, 3.0))
    return np.clip(x, -3.5, 3.5)


TO_KW = dict(mn=0.001, alt_ft=0.0, n1_rpm=4813.0)


def worker(args):
    seed, n, family = args
    rng = np.random.default_rng(seed)

    def build(dts_c=0.0):
        if family == 'cruise':
            return build_study_problem()
        from ehmbrain.perf.cycle import OD_GUESSES
        to_guess = OD_GUESSES['A2_takeoff'] | {'hpt_PR': 3.7, 'lpt_PR': 4.3}
        return build_study_problem(dTs=dts_c * 1.8, guesses=to_guess, **TO_KW)

    prob = build()
    prob.set_solver_print(level=-1)
    prob.run_model()
    rows = []
    for _ in range(n):
        x = sample_x(rng)
        try:
            if family == 'cruise':
                u = float(rng.uniform(4400.0, 4666.0))
                prob.set_val('OD.N1_target', u, units='rpm')
            else:
                u = float(rng.uniform(-20.0, 35.0))
                prob.set_val('OD.fc.dTs', u * 1.8, units='degR')
            set_health(prob, {p: v / 100.0 for p, v in zip(HEALTH_PARAMS, x)})
            prob.run_model()
            s = snapshot(prob, 'OD')
            n1_expect = u if family == 'cruise' else 4813.0
            if abs(s['N1_rpm'] - n1_expect) > 5.0:   # unconverged
                raise RuntimeError('n1 miss')
            rows.append(list(x) + [u] + [s[c] for c in CHANNELS])
        except Exception:
            prob = build()                           # norm N2: cold rebuild
            prob.set_solver_print(level=-1)
            prob.run_model()
    return rows


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    family = sys.argv[2] if len(sys.argv) > 2 else 'cruise'
    OUT.mkdir(parents=True, exist_ok=True)
    n_workers = 8
    per = n_total // n_workers
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunks = list(pool.map(worker,
                               [(100 + i, per, family) for i in range(n_workers)]))
    rows = [r for ch in chunks for r in ch]
    ucol = 'N1_cmd' if family == 'cruise' else 'dTs_C'
    cols = ([p.replace('.', '_') for p in HEALTH_PARAMS] + [ucol] + CHANNELS)
    df = pd.DataFrame(rows, columns=cols)
    suffix = '' if family == 'cruise' else '_takeoff'
    df.to_parquet(OUT / f'surrogate_data{suffix}.parquet', index=False)
    print(f'{len(df)} solves -> {OUT}/surrogate_data{suffix}.parquet')


if __name__ == '__main__':
    main()
