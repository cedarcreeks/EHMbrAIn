"""F8/L1: training data for the differentiable neural surrogate of the twin.

Samples (health state x, commanded N1) at the cruise snapshot family and
solves the FULL pyCycle Study model for each — the ground truth the surrogate
must reproduce. Parallel workers (norm N1), cold rebuild on failure (norm N2).

x sampling: per-parameter U(-3 %, +1 %) for eta / U(-3 %, +3 %) for flow,
covering the fleet's realistic envelope with margin; 15 % healthy-ish draws
(x ~ N(0, 0.3)) so the surrogate is accurate near zero too.

Output: data/processed/f8/surrogate_data.parquet
Usage: uv run python scripts/f8_surrogate_data.py [n_samples]
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


def worker(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    prob = build_study_problem()          # cruise family, N1 throttle
    prob.set_solver_print(level=-1)
    prob.run_model()
    rows = []
    for _ in range(n):
        x = sample_x(rng)
        n1 = float(rng.uniform(4400.0, 4666.0))
        try:
            prob.set_val('OD.N1_target', n1, units='rpm')
            set_health(prob, {p: v / 100.0 for p, v in zip(HEALTH_PARAMS, x)})
            prob.run_model()
            s = snapshot(prob, 'OD')
            if abs(s['N1_rpm'] - n1) > 5.0:          # unconverged
                raise RuntimeError('n1 miss')
            rows.append(list(x) + [n1] + [s[c] for c in CHANNELS])
        except Exception:
            prob = build_study_problem()             # norm N2: cold rebuild
            prob.set_solver_print(level=-1)
            prob.run_model()
    return rows


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    OUT.mkdir(parents=True, exist_ok=True)
    n_workers = 8
    per = n_total // n_workers
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunks = list(pool.map(worker, [(100 + i, per) for i in range(n_workers)]))
    rows = [r for ch in chunks for r in ch]
    cols = ([p.replace('.', '_') for p in HEALTH_PARAMS] + ['N1_cmd'] + CHANNELS)
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(OUT / 'surrogate_data.parquet', index=False)
    print(f'{len(df)} solves -> {OUT}/surrogate_data.parquet')


if __name__ == '__main__':
    main()
