"""WP2.4 audit 1: nonlinearity of the fleet generator's linearization.

The generator predicts snapshots as z = baseline * (1 + H.x/100). This audit
re-solves full pyCycle at health states actually sampled from the generated
fleet (stratified by life fraction, over-weighting end of life where |x| is
largest) and reports the linearization error per channel, next to the sensor
noise floor it must be judged against.

Conditions are exact ICM grid points (cruise, takeoff_hot), so the measured
error is pure nonlinearity, not interpolation. One worker per condition chunk
(norm N1); recovery on divergence is a cold rebuild (norm N2).

Output: data/processed/fleet/audit_nonlinearity.json
Usage: uv run python scripts/audit_nonlinearity.py [n_states_per_condition]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.cycle import OD_GUESSES, build_study_problem, set_health, snapshot
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'

CONDITIONS = {
    'cruise': dict(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4666.0),
    'takeoff_hot': dict(mn=0.001, alt_ft=0.0, dTs=54.0, n1_rpm=4813.0,
                        guesses=OD_GUESSES['A2_takeoff'] | {'hpt_PR': 3.7, 'lpt_PR': 4.3}),
}
X_COLS = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]


def sample_states(df, n, rng):
    """Stratified by life fraction: 20 % early, 30 % mid, 50 % late (worst case)."""
    frac = df.cycle / df.groupby('engine_id').cycle.transform('max')
    parts = [df[frac < 0.3], df[(frac >= 0.3) & (frac < 0.7)], df[frac >= 0.7]]
    weights = [0.2, 0.3, 0.5]
    picks = [p.sample(int(n * w), random_state=rng) for p, w in zip(parts, weights)]
    return pd.concat(picks)[X_COLS].to_numpy(dtype=float)


def solve_chunk(args):
    """Worker: full pyCycle solves for a chunk of health states at one condition."""
    cond_name, states = args
    op = CONDITIONS[cond_name]
    H, channels, baseline = load_icm(cond_name)

    prob = build_study_problem(**op)
    prob.set_solver_print(level=-1)
    prob.run_model()

    out = []
    for x in states:
        deltas = {p: v / 100.0 for p, v in zip(HEALTH_PARAMS, x)}
        set_health(prob, deltas)
        try:
            prob.run_model()
            full = snapshot(prob, 'OD')
        except Exception:
            prob = build_study_problem(**op)   # norm N2: rebuild after failure
            prob.set_solver_print(level=-1)
            prob.run_model()
            out.append(None)
            continue
        lin = {c: baseline[c] * (1.0 + float(np.dot(H[i], x)) / 100.0)
               for i, c in enumerate(channels)}
        out.append({'x_maxabs': float(np.max(np.abs(x))),
                    'full': {c: full[c] for c in channels},
                    'linear': lin})
    return cond_name, out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df = pd.read_parquet(FLEET / 'snapshots.parquet', columns=['engine_id', 'cycle'] + X_COLS)
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())

    jobs = []
    for cond in CONDITIONS:
        states = sample_states(df, n, np.random.default_rng(11))
        for chunk in np.array_split(states, 5):        # 5 workers per condition
            jobs.append((cond, chunk))

    results = {c: [] for c in CONDITIONS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for cond, out in pool.map(solve_chunk, jobs):
            results[cond].extend([r for r in out if r is not None])

    report = {}
    for cond, rows in results.items():
        channels = list(rows[0]['full'])
        errs = {c: [] for c in channels}
        for r in rows:
            for c in channels:
                errs[c].append(abs(r['linear'][c] - r['full'][c]) / abs(r['full'][c]) * 100)
        noise = {}
        for c in channels:
            spec = catalog['sensors'].get(c, {})
            base = np.mean([r['full'][c] for r in rows])
            noise[c] = spec.get('sigma_pct', 100.0 * spec.get('sigma', 0.0) / base)
        report[cond] = {
            'n_attempted': n,
            'n_solved': len(rows),
            'per_channel_err_pct': {
                c: {'median': float(np.median(e)), 'p95': float(np.percentile(e, 95)),
                    'max': float(np.max(e)), 'noise_sigma_pct': float(noise[c])}
                for c, e in errs.items()},
        }

    (FLEET / 'audit_nonlinearity.json').write_text(json.dumps(report, indent=2))
    for cond, r in report.items():
        print(f'\n== {cond} (n={r["n_solved"]}) ==')
        for c, e in r['per_channel_err_pct'].items():
            print(f'  {c:10s} med {e["median"]:.3f}%  p95 {e["p95"]:.3f}%  '
                  f'max {e["max"]:.3f}%   (noise sigma ~{e["noise_sigma_pct"]:.2f}%)')


if __name__ == '__main__':
    main()
