"""F8/L2 audit: fidelity of the v2 (surrogate) generator vs the v1 (linear)
generator, both against full pyCycle, at states sampled from the v2 fleet.

Re-solves full pyCycle at fleet-realistic health states and compares each
generator's cockpit-channel prediction against it. The headline L2 number:
how much closer to true physics v2 is than v1. Output:
data/processed/fleet_v2/audit_v2_fidelity.json
Usage: uv run python scripts/audit_v2_fidelity.py [n]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.cycle import build_study_problem, set_health, snapshot
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
V2 = REPO_ROOT / 'data' / 'processed' / 'fleet_v2'
X_COLS = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
# cruise family surrogate is exercised (u = N1); design N1 = 4666 rpm
COND = dict(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4666.0)


def solve_chunk(states):
    from ehmbrain.perf.surrogate import SurrogateEmitter, SURR_CHANNELS
    H, channels, baseline = load_icm('cruise')
    surr = SurrogateEmitter.cached()
    prob = build_study_problem(**COND)
    prob.set_solver_print(level=-1)
    prob.run_model()
    out = []
    for x in states:
        set_health(prob, {p: v / 100.0 for p, v in zip(HEALTH_PARAMS, x)})
        try:
            prob.run_model()
            full = snapshot(prob, 'OD')
        except Exception:
            prob = build_study_problem(**COND)
            prob.set_solver_print(level=-1)
            prob.run_model()
            out.append(None)
            continue
        lin = {c: baseline[c] * (1.0 + float(np.dot(H[i], x)) / 100.0)
               for i, c in enumerate(channels)}
        sp = surr.predict(x[None], np.array([4666.0]), 'cruise')[0]
        srg = {c: float(sp[SURR_CHANNELS.index(c)]) for c in COCKPIT}
        out.append({'full': {c: full[c] for c in COCKPIT},
                    'linear': {c: lin[c] for c in COCKPIT}, 'surrogate': srg})
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = pd.read_parquet(V2 / 'snapshots.parquet',
                         columns=['engine_id', 'cycle'] + X_COLS)
    # over-weight end of life where |x| is largest
    df['life'] = df.groupby('engine_id')['cycle'].transform('max')
    df['frac'] = df['cycle'] / df['life']
    late = df[df.frac > 0.7].sample(n // 2, random_state=1)
    rest = df[df.frac <= 0.7].sample(n - n // 2, random_state=1)
    states = pd.concat([late, rest])[X_COLS].to_numpy()

    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for out in pool.map(solve_chunk, np.array_split(states, 8)):
            rows.extend([r for r in out if r is not None])

    report = {'n_solved': len(rows), 'per_channel': {}}
    for c in COCKPIT:
        le = [abs(r['linear'][c] - r['full'][c]) / abs(r['full'][c]) * 100 for r in rows]
        se = [abs(r['surrogate'][c] - r['full'][c]) / abs(r['full'][c]) * 100 for r in rows]
        report['per_channel'][c] = {
            'v1_linear': {'median': float(np.median(le)), 'p95': float(np.percentile(le, 95))},
            'v2_surrogate': {'median': float(np.median(se)), 'p95': float(np.percentile(se, 95))},
            'improvement_factor_median': float(np.median(le) / max(np.median(se), 1e-9))}
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / 'audit_v2_fidelity.json').write_text(json.dumps(report, indent=2))
    for c, r in report['per_channel'].items():
        print(f"{c:10s} v1 {r['v1_linear']['median']:.3f}/{r['v1_linear']['p95']:.3f}%  "
              f"v2 {r['v2_surrogate']['median']:.3f}/{r['v2_surrogate']['p95']:.3f}%  "
              f"({r['improvement_factor_median']:.1f}x better)")


if __name__ == '__main__':
    main()
