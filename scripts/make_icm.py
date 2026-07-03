"""WP1.4: generate the Influence Coefficient Matrix at the reference operating
points, with the SVD observability analysis and the confusable-pair report.

Output: data/processed/icm/icm_<point>.npz + icm_report.json.
Usage: uv run python scripts/make_icm.py
"""

import json
import os
from pathlib import Path

import numpy as np

from ehmbrain.perf.cycle import OD_GUESSES
from ehmbrain.perf.icm import (COCKPIT, EXTENDED, HEALTH_PARAMS, compute_icm,
                               confusable_pairs, signature_angles, svd_report)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'data' / 'processed' / 'icm'

# Operating-point grid: the two EHM snapshot conditions (cruise report,
# takeoff report) plus variations in power, altitude, climb and hot-day —
# enough to quantify how much the ICM changes across the envelope (F3
# interpolates it over this grid).
TO_GUESS = OD_GUESSES['A2_takeoff'] | {'hpt_PR': 3.7, 'lpt_PR': 4.3}
CLIMB_GUESS = dict(FAR=0.026, W=550.0, BPR=5.2, lp_Nmech=4750.0, hp_Nmech=14100.0,
                   hpt_PR=3.6, lpt_PR=4.3)
POINTS = {
    'cruise': dict(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4666.0),
    'cruise_lowpwr': dict(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4400.0),
    'cruise_39k': dict(mn=0.78, alt_ft=39000.0, dTs=0.0, n1_rpm=4666.0),
    'climb': dict(mn=0.40, alt_ft=10000.0, dTs=0.0, n1_rpm=4750.0,
                  guesses=CLIMB_GUESS),
    'takeoff': dict(mn=0.001, alt_ft=0.0, dTs=0.0, n1_rpm=4813.0, guesses=TO_GUESS),
    'takeoff_hot': dict(mn=0.001, alt_ft=0.0, dTs=54.0, n1_rpm=4813.0,  # ISA+30 C
                        guesses=TO_GUESS),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {'health_params': HEALTH_PARAMS, 'channels_extended': EXTENDED,
              'channels_cockpit': COCKPIT, 'points': {}}

    for name, op in POINTS.items():
        print(f'== {name} ==', flush=True)
        H, base = compute_icm(**op, verbose=True,
                              n_workers=max(1, (os.cpu_count() or 2) - 1))
        np.savez(OUT_DIR / f'icm_{name}.npz', H=H, params=HEALTH_PARAMS,
                 channels=EXTENDED, baseline=json.dumps(base), op=json.dumps(
                     {k: v for k, v in op.items() if k != 'guesses'}))

        Hc = H[:len(COCKPIT), :]   # cockpit-only rows (N2, WF, EGT)
        report['points'][name] = {
            'operating_point': {k: v for k, v in op.items() if k != 'guesses'},
            'H_extended': H.tolist(),
            'svd_cockpit': svd_report(Hc),
            'svd_extended': svd_report(H),
            'confusable_cockpit_15deg': confusable_pairs(Hc),
            'min_angle_cockpit_deg': float(np.nanmin(signature_angles(Hc))),
            'min_angle_extended_deg': float(np.nanmin(signature_angles(H))),
        }

    (OUT_DIR / 'icm_report.json').write_text(json.dumps(report, indent=2))
    for name, r in report['points'].items():
        print(f"\n{name}: cond(cockpit) = {r['svd_cockpit']['condition_number']:.1f}, "
              f"cond(extended) = {r['svd_extended']['condition_number']:.1f}, "
              f"min angle cockpit = {r['min_angle_cockpit_deg']:.1f} deg, "
              f"confusable pairs = {len(r['confusable_cockpit_15deg'])}")


if __name__ == '__main__':
    main()
