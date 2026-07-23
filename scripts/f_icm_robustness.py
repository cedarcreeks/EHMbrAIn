"""L-ICM: is the observability verdict robust to the twin's calibration?

The classical GPA sensitivity study the project owes its own premise. The model
chapter asserts that the AI-vs-traditional comparison "rests on the model's
sensitivity structure (the ICM), not on its absolute values"; this script
measures whether that holds by perturbing every calibration knob inside the
tolerance its own contract declares, recomputing the ICM, and asking whether the
observability verdict (rank, minimum signature angle, confusable-pair set)
survives. It also sweeps the finite-difference step, a class-[C] choice that was
reasoned but never measured.

Pre-registered: docs/prereg-v13.md, tag prereg-v13 (thresholds frozen first).
Output: data/processed/icm/icm_robustness.json
Usage: uv run python scripts/f_icm_robustness.py
"""

import json
import os
from pathlib import Path

import numpy as np

from ehmbrain.perf.cycle import OD_GUESSES
from ehmbrain.perf.icm import (COCKPIT, HEALTH_PARAMS, compute_icm,
                               confusable_pairs, signature_angles, svd_report)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'data' / 'processed' / 'icm'

TO_GUESS = OD_GUESSES['A2_takeoff'] | {'hpt_PR': 3.7, 'lpt_PR': 4.3}
POINTS = {
    'cruise': dict(mn=0.78, alt_ft=35000.0, dTs=0.0, n1_rpm=4666.0),
    'takeoff_hot': dict(mn=0.001, alt_ft=0.0, dTs=54.0, n1_rpm=4813.0,
                        guesses=TO_GUESS),
}

# Calibration perturbations, each inside the tolerance its contract declares
# (conf/cfm56_7b_targets.yaml + appendix A1). One at a time, then a joint
# adverse-sign draw. Values are absolute (name -> (value, units)).
EFFS = {'DESIGN.fan.eff': 0.8948, 'DESIGN.lpc.eff': 0.9243, 'DESIGN.hpc.eff': 0.8707,
        'DESIGN.hpt.eff': 0.8888, 'DESIGN.lpt.eff': 0.8996}


def _scaled(name, nominal, rel, units=None):
    return {name: (nominal * (1.0 + rel), units)}


def perturbations():
    """The frozen perturbation set of prereg-v13."""
    p = {}
    for sign, tag in ((+1, '+3%'), (-1, '-3%')):
        p[f'hpc.PR {tag}'] = _scaled('DESIGN.hpc.PR', 9.35, sign * 0.03)
        p[f'HP_Nmech {tag}'] = _scaled('DESIGN.HP_Nmech', 13940.0, sign * 0.03, 'rpm')
        p[f'Fn_DES {tag}'] = _scaled('DESIGN.Fn_DES', 5480.0, sign * 0.03, 'lbf')
        p[f'T4_MAX {tag}'] = _scaled('DESIGN.T4_MAX', 2857.0, sign * 0.03, 'degR')
    for sign, tag in ((+1, '+0.2'), (-1, '-0.2')):
        p[f'BPR {tag}'] = {'DESIGN.splitter.BPR': (5.1 + sign * 0.2, None)}
    for sign, tag in ((+1, '+2%'), (-1, '-2%')):
        p[f'all component eff {tag}'] = {
            n: (v * (1.0 + sign * 0.02), None) for n, v in EFFS.items()}
    # joint adverse: every knob at the sign that moves the cycle the same way
    joint = {}
    joint.update(_scaled('DESIGN.hpc.PR', 9.35, -0.03))
    joint.update(_scaled('DESIGN.HP_Nmech', 13940.0, +0.03, 'rpm'))
    joint.update(_scaled('DESIGN.Fn_DES', 5480.0, -0.03, 'lbf'))
    joint.update(_scaled('DESIGN.T4_MAX', 2857.0, +0.03, 'degR'))
    joint['DESIGN.splitter.BPR'] = (5.1 - 0.2, None)
    joint.update({n: (v * 0.98, None) for n, v in EFFS.items()})
    p['joint adverse'] = joint
    return p


def column_shift_deg(H_ref, H):
    """Per-fault direction shift: angle between each perturbed signature and
    its nominal counterpart (deg)."""
    out = []
    for j in range(H.shape[1]):
        u, v = H_ref[:, j], H[:, j]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        out.append(float(np.degrees(np.arccos(np.clip(abs(u @ v) / denom, 0, 1)))))
    return out


def summarize(Hc, H_ref=None):
    """Observability verdict for one cockpit ICM."""
    conf = confusable_pairs(Hc)
    s = {
        'rank': svd_report(Hc)['rank'],
        'condition_number': svd_report(Hc)['condition_number'],
        'min_angle_deg': float(np.nanmin(signature_angles(Hc))),
        'n_confusable': len(conf),
        'confusable_set': sorted(f'{a}~{b}' for a, b, _ in conf),
        'top_pair': f'{conf[0][0]}~{conf[0][1]}' if conf else None,
    }
    if H_ref is not None:
        shifts = column_shift_deg(H_ref, Hc)
        s['column_shift_deg'] = dict(zip(HEALTH_PARAMS, shifts))
        s['column_shift_median_deg'] = float(np.median(shifts))
        s['column_shift_max_deg'] = float(np.max(shifts))
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    report = {'prereg': 'docs/prereg-v13.md (tag prereg-v13)',
              'health_params': HEALTH_PARAMS, 'channels': COCKPIT, 'points': {}}

    for point, op in POINTS.items():
        print(f'===== {point} =====', flush=True)
        print('nominal...', flush=True)
        H_ref, _ = compute_icm(**op, channels=COCKPIT, n_workers=workers)
        entry = {'nominal': summarize(H_ref), 'perturbations': {}, 'step_sweep': {}}

        for label, ov in perturbations().items():
            print(f'  {label}', flush=True)
            H, _ = compute_icm(**op, channels=COCKPIT, n_workers=workers, overrides=ov)
            entry['perturbations'][label] = summarize(H, H_ref) | {
                'overrides': {k: v[0] for k, v in ov.items()}}

        for step in (0.0025, 0.01):   # nominal is 0.005
            print(f'  step {step:.4f}', flush=True)
            H, _ = compute_icm(**op, channels=COCKPIT, n_workers=workers, step=step)
            mag_ratio = [float(np.linalg.norm(H[:, j]) / np.linalg.norm(H_ref[:, j]))
                         for j in range(H.shape[1])]
            entry['step_sweep'][f'{step * 100:.2f}%'] = summarize(H, H_ref) | {
                'magnitude_ratio_max_pct_dev': float(
                    100 * np.max(np.abs(np.array(mag_ratio) - 1.0)))}
        report['points'][point] = entry

    # ---- pre-registered verdicts (prereg-v13) ------------------------------
    verdicts = {}
    all_pert = [(pt, lbl, s) for pt, e in report['points'].items()
                for lbl, s in e['perturbations'].items()]
    nominal_top = {pt: e['nominal']['top_pair'] for pt, e in report['points'].items()}
    nominal_set = {pt: e['nominal']['confusable_set'] for pt, e in report['points'].items()}

    h1_fail = [f'{pt}/{lbl}' for pt, lbl, s in all_pert
               if s['rank'] != 3 or s['min_angle_deg'] >= 15.0
               or s['top_pair'] != nominal_top[pt]]
    verdicts['H-ICM.1'] = {
        'criterion': 'every perturbation: rank 3, min angle < 15 deg, top pair unchanged',
        'failures': h1_fail,
        'confirmed': not h1_fail,
        'min_angle_range_deg': [min(s['min_angle_deg'] for _, _, s in all_pert),
                                max(s['min_angle_deg'] for _, _, s in all_pert)],
    }

    med_shift = float(np.median([s['column_shift_median_deg'] for _, _, s in all_pert]))
    set_same = [nominal_set[pt] == s['confusable_set'] for pt, _, s in all_pert]
    frac_same = float(np.mean(set_same))
    verdicts['H-ICM.2'] = {
        'criterion': 'median column shift <= 5 deg AND confusable set unchanged in >= 90 %',
        'median_column_shift_deg': med_shift,
        'max_column_shift_deg': max(s['column_shift_max_deg'] for _, _, s in all_pert),
        'confusable_set_unchanged_frac': frac_same,
        'confirmed': bool(med_shift <= 5.0 and frac_same >= 0.90),
    }

    steps = [(pt, lbl, s) for pt, e in report['points'].items()
             for lbl, s in e['step_sweep'].items()]
    max_dir = max(s['column_shift_max_deg'] for _, _, s in steps)
    max_mag = max(s['magnitude_ratio_max_pct_dev'] for _, _, s in steps)
    verdicts['H-ICM.3'] = {
        'criterion': 'step 0.25 %/1.0 % vs 0.5 %: < 2 deg direction and < 5 % magnitude',
        'max_direction_deg': max_dir, 'max_magnitude_dev_pct': max_mag,
        'confirmed': bool(max_dir < 2.0 and max_mag < 5.0),
    }
    report['verdicts'] = verdicts

    (OUT_DIR / 'icm_robustness.json').write_text(json.dumps(report, indent=2))
    print('\n===== pre-registered verdicts =====')
    for k, v in verdicts.items():
        print(f"{k}: {'CONFIRMED' if v['confirmed'] else 'REFUTED'} — {v['criterion']}")
    print(f"  min angle across all perturbations: "
          f"{verdicts['H-ICM.1']['min_angle_range_deg'][0]:.2f}"
          f"–{verdicts['H-ICM.1']['min_angle_range_deg'][1]:.2f} deg")
    print(f"  median column shift: {med_shift:.2f} deg, "
          f"max {verdicts['H-ICM.2']['max_column_shift_deg']:.2f} deg")
    print(f"  step sweep: {max_dir:.2f} deg, {max_mag:.2f} % magnitude")
    print(f'-> {OUT_DIR / "icm_robustness.json"}')


if __name__ == '__main__':
    main()
