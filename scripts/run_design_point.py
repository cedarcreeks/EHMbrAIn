"""Solve the CFM56-7B26 design point and check it against the calibration contract.

Usage: uv run python scripts/run_design_point.py
Exit code 0 iff the point converges and all WP1.1 windows hold.
"""

import sys
from pathlib import Path

import yaml

from ehmbrain.perf.cycle import build_design_problem, design_summary

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_windows(summary, targets):
    dp = targets['design_point']
    checks = []

    bpr_t = dp['targets']['bpr']
    checks.append(('BPR', abs(summary['BPR'] - bpr_t['value']) <= bpr_t['tol_abs'],
                   f"{summary['BPR']:.3f} vs {bpr_t['value']} ± {bpr_t['tol_abs']}"))

    lo, hi = dp['targets']['opr']['window']
    checks.append(('OPR', lo <= summary['OPR'] <= hi, f"{summary['OPR']:.2f} in [{lo}, {hi}]"))

    lo, hi = dp['windows']['t4_degR']
    checks.append(('T4', lo <= summary['T4_degR'] <= hi, f"{summary['T4_degR']:.0f} degR in [{lo}, {hi}]"))

    lo, hi = dp['windows']['cooling_total_frac']
    checks.append(('cooling', lo <= summary['cooling_total_frac'] <= hi,
                   f"{summary['cooling_total_frac']:.3f} in [{lo}, {hi}]"))

    fn_t = dp['targets']['fn_lbf']
    rel = abs(summary['Fn_lbf'] - fn_t['value']) / fn_t['value'] * 100
    checks.append(('Fn', rel <= fn_t['tol_pct'], f"{summary['Fn_lbf']:.0f} lbf ({rel:.2f} % off target)"))

    return checks


def main():
    targets = yaml.safe_load((REPO_ROOT / 'conf' / 'cfm56_7b_targets.yaml').read_text())

    prob = build_design_problem()
    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    summary = design_summary(prob)

    print('\n=== CFM56-7B26 design point (M0.78 / 35 kft / ISA) ===')
    for k, v in summary.items():
        print(f'  {k:20s} {v:.4f}')

    print('\n=== WP1.1 window checks ===')
    ok = True
    for name, passed, detail in check_windows(summary, targets):
        print(f'  [{"PASS" if passed else "FAIL"}] {name:8s} {detail}')
        ok &= passed

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
