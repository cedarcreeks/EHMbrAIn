"""Solve the CFM56-7B26 design point + SLS anchors and compare fuel-flow
predictions against the ICAO EEDB measurements (calibration contract).

Usage: uv run python scripts/run_anchors.py
Exit code 0 iff every *gated* anchor converges and meets its tolerance
(A5 idle is reported but not gated — see conf/cfm56_7b_targets.yaml).
"""

import sys
from pathlib import Path

import yaml

from ehmbrain.perf.cycle import MPCFM56, anchor_summary, build_problem, solve_anchors

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    targets = yaml.safe_load((REPO_ROOT / 'conf' / 'cfm56_7b_targets.yaml').read_text())
    anchors_t = targets['anchors_offdesign']

    prob = build_problem(od=True)
    prob.set_solver_print(level=-1)
    converged = solve_anchors(prob)

    ok = True
    print('\n=== SLS anchors vs ICAO EEDB (CFM56-7B26) ===')
    print(f'{"anchor":14s} {"conv":5s} {"Fn lbf":>9s} {"WF kg/s":>8s} {"EEDB":>6s} {"err%":>6s} '
          f'{"OPR":>6s} {"N1 rpm":>7s} {"N2 rpm":>7s} {"T4 degR":>8s}')

    for name in MPCFM56.OD_ANCHORS:
        s = anchor_summary(prob, name)
        t = anchors_t[name]
        gated = t.get('gated', True)
        wf_target = t['wf_kgps']['value']
        wf_err = (s['WF_kgps'] - wf_target) / wf_target * 100
        line_ok = converged[name] and abs(wf_err) <= t['wf_kgps']['tol_pct']

        if name == 'A2_takeoff':
            opr_t = t['opr']
            line_ok &= abs(s['OPR'] - opr_t['value']) / opr_t['value'] * 100 <= opr_t['tol_pct']
            line_ok &= s['N1_rpm'] <= t['limits']['n1_rpm_max']
            line_ok &= s['N2_rpm'] <= t['limits']['n2_rpm_max']

        if gated:
            ok &= line_ok
        verdict = 'PASS' if line_ok else ('WARN (not gated)' if not gated else 'FAIL')
        print(f'{name:14s} {"yes" if converged[name] else "NO":5s} {s["Fn_lbf"]:9.0f} '
              f'{s["WF_kgps"]:8.3f} {wf_target:6.3f} {wf_err:+6.2f} {s["OPR"]:6.2f} '
              f'{s["N1_rpm"]:7.0f} {s["N2_rpm"]:7.0f} {s["T4_degR"]:8.0f}  [{verdict}]')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
