"""WP1.2 / gate H1: off-design SLS anchors vs measured ICAO EEDB data.

Gated anchors (A2 takeoff, A3 climb-out, A4 approach) must converge and
predict fuel flow within the contract tolerance; A2 must also match the
measured engine pressure ratio and respect the TCDS rotor speed redlines.
A5 (idle) must converge via the PC continuation but its fuel-flow error is
reported only (generic-map limitation, gated: false in the contract).
"""

from pathlib import Path

import pytest
import yaml

from ehmbrain.perf.cycle import MPCFM56, anchor_summary, build_problem, solve_anchors

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def solved():
    targets = yaml.safe_load((REPO_ROOT / 'conf' / 'cfm56_7b_targets.yaml').read_text())
    prob = build_problem(od=True)
    prob.set_solver_print(level=-1)
    converged = solve_anchors(prob)
    return prob, converged, targets


def test_all_anchors_converge(solved):
    _, converged, _ = solved
    assert all(converged.values()), f'non-converged anchors: {converged}'


@pytest.mark.parametrize('name', ['A2_takeoff', 'A3_climbout', 'A4_approach'])
def test_gated_fuel_flow_within_tolerance(solved, name):
    prob, _, targets = solved
    t = targets['anchors_offdesign'][name]
    wf = anchor_summary(prob, name)['WF_kgps']
    err_pct = abs(wf - t['wf_kgps']['value']) / t['wf_kgps']['value'] * 100
    assert err_pct <= t['wf_kgps']['tol_pct'], f'{name}: WF {wf:.3f} kg/s, err {err_pct:.2f} %'


def test_takeoff_opr_and_redlines(solved):
    prob, _, targets = solved
    t = targets['anchors_offdesign']['A2_takeoff']
    s = anchor_summary(prob, 'A2_takeoff')
    opr_err = abs(s['OPR'] - t['opr']['value']) / t['opr']['value'] * 100
    assert opr_err <= t['opr']['tol_pct']
    assert s['N1_rpm'] <= t['limits']['n1_rpm_max']
    assert s['N2_rpm'] <= t['limits']['n2_rpm_max']
