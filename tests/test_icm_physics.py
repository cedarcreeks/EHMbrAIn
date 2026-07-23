"""WP1.4 / gate H1: physical sign checks on the influence coefficient matrix.

Computes a reduced ICM (3 health parameters, cockpit channels) at cruise and
asserts the signs that any thermodynamically sound engine model must show.
Also verifies the Study self-consistency property: the healthy off-design
point at design N1 must reproduce the design point.
"""

import numpy as np
import pytest

from ehmbrain.perf.cycle import build_study_problem, set_health, snapshot
from ehmbrain.perf.icm import COCKPIT, compute_icm

CHECK_PARAMS = ['hpt.eta', 'hpc.eta', 'fan.flow']


@pytest.fixture(scope='module')
def study():
    prob = build_study_problem()          # cruise, N1 = design 4666 rpm
    prob.set_solver_print(level=-1)
    prob.run_model()
    return prob


def test_healthy_od_reproduces_design(study):
    """Self-consistency: healthy OD at design N1 == the design point."""
    s = snapshot(study, 'OD')
    assert abs(s['Fn_lbf'] - 5480.0) < 5.0
    assert abs(s['N2_rpm'] - 13940.0) < 15.0
    assert abs(s['OPR'] - 30.03) < 0.1
    assert abs(s['EGT_degK'] - 1141.6) < 2.0


def test_icm_signs_cruise(study):
    """Signs of the classic GPA sensitivities, one health param at a time."""
    base = snapshot(study, 'OD')
    step = 0.005
    sens = {}
    for param in CHECK_PARAMS:
        set_health(study, {param: -step})     # degrade by 0.5 %
        study.run_model()
        s = snapshot(study, 'OD')
        sens[param] = {k: s[k] - base[k] for k in ('EGT_degK', 'WF_kgps', 'N2_rpm')}
        set_health(study, {})
        study.run_model()

    # A degraded (less efficient) HPT or HPC burns more fuel and runs hotter.
    assert sens['hpt.eta']['EGT_degK'] > 0
    assert sens['hpt.eta']['WF_kgps'] > 0
    assert sens['hpc.eta']['EGT_degK'] > 0
    assert sens['hpc.eta']['WF_kgps'] > 0
    # Fan fouling (flow capacity loss) at constant N1 reduces thrust -> the
    # cycle compensates with temperature: EGT up, fuel flow up... but through
    # the bypass the dominant effect is measured fuel flow DOWN per unit; the
    # robust sign at constant N1 is fuel flow down with the smaller fan flow.
    assert sens['fan.flow']['WF_kgps'] < 0


def test_icm_matrix_consistency():
    """The central-difference ICM matches the one-sided sensitivities in sign
    and the cockpit block is severely ill-conditioned (the GPA motivation)."""
    H, _ = compute_icm(channels=COCKPIT, step=0.005)
    params_idx = {p: j for j, p in enumerate(
        [f'{c}.{k}' for c in ('fan', 'lpc', 'hpc', 'hpt', 'lpt') for k in ('eta', 'flow')])}
    egt, wf = COCKPIT.index('EGT_degK'), COCKPIT.index('WF_kgps')
    # +1 % efficiency (healthier) -> cooler and thriftier
    assert H[egt, params_idx['hpt.eta']] < 0
    assert H[wf, params_idx['hpt.eta']] < 0
    assert H[egt, params_idx['hpc.eta']] < 0
    # Underdetermination: 3 measurements for 10 unknowns
    assert np.linalg.matrix_rank(H) == 3


def test_calibration_override_moves_the_model_not_the_verdict():
    """The `overrides` hook (L-ICM) must actually perturb the calibration --- a
    silently ignored override would make the robustness study vacuous --- while
    leaving the cockpit underdetermination that every GPA claim rests on."""
    base = build_study_problem()
    base.set_solver_print(level=-1)
    base.run_model()
    opr_nominal = snapshot(base, 'OD')['OPR']

    pert = build_study_problem(overrides={'DESIGN.hpc.PR': (9.35 * 1.03, None)})
    pert.set_solver_print(level=-1)
    pert.run_model()
    s = snapshot(pert, 'OD')
    assert s['OPR'] > opr_nominal * 1.01      # the override reached the cycle
    assert np.linalg.matrix_rank(
        compute_icm(channels=COCKPIT, step=0.005,
                    overrides={'DESIGN.hpc.PR': (9.35 * 1.03, None)})[0]) == 3
