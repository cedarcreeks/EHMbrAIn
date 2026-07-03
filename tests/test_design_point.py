"""WP1.1 DoD: the CFM56-7B26 design point converges and sits inside the
sanity windows of the calibration contract (conf/cfm56_7b_targets.yaml).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ehmbrain.perf.cycle import build_design_problem, design_summary

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def solved():
    prob = build_design_problem()
    prob.set_solver_print(level=-1)
    prob.run_model()
    targets = yaml.safe_load((REPO_ROOT / 'conf' / 'cfm56_7b_targets.yaml').read_text())
    return prob, design_summary(prob), targets


def test_converged(solved):
    prob, summary, _ = solved
    # Thrust balance actually met => Newton converged, not just ran out of iterations.
    fn_err = abs(summary['Fn_lbf'] - prob.get_val('DESIGN.Fn_DES', units='lbf')[0])
    assert fn_err < 1.0, f'thrust balance off by {fn_err} lbf'
    assert np.isfinite(list(summary.values())).all()


def test_design_windows(solved):
    _, s, targets = solved
    dp = targets['design_point']

    bpr = dp['targets']['bpr']
    assert abs(s['BPR'] - bpr['value']) <= bpr['tol_abs']

    lo, hi = dp['targets']['opr']['window']
    assert lo <= s['OPR'] <= hi

    lo, hi = dp['windows']['t4_degR']
    assert lo <= s['T4_degR'] <= hi

    lo, hi = dp['windows']['cooling_total_frac']
    assert lo <= s['cooling_total_frac'] <= hi


def test_turbine_prs_physical(solved):
    _, s, _ = solved
    # Two-spool HBTF: HPT drives the HPC (PR ~3-4), LPT drives fan+booster (PR ~4-5).
    assert 2.5 <= s['HPT_PR'] <= 4.5
    assert 3.5 <= s['LPT_PR'] <= 5.5
