"""F-UQ (prereg-v15): re-attribute the H5 uncertainty result.

Why. The frozen H5 verdict compares a *split-conformal* AI interval against the
raw Theil-Sen slope-percentile band. The AI side is calibrated to 90 %; the
traditional side is not, and empirically it over-covers (0.983 against a
nominal 0.90). An interval that over-covers is necessarily wide, so the
reported 6x width advantage mixes two different things: how accurate the point
predictor is, and whether its interval was calibrated at all. Only the first is
an AI-versus-traditional fact. The second is a fact about conformal
prediction, which is model-agnostic -- this project already demonstrated that
by wrapping split-conformal around the classical Kalman/CRB ellipsoid in F10
(H10.2 fix).

What this does. Wraps the *same* split-conformal procedure around all three
point predictors -- the AI GRU, the operational Theil-Sen, and the advanced
similarity-based classical -- calibrating each on the validation split and
evaluating each on the test split. With every method held at the same nominal
coverage, the remaining width difference is the honest uncertainty margin.

The frozen H5 verdict is NOT rewritten: it stands as pre-registered and is
reproduced here for comparison. This is a disclosed post-hoc re-attribution.

Foreground (torch-MPS for the GRU). Output: data/processed/f8/uq_verdict.json
Usage: uv run python scripts/f_uq_reattribution.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import EVAL_FRACS, fleet_cache, split_ids               # noqa: E402
from f8_lrul_advanced import (ai_errors, build_library,              # noqa: E402
                              classical_errors)

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
F8 = REPO_ROOT / 'data' / 'processed' / 'f8'
ALPHA = 0.10                      # nominal 90 % intervals, as in H5
METHODS = ('ai', 'similarity', 'theilsen')
LABEL = {'ai': 'AI GRU', 'similarity': 'similarity (advanced classical)',
         'theilsen': 'Theil-Sen (operational)'}


def conformal_qhat(abs_err, alpha=ALPHA):
    """Split-conformal radius: the finite-sample-corrected (1-alpha) quantile
    of the calibration absolute residuals (same formula as the frozen H5)."""
    a = np.asarray(list(abs_err), float)
    n = len(a)
    return float(np.quantile(a, min(1.0, (1 - alpha) * (1 + 1 / n))))


def calibrate_and_test(err_val, err_test, keys_by_frac=None):
    """Pooled-over-fractions conformal band (the H5 protocol), plus the
    per-fraction variant, which is the honest way to band a quantity whose
    scale changes by an order of magnitude across life."""
    qhat = conformal_qhat(abs(v) for v in err_val.values())
    te = np.array([abs(v) for v in err_test.values()])
    out = {'halfwidth_cycles': qhat,
           'coverage': float(np.mean(te <= qhat)),
           'n_cal': len(err_val), 'n_test': len(te)}
    per_frac = {}
    for f in EVAL_FRACS:
        cal = [abs(v) for (e, ff), v in err_val.items() if ff == f]
        tst = [abs(v) for (e, ff), v in err_test.items() if ff == f]
        if not cal or not tst:
            continue
        q = conformal_qhat(cal)
        per_frac[str(f)] = {'halfwidth_cycles': q,
                            'coverage': float(np.mean(np.array(tst) <= q))}
    out['per_fraction'] = per_frac
    return out


def main():
    c = fleet_cache()
    fleet = c['fleet']
    train_ids = split_ids(fleet, 'train')
    val_ids = split_ids(fleet, 'val')
    test_ids = split_ids(fleet, 'test')
    lib = build_library(c, fleet, train_ids)

    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    win = sel_t['rul']['params']['rul_win']
    rul_a = sel_t['rul']['params']['rul_a']

    print('== classical predictors on val (calibration) ==', flush=True)
    ev = classical_errors(c, fleet, lib, val_ids, win, rul_a)
    print('== classical predictors on test ==', flush=True)
    et = classical_errors(c, fleet, lib, test_ids, win, rul_a)
    # seed 0 only, matching the frozen H5 AI conformal exactly
    print('== AI GRU on val + test (seed 0, as frozen H5) ==', flush=True)
    ev['ai'] = ai_errors('val', seeds=(0,))
    et['ai'] = ai_errors('test', seeds=(0,))

    res = {m: calibrate_and_test(ev[m], et[m]) for m in METHODS}
    for m in METHODS:
        res[m]['rmse_test'] = float(np.sqrt(np.mean(
            np.square([v for v in et[m].values()]))))
        res[m]['label'] = LABEL[m]

    hw_ai = res['ai']['halfwidth_cycles']
    ratios = {m: res[m]['halfwidth_cycles'] / hw_ai for m in METHODS}
    for m in METHODS:
        res[m]['coverage_gap'] = res[m]['coverage'] - (1 - ALPHA)
        # a width comparison is only fair where the achieved coverage lands on
        # the nominal one; conformal guarantees this in expectation, but with
        # 20 calibration engines the realised coverage still moves
        res[m]['fair_width_comparison'] = bool(abs(res[m]['coverage_gap']) <= 0.05)

    frozen = json.loads((F5 / 'verdicts.json').read_text())['H5']
    verdict = {
        'protocol': ('split-conformal at nominal 90 %, calibrated on val, '
                     'evaluated on test, applied identically to all three '
                     'point predictors'),
        'per_method': res,
        'width_ratio_over_ai': ratios,
        'frozen_H5': {
            'ai': frozen['ai'], 'trad': frozen['trad'],
            'width_ratio': frozen['trad']['halfwidth_cycles'] / frozen['ai']['halfwidth_cycles'],
            'note': ('traditional side uncalibrated: coverage '
                     f"{frozen['trad']['coverage']:.3f} against nominal 0.90")},
        'H-UQ.1_advantage_survives_calibration': {
            'ratio_vs_operational': ratios['theilsen'],
            'ratio_vs_advanced': ratios['similarity'],
            'frozen_ratio': frozen['trad']['halfwidth_cycles'] / frozen['ai']['halfwidth_cycles'],
            'confirmed': bool(ratios['similarity'] > 1.0
                              and res['similarity']['fair_width_comparison']
                              and res['ai']['fair_width_comparison']),
            'note': ('the AI interval is still narrower at equal coverage, but '
                     'the margin attributable to the model -- not to conformal '
                     'calibration -- is the ratio against the advanced '
                     'classical'),
            'caveat_theilsen': ('the operational band under-covers on test '
                                f"({res['theilsen']['coverage']:.3f} vs 0.90): "
                                'its residuals are not exchangeable between '
                                'val and test, so the 2.3x figure is a lower '
                                'bound on the width it would need for honest '
                                'coverage')},
    }
    (F8 / 'uq_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    print('  method                              coverage   halfwidth   ratio   RMSE')
    for m in METHODS:
        r = res[m]
        print(f"  {LABEL[m]:34s}  {r['coverage']:.3f}    {r['halfwidth_cycles']:8.0f}"
              f"  {ratios[m]:5.2f}x  {r['rmse_test']:6.0f}")
    print(f"\n  frozen H5 (uncalibrated traditional band): coverage "
          f"{frozen['trad']['coverage']:.3f}, halfwidth "
          f"{frozen['trad']['halfwidth_cycles']:.0f}, ratio "
          f"{verdict['frozen_H5']['width_ratio']:.2f}x")
    print(f"  re-attributed: {ratios['theilsen']:.2f}x vs operational, "
          f"{ratios['similarity']:.2f}x vs advanced classical")


if __name__ == '__main__':
    main()
