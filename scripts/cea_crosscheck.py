"""WP1.4: cross-check the tabular thermodynamics against CEA at the design point.

The project runs with TABULAR thermo for speed (decision D2). This script
re-solves the design point with full chemical-equilibrium (CEA) thermodynamics
and reports the deltas; the spec expects < 0.5 % on the key quantities.

Output: data/processed/cea_crosscheck.json
Usage: uv run python scripts/cea_crosscheck.py   (takes a few minutes: CEA is slow)
"""

import json
from pathlib import Path

import openmdao.api as om

from ehmbrain.perf.cycle import MPCFM56, build_problem, design_summary

REPO_ROOT = Path(__file__).resolve().parents[1]


class MPCFM56CEA(MPCFM56):
    THERMO = 'CEA'


def main():
    results = {}
    for name, cls in (('tabular', MPCFM56), ('cea', MPCFM56CEA)):
        prob = om.Problem()
        prob.model = cls(od=False)
        prob.setup()
        from ehmbrain.perf.cycle import DESIGN_INPUTS
        for var, (val, units) in DESIGN_INPUTS.items():
            prob.set_val(var, val, units=units)
        prob['DESIGN.balance.FAR'] = 0.025
        prob['DESIGN.balance.W'] = 100.
        prob['DESIGN.balance.lpt_PR'] = 4.0
        prob['DESIGN.balance.hpt_PR'] = 3.0
        prob['DESIGN.fc.balance.Pt'] = 5.2
        prob['DESIGN.fc.balance.Tt'] = 440.0
        prob.set_solver_print(level=-1)
        prob.run_model()
        results[name] = design_summary(prob)
        print(f'{name}: {results[name]}')

    deltas = {k: 100.0 * (results['cea'][k] - results['tabular'][k]) / results['tabular'][k]
              for k in results['tabular'] if abs(results['tabular'][k]) > 1e-12}
    out = {'tabular': results['tabular'], 'cea': results['cea'], 'delta_pct': deltas}
    (REPO_ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / 'data' / 'processed' / 'cea_crosscheck.json').write_text(
        json.dumps(out, indent=2))
    print('\ndeltas CEA vs tabular [%]:')
    for k, v in deltas.items():
        print(f'  {k:22s} {v:+.3f}')


if __name__ == '__main__':
    main()
