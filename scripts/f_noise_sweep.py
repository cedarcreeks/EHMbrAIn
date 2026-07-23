"""The noise axis of the determinability map (contribution C6).

Every result in this document lives at one sensor-quality setting. This sweep
regenerates the fleet with each sensor sigma scaled by {0.5, 1, 2, 4} -- same
seed, same health trajectories, same splits, same faults, so measurement
quality is the only variable that moves -- and re-scores both families at their
F5-tuned configurations, unchanged.

That last choice is the honest reading of the output: this measures how the
systems an operator would actually have fielded degrade as sensors change, not
the best score reachable at each noise level (which would need eight more
Optuna campaigns and answers a different question).

Pre-registered: docs/prereg-v14.md, tag prereg-v14 (thresholds frozen first).
Output: data/processed/f5/noise_sweep.json
Usage:  uv run python scripts/f_noise_sweep.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from tune_f5 import (EVAL_FRACS, eval_ai, eval_trad,  # noqa: E402
                     fleet_cache, split_ids, use_fleet)

F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
PROC = REPO_ROOT / 'data' / 'processed'
LEVELS = (0.5, 1.0, 2.0, 4.0)
RUL_SEEDS = (0, 1, 2)


def fleet_dir_for(mult):
    return PROC / 'fleet' if mult == 1.0 else PROC / f'fleet_noise{mult:g}'


def ensure_fleet(mult):
    """Regenerate the fleet at this noise level if it is not on disk. The
    m = 1 case reuses the frozen v1.1 fleet -- never regenerated, its hashes
    anchor prereg-v1."""
    d = fleet_dir_for(mult)
    if (d / 'snapshots.parquet').exists():
        print(f'  fleet {d.name} present')
        return d
    print(f'  generating {d.name} ...', flush=True)
    subprocess.run([sys.executable, str(REPO_ROOT / 'scripts' / 'make_fleet.py'),
                    f'noise={mult:g}'], check=True, cwd=REPO_ROOT)
    return d


def rmse_at(rows, frac):
    e = [r['err'] for r in rows if abs(r['frac'] - frac) < 1e-9]
    return float(np.sqrt(np.mean(np.square(e)))) if e else float('nan')


def score_level(mult, sel_t, sel_a):
    use_fleet(ensure_fleet(mult))
    c = fleet_cache()
    n_test = len(split_ids(c['fleet'], 'test'))

    print('  traditional...', flush=True)
    t_det = eval_trad(sel_t['detection']['params'], 'test')
    t_iso = eval_trad(sel_t['isolation']['params'], 'test')
    t_rul = eval_trad(sel_t['rul']['params'], 'test')

    print('  AI...', flush=True)
    a_det = eval_ai(sel_a['detection']['params'], 'train', 'test', seed=0)
    a_iso = eval_ai(sel_a['isolation']['params'], 'train', 'test', seed=0)
    a_rul = [eval_ai(sel_a['rul']['params'], 'train', 'test', seed=s)
             for s in RUL_SEEDS]

    def rul_block(rows_list):
        return {f'{f:g}': float(np.mean([rmse_at(r, f) for r in rows_list]))
                for f in EVAL_FRACS}

    return {
        'n_test_engines': n_test,
        'n_episodes': t_det['n_episodes'],
        'traditional': {
            'recall': t_det['recall'], 'false_alarms': t_det['fa'],
            'confusable_isolation': t_iso['conf_acc'],
            'rul_rmse': rul_block([t_rul['rul_rows']]),
        },
        'ai': {
            'recall': a_det['recall'], 'false_alarms': a_det['fa'],
            'confusable_isolation': a_iso['conf_acc'],
            'rul_rmse': rul_block([r['rul_rows'] for r in a_rul]),
            'rul_rmse_seed_spread': {
                f'{f:g}': float(np.std([rmse_at(r['rul_rows'], f) for r in a_rul]))
                for f in EVAL_FRACS},
        },
    }


def main():
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']

    levels = {}
    for m in LEVELS:
        print(f'===== sensor noise x{m:g} =====', flush=True)
        levels[f'{m:g}'] = score_level(m, sel_t, sel_a)

    # ---- pre-registered verdicts (prereg-v14) ---------------------------
    ai_wins = {k: v['ai']['rul_rmse']['0.9'] < v['traditional']['rul_rmse']['0.9']
               for k, v in levels.items()}
    gaps = {k: abs(v['ai']['confusable_isolation'] - v['traditional']['confusable_isolation'])
            for k, v in levels.items()}
    quiet = levels['0.5']
    verdicts = {
        'H-N.1': {
            'criterion': 'AI RUL RMSE at 90 % life below traditional at every noise level',
            'per_level': ai_wins,
            'confirmed': all(ai_wins.values()),
        },
        'H-N.2': {
            'criterion': 'confusable-isolation gap between families <= 10 pp at every level',
            'gap_pp': {k: 100 * g for k, g in gaps.items()},
            'confirmed': all(g <= 0.10 for g in gaps.values()),
        },
        'H-N.3': {
            'criterion': 'at halved noise neither family exceeds 50 % confusable isolation',
            'traditional': quiet['traditional']['confusable_isolation'],
            'ai': quiet['ai']['confusable_isolation'],
            'confirmed': (quiet['traditional']['confusable_isolation'] <= 0.5
                          and quiet['ai']['confusable_isolation'] <= 0.5),
        },
    }

    out = {'prereg': 'docs/prereg-v14.md (tag prereg-v14)',
           'note': 'both families at their F5-tuned configs, not re-tuned per level',
           'levels': levels, 'verdicts': verdicts}
    (F5 / 'noise_sweep.json').write_text(json.dumps(out, indent=2))

    print('\n===== determinability map: the noise axis =====')
    print(f"{'sigma':>6} {'trad RUL@90':>12} {'AI RUL@90':>10} "
          f"{'trad conf':>10} {'AI conf':>8} {'trad rec':>9} {'AI rec':>7}")
    for k, v in levels.items():
        print(f"{k:>6} {v['traditional']['rul_rmse']['0.9']:12.0f} "
              f"{v['ai']['rul_rmse']['0.9']:10.0f} "
              f"{v['traditional']['confusable_isolation']:10.2f} "
              f"{v['ai']['confusable_isolation']:8.2f} "
              f"{v['traditional']['recall']:9.2f} {v['ai']['recall']:7.2f}")
    print()
    for k, v in verdicts.items():
        print(f"{k}: {'CONFIRMED' if v['confirmed'] else 'REFUTED'} — {v['criterion']}")
    print(f'-> {F5 / "noise_sweep.json"}')


if __name__ == '__main__':
    main()
