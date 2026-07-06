"""F11 (prereg-v11): the prognostic irreducibility floor.

Decomposes remaining-life uncertainty into ALEATORIC (irreducible: the engine's
future is stochastic) and EPISTEMIC (reducible with better data/models),
validated against ground truth. The aleatoric floor at a life fraction = the
spread of TRUE remaining life among engines with near-identical TRUE current
health (kNN on the 10-dim health state) -- conditioning on the exact current
state removes epistemic ignorance, so the residual spread is irreducible.

Foreground. Output: data/processed/f11/prognostic_floor.json + report figure.
Usage: uv run python scripts/f11_prognostic_floor.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache                              # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                 # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'f11'
FIG = REPO_ROOT / 'paper' / 'report' / 'figures'
FRACS = (0.5, 0.7, 0.9)
K = 10
# best achieved RMSE per fraction (F5 AI is best; from verdicts, disclosed)
AI_RMSE = {0.5: 1694.0, 0.7: 1042.0, 0.9: 858.0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = fleet_cache()
    fleet = c['fleet']
    Xc = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]
    snap = pd.read_parquet(FLEET / 'snapshots.parquet',
                           columns=['engine_id', 'cycle'] + Xc)

    res = {}
    for frac in FRACS:
        X, REM = [], []
        for eid, v in fleet.items():
            e = snap[snap.engine_id == eid].sort_values('cycle')
            i = int(frac * v['life'])
            if i >= len(e):
                continue
            X.append(e[Xc].to_numpy()[i]); REM.append(v['life'] - i)
        X = np.array(X); REM = np.array(REM, float)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
        floors = []
        for i in range(len(Xn)):
            d = np.sum((Xn - Xn[i]) ** 2, axis=1)
            nn = np.argsort(d)[:K]
            floors.append(np.std(REM[nn]))
        floor = float(np.median(floors))
        marg = float(REM.std())
        res[frac] = {'aleatoric_floor': floor, 'marginal_std': marg,
                     'best_rmse': AI_RMSE[frac],
                     'irreducible_share': floor / marg,
                     'method_over_floor': AI_RMSE[frac] / floor}

    h11_1 = bool(res[0.5]['irreducible_share'] >= 0.60)
    h11_2 = bool(res[0.9]['method_over_floor'] > res[0.5]['method_over_floor'])
    verdict = {
        'per_fraction': {str(f): res[f] for f in FRACS},
        'H11.1_floor_dominant_early': {
            'irreducible_share_50': res[0.5]['irreducible_share'],
            'confirmed': h11_1},
        'H11.2_headroom_late': {
            'method_over_floor_50': res[0.5]['method_over_floor'],
            'method_over_floor_90': res[0.9]['method_over_floor'],
            'confirmed': h11_2},
    }
    (OUT / 'prognostic_floor.json').write_text(json.dumps(verdict, indent=2))

    # figure: stacked aleatoric floor + epistemic gap per life fraction
    INK, BLUE, RED = '#212529', '#4263EB', '#A61E4D'
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#E9ECEF',
                         'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = np.arange(len(FRACS))
    floors = [res[f]['aleatoric_floor'] for f in FRACS]
    gaps = [res[f]['best_rmse'] - res[f]['aleatoric_floor'] for f in FRACS]
    ax.bar(xs, floors, color=RED, label='aleatoric floor (irreducible)')
    ax.bar(xs, gaps, bottom=floors, color=BLUE, alpha=0.55,
           label='epistemic gap (reducible)')
    for i, f in enumerate(FRACS):
        ax.annotate(f"{res[f]['best_rmse']:.0f}", (i, res[f]['best_rmse']),
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(['50\\%', '70\\%', '90\\%'])
    ax.set_xlabel('life fraction at prediction')
    ax.set_ylabel('RUL error [cycles]')
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title('early: mostly irreducible; late: headroom to improve', fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'prognostic_floor.pdf'); plt.close(fig)

    for f in FRACS:
        r = res[f]
        print(f"{f:.0%}: floor {r['aleatoric_floor']:.0f}  best {r['best_rmse']:.0f}  "
              f"irreducible {r['irreducible_share']:.0%}  method/floor {r['method_over_floor']:.1f}x")
    print(f"H11.1 floor dominant early: {h11_1}  |  H11.2 headroom late: {h11_2}")


if __name__ == '__main__':
    main()
