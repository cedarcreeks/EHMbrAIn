"""WP7.1/H7.1: observability recovery under REALISTIC in-service scatter.

The grid check (proposal) used 6 widely-spread designed points. The decisive
question: does the fleet's actual per-flight condition scatter — takeoff
dTs ~ clipped N(10, 8) °C and cruise N1 ~ U[4450, 4666] rpm, two snapshots per
flight — recover identifiability over a K-flight fusion window?

Method: draw K flights from the generator's condition distributions, build
the stacked cockpit ICM [H_to(dTs_1); H_cr(N1_1); ...] via the same linear
interpolation the generator itself uses, and measure rank and confusable-pair
angles as K grows. Repeated draws give scatter bands.

Output: data/processed/f7/observability.json + report figure.
Usage: uv run python scripts/f7_observability.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ICM = REPO_ROOT / 'data' / 'processed' / 'icm'
OUT = REPO_ROOT / 'data' / 'processed' / 'f7'
FIG = REPO_ROOT / 'paper' / 'report' / 'figures'

COCKPIT = ['N2_rpm', 'WF_kgps', 'EGT_degK']
PAIRS = [('hpc.eta', 'hpt.eta'), ('hpt.eta', 'hpt.flow'), ('fan.eta', 'lpt.eta')]
KS = [1, 2, 4, 8, 16, 32, 64]
N_DRAWS = 200


def load(pt):
    z = np.load(ICM / f'icm_{pt}.npz', allow_pickle=True)
    ch = [str(c) for c in z['channels']]
    return z['H'][[ch.index(c) for c in COCKPIT]], [str(p) for p in z['params']]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    H_to0, params = load('takeoff')
    H_to30, _ = load('takeoff_hot')
    H_cr, _ = load('cruise')
    H_crlo, _ = load('cruise_lowpwr')
    idx = {p: i for i, p in enumerate(params)}
    rng = np.random.default_rng(77)

    def draw_flight():
        dts = float(np.clip(rng.normal(10.0, 8.0), -20.0, 35.0))
        n1 = float(rng.uniform(4450.0, 4666.0))
        w_to = np.clip(dts / 30.0, 0.0, 1.0)     # interp clamps like the generator
        w_cr = (n1 - 4666.0) / (4400.0 - 4666.0)
        Hto = H_to0 * (1 - w_to) + H_to30 * w_to
        Hcr = H_cr * (1 - w_cr) + H_crlo * w_cr
        return Hto, Hcr

    def angle(H, a, b):
        u, v = H[:, idx[a]], H[:, idx[b]]
        return float(np.degrees(np.arccos(np.clip(
            abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1))))

    results = {}
    for K in KS:
        ranks, angs = [], {f'{a}~{b}': [] for a, b in PAIRS}
        for _ in range(N_DRAWS):
            blocks = []
            for _ in range(K):
                Hto, Hcr = draw_flight()
                blocks += [Hto, Hcr]
            Hs = np.vstack(blocks)
            ranks.append(int(np.linalg.matrix_rank(Hs, tol=1e-6)))
            for a, b in PAIRS:
                angs[f'{a}~{b}'].append(angle(Hs, a, b))
        results[str(K)] = {
            'rank_median': float(np.median(ranks)),
            'rank_p10': float(np.percentile(ranks, 10)),
            'angles': {k: {'median': float(np.median(v)),
                           'p10': float(np.percentile(v, 10)),
                           'p90': float(np.percentile(v, 90))}
                       for k, v in angs.items()}}
        print(f'K={K:3d} flights  rank {results[str(K)]["rank_median"]:.0f} '
              + '  '.join(f'{k}:{results[str(K)]["angles"][k]["median"]:.2f}°'
                          for k in angs), flush=True)

    # H7.1 verdict on realistic scatter
    base = results['1']['angles']
    final = results['64']['angles']
    verdict = {
        'rank_recovered_ge8': results['64']['rank_p10'] >= 8,
        'u_breakable_pairs_2x': {
            k: final[k]['median'] >= 2.0 * base[k]['median']
            for k in base},
    }
    out = {'results': results, 'h71_verdict': verdict,
           'scatter': 'takeoff dTs clipped N(10,8) C; cruise N1 U[4450,4666] rpm'}
    (OUT / 'observability.json').write_text(json.dumps(out, indent=2))

    # figure
    INK, BLUE, RED, TEAL = '#212529', '#4263EB', '#A61E4D', '#1098AD'
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#E9ECEF',
                         'figure.dpi': 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax1.plot(KS, [results[str(k)]['rank_median'] for k in KS], 'o-', color=BLUE)
    ax1.axhline(10, color=INK, lw=0.6, ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Fusion window [flights]')
    ax1.set_ylabel('Identifiable rank /10')
    for (k, color) in zip(['hpc.eta~hpt.eta', 'hpt.eta~hpt.flow',
                           'fan.eta~lpt.eta'], (RED, TEAL, BLUE)):
        med = [results[str(K)]['angles'][k]['median'] for K in KS]
        lo = [results[str(K)]['angles'][k]['p10'] for K in KS]
        hi = [results[str(K)]['angles'][k]['p90'] for K in KS]
        ax2.plot(KS, med, 'o-', color=color, label=k, lw=1.3, markersize=3)
        ax2.fill_between(KS, lo, hi, color=color, alpha=0.15, lw=0)
    ax2.axhline(15, color=INK, lw=0.6, ls=':')
    ax2.set_xscale('log')
    ax2.set_xlabel('Fusion window [flights]')
    ax2.set_ylabel('Signature angle [deg]')
    ax2.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / 'f7_observability.pdf')
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
