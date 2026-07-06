"""F-OPS (prereg-v12): unscheduled->scheduled conversion per method.

Reads the RUL prediction errors already produced by F5 (no retraining) and turns
them into an operational KPI: what fraction of run-to-failure removals each
approach converts to scheduled removals, at a logistics horizon L, net of
wasteful early removals, against the F11 aleatoric-floor ceiling.

Conversion condition: an engine reaches its booked slot without failing iff the
RUL over-prediction e = pred - true satisfies e <= L. It is wastefully early iff
e < -W. Net conversion = fraction with -W <= e <= L (false-alarm-adjusted).

Output: data/processed/f_ops/conversion.json + figure. Usage:
uv run python scripts/f_ops_conversion.py
"""
import json
from math import erf
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
F5 = REPO / 'data' / 'processed' / 'f5' / 'rul_errors.json'
OUT = REPO / 'data' / 'processed' / 'f_ops'
FIG = REPO / 'paper' / 'report' / 'figures'
FRACS = ('0.5', '0.7', '0.9')
LS = (200, 400, 800)
L_NOM, W = 400, 800
FLOOR = {'0.5': 1053.0, '0.7': 615.0, '0.9': 212.0}   # F11 aleatoric floor sigma
RNG = np.random.default_rng(12)


def phi(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def bca_ci(mask, n_boot=5000):
    """BCa 95% CI for the mean of a 0/1 array (net-conversion indicator)."""
    from scipy.stats import norm
    x = mask.astype(float)
    n = len(x)
    if n < 2:
        return [float('nan'), float('nan')]
    theta = x.mean()
    boot = np.array([RNG.choice(x, n, replace=True).mean() for _ in range(n_boot)])
    frac_lt = np.mean(boot < theta)
    z0 = norm.ppf(frac_lt) if 0 < frac_lt < 1 else 0.0
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jm = jack.mean()
    denom = 6 * (((jm - jack) ** 2).sum() ** 1.5)
    a = ((jm - jack) ** 3).sum() / denom if denom != 0 else 0.0

    def pct(alpha):
        zc = norm.ppf(alpha)
        z = z0 + (z0 + zc) / (1 - a * (z0 + zc))
        return float(norm.cdf(z))
    return [float(np.quantile(boot, pct(0.025))),
            float(np.quantile(boot, pct(0.975)))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(F5))
    res = {}
    for f in FRACS:
        tr = np.array(d['traditional'][f], float)
        ai = np.array(d['ai'][f], float)
        sig = FLOOR[f]
        entry = {'n': len(ai), 'floor_sigma': sig, 'by_L': {}}
        for L in LS:
            row = {}
            for name, e in (('trad', tr), ('ai', ai)):
                gross = float((e <= L).mean())
                waste = float((e < -W).mean())
                netmask = (e <= L) & (e >= -W)
                net = float(netmask.mean())
                row[name] = {'gross': gross, 'wasteful': waste, 'net': net,
                             'net_ci': bca_ci(netmask)}
            row['baseline'] = 0.0
            row['ceiling'] = float(phi(L / sig))
            row['ceiling_gap_ai'] = row['ceiling'] - row['ai']['gross']
            entry['by_L'][str(L)] = row
        res[f] = entry

    nom = {f: res[f]['by_L'][str(L_NOM)] for f in FRACS}
    h1 = all(nom[f]['ai']['net'] > nom[f]['trad']['net'] for f in FRACS)
    h2 = nom['0.9']['ceiling_gap_ai'] > nom['0.5']['ceiling_gap_ai']
    h3 = (nom['0.5']['ai']['net'] <= 0.6 * nom['0.5']['ai']['gross']
          and nom['0.5']['ai']['net'] > nom['0.5']['trad']['net'])
    verdict = {
        'params': {'L_nominal': L_NOM, 'W': W, 'Ls': LS},
        'per_fraction': res,
        'H-OPS.1_ai_converts_more_net': {
            'net_ai': {f: nom[f]['ai']['net'] for f in FRACS},
            'net_trad': {f: nom[f]['trad']['net'] for f in FRACS},
            'confirmed': bool(h1)},
        'H-OPS.2_headroom_late': {
            'ceiling_gap_50': nom['0.5']['ceiling_gap_ai'],
            'ceiling_gap_90': nom['0.9']['ceiling_gap_ai'],
            'confirmed': bool(h2)},
        'H-OPS.3_honest_downside_survives': {
            'gross_ai_50': nom['0.5']['ai']['gross'],
            'net_ai_50': nom['0.5']['ai']['net'],
            'confirmed': bool(h3)},
    }
    (OUT / 'conversion.json').write_text(json.dumps(verdict, indent=2))

    _figure(res)
    for f in FRACS:
        n = res[f]['by_L'][str(L_NOM)]
        print(f"{f}: net trad {n['trad']['net']:.0%}  net ai {n['ai']['net']:.0%}  "
              f"gross ai {n['ai']['gross']:.0%}  ceiling {n['ceiling']:.0%}  "
              f"gap {n['ceiling_gap_ai']:.0%}")
    print(f"H-OPS.1 {h1}  |  H-OPS.2 {h2}  |  H-OPS.3 {h3}")


def _figure(res):
    INK, BLUE, RED, GREY = '#212529', '#4263EB', '#A61E4D', '#ADB5BD'
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'axes.grid.axis': 'y',
                         'grid.color': '#E9ECEF', 'figure.dpi': 150})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, f in zip(axes, FRACS):
        Ls = np.array(LS)
        ai = [res[f]['by_L'][str(L)]['ai']['net'] * 100 for L in LS]
        tr = [res[f]['by_L'][str(L)]['trad']['net'] * 100 for L in LS]
        ceil = [res[f]['by_L'][str(L)]['ceiling'] * 100 for L in LS]
        ax.fill_between(Ls, ceil, 100, color=GREY, alpha=0.18, lw=0,
                        label='above floor ceiling\n(unreachable)')
        ax.plot(Ls, ceil, color=GREY, ls=':', lw=1.2, label='floor ceiling')
        ax.plot(Ls, ai, 'o-', color=BLUE, lw=1.8, ms=4, label='AI (net)')
        ax.plot(Ls, tr, 's-', color=RED, lw=1.6, ms=4, label='traditional (net)')
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f'inspection at {int(float(f)*100)}\\% life', fontsize=8)
        ax.set_xlabel('logistics horizon $L$ [cy]')
        ax.set_xticks(LS)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel('unscheduled$\\to$scheduled [\\%]')
    axes[-1].legend(frameon=False, fontsize=6.3, loc='center right')
    fig.tight_layout()
    fig.savefig(FIG / 'ops_conversion.pdf')
    plt.close(fig)
    print('  ops_conversion figure done')


if __name__ == '__main__':
    main()
