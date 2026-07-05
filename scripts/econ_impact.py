"""F-ECON: economic-impact Monte Carlo, grounded in the MEASURED RUL errors.

The ONLY mechanism quantified is the one the results directly support: a
less-biased prognosis converts a fraction of UNSCHEDULED engine removals into
SCHEDULED ones. Everything else (fuel, capacity smoothing, OEM sensor value)
is discussed qualitatively in the chapter, not monetized here.

Two-layer model, deliberately conservative:
  (1) GROUNDING (transparency only): a raw removal rule -- trigger when predicted
      RUL <= L, unscheduled iff signed error e > L -- read from each family's
      EMPIRICAL late-life error distribution (tuned F5 configs, test fleet). On
      the synthetic fleet this gives trad ~40 %, AI ~0 % unscheduled. That
      OVERSTATES real practice (real trend monitoring uses conservative margins
      and more signals), so it is NOT used for the headline; it only establishes
      the DIRECTION and rough size of the improvement.
  (2) HEADLINE: the absolute unscheduled rate is ANCHORED to the industry band,
      and better prognosis converts a ranged, conservative fraction of those to
      scheduled. Savings = converted removals * shop_visit * (premium - 1).

All cost/policy inputs come from conf/econ_assumptions.yaml as RANGES; the
Monte Carlo propagates them into the output intervals. Parametric uncertainty
only -- structural risks (integration failure, false-alarm trust erosion, the
mechanism not materializing) live in the chapter's honest-limits section.

Foreground. Output: data/processed/econ/impact.json + report figure.
Usage: uv run python scripts/econ_impact.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import eval_trad, eval_ai, fleet_cache      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
OUT = REPO_ROOT / 'data' / 'processed' / 'econ'
FIG = REPO_ROOT / 'paper' / 'report' / 'figures'
MEAN_LIFE = 15139.0        # fleet median life [cycles] (data-derived, ch5)


def measured_errors():
    """Signed RUL errors [cycles] at 90 % life, tuned configs, test fleet."""
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    fleet_cache()
    mt = eval_trad(sel_t['rul']['params'], 'test')
    ma = eval_ai(sel_a['rul']['params'], 'train', 'test', seed=0)
    et = np.array([r['err'] for r in mt['rul_rows'] if abs(r['frac'] - 0.9) < 1e-6])
    ea = np.array([r['err'] for r in ma['rul_rows'] if abs(r['frac'] - 0.9) < 1e-6])
    return et, ea


def u(rng, lohi):
    return rng.uniform(lohi[0], lohi[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((REPO_ROOT / 'conf' / 'econ_assumptions.yaml').read_text())
    et, ea = measured_errors()
    cc, co, po, mc = (cfg['concrete_case'], cfg['costs_usd'],
                      cfg['policy'], cfg['monte_carlo'])
    n_eng = cc['n_aircraft'] * cc['engines_per_aircraft']
    rng = np.random.default_rng(mc['seed'])

    # Transparency: the RAW removal model on the synthetic fleet (a CEILING that
    # overstates real practice, hence NOT used for the headline).
    L_mid = float(np.mean(po['lead_time_cycles']))
    raw_uns_trad = float(np.mean(et > L_mid))
    raw_uns_ai = float(np.mean(ea > L_mid))
    # Measured relative reduction (direction + rough size) that grounds the
    # conversion-fraction prior; conservatively capped at the assumed range.
    raw_ratio = raw_uns_ai / raw_uns_trad if raw_uns_trad > 0 else 0.0
    measured_conversion = 1.0 - raw_ratio    # what the raw model implies

    savings, contribs = [], []
    for _ in range(mc['n_trials']):
        sv = u(rng, co['shop_visit'])
        prem = u(rng, co['unscheduled_premium_mult'])
        cyc = u(rng, cc['annual_cycles_per_engine'])
        integ = u(rng, co['integration_annual'])
        uns_base = u(rng, po['current_unscheduled_fraction'])   # anchored to industry
        conv = u(rng, po['conversion_fraction'])                # grounded-conservative
        annual_removals = n_eng * cyc / MEAN_LIFE
        converted = annual_removals * uns_base * conv           # unsched -> sched
        save = converted * sv * (prem - 1.0) - integ            # extra cost avoided
        savings.append(save)
        contribs.append({'sv': sv, 'prem': prem, 'cyc': cyc, 'integ': integ,
                         'uns_base': uns_base, 'conv': conv, 'save': save})
    savings = np.array(savings)

    # sensitivity: Spearman of each input vs the outcome (tornado)
    from scipy.stats import spearmanr
    keys = ['sv', 'prem', 'uns_base', 'conv', 'cyc', 'integ']
    sens = {k: float(spearmanr([c[k] for c in contribs], savings).statistic)
            for k in keys}

    report = {
        'concrete_case': f"{cc['n_aircraft']} aircraft, {n_eng} CFM56-7B engines",
        'mechanism': 'less-biased prognosis converts unscheduled removals to scheduled',
        'measured_grounding': {
            'n_test_engines': int(len(et)),
            'raw_model_unscheduled_trad': raw_uns_trad,
            'raw_model_unscheduled_ai': raw_uns_ai,
            'raw_implied_conversion': measured_conversion,
            'note': ('raw removal model on the synthetic fleet OVERSTATES real '
                     'practice; used only to ground the conversion direction. '
                     'The headline anchors the absolute rate to the industry band.')},
        'annual_savings_usd': {
            'p10': float(np.percentile(savings, 10)),
            'p50': float(np.percentile(savings, 50)),
            'p90': float(np.percentile(savings, 90)),
            'prob_net_negative': float(np.mean(savings < 0)),
            'per_engine_p50': float(np.percentile(savings, 50) / n_eng)},
        'sensitivity_spearman': sens,
    }
    (OUT / 'impact.json').write_text(json.dumps(report, indent=2))

    # figure: savings distribution + sensitivity tornado
    INK, BLUE, RED = '#212529', '#4263EB', '#A61E4D'
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax1.hist(savings / 1e6, bins=60, color=BLUE, alpha=0.8)
    ax1.axvline(0, color=RED, lw=1.2)
    ax1.axvline(np.median(savings) / 1e6, color=INK, lw=1.0, ls='--')
    ax1.set_xlabel('Annual fleet savings [\\$M]'.replace('\\', ''))
    ax1.set_ylabel('Monte-Carlo trials')
    ax1.set_title(f"median \\${np.median(savings)/1e6:.1f}M; "
                  f"P(net$<$0)={np.mean(savings<0):.0%}".replace('\\', ''), fontsize=8)
    labels = {'sv': 'shop-visit cost', 'prem': 'unsched. premium',
              'uns_base': 'base unsched. rate', 'conv': 'conversion fraction',
              'cyc': 'utilization', 'integ': 'integration cost'}
    order = sorted(keys, key=lambda k: abs(sens[k]))
    ax2.barh([labels[k] for k in order], [sens[k] for k in order],
             color=[RED if sens[k] < 0 else BLUE for k in order])
    ax2.axvline(0, color=INK, lw=0.6)
    ax2.set_xlabel('Sensitivity (Spearman with savings)')
    ax2.set_title('What drives the outcome', fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'econ_impact.pdf'); plt.close(fig)

    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
