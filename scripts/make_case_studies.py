"""F6: five narrated case studies from the TEST fleet, with figures.

Engines are selected by criterion from the frozen fleet artifacts; both tuned
families are evaluated once (same code as F5) to annotate each case with what
each pipeline saw/said. Figures to paper/report/figures/case_*.pdf; facts to
data/processed/f6/case_facts.json (norm N4: chapter numbers come from there).

Foreground (MPS). Usage: uv run python scripts/make_case_studies.py
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
from tune_f5 import (eval_ai, eval_trad, fleet_cache, split_ids,
                     trad_detect_engine)                     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
F6 = REPO_ROOT / 'data' / 'processed' / 'f6'
FIG = REPO_ROOT / 'paper' / 'report' / 'figures'

INK, BLUE, RED, GRAY = '#212529', '#4263EB', '#A61E4D', '#868E96'
plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#E9ECEF',
                     'figure.dpi': 150})


def engine_frame(eid):
    return pd.read_parquet(FLEET / 'snapshots.parquet',
                           filters=[('engine_id', '==', eid)]).sort_values('cycle')


def main():
    F6.mkdir(parents=True, exist_ok=True)
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    c = fleet_cache()
    fleet = c['fleet']
    test_ids = split_ids(fleet, 'test')
    events = pd.read_parquet(FLEET / 'events.parquet')
    index = {e['engine_id']: e for e in json.loads(
        (FLEET / 'fleet_index.json').read_text())['engines']}

    print('evaluating tuned families on test (case annotations)...', flush=True)
    trad_iso = eval_trad(sel_t['isolation']['params'], 'test')
    ai_iso = eval_ai(sel_a['isolation']['params'], 'train', 'test', seed=0)
    ai_rul = eval_ai(sel_a['rul']['params'], 'train', 'test', seed=0)
    trad_rul = eval_trad(sel_t['rul']['params'], 'test')

    facts = {}

    def ev_of(eid, typ):
        return events[(events.engine_id == eid) & (events.type == typ)]

    # ---- Case A: fouling + washes (clean engine) --------------------------
    clean = [e for e in test_ids if not fleet[e]['episodes']
             and not index[e]['drift_channel']]
    eid_a = clean[0]
    e = engine_frame(eid_a)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(e.cycle / 1000, e.egtm_C, color=BLUE, lw=1.2)
    for _, w in ev_of(eid_a, 'wash').iterrows():
        ax.axvline(w.cycle / 1000, color=GRAY, lw=0.5)
    ax.set_xlabel('Flight cycles [thousands]')
    ax.set_ylabel('EGT margin [°C]')
    fig.tight_layout()
    fig.savefig(FIG / 'case_a_fouling.pdf')
    plt.close(fig)
    facts['A'] = {'engine': int(eid_a), 'life': int(fleet[eid_a]['life']),
                  'n_washes': int(len(ev_of(eid_a, 'wash')))}

    # ---- Case B: prognosis on a hot-section degrader ----------------------
    eid_b = max(clean, key=lambda e_: index[e_]['multipliers']['hot_section'])
    e = engine_frame(eid_b)
    life = fleet[eid_b]['life']
    t_pred = {r['frac']: r['err'] for r in trad_rul['rul_rows']
              if r['engine'] == eid_b}
    a_pred = {r['frac']: r['err'] for r in ai_rul['rul_rows']
              if r['engine'] == eid_b}
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(e.cycle / 1000, e.egtm_C, color=BLUE, lw=1.2, label='EGT margin')
    for f, color, lbl in ((0.5, GRAY, '50 %'), (0.7, '#7048E8', '70 %'),
                          (0.9, RED, '90 %')):
        ax.axvline(f * life / 1000, color=color, lw=0.7, ls='--')
    ax.set_xlabel('Flight cycles [thousands]')
    ax.set_ylabel('EGT margin [°C]')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'case_b_prognosis.pdf')
    plt.close(fig)
    facts['B'] = {'engine': int(eid_b), 'life': int(life),
                  'trad_err': {str(k): round(v) for k, v in t_pred.items()},
                  'ai_err': {str(k): round(v) for k, v in a_pred.items()},
                  'hot_mult': round(index[eid_b]['multipliers']['hot_section'], 2)}

    # ---- Case C: sensor drift --------------------------------------------
    drifted = [e for e in test_ids if index[e]['drift_channel'] == 'EGT_degK']
    eid_c = drifted[0]
    e = engine_frame(eid_c)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    meas = e.cr_EGT_degK - e.cr_EGT_degK_true
    sm = pd.Series(meas.to_numpy()).rolling(301, center=True, min_periods=30).mean()
    ax.plot(e.cycle / 1000, sm, color=RED, lw=1.2,
            label='measured minus true EGT (smoothed bias)')
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel('Flight cycles [thousands]')
    ax.set_ylabel('Sensor bias [K]')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'case_c_drift.pdf')
    plt.close(fig)
    facts['C'] = {'engine': int(eid_c),
                  'final_bias_K': float(sm.dropna().iloc[-1])}

    # ---- Case D: FOD step --------------------------------------------------
    fod_engines = [e for e in test_ids if len(ev_of(e, 'fod'))]
    eid_d = fod_engines[0]
    fodc = float(ev_of(eid_d, 'fod').cycle.iloc[0])
    dz = c['dev'][eid_d]
    det = trad_detect_engine(dz, sel_t['detection']['params'])
    e = engine_frame(eid_d)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    sm = pd.Series(dz[:, 2]).rolling(151, center=True, min_periods=20).mean()
    ax.plot(e.cycle / 1000, sm, color=BLUE, lw=1.2, label='cruise EGT deviation [%]')
    ax.axvline(fodc / 1000, color=INK, ls='--', lw=0.9, label='FOD event')
    if det is not None:
        ax.axvline(det / 1000, color=RED, ls=':', lw=1.2, label='traditional alarm')
    ax.set_xlabel('Flight cycles [thousands]')
    ax.set_ylabel('dEGT [%]')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'case_d_fod.pdf')
    plt.close(fig)
    facts['D'] = {'engine': int(eid_d), 'fod_cycle': int(fodc),
                  'trad_alarm': int(det) if det is not None else None,
                  'fod_step_pct': float(ev_of(eid_d, 'fod').step_eta_pct.iloc[0])}

    # ---- Case E: the confusable episode ------------------------------------
    def iso_answer(rows, eid):
        return [r for r in rows if r['engine'] == eid]

    conf_eng = [e for e in test_ids
                if any(p in ('hpc.eta', 'hpt.eta', 'hpt.flow')
                       for _, p in fleet[e]['episodes'])]
    eid_e = conf_eng[0]
    ep = [x for x in fleet[eid_e]['episodes']
          if x[1] in ('hpc.eta', 'hpt.eta', 'hpt.flow')][0]
    facts['E'] = {'engine': int(eid_e), 'param': ep[1], 'onset': int(ep[0]),
                  'trad_said': iso_answer(trad_iso['iso_rows'], eid_e),
                  'ai_said': [dict(r) for r in iso_answer(ai_iso['iso_rows'], eid_e)]}
    e = engine_frame(eid_e)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for col, color, lbl in ((f"x_{ep[1].replace('.', '_')}", RED, f'true {ep[1]}'),
                            ('x_hpc_eta' if ep[1] != 'hpc.eta' else 'x_hpt_eta',
                             BLUE, 'confusable partner')):
        ax.plot(e.cycle / 1000, e[col], color=color, lw=1.2, label=lbl)
    ax.axvline(ep[0] / 1000, color=INK, ls='--', lw=0.8)
    ax.set_xlabel('Flight cycles [thousands]')
    ax.set_ylabel('Health deviation [%]')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'case_e_confusable.pdf')
    plt.close(fig)

    (F6 / 'case_facts.json').write_text(json.dumps(facts, indent=2, default=str))
    print(json.dumps(facts, indent=1, default=str))


if __name__ == '__main__':
    main()
