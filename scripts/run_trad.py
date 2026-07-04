"""Run the traditional EHM pipeline over the fleet (phase F3).

Per engine (one worker per engine, norm N1): cruise-deviation smoothing,
CUSUM event detection, nearest-signature isolation of the detected step,
Kalman-GPA health tracking (wash-aware), and Theil-Sen RUL from the tracked
takeoff EGT margin. Metrics aggregated on the TEST split only.

Output: data/processed/trad/trad_metrics.json (+ per-engine parquet)
Usage: uv run python scripts/run_trad.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.pipeline import (COCKPIT, BaselineModel, cusum, holt_smooth,
                                    gap_alert, isolate_step, kalman_gpa,
                                    theil_sen_rul)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'
OUT = REPO_ROOT / 'data' / 'processed' / 'trad'

R_DIAG = [0.07, 0.5, 0.23]          # % noise sigmas: N2, WF, EGT (2.5 K / ~1100 K)
EGTM_NOMINAL = 85.0
EVAL_FRACS = (0.5, 0.7, 0.9)
X_COLS = [f'x_{p.replace(".", "_")}' for p in HEALTH_PARAMS]


def takeoff_egt_baseline(dts_c):
    """EGT takeoff baseline interpolated in dTs between the two ICM points."""
    _, ch, b0 = load_icm('takeoff')
    _, _, b30 = load_icm('takeoff_hot')
    w = np.asarray(dts_c, float) / 30.0
    return b0['EGT_degK'] * (1 - w) + b30['EGT_degK'] * w


def process_engine(args):
    eid, group_path, events_rows = args
    e = pd.read_parquet(group_path, filters=[('engine_id', '==', eid)])
    e = e.sort_values('cycle').reset_index(drop=True)
    n = len(e)
    bm = BaselineModel()

    measured = e[[f'cr_{c}' for c in COCKPIT]].to_numpy(float)
    dz = bm.deviations(measured, e.cr_N1_cmd.to_numpy())
    sm = [holt_smooth(dz[:, j]) for j in range(3)]
    dz_s = np.column_stack([sm[j][0] for j in range(3)])

    # --- detection: CUSUM on Holt innovations (steps) OR trend-shift (ramps).
    # Running CUSUM on the raw deviation would integrate the chronic trend and
    # alarm on every engine mid-life (observed in the first run).
    det_step = cusum(sm[2][2], drift_k=0.75, h=8.0)
    det_ramp_egt = gap_alert(dz[:, 2])
    det_ramp_wf = gap_alert(dz[:, 1])
    det = min([d for d in (det_step, det_ramp_egt, det_ramp_wf) if d is not None],
              default=None)

    # --- isolation, oracle-timed per episode (v1.1 protocol: the isolation
    # question is asked at onset+500 for BOTH families, decoupled from each
    # family's detection performance) ---
    acutes = sorted([r for r in events_rows if r['type'] == 'acute'],
                    key=lambda r: r['cycle'])
    bounds = [int(r['cycle']) for r in acutes] + [n]
    Ha, Hb, w = bm.cruise(e.cr_N1_cmd.to_numpy())[1]
    episode_iso = []
    for k, r in enumerate(acutes):
        onset = int(r['cycle'])
        t = min(bounds[k + 1] - 1, n - 1, onset + 500)
        if t <= onset + 100:
            continue
        pre = np.nanmean(dz_s[max(0, onset - 300):max(1, onset - 20)], axis=0)
        post = np.nanmean(dz_s[max(0, t - 300):t], axis=0)
        Ht = Ha * (1 - w[t]) + Hb * w[t]
        episode_iso.append({'param': r['param'],
                            'isolated': isolate_step(post - pre, Ht)})

    # --- Kalman-GPA health tracking ---
    washes = [r['cycle'] for r in events_rows if r['type'] == 'wash']
    xs = kalman_gpa(dz, lambda i: Ha * (1 - w[i]) + Hb * w[i], R_DIAG,
                    q=2e-4, wash_cycles=washes)
    x_true = e[X_COLS].to_numpy(float)
    kf_rmse = float(np.sqrt(np.nanmean((xs[n // 2:] - x_true[n // 2:]) ** 2)))

    # smearing on acute engines: share of |x_hat| on healthy components
    acute = acutes[0] if acutes else None
    smear = None
    if acute is not None:
        j_true = HEALTH_PARAMS.index(acute['param'])
        onset = int(np.ceil(acute['cycle']))
        seg = xs[min(n - 1, onset + 500):]
        step_est = np.nanmean(seg, axis=0) - np.nanmean(xs[max(0, onset - 500):onset], axis=0)
        tot = np.sum(np.abs(step_est))
        smear = float(1.0 - abs(step_est[j_true]) / tot) if tot > 0 else None

    # --- RUL from the tracked takeoff EGT margin ---
    to_base = takeoff_egt_baseline(e.to_dTs_C.to_numpy())
    dev_K = e.to_EGT_degK.to_numpy(float) - to_base
    dev_s, _, _ = holt_smooth(dev_K, alpha=0.08, beta=0.02)
    egtm_est = EGTM_NOMINAL - dev_s
    rul_rows = []
    for frac in EVAL_FRACS:
        i = int(frac * n)
        pred = theil_sen_rul(egtm_est[:i])
        if pred is not None:
            pred = min(pred, 25000.0)      # horizon cap (NASA-style)
        rul_rows.append({'frac': frac, 'rul_true': n - i,
                         'rul_pred': pred})

    return {'engine_id': int(eid), 'n': n, 'split': e.split.iloc[0],
            'det_cycle': det, 'episode_iso': episode_iso,
            'acute_param': acute['param'] if acute else None,
            'acute_onset': float(acute['cycle']) if acute else None,
            'kf_rmse_pct': kf_rmse, 'smearing_index': smear,
            'rul': rul_rows}


def nasa_score(d):
    return float(np.exp(-d / 13.0) - 1.0) if d < 0 else float(np.exp(d / 10.0) - 1.0)


def aggregate(results):
    test = [r for r in results if r['split'] == 'test']
    acute = [r for r in test if r['acute_param']]
    clean = [r for r in test if not r['acute_param']]

    detected = [r for r in acute if r['det_cycle'] is not None
                and r['det_cycle'] >= r['acute_onset']]
    delays = [r['det_cycle'] - r['acute_onset'] for r in detected]
    false_alarms = [r for r in clean if r['det_cycle'] is not None]

    episodes = [ep for r in test for ep in r['episode_iso']]
    iso_ok = [ep for ep in episodes if ep['isolated'] == ep['param']]
    confus = [ep for ep in episodes
              if ep['param'] in ('hpc.eta', 'hpt.eta', 'hpt.flow')]
    confus_ok = [ep for ep in confus if ep['isolated'] == ep['param']]

    rul_errs, scores = {f: [] for f in EVAL_FRACS}, {f: [] for f in EVAL_FRACS}
    for r in test:
        for row in r['rul']:
            if row['rul_pred'] is not None:
                d = row['rul_pred'] - row['rul_true']
                rul_errs[row['frac']].append(d)
                scores[row['frac']].append(nasa_score(d / 100.0))  # d in units of 100 cycles

    return {
        'n_test': len(test), 'n_acute_test': len(acute),
        'detection': {
            'recall': len(detected) / len(acute) if acute else None,
            'median_delay_cycles': float(np.median(delays)) if delays else None,
            'false_alarm_engines': len(false_alarms), 'n_clean': len(clean)},
        'isolation': {
            'accuracy': len(iso_ok) / len(episodes) if episodes else None,
            'confusable_accuracy': len(confus_ok) / len(confus) if confus else None,
            'n_confusable': len(confus), 'n_episodes_test': len(episodes)},
        'health_tracking': {
            'kf_rmse_pct_median': float(np.median([r['kf_rmse_pct'] for r in test])),
            'smearing_index_median': float(np.median(
                [r['smearing_index'] for r in acute if r['smearing_index'] is not None]))},
        'rul': {str(f): {
            'n': len(rul_errs[f]),
            'rmse_cycles': float(np.sqrt(np.mean(np.square(rul_errs[f])))) if rul_errs[f] else None,
            'median_err_cycles': float(np.median(rul_errs[f])) if rul_errs[f] else None,
            'nasa_score_mean': float(np.mean(scores[f])) if scores[f] else None}
            for f in EVAL_FRACS},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(FLEET / 'events.parquet')
    index = json.loads((FLEET / 'fleet_index.json').read_text())
    ids = [e['engine_id'] for e in index['engines']]
    ev_by_engine = {eid: events[events.engine_id == eid].to_dict('records')
                    for eid in ids}
    snap_path = str(FLEET / 'snapshots.parquet')

    jobs = [(eid, snap_path, ev_by_engine[eid]) for eid in ids]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(process_engine, jobs))

    metrics = aggregate(results)
    (OUT / 'trad_results.json').write_text(json.dumps(results, indent=1, default=float))
    (OUT / 'trad_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
