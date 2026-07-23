"""F5 tuning campaigns (prereg-v1 §4): 50 Optuna trials per family, TPE
sampler seed 0, objectives evaluated on the VALIDATION split only.

One trial samples a full family configuration and records every task metric
as user attributes; per-task winners are selected from the same 50-trial pool
(the frozen budget). SQLite storage so campaigns run in resumable batches
(torch-MPS requires foreground execution).

Shared evaluation functions here are reused verbatim by the confirmatory
script (f5_confirm.py) so tuning and confirmation exercise the same code.

Usage:
    uv run python scripts/tune_f5.py trad 50     # family, target trial count
    uv run python scripts/tune_f5.py ai 10       # ...repeat until 50
"""

import json
import sys
from pathlib import Path

import numpy as np

from ehmbrain.ai.data import load_fleet_features, normalization
from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.pipeline import (COCKPIT, BaselineModel, cusum, ewma,
                                    holt_smooth, isolate_step, theil_sen_rul)

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
FLEET_DIR = REPO_ROOT / 'data' / 'processed' / 'fleet'
_FLEET_DEFAULT = FLEET_DIR

CONFUSABLE = ('hpc.eta', 'hpt.eta', 'hpt.flow')
EVAL_FRACS = (0.5, 0.7, 0.9)
RUL_CAP_CY = 12000.0

# ---------------------------------------------------------------------------
# Shared engine data (loaded once per process)
# ---------------------------------------------------------------------------
_CACHE = {}


def use_fleet(fleet_dir):
    """Point the shared evaluators at another fleet directory and clear the
    cache. Used only by the noise sweep (C6); the F5/F7 paths never call it, so
    their frozen behaviour is unchanged."""
    global FLEET_DIR
    FLEET_DIR = Path(fleet_dir)
    _CACHE.clear()


def fleet_cache():
    if 'fleet' not in _CACHE:
        fleet = load_fleet_features(FLEET_DIR if FLEET_DIR != _FLEET_DEFAULT else None)
        _CACHE['fleet'] = fleet
        _CACHE['norm'] = normalization(fleet)
        bm = BaselineModel()
        _CACHE['bm'] = bm
        # per-engine cruise cockpit deviations + takeoff EGT dev (channel 3)
        import pandas as pd
        dev = {}
        for eid, v in fleet.items():
            dev[eid] = v['F']          # (n,4): 3 cruise dz % + takeoff EGT dev K
        _CACHE['dev'] = dev
        _CACHE['Hs'] = {}
        df_n1 = {}
        snap = pd.read_parquet(FLEET_DIR / 'snapshots.parquet',
                               columns=['engine_id', 'cycle', 'cr_N1_cmd'])
        for eid in fleet:
            e = snap[snap.engine_id == eid].sort_values('cycle')
            Ha, Hb, w = bm.cruise(e.cr_N1_cmd.to_numpy())[1]
            _CACHE['Hs'][eid] = (Ha, Hb, w)
    return _CACHE


def episodes_of(v):
    return v['episodes']


def split_ids(fleet, split):
    return sorted(e for e, v in fleet.items() if v['split'] == split)


# ---------------------------------------------------------------------------
# Traditional family: configurable evaluation
# ---------------------------------------------------------------------------

def trad_detect_engine(dz, cfg):
    _, _, innov = holt_smooth(dz[:, 2], cfg['holt_a'], cfg['holt_b'])
    d1 = cusum(innov, drift_k=cfg['cusum_k'], h=cfg['cusum_h'])
    alarms = [d for d in (d1,) if d is not None]
    for ch in (2, 1):
        fast, slow = ewma(dz[:, ch], cfg['gap_fast']), ewma(dz[:, ch], cfg['gap_slow'])
        gap = fast - slow
        base = gap[200:2500]
        med = np.nanmedian(base)
        mad = np.nanmedian(np.abs(base - med)) or 1e-9
        z = (gap - med) / (1.4826 * mad)
        exceed = z > cfg['gap_nsig']
        exceed[:2500] = False
        counts = np.convolve(exceed.astype(int), np.ones(cfg['pers_n'], int),
                             'full')[:len(z)]
        hits = np.nonzero(counts >= cfg['pers_k'])[0]
        if len(hits):
            alarms.append(int(hits[0]))
    return min(alarms) if alarms else None


def trad_isolate_episode(dz_s, H_at, onset, t, cfg):
    pre = np.nanmean(dz_s[max(0, onset - cfg['iso_pre']):max(1, onset - 20)], axis=0)
    post = np.nanmean(dz_s[max(0, t - cfg['iso_post']):t], axis=0)
    return isolate_step(post - pre, H_at, min_norm=cfg['iso_min_norm'])


def eval_trad(cfg, split):
    c = fleet_cache()
    fleet = c['fleet']
    ids = split_ids(fleet, split)
    hits = tot = fa = 0
    iso_rows = []
    rul_err = []
    for eid in ids:
        v = fleet[eid]
        dz = c['dev'][eid][:, :3]
        det = trad_detect_engine(c['dev'][eid], cfg)
        eps = episodes_of(v)
        if eps:
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, param) in enumerate(eps):
                tot += 1
                if det is not None and onset <= det < bounds[k + 1]:
                    hits += 1
            if det is not None and det < eps[0][0]:
                fa += 1
        elif det is not None:
            fa += 1
        # isolation (oracle-timed)
        lvl = np.column_stack([holt_smooth(dz[:, j], cfg['holt_a'], cfg['holt_b'])[0]
                               for j in range(3)])
        Ha, Hb, w = c['Hs'][eid]
        if eps:
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, param) in enumerate(eps):
                t = min(bounds[k + 1] - 1, v['life'] - 1, int(onset) + 500)
                if t <= int(onset) + 100:
                    continue
                Ht = Ha * (1 - w[t]) + Hb * w[t]
                pred = trad_isolate_episode(lvl, Ht, int(onset), t, cfg)
                iso_rows.append({'param': param, 'pred': pred, 'engine': eid})
        # RUL
        to_dev = c['dev'][eid][:, 3]
        dev_s, _, _ = holt_smooth(to_dev, cfg['rul_a'], cfg['rul_a'] / 3)
        egtm = 85.0 - dev_s
        for f in EVAL_FRACS:
            i = int(f * v['life'])
            pred = theil_sen_rul(egtm[:i], window=cfg['rul_win'])
            if pred is None:
                pred = RUL_CAP_CY
            pred = min(pred, 25000.0)
            rul_err.append({'engine': eid, 'frac': f,
                            'err': min(pred, RUL_CAP_CY) - min(v['life'] - i, RUL_CAP_CY)})
    recall = hits / tot if tot else 0.0
    iso_conf = [r for r in iso_rows if r['param'] in CONFUSABLE]
    conf_acc = (np.mean([r['pred'] == r['param'] for r in iso_conf])
                if iso_conf else 0.0)
    rmse = float(np.sqrt(np.mean([r['err'] ** 2 for r in rul_err])))
    return {'recall': recall, 'fa': fa, 'conf_acc': float(conf_acc),
            'rul_rmse': rmse, 'iso_rows': iso_rows, 'rul_rows': rul_err,
            'n_episodes': tot}


def trad_space(trial):
    return {
        'holt_a': trial.suggest_float('holt_a', 0.05, 0.3),
        'holt_b': trial.suggest_float('holt_b', 0.01, 0.1),
        'cusum_k': trial.suggest_float('cusum_k', 0.4, 1.2),
        'cusum_h': trial.suggest_float('cusum_h', 4.0, 12.0),
        'gap_fast': trial.suggest_float('gap_fast', 0.02, 0.2, log=True),
        'gap_slow': trial.suggest_float('gap_slow', 0.001, 0.01, log=True),
        'gap_nsig': trial.suggest_float('gap_nsig', 3.0, 8.0),
        'pers_k': trial.suggest_int('pers_k', 5, 20),
        'pers_n': trial.suggest_int('pers_n', 8, 28),
        'iso_pre': trial.suggest_int('iso_pre', 150, 500),
        'iso_post': trial.suggest_int('iso_post', 150, 500),
        'iso_min_norm': trial.suggest_float('iso_min_norm', 0.05, 0.3),
        'rul_a': trial.suggest_float('rul_a', 0.02, 0.15),
        'rul_win': trial.suggest_int('rul_win', 800, 3000),
    }


# ---------------------------------------------------------------------------
# AI family: configurable evaluation
# ---------------------------------------------------------------------------

def ai_step_features(F, t, short, long_w):
    pre = F[max(0, t - short - long_w):max(1, t - short)].mean(axis=0)
    post = F[max(0, t - short):t].mean(axis=0)
    slope = (F[max(0, t - short // 2):t].mean(axis=0)
             - F[max(0, t - short):max(1, t - short // 2)].mean(axis=0))
    return np.concatenate([post - pre, slope, post])


def eval_ai(cfg, fit_split, eval_split, seed=0):
    import torch
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import HistGradientBoostingClassifier
    from ehmbrain.ai.models import RULNet, predict_torch, train_torch

    c = fleet_cache()
    fleet = c['fleet']
    mu, sd = c['norm']
    rng = np.random.default_rng(1000 + seed)
    fit_ids = split_ids(fleet, fit_split)
    ev_ids = split_ids(fleet, eval_split)
    short, long_w = cfg['det_short'], cfg['det_long']

    def nf(eid):
        return (c['dev'][eid] - mu) / sd

    # --- detection: Mahalanobis on configurable-window step features ---
    Xtr = []
    for eid in fit_ids:
        v = fleet[eid]
        hi = int(v['episodes'][0][0]) if v['episodes'] else v['life']
        lo = short + long_w + 50
        if hi - 50 <= lo:
            continue
        for t in rng.integers(lo, hi - 50, size=40):
            Xtr.append(ai_step_features(nf(eid), int(t), short, long_w))
    lw = LedoitWolf().fit(np.array(Xtr))

    def maha_series(eid, stride=25):
        v = fleet[eid]
        ts = np.arange(short + long_w + 50, v['life'], stride)
        X = np.stack([ai_step_features(nf(eid), int(t), short, long_w)
                      for t in ts]) - lw.location_
        return ts, np.einsum('ij,jk,ik->i', X, lw.precision_, X)

    # threshold: FA budget on the FIT split's clean engines (selection data)
    pool = [maha_series(eid)[1] for eid in fit_ids
            if not fleet[eid]['episodes']]
    thr = float(np.percentile(np.concatenate(pool), cfg['det_pct']))

    hits = tot = fa = 0
    for eid in ev_ids:
        v = fleet[eid]
        ts, sc = maha_series(eid)
        exceed = sc > thr
        run = 0
        det = None
        for i, ex in enumerate(exceed):
            run = run + 1 if ex else 0
            if run >= cfg['det_k']:
                det = int(ts[i])
                break
        eps = episodes_of(v)
        if eps:
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, _) in enumerate(eps):
                tot += 1
                if det is not None and onset <= det < bounds[k + 1]:
                    hits += 1
            if det is not None and det < eps[0][0]:
                fa += 1
        elif det is not None:
            fa += 1

    # --- diagnosis ---
    CLASSES = ['none'] + [f'acute_{p}' for p in
                          ('fan.eta', 'lpc.eta', 'hpc.eta', 'hpt.eta',
                           'hpt.flow', 'lpt.eta')]
    Xd, yd = [], []
    for eid in fit_ids:
        v = fleet[eid]
        Fn = nf(eid)
        eps = episodes_of(v)
        if eps:
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, param) in enumerate(eps):
                lo = min(int(onset) + 500, bounds[k + 1] - 2)
                if lo >= bounds[k + 1]:
                    continue
                for t in rng.integers(lo, bounds[k + 1],
                                      size=cfg['diag_samples']):
                    Xd.append(ai_step_features(Fn, int(t), cfg['diag_short'],
                                               cfg['diag_long']))
                    yd.append(CLASSES.index(f'acute_{param}'))
            if bounds[0] - 100 > 900:
                for t in rng.integers(900, bounds[0] - 100, size=15):
                    Xd.append(ai_step_features(Fn, int(t), cfg['diag_short'],
                                               cfg['diag_long']))
                    yd.append(0)
        else:
            for t in rng.integers(900, v['life'], size=25):
                Xd.append(ai_step_features(Fn, int(t), cfg['diag_short'],
                                           cfg['diag_long']))
                yd.append(0)
    clf = HistGradientBoostingClassifier(
        max_iter=cfg['diag_iters'], max_depth=cfg['diag_depth'],
        learning_rate=cfg['diag_lr'], random_state=seed,
        class_weight='balanced')
    clf.fit(np.array(Xd), np.array(yd))

    iso_rows = []
    for eid in ev_ids:
        v = fleet[eid]
        eps = episodes_of(v)
        if not eps:
            continue
        bounds = [int(o) for o, _ in eps] + [v['life']]
        for k, (onset, param) in enumerate(eps):
            t = min(bounds[k + 1] - 1, v['life'] - 1, int(onset) + 500)
            if t <= int(onset) + 100:
                continue
            pred = CLASSES[int(clf.predict(
                ai_step_features(nf(eid), t, cfg['diag_short'],
                                 cfg['diag_long'])[None])[0])]
            iso_rows.append({'param': param,
                             'pred': pred.replace('acute_', ''),
                             'engine': eid})

    # --- RUL GRU ---
    ds, seq = cfg['rul_ds'], cfg['rul_seq']

    def sequences(ids, cuts_per):
        X, y, meta = [], [], []
        for eid in ids:
            v = fleet[eid]
            Fn = nf(eid)[::ds]
            cuts = rng.integers(seq * ds, v['life'], size=cuts_per)
            for cut in cuts:
                i = int(cut) // ds
                if i < seq:
                    continue
                X.append(Fn[i - seq:i])
                y.append(min((v['life'] - int(cut)) / 1000.0, 12.0))
                meta.append(eid)
        return np.array(X, np.float32), np.array(y, np.float32), meta

    Xr, yr, _ = sequences(fit_ids, 40)
    net = train_torch(RULNet(ch=4, hidden=cfg['rul_hidden'],
                             layers=cfg['rul_layers']),
                      Xr, yr, epochs=cfg['rul_epochs'], lr=cfg['rul_lr'],
                      seed=seed)
    cpu = torch.device('cpu')
    rul_rows = []
    for eid in ev_ids:
        v = fleet[eid]
        Fn = nf(eid)[::ds]
        for f in EVAL_FRACS:
            cut = int(f * v['life'])
            i = cut // ds
            if i < seq:
                continue
            pred = float(predict_torch(net, Fn[i - seq:i][None], dev=cpu)[0])
            rul_rows.append({'engine': eid, 'frac': f,
                             'err': (min(pred, 12.0)
                                     - min((v['life'] - cut) / 1000.0, 12.0)) * 1000.0})

    recall = hits / tot if tot else 0.0
    iso_conf = [r for r in iso_rows if r['param'] in CONFUSABLE]
    conf_acc = (np.mean([r['pred'] == r['param'] for r in iso_conf])
                if iso_conf else 0.0)
    rmse = float(np.sqrt(np.mean([r['err'] ** 2 for r in rul_rows])))
    return {'recall': recall, 'fa': fa, 'conf_acc': float(conf_acc),
            'rul_rmse': rmse, 'iso_rows': iso_rows, 'rul_rows': rul_rows,
            'n_episodes': tot, 'net': net, 'maha': (lw, thr, cfg), 'clf': clf}


def ai_space(trial):
    return {
        'det_short': trial.suggest_int('det_short', 100, 800),
        'det_long': trial.suggest_int('det_long', 300, 2000),
        'det_pct': trial.suggest_float('det_pct', 97.0, 99.9),
        'det_k': trial.suggest_int('det_k', 3, 12),
        'diag_short': trial.suggest_int('diag_short', 100, 500),
        'diag_long': trial.suggest_int('diag_long', 200, 1000),
        'diag_samples': trial.suggest_int('diag_samples', 15, 50),
        'diag_iters': trial.suggest_int('diag_iters', 150, 500),
        'diag_depth': trial.suggest_int('diag_depth', 3, 8),
        'diag_lr': trial.suggest_float('diag_lr', 0.02, 0.2, log=True),
        'rul_hidden': trial.suggest_categorical('rul_hidden', [32, 64, 96]),
        'rul_layers': trial.suggest_int('rul_layers', 1, 2),
        'rul_lr': trial.suggest_float('rul_lr', 3e-4, 3e-3, log=True),
        'rul_epochs': trial.suggest_int('rul_epochs', 15, 45),
        'rul_ds': trial.suggest_categorical('rul_ds', [10, 20, 30]),
        'rul_seq': trial.suggest_categorical('rul_seq', [32, 64, 96]),
    }


# ---------------------------------------------------------------------------
# Campaign driver
# ---------------------------------------------------------------------------

def objective_factory(family):
    def objective(trial):
        if family == 'trad':
            cfg = trad_space(trial)
            m = eval_trad(cfg, 'val')
        else:
            cfg = ai_space(trial)
            m = eval_ai(cfg, 'train', 'val')
        det_score = m['recall'] if m['fa'] <= 1 else -1.0
        trial.set_user_attr('recall', m['recall'])
        trial.set_user_attr('fa', m['fa'])
        trial.set_user_attr('det_score', det_score)
        trial.set_user_attr('conf_acc', m['conf_acc'])
        trial.set_user_attr('rul_rmse', m['rul_rmse'])
        # composite only guides TPE; per-task winners picked from the pool
        return det_score + m['conf_acc'] - m['rul_rmse'] / 5000.0
    return objective


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    family, target = sys.argv[1], int(sys.argv[2])
    F5.mkdir(parents=True, exist_ok=True)
    storage = f'sqlite:///{F5}/optuna.db'
    study = optuna.create_study(
        study_name=f'{family}-v1', storage=storage, load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0))
    done = len([t for t in study.trials if t.state.is_finished()])
    todo = max(0, target - done)
    print(f'{family}: {done} done, running {todo} more', flush=True)
    if todo:
        study.optimize(objective_factory(family), n_trials=todo,
                       show_progress_bar=False)

    finished = [t for t in study.trials if t.state.is_finished()]
    sel = {}

    def best_detection(trials):
        # prereg: max recall s.t. val FA <= 1. If infeasible across the whole
        # frozen pool, fall back lexicographically: min FA, then max recall
        # (interpretive rule for the infeasible case; disclosed in the report).
        feasible = [t for t in trials if t.user_attrs.get('fa', 99) <= 1]
        pool = feasible or trials
        return max(pool, key=lambda t: (-t.user_attrs.get('fa', 99),
                                        t.user_attrs.get('recall', -1)))

    for task, key, better in (('detection', None, None),
                              ('isolation', 'conf_acc', max),
                              ('rul', 'rul_rmse', min)):
        if task == 'detection':
            best = best_detection(finished)
        else:
            best = better(finished, key=lambda t: t.user_attrs.get(
                key, -1e9 if better is max else 1e9))
        sel[task] = {'trial': best.number, 'params': best.params,
                     'val': {k: best.user_attrs.get(k) for k in
                             ('recall', 'fa', 'conf_acc', 'rul_rmse')}}
    out = F5 / f'selected_{family}.json'
    out.write_text(json.dumps({'n_trials': len(finished), 'selected': sel},
                              indent=2))
    print(json.dumps({k: v['val'] for k, v in sel.items()}, indent=1))


if __name__ == '__main__':
    main()
