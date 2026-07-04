"""Run the AI EHM suite (phase F4, v0) and score it with the same protocol
as the traditional pipeline.

Tasks:
  detection   WindowAE trained on acute-free train windows; threshold set on
              val clean engines (max score quantile), persistence like trad.
  diagnosis   XGBoost on step features at evaluation time onset+500 (the same
              timing the traditional isolation rule gets).
  RUL         GRU on downsampled deviation history (MPS), target capped;
              split-conformal 90 % intervals calibrated on val.

Outputs: data/processed/ai/ai_metrics.json
Usage: uv run python scripts/run_ai.py
"""

import faulthandler
import json
from pathlib import Path

faulthandler.enable()

import numpy as np

from ehmbrain.ai.data import load_fleet_features, normalization, windows_from
from ehmbrain.ai.models import (RULNet, WindowAE, ae_scores, device,
                                predict_torch, train_torch)
from ehmbrain.perf.icm import HEALTH_PARAMS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'ai'

W_AE = 50
AE_STRIDE = 25
RUL_DS = 20          # downsample factor
RUL_SEQ = 64         # sequence length -> covers 1280 cycles
RUL_CAP = 12.0       # kcycles
EVAL_FRACS = (0.5, 0.7, 0.9)
CLASSES = ['none'] + [f'acute_{p}' for p in HEALTH_PARAMS
                      if p in ('fan.eta', 'lpc.eta', 'hpc.eta',
                               'hpt.eta', 'hpt.flow', 'lpt.eta')]


def norm_f(F, mu, sd):
    return (F - mu) / sd


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def detection_task(fleet, mu, sd):
    Xtr = []
    for v in fleet.values():
        if v['split'] != 'train':
            continue
        hi = int(v['acute_onset']) - W_AE if v['acute_param'] else v['life']
        _, w = windows_from(norm_f(v['F'], mu, sd), W_AE, AE_STRIDE, hi=hi)
        Xtr.append(w)
    Xtr = np.concatenate(Xtr)
    print(f'AE train windows: {len(Xtr):,}', flush=True)
    ae = train_torch(WindowAE(W_AE), Xtr, epochs=20, verbose=True)
    # Inference on CPU: MPS segfaults when switching to large-batch inference
    # after a long training session (torch 2.12, macOS); the AE is tiny, so
    # CPU scoring costs seconds. Training stays on the GPU.
    import torch
    cpu = torch.device('cpu')   # inference on CPU (MPS large-batch inference
                                # after long training segfaults on torch 2.12)

    def engine_scores(v):
        ends, w = windows_from(norm_f(v['F'], mu, sd), W_AE, 5)
        return ends, ae_scores(ae, w, dev=cpu)

    # threshold: val clean engines, no false alarm allowed at k-consecutive
    val_max = []
    for v in fleet.values():
        if v['split'] == 'val' and not v['acute_param']:
            _, s = engine_scores(v)
            val_max.append(np.percentile(s, 99.9))
    thr = max(val_max) * 1.05

    def first_alarm(v, k=6):
        ends, s = engine_scores(v)
        exceed = s > thr
        run = 0
        for i, ex in enumerate(exceed):
            run = run + 1 if ex else 0
            if run >= k:
                return int(ends[i])
        return None

    det = {}
    for eid, v in fleet.items():
        if v['split'] == 'test':
            det[eid] = first_alarm(v)
    return det, float(thr)


def detection_task_if(fleet, mu, sd, rng):
    """Isolation Forest on step features. The naive window-AE scores recall 0
    (it learns a near-identity map and reconstructs post-ramp windows
    perfectly - recorded as a finding); step features expose exactly the
    change the acute episodes make."""
    from sklearn.ensemble import IsolationForest
    Xtr = []
    for v in fleet.values():
        if v['split'] != 'train':
            continue
        Fn = norm_f(v['F'], mu, sd)
        hi = int(v['acute_onset']) if v['acute_param'] else v['life']
        for t in rng.integers(700, max(701, hi - 50), size=60):
            Xtr.append(step_features(Fn, int(t)))
    forest = IsolationForest(n_estimators=300, contamination='auto',
                             random_state=0).fit(np.array(Xtr))

    def first_alarm(v, k=8, stride=10):
        Fn = norm_f(v['F'], mu, sd)
        ts = np.arange(700, v['life'], stride)
        scores = -forest.score_samples(np.stack([step_features(Fn, int(t))
                                                 for t in ts]))
        exceed = scores > first_alarm.thr
        run = 0
        for i, ex in enumerate(exceed):
            run = run + 1 if ex else 0
            if run >= k:
                return int(ts[i])
        return None

    # threshold: val clean engines, worst score seen + margin
    val_scores = []
    for v in fleet.values():
        if v['split'] == 'val' and not v['acute_param']:
            Fn = norm_f(v['F'], mu, sd)
            ts = np.arange(700, v['life'], 10)
            val_scores.append(-forest.score_samples(
                np.stack([step_features(Fn, int(t)) for t in ts])))
    first_alarm.thr = float(np.percentile(np.concatenate(val_scores), 99.8))

    det = {}
    for eid, v in fleet.items():
        if v['split'] == 'test':
            det[eid] = first_alarm(v)
    return det, first_alarm.thr


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------

def step_features(F, t, short=100, long_lo=600, long_hi=500):
    pre = F[max(0, t - long_lo):max(1, t - long_hi)].mean(axis=0)
    post = F[max(0, t - short):t].mean(axis=0)
    return np.concatenate([post - pre, post])


def diag_features(F, t):
    """Richer isolation features: 300-cycle averaged step (noise /sqrt(3) vs
    100), recent slope, and absolute level."""
    pre = F[max(0, t - 800):max(1, t - 500)].mean(axis=0)
    post = F[max(0, t - 300):t].mean(axis=0)
    slope = (F[max(0, t - 150):t].mean(axis=0)
             - F[max(0, t - 300):max(1, t - 150)].mean(axis=0))
    return np.concatenate([post - pre, slope, post])


def diagnosis_task(fleet, mu, sd, rng):
    # Histogram gradient boosting (sklearn): XGBoost-class model without the
    # second OpenMP runtime that segfaults next to torch-MPS on macOS.
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = [], []
    for v in fleet.values():
        if v['split'] not in ('train', 'val'):
            continue
        Fn = norm_f(v['F'], mu, sd)
        if v['acute_param']:
            onset = int(v['acute_onset'])
            lo = min(onset + 500, v['life'] - 2)
            for t in rng.integers(lo, v['life'], size=40):
                Xtr.append(diag_features(Fn, int(t)))
                ytr.append(CLASSES.index(f"acute_{v['acute_param']}"))
            if onset - 100 > 900:
                for t in rng.integers(900, onset - 100, size=15):
                    Xtr.append(diag_features(Fn, int(t)))
                    ytr.append(0)
        else:
            for t in rng.integers(900, v['life'], size=25):
                Xtr.append(diag_features(Fn, int(t)))
                ytr.append(0)
    clf = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                         learning_rate=0.06, random_state=0,
                                         class_weight='balanced')
    clf.fit(np.array(Xtr), np.array(ytr))

    correct = confus_ok = n_conf = n_acute = 0
    for v in fleet.values():
        if v['split'] != 'test' or not v['acute_param']:
            continue
        n_acute += 1
        t = min(v['life'] - 1, int(v['acute_onset']) + 500)
        pred = CLASSES[int(clf.predict(diag_features(
            norm_f(v['F'], mu, sd), t)[None])[0])]
        truth = f"acute_{v['acute_param']}"
        confusable = v['acute_param'] in ('hpc.eta', 'hpt.eta', 'hpt.flow')
        if pred == truth:
            correct += 1
            if confusable:
                confus_ok += 1
        if confusable:
            n_conf += 1
    return {'accuracy': correct / n_acute if n_acute else None,
            'confusable_accuracy': confus_ok / n_conf if n_conf else None,
            'n_confusable': n_conf, 'n_acute_test': n_acute}


# --------------------------------------------------------------------------
# RUL + conformal
# --------------------------------------------------------------------------

def rul_sequences(v, mu, sd, cuts):
    Fn = norm_f(v['F'], mu, sd)[::RUL_DS]
    seqs, targets = [], []
    for cut in cuts:
        i = cut // RUL_DS
        if i < RUL_SEQ:
            continue
        seqs.append(Fn[i - RUL_SEQ:i])
        targets.append(min((v['life'] - cut) / 1000.0, RUL_CAP))
    return seqs, targets


def rul_task(fleet, mu, sd, rng):
    Xtr, ytr = [], []
    for v in fleet.values():
        if v['split'] != 'train':
            continue
        cuts = rng.integers(RUL_SEQ * RUL_DS, v['life'], size=40)
        s, t = rul_sequences(v, mu, sd, [int(c) for c in cuts])
        Xtr += s
        ytr += t
    print(f'RUL train sequences: {len(Xtr):,} (device: {device()})', flush=True)
    net = train_torch(RULNet(), np.array(Xtr), np.array(ytr),
                      epochs=40, verbose=True)
    import torch
    cpu = torch.device('cpu')
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # split conformal on val
    Xv, yv = [], []
    for v in fleet.values():
        if v['split'] != 'val':
            continue
        cuts = rng.integers(RUL_SEQ * RUL_DS, v['life'], size=40)
        s, t = rul_sequences(v, mu, sd, [int(c) for c in cuts])
        Xv += s
        yv += t
    resid = np.abs(predict_torch(net, np.array(Xv), dev=cpu) - np.array(yv))
    qhat = float(np.quantile(resid, 0.9 * (1 + 1 / len(resid))))

    per_frac = {f: {'err': [], 'cover': []} for f in EVAL_FRACS}
    for v in fleet.values():
        if v['split'] != 'test':
            continue
        for f in EVAL_FRACS:
            cut = int(f * v['life'])
            s, t = rul_sequences(v, mu, sd, [cut])
            if not s:
                continue
            pred = float(predict_torch(net, np.array(s), dev=cpu)[0])
            true = (v['life'] - cut) / 1000.0
            err_cy = (min(pred, RUL_CAP) - min(true, RUL_CAP)) * 1000.0
            per_frac[f]['err'].append(err_cy)
            per_frac[f]['cover'].append(abs(pred - min(true, RUL_CAP)) <= qhat)

    def nasa(d):
        return float(np.exp(-d / 13.0) - 1) if d < 0 else float(np.exp(d / 10.0) - 1)

    out = {}
    for f, d in per_frac.items():
        e = np.array(d['err'])
        out[str(f)] = {'n': len(e),
                       'rmse_cycles': float(np.sqrt((e ** 2).mean())),
                       'median_err_cycles': float(np.median(e)),
                       'nasa_score_mean': float(np.mean([nasa(x / 100) for x in e])),
                       'conformal_coverage': float(np.mean(d['cover'])),
                       'conformal_halfwidth_cycles': qhat * 1000.0}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    print('Loading fleet features...', flush=True)
    fleet = load_fleet_features()
    mu, sd = normalization(fleet)

    det_ae, thr_ae = detection_task(fleet, mu, sd)
    detection_ae = aggregate_detection(fleet, det_ae)
    det_if, thr_if = detection_task_if(fleet, mu, sd, rng)
    detection = aggregate_detection(fleet, det_if)
    diagnosis = diagnosis_task(fleet, mu, sd, rng)
    rul = rul_task(fleet, mu, sd, rng)

    metrics = {'detection': detection, 'detection_naive_ae': detection_ae,
               'diagnosis': diagnosis, 'rul': rul,
               'ae_threshold': thr_ae, 'if_threshold': thr_if,
               'device': str(device())}
    (OUT / 'ai_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def aggregate_detection(fleet, det):
    acute = {e: v for e, v in fleet.items()
             if v['split'] == 'test' and v['acute_param']}
    clean = {e: v for e, v in fleet.items()
             if v['split'] == 'test' and not v['acute_param']}
    detected = {e: det[e] for e in acute
                if det.get(e) is not None and det[e] >= acute[e]['acute_onset']}
    early = [e for e in acute if det.get(e) is not None
             and det[e] < acute[e]['acute_onset']]
    delays = [det[e] - acute[e]['acute_onset'] for e in detected]
    fa = [e for e in clean if det.get(e) is not None]
    return {'recall': len(detected) / len(acute) if acute else None,
            'median_delay_cycles': float(np.median(delays)) if delays else None,
            'false_alarm_engines': len(fa) + len(early),
            'n_clean': len(clean)}


if __name__ == '__main__':
    main()
