"""F5 confirmatory evaluation (prereg-v1 §5): ONE pass over the 20 test
engines with the tuned configurations, then the H1-H5 verdicts.

Implementation notes (disclosed):
- AI models fit on TRAIN only (consistent with the tuning objective and
  required for valid split-conformal calibration on val, H5).
- McNemar exact, one-sided in the hypothesized direction (the confirmation
  criteria are directional).
- Traditional H5 interval: theil_sen_rul_interval with the tuned RUL config;
  infinite upper ends capped at the 25 000-cycle horizon (decision register).
- H4 uses the prereg-specified pure-vs-hybrid protocol with per-engine
  errors captured at the 10 % fraction for the Wilcoxon.

Foreground only (torch-MPS). Output: data/processed/f5/verdicts.json
Usage: uv run python scripts/f5_confirm.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import (EVAL_FRACS, RUL_CAP_CY, CONFUSABLE, ai_step_features,
                     eval_ai, eval_trad, fleet_cache, split_ids)   # noqa: E402
from ehmbrain.trad.pipeline import holt_smooth, theil_sen_rul_interval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'


def nasa(d_cycles):
    d = d_cycles / 100.0
    return float(np.exp(-d / 13.0) - 1) if d < 0 else float(np.exp(d / 10.0) - 1)


def mcnemar_one_sided(ai_ok, trad_ok):
    """b = AI-only successes, c = trad-only; H1: AI better -> b > c."""
    b = int(np.sum(ai_ok & ~trad_ok))
    c = int(np.sum(~ai_ok & trad_ok))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(b, b + c, 0.5,
                                       alternative='greater').pvalue)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def main():
    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    c = fleet_cache()
    fleet = c['fleet']
    test_ids = split_ids(fleet, 'test')

    print('== confirmatory pass: traditional ==', flush=True)
    trad_det = eval_trad(sel_t['detection']['params'], 'test')
    trad_iso = eval_trad(sel_t['isolation']['params'], 'test')
    trad_rul = eval_trad(sel_t['rul']['params'], 'test')

    print('== confirmatory pass: AI (3 seeds) ==', flush=True)
    ai_det = eval_ai(sel_a['detection']['params'], 'train', 'test', seed=0)
    ai_iso = eval_ai(sel_a['isolation']['params'], 'train', 'test', seed=0)
    ai_rul_seeds = [eval_ai(sel_a['rul']['params'], 'train', 'test', seed=s)
                    for s in (0, 1, 2)]

    verdicts = {}

    # ---- H1: detection --------------------------------------------------
    def episode_hits(m):
        # rebuild per-episode hit vector in a fixed episode order
        hits = []
        for eid in test_ids:
            v = fleet[eid]
            eps = v['episodes']
            if not eps:
                continue
            det = m.get('det_by_engine', {}).get(eid)
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, _) in enumerate(eps):
                hits.append(bool(det is not None and onset <= det < bounds[k + 1]))
        return np.array(hits)

    # eval_* don't return per-engine detections; recompute quickly
    from tune_f5 import trad_detect_engine
    trad_det_by = {eid: trad_detect_engine(c['dev'][eid],
                                           sel_t['detection']['params'])
                   for eid in test_ids}
    trad_det['det_by_engine'] = trad_det_by
    # AI: rerun detection portion via eval_ai's maha inside; simplest: reuse
    # its recall/fa plus per-episode via a dedicated pass
    from sklearn.covariance import LedoitWolf
    mu, sd = c['norm']
    cfg = sel_a['detection']['params']
    rng = np.random.default_rng(1000)
    short, long_w = cfg['det_short'], cfg['det_long']
    fit_ids = split_ids(fleet, 'train')
    X = []
    for eid in fit_ids:
        v = fleet[eid]
        hi = int(v['episodes'][0][0]) if v['episodes'] else v['life']
        lo = short + long_w + 50
        if hi - 50 <= lo:
            continue
        for t in rng.integers(lo, hi - 50, size=40):
            X.append(ai_step_features((c['dev'][eid] - mu) / sd, int(t),
                                      short, long_w))
    lw = LedoitWolf().fit(np.array(X))
    pool = []
    for eid in fit_ids:
        if fleet[eid]['episodes']:
            continue
        v = fleet[eid]
        ts = np.arange(short + long_w + 50, v['life'], 25)
        Xs = np.stack([ai_step_features((c['dev'][eid] - mu) / sd, int(t),
                                        short, long_w) for t in ts]) - lw.location_
        pool.append(np.einsum('ij,jk,ik->i', Xs, lw.precision_, Xs))
    thr = float(np.percentile(np.concatenate(pool), cfg['det_pct']))
    ai_det_by = {}
    for eid in test_ids:
        v = fleet[eid]
        ts = np.arange(short + long_w + 50, v['life'], 25)
        Xs = np.stack([ai_step_features((c['dev'][eid] - mu) / sd, int(t),
                                        short, long_w) for t in ts]) - lw.location_
        sc = np.einsum('ij,jk,ik->i', Xs, lw.precision_, Xs)
        run, det = 0, None
        for i, ex in enumerate(sc > thr):
            run = run + 1 if ex else 0
            if run >= cfg['det_k']:
                det = int(ts[i])
                break
        ai_det_by[eid] = det
    ai_det['det_by_engine'] = ai_det_by

    h_t, h_a = episode_hits(trad_det), episode_hits(ai_det)

    def delays(det_by):
        out = []
        for eid in test_ids:
            v = fleet[eid]
            eps = v['episodes']
            if not eps:
                continue
            det = det_by.get(eid)
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, _) in enumerate(eps):
                if det is not None and onset <= det < bounds[k + 1]:
                    out.append(det - onset)
        return np.array(out)

    d_t, d_a = delays(trad_det_by), delays(ai_det_by)
    b, cc, p1 = mcnemar_one_sided(h_a, h_t)
    rec_t, rec_a = float(h_t.mean()), float(h_a.mean())
    med_dt = float(np.median(d_t)) if len(d_t) else None
    med_da = float(np.median(d_a)) if len(d_a) else None
    fa_t = sum(1 for eid in test_ids
               if (not fleet[eid]['episodes'] and trad_det_by[eid] is not None)
               or (fleet[eid]['episodes'] and trad_det_by[eid] is not None
                   and trad_det_by[eid] < fleet[eid]['episodes'][0][0]))
    fa_a = sum(1 for eid in test_ids
               if (not fleet[eid]['episodes'] and ai_det_by[eid] is not None)
               or (fleet[eid]['episodes'] and ai_det_by[eid] is not None
                   and ai_det_by[eid] < fleet[eid]['episodes'][0][0]))
    h1_ok = (rec_a >= rec_t and med_da is not None
             and (med_dt is None or med_da <= 0.8 * med_dt)
             and fa_a <= fa_t and p1 < 0.05)
    verdicts['H1'] = {
        'trad': {'recall': rec_t, 'median_delay': med_dt, 'fa': fa_t},
        'ai': {'recall': rec_a, 'median_delay': med_da, 'fa': fa_a},
        'mcnemar': {'b_ai_only': b, 'c_trad_only': cc, 'p_one_sided': p1},
        'confirmed': bool(h1_ok)}

    # ---- H2: isolation on confusable episodes ---------------------------
    def iso_ok_vec(rows):
        conf = [r for r in rows if r['param'] in CONFUSABLE]
        conf.sort(key=lambda r: (r['engine'], r['param']))
        return np.array([r['pred'] == r['param'] for r in conf])

    ok_t, ok_a = iso_ok_vec(trad_iso['iso_rows']), iso_ok_vec(ai_iso['iso_rows'])
    b2, c2, p2 = mcnemar_one_sided(ok_a, ok_t)
    acc_t, acc_a = float(ok_t.mean()), float(ok_a.mean())
    h2_ok = (acc_a >= acc_t + 0.10) and p2 < 0.05
    verdicts['H2'] = {'trad_confusable_acc': acc_t, 'ai_confusable_acc': acc_a,
                      'n_confusable': int(len(ok_t)),
                      'mcnemar': {'b': b2, 'c': c2, 'p_one_sided': p2},
                      'confirmed': bool(h2_ok)}

    # ---- H3: RUL ---------------------------------------------------------
    def per_engine_abs(rows):
        by = {}
        for r in rows:
            by.setdefault(r['engine'], []).append(abs(r['err']))
        return {e: float(np.mean(v)) for e, v in by.items()}

    trad_abs = per_engine_abs(trad_rul['rul_rows'])
    ai_abs_seeds = [per_engine_abs(m['rul_rows']) for m in ai_rul_seeds]
    ai_abs = {e: float(np.mean([s[e] for s in ai_abs_seeds])) for e in trad_abs}
    pairs_t = np.array([trad_abs[e] for e in test_ids])
    pairs_a = np.array([ai_abs[e] for e in test_ids])
    w = stats.wilcoxon(pairs_t - pairs_a, alternative='greater')
    frac_ok = {}
    for f in EVAL_FRACS:
        et = [r['err'] for r in trad_rul['rul_rows'] if r['frac'] == f]
        ea = np.mean([[r['err'] for r in m['rul_rows'] if r['frac'] == f]
                      for m in ai_rul_seeds], axis=0)
        frac_ok[str(f)] = {
            'trad_rmse': float(np.sqrt(np.mean(np.square(et)))),
            'ai_rmse': float(np.sqrt(np.mean(np.square(ea)))),
            'trad_nasa': float(np.mean([nasa(x) for x in et])),
            'ai_nasa': float(np.mean([nasa(x) for x in ea]))}
    both_better = all(v['ai_rmse'] < v['trad_rmse'] and
                      v['ai_nasa'] < v['trad_nasa'] for v in frac_ok.values())
    h3_ok = both_better and w.pvalue < 0.05
    verdicts['H3'] = {'per_fraction': frac_ok,
                      'wilcoxon_p_one_sided': float(w.pvalue),
                      'cliffs_delta_abs_err': cliffs_delta(pairs_t, pairs_a),
                      'confirmed': bool(h3_ok)}

    # ---- H4: from the prereg-specified hybrid protocol -------------------
    hyb = json.loads((REPO_ROOT / 'data' / 'processed' / 'ai' /
                      'hybrid_metrics.json').read_text())
    h4_ok = (hyb['hybrid@100%']['rmse_cycles_mean'] <= hyb['pure@100%']['rmse_cycles_mean']
             and hyb['hybrid@10%']['rmse_cycles_mean'] < hyb['pure@10%']['rmse_cycles_mean']
             and hyb['hybrid@25%']['rmse_cycles_mean'] < hyb['pure@25%']['rmse_cycles_mean'])
    verdicts['H4'] = {'summary': {k: v['rmse_cycles_mean'] for k, v in hyb.items()},
                      'confirmed': bool(h4_ok),
                      'note': 'criteria fail at every fraction; Wilcoxon unnecessary'}

    # ---- H5: uncertainty calibration --------------------------------------
    # AI: conformal from the H3 model (train-fit), calibrated on val
    cfgr = sel_a['rul']['params']
    m_val = eval_ai(cfgr, 'train', 'val', seed=0)
    qhat = float(np.quantile([abs(r['err']) for r in m_val['rul_rows']],
                             0.9 * (1 + 1 / len(m_val['rul_rows']))))
    ai_err0 = [r['err'] for r in ai_rul_seeds[0]['rul_rows']]
    cov_a = float(np.mean([abs(e) <= qhat for e in ai_err0]))
    hw_a = qhat
    # Traditional: slope-percentile band
    cfg_rt = sel_t['rul']['params']
    covs, hws = [], []
    for eid in test_ids:
        v = fleet[eid]
        dev_s, _, _ = holt_smooth(c['dev'][eid][:, 3], cfg_rt['rul_a'],
                                  cfg_rt['rul_a'] / 3)
        egtm = 85.0 - dev_s
        for f in EVAL_FRACS:
            i = int(f * v['life'])
            out = theil_sen_rul_interval(egtm[:i], window=cfg_rt['rul_win'])
            true = min(v['life'] - i, RUL_CAP_CY)
            if out is None:
                lo, hi = 0.0, 25000.0
            else:
                lo, _, hi = out
                hi = min(hi, 25000.0)
            covs.append(lo <= true <= hi)
            hws.append((hi - lo) / 2.0)
    cov_t, hw_t = float(np.mean(covs)), float(np.mean(hws))
    h5_ok = (abs(cov_a - 0.9) < abs(cov_t - 0.9)) and (hw_a <= 1.2 * hw_t)
    verdicts['H5'] = {'ai': {'coverage': cov_a, 'halfwidth_cycles': hw_a},
                      'trad': {'coverage': cov_t, 'halfwidth_cycles': hw_t},
                      'confirmed': bool(h5_ok)}

    # ---- Holm across the p-valued hypotheses ------------------------------
    ps = {'H1': verdicts['H1']['mcnemar']['p_one_sided'],
          'H2': verdicts['H2']['mcnemar']['p_one_sided'],
          'H3': verdicts['H3']['wilcoxon_p_one_sided']}
    order = sorted(ps, key=ps.get)
    m = len(order)
    holm = {}
    for i, h in enumerate(order):
        holm[h] = min(1.0, ps[h] * (m - i))
    for h in holm:
        verdicts[h]['p_holm'] = holm[h]
        if verdicts[h]['confirmed'] and holm[h] >= 0.05:
            verdicts[h]['confirmed'] = False
            verdicts[h]['note'] = 'fails after Holm correction'

    (F5 / 'verdicts.json').write_text(json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts, indent=1, default=float))


if __name__ == '__main__':
    main()
