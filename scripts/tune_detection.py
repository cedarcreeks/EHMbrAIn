"""Validation sweep of the AI Mahalanobis detector (threshold percentile x
persistence). Selection on the val split only (recall on val episodes with a
false-alarm budget); the winning setting is then applied once to test.

This is a [T]-class threshold choice exercised on validation - the same
freedom the traditional pipeline's thresholds enjoy; the full symmetric
budget arrives in F5.

Output: data/processed/ai/detection_tuned.json
Usage (foreground): uv run python scripts/tune_detection.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from ehmbrain.ai.data import load_fleet_features, normalization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ai import aggregate_detection, norm_f, step_features   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'ai'

GRID_PCT = (97.0, 98.5, 99.0, 99.5, 99.8)
GRID_K = (4, 6, 10, 16)


def main():
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(7)
    fleet = load_fleet_features()
    mu, sd = normalization(fleet)

    Xtr = []
    for v in fleet.values():
        if v['split'] != 'train':
            continue
        Fn = norm_f(v['F'], mu, sd)
        hi = int(v['episodes'][0][0]) if v['episodes'] else v['life']
        for t in rng.integers(700, max(701, hi - 50), size=60):
            Xtr.append(step_features(Fn, int(t)))
    lw = LedoitWolf().fit(np.array(Xtr))

    def scores_for(v, stride=10):
        Fn = norm_f(v['F'], mu, sd)
        ts = np.arange(700, v['life'], stride)
        X = np.stack([step_features(Fn, int(t)) for t in ts]) - lw.location_
        return ts, np.einsum('ij,jk,ik->i', X, lw.precision_, X)

    cached = {eid: scores_for(v) for eid, v in fleet.items()
              if v['split'] in ('val', 'test')}
    val_pool = np.concatenate([cached[e][1] for e, v in fleet.items()
                               if v['split'] == 'val' and not v['episodes']])

    def first_alarm(eid, thr, k):
        ts, sc = cached[eid]
        run = 0
        for i, ex in enumerate(sc > thr):
            run = run + 1 if ex else 0
            if run >= k:
                return int(ts[i])
        return None

    def split_metrics(split, thr, k):
        det = {e: first_alarm(e, thr, k) for e, v in fleet.items()
               if v['split'] == split}
        # per-episode recall: alarm within [onset, next bound)
        hits = total = 0
        fa = 0
        for e, v in fleet.items():
            if v['split'] != split:
                continue
            d = det[e]
            if v['episodes']:
                bounds = [o for o, _ in v['episodes']] + [v['life']]
                for kk, (onset, _) in enumerate(v['episodes']):
                    total += 1
                    if d is not None and onset <= d < bounds[kk + 1]:
                        hits += 1
                if d is not None and d < v['episodes'][0][0]:
                    fa += 1
            elif d is not None:
                fa += 1
        return hits / total if total else None, fa, total

    best = None
    for pct in GRID_PCT:
        thr = float(np.percentile(val_pool, pct))
        for k in GRID_K:
            rec, fa, tot = split_metrics('val', thr, k)
            # budget: at most 1 false alarm on the val split
            score = (-1 if fa > 1 else rec)
            print(f'pct {pct:5.1f}  k {k:2d}  val recall {rec:.2f}  FA {fa}',
                  flush=True)
            if best is None or score > best['score']:
                best = {'pct': pct, 'k': k, 'thr': thr, 'val_recall': rec,
                        'val_fa': fa, 'score': score}

    rec_t, fa_t, tot_t = split_metrics('test', best['thr'], best['k'])
    result = {'selected': {kk: best[kk] for kk in ('pct', 'k', 'thr',
                                                   'val_recall', 'val_fa')},
              'test': {'recall_episodes': rec_t, 'false_alarm_engines': fa_t,
                       'n_episodes': tot_t}}
    (OUT / 'detection_tuned.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
