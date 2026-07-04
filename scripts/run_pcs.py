"""Physics-Consistency Score (contribution C5) on the v1.1 diagnosis task.

For each oracle-timed test episode, SHAP attributions of the diagnosis
classifier's predicted class are read out on the STEP block of the feature
vector (the three cockpit deviation channels) — a direction in measurement
space that represents the model's evidence. That direction is projected into
health-parameter space through the pseudo-inverse of the cockpit ICM and
compared (absolute cosine) with the true fault's unit direction:

    PCS = | cos( H_cockpit^+  m_shap ,  e_true ) |

High PCS = the model's reasoning points at the physically correct component,
regardless of whether the argmax label was right. Reported: PCS distribution
for correct vs incorrect predictions, and the point-biserial correlation
between PCS and correctness.

Foreground only (torch-MPS background segfault, recorded platform finding).
Output: data/processed/ai/pcs_metrics.json
Usage: uv run python scripts/run_pcs.py
"""

import json
from pathlib import Path

import numpy as np

from ehmbrain.ai.data import load_fleet_features, normalization
from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ai import CLASSES, diag_features, norm_f          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'ai'
COCKPIT_ICM = ['N2_rpm', 'WF_kgps', 'EGT_degK']


def train_classifier(fleet, mu, sd, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = [], []
    for v in fleet.values():
        if v['split'] not in ('train', 'val'):
            continue
        Fn = norm_f(v['F'], mu, sd)
        eps = v['episodes']
        if eps:
            bounds = [int(o) for o, _ in eps] + [v['life']]
            for k, (onset, param) in enumerate(eps):
                lo = min(int(onset) + 500, bounds[k + 1] - 2)
                if lo >= bounds[k + 1]:
                    continue
                for t in rng.integers(lo, bounds[k + 1], size=25):
                    Xtr.append(diag_features(Fn, int(t)))
                    ytr.append(CLASSES.index(f'acute_{param}'))
            if bounds[0] - 100 > 900:
                for t in rng.integers(900, bounds[0] - 100, size=15):
                    Xtr.append(diag_features(Fn, int(t)))
                    ytr.append(0)
        else:
            for t in rng.integers(900, v['life'], size=25):
                Xtr.append(diag_features(Fn, int(t)))
                ytr.append(0)
    clf = HistGradientBoostingClassifier(max_iter=400, max_depth=6,
                                         learning_rate=0.06, random_state=0,
                                         class_weight='balanced')
    Xtr = np.array(Xtr)
    clf.fit(Xtr, np.array(ytr))
    return clf, Xtr


def main():
    import shap
    rng = np.random.default_rng(7)
    print('Loading fleet...', flush=True)
    fleet = load_fleet_features()
    mu, sd = normalization(fleet)
    clf, Xtr = train_classifier(fleet, mu, sd, rng)

    H, ch, _ = load_icm('cruise')
    Hc = H[[ch.index(c) for c in COCKPIT_ICM]]          # (3, 10)
    Hp = np.linalg.pinv(Hc)                              # (10, 3)

    background = shap.utils.sample(Xtr, 100, random_state=0)
    explainer = shap.PermutationExplainer(clf.predict_proba, background)

    rows = []
    for v in fleet.values():
        if v['split'] != 'test' or not v['episodes']:
            continue
        bounds = [int(o) for o, _ in v['episodes']] + [v['life']]
        Fn = norm_f(v['F'], mu, sd)
        for k, (onset, param) in enumerate(v['episodes']):
            t = min(bounds[k + 1] - 1, v['life'] - 1, int(onset) + 500)
            if t <= int(onset) + 100:
                continue
            x = diag_features(Fn, t)
            pred_idx = int(clf.predict(x[None])[0])
            sv = explainer(x[None]).values[0]            # (12, n_classes)
            phi = sv[:, pred_idx]
            # step block = first 4 features; cockpit channels = first 3 of them
            m = phi[:3]
            x_attr = Hp @ m                              # (10,) health space
            j_true = HEALTH_PARAMS.index(param)
            denom = np.linalg.norm(x_attr) * 1.0
            pcs = float(abs(x_attr[j_true]) / denom) if denom > 0 else 0.0
            rows.append({'param': param,
                         'pred': CLASSES[pred_idx],
                         'correct': CLASSES[pred_idx] == f'acute_{param}',
                         'pcs': pcs})
            print(f'  {param:9s} pred={CLASSES[pred_idx]:15s} PCS={pcs:.3f}',
                  flush=True)

    pcs = np.array([r['pcs'] for r in rows])
    ok = np.array([r['correct'] for r in rows])
    corr = float(np.corrcoef(pcs, ok.astype(float))[0, 1]) if ok.std() > 0 else None
    report = {
        'n_episodes': len(rows),
        'pcs_mean_correct': float(pcs[ok].mean()) if ok.any() else None,
        'pcs_mean_incorrect': float(pcs[~ok].mean()) if (~ok).any() else None,
        'pcs_correctness_correlation': corr,
        'episodes': rows,
    }
    (OUT / 'pcs_metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'},
                     indent=2))


if __name__ == '__main__':
    main()
