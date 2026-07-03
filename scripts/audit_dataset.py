"""WP2.4 audits 2+3: difficulty and realism of the generated fleet.

Difficulty: a trivial classifier (multinomial logistic regression on the raw
measured snapshot channels) must NOT isolate faults well — if it does, the
dataset is too easy and the AI-vs-traditional comparison would be trivial.
Gate criterion: accuracy < ~60 % (and macro-F1 reported, since classes are
imbalanced).

Realism: quantitative checks of the fleet's macroscopic behavior — life
distribution, EGT-margin sawtooth at wash events, deterioration signs of the
deviations at end of life, and measured-noise levels against the catalog.

Output: data/processed/fleet/audit_dataset.json
Usage: uv run python scripts/audit_dataset.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET = REPO_ROOT / 'data' / 'processed' / 'fleet'

MEASURED = [f'{p}_{c}' for p in ('to', 'cr')
            for c in ('N1_rpm', 'N2_rpm', 'WF_kgps', 'EGT_degK',
                      'P25_bar', 'T25_degK', 'PS3_bar', 'T3_degK')]


def _fit_trivial(train_X, train_y, test_X, test_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(train_X), train_y)
    pred = clf.predict(scaler.transform(test_X))
    return (float(accuracy_score(test_y, pred)),
            float(f1_score(test_y, pred, average='macro')))


def difficulty_audit(df):
    """Gated task: isolate the acute-fault parameter (the H2 task) from raw
    measured channels with a trivial classifier. The chronic-mechanism label
    is reported as a secondary metric but NOT gated: chronic mechanisms
    co-evolve with age in every engine, so that label is largely an age proxy
    and high accuracy on it is expected and uninformative."""
    d = df.dropna(subset=MEASURED)

    def task_frames(frame, per_class_cap=40_000, rs=0):
        acute = frame[frame.label.str.startswith('acute_')]
        none = frame[~frame.label.str.startswith('acute_')].sample(
            min(len(acute), len(frame)), random_state=rs)
        data = pd.concat([acute, none])
        y = np.where(data.label.str.startswith('acute_'), data.label, 'none')
        if len(data) > per_class_cap * 8:
            keep = data.sample(per_class_cap * 8, random_state=rs).index
            data, y = data.loc[keep], pd.Series(y, index=data.index).loc[keep].to_numpy()
        return data[MEASURED], y

    Xtr, ytr = task_frames(d[d.split == 'train'], rs=0)
    Xte, yte = task_frames(d[d.split == 'test'], rs=1)
    acc, f1 = _fit_trivial(Xtr, ytr, Xte, yte)

    # Secondary: chronic label (age proxy, not gated)
    tr = d[d.split == 'train'].sample(min(150_000, (d.split == 'train').sum()),
                                      random_state=0)
    te = d[d.split == 'test']
    acc_chr, f1_chr = _fit_trivial(tr[MEASURED], tr.label_chronic,
                                   te[MEASURED], te.label_chronic)

    return {
        'classifier': 'multinomial logistic on 16 raw measured channels',
        'gated_task': 'acute-fault isolation (none + acute_<param> classes)',
        'acute_isolation': {'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
                            'accuracy': acc, 'macro_f1': f1,
                            'classes': sorted(set(ytr))},
        'chronic_label_secondary': {'accuracy': acc_chr, 'macro_f1': f1_chr,
                                    'note': 'age proxy, expected high, not gated'},
    }


def realism_audit(df, index, events, catalog):
    lives = np.array([e['life_cycles'] for e in index['engines']])

    # EGTM sawtooth: margin must improve across wash events (recovery visible).
    washes = events[events.type == 'wash']
    jumps = []
    for _, w in washes.sample(min(len(washes), 400), random_state=1).iterrows():
        e = df[df.engine_id == w.engine_id]
        i = int(np.ceil(w.cycle))
        pre = e[e.cycle.between(i - 5, i - 1)].egtm_C.mean()
        post = e[e.cycle.between(i, i + 4)].egtm_C.mean()
        if np.isfinite(pre) and np.isfinite(post):
            jumps.append(post - pre)
    jumps = np.array(jumps)

    # End-of-life deviation signs at cruise (true channels vs cycle-0 value).
    last = df.groupby('engine_id').tail(50).groupby('engine_id').mean(numeric_only=True)
    first = df.groupby('engine_id').head(50).groupby('engine_id').mean(numeric_only=True)
    degt = (last.cr_EGT_degK_true - first.cr_EGT_degK_true)
    dwf = (last.cr_WF_kgps_true - first.cr_WF_kgps_true) / first.cr_WF_kgps_true * 100

    # Measured-noise check on one channel (drift-free engines only).
    no_drift = [e['engine_id'] for e in index['engines'] if not e['drift_channel']]
    nd = df[df.engine_id.isin(no_drift)]
    resid = (nd.cr_EGT_degK - nd.cr_EGT_degK_true).dropna()

    return {
        'lives': {'median': float(np.median(lives)), 'min': int(lives.min()),
                  'max': int(lives.max()), 'censored': int(sum(e['censored'] for e in index['engines']))},
        'wash_sawtooth': {'n_checked': len(jumps),
                          'mean_egtm_recovery_C': float(jumps.mean()),
                          'frac_positive': float((jumps > 0).mean())},
        'eol_deviation_signs': {'degt_K_mean': float(degt.mean()),
                                'degt_positive_frac': float((degt > 0).mean()),
                                'dwf_pct_mean': float(dwf.mean()),
                                'dwf_positive_frac': float((dwf > 0).mean())},
        'noise_check_EGT': {'resid_std_K': float(resid.std()),
                            'catalog_sigma_K': catalog['sensors']['EGT_degK']['sigma'],
                            'quant_K': catalog['sensors']['EGT_degK']['quant']},
    }


def main():
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    df = pd.read_parquet(FLEET / 'snapshots.parquet')
    events = pd.read_parquet(FLEET / 'events.parquet')
    index = json.loads((FLEET / 'fleet_index.json').read_text())

    report = {'difficulty': difficulty_audit(df),
              'realism': realism_audit(df, index, events, catalog)}
    (FLEET / 'audit_dataset.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
