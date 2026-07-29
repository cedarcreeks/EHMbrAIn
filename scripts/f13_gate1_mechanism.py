"""F13 gate one: is the chronic-degradation MECHANISM MIX recoverable from the
cockpit deviation trajectory -- and if so, does it need a sequence model?

Motivation. The identifiability wall (H2) is a statement about the INSTANTANEOUS
estimand: two health directions 1.3 deg apart in a rank-3 measurement space are
confusable at any single moment, however precisely measured. But the operational
question behind a shop visit is not "what is the 10-dim state now"; it is "which
mechanism is consuming this engine", and the generator's mechanisms are
5 scalars per engine with sharply different TIME signatures:

  fouling      saturating exponential + wash sawtooth (recoverable)
  erosion      linear
  clearance    bilinear: fast break-in, then slow
  hot_section  linear accelerating, eta down AND flow capacity up
  lpt_wear     linear, eta down and flow up

So a quantity that is unidentifiable instant-by-instant may be identifiable over
a history. This gate asks whether that is true here, and -- critically -- whether
a hand-built physics feature set already captures it. If the hand features win,
the finding is a physics result and there is no AI result, exactly as the L4
line (recoverable fraction, R2 = 0.86 from shape features) suggests is possible.
That question is settled BEFORE any milestone is built on it.

Ground truth is replayed deterministically from each engine's seed (the L4
pattern): per-mechanism health contributions projected through the takeoff-hot
EGT row give each mechanism's share of the lost EGT margin at any cycle.

Both families see identical inputs (the 4 cockpit deviation channels every other
experiment in this project uses), identical train/val/test splits, and a
SYMMETRIC tuning budget (the same number of Optuna trials each, selected on val,
reported on test) -- the fairness discipline of sec:fair-design. A first pass
without the symmetric budget gave the hand features a tuned booster over 45
physics-designed descriptors against a single untuned GRU configuration, and its
G1b verdict is not reported: this project's own H1 inverted under tuning, so an
untuned comparison here would be worthless in either direction.

Foreground (torch-MPS for the GRU).
Output: data/processed/f13/gate1_verdict.json
Usage: uv run python scripts/f13_gate1_mechanism.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                     # noqa: E402
from ehmbrain.datagen.fleet import generate_engine, load_icm   # noqa: E402
from ehmbrain.perf.icm import HEALTH_PARAMS                    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f13'
MECHANISMS = ['fouling', 'erosion', 'clearance', 'hot_section', 'lpt_wear']
CUT_MIN = 2500          # cycles: below this the margin loss is too small to share out
CUTS_PER_ENGINE = 40
SEQ_LEN = 256           # trajectory resampled to a fixed length (shape, not rate)
MIN_LOSS_C = 3.0        # degC of lost margin required for a share to be meaningful
N_TRIALS = 25           # symmetric tuning budget, per family (sec:fair-design)


# --------------------------------------------------------------------------
# ground truth: per-mechanism share of lost EGT margin, replayed from seed
# --------------------------------------------------------------------------

def mechanism_shares(catalog, eid, H, ch, base):
    """(life, n_mech) share of EGT-margin loss per mechanism, plus the total
    loss in degC so that near-zero denominators can be excluded."""
    seed = catalog['fleet']['seed']
    rng = np.random.default_rng(np.random.SeedSequence([seed, eid, 0]))
    eng = generate_engine(eid, catalog, H, ch, base, rng)
    Hegt = H[ch.index('EGT_degK')]                     # (10,)
    total = eng['x'] @ Hegt                            # (life,) EGT rise
    contribs = eng['contributions']
    shares = np.zeros((len(total), len(MECHANISMS)))
    for j, m in enumerate(MECHANISMS):
        c = contribs.get(m)
        if c is None:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            shares[:, j] = np.where(np.abs(total) > 1e-9, (c @ Hegt) / total, 0.0)
    return eng['life_cycles'], shares, total


# --------------------------------------------------------------------------
# family A: physics-targeted hand features (the baseline that must be beaten)
# --------------------------------------------------------------------------

def hand_feats(dz, t):
    """Shape descriptors of the whole history up to cut t, each chosen for a
    mechanism the catalog actually contains. Deliberately generous: this is the
    baseline whose success would mean no AI result."""
    seg = dz[:t]
    n = len(seg)
    if n < CUT_MIN:
        return None
    f = []
    for j in range(seg.shape[1]):
        y = seg[:, j]
        lo, hi = y[:800], y[-400:]
        third = n // 3
        s_early = (y[third] - y[0]) / max(third, 1)
        s_late = (y[-1] - y[2 * third]) / max(n - 2 * third, 1)
        # detrended residual: wash sawtooth lives here (fouling)
        tt = np.arange(n)
        fit = np.polyfit(tt, y, 1)
        resid = y - np.polyval(fit, tt)
        jumps = np.diff(y[::25])
        f += [
            float(hi.mean()),                       # level
            float(fit[0] * 1000),                   # global slope per kcycle
            float(s_late - s_early),                # acceleration -> hot_section
            float(s_early / (abs(s_late) + 1e-9)),  # break-in dominance -> clearance
            float(np.std(resid)),                   # sawtooth amplitude -> fouling
            float(np.abs(np.diff(y[::25])).mean()), # roughness
            float(np.sum(jumps < -np.std(jumps))),  # downward steps = wash count
            float(lo.mean() - hi.mean()),           # total excursion
            float(np.polyfit(tt, y, 2)[0] * 1e6),   # quadratic term -> curvature
        ]
    # cross-channel: eta-down-with-flow-up is the hot_section / lpt_wear tell
    sl = [np.polyfit(np.arange(n), seg[:, j], 1)[0] for j in range(seg.shape[1])]
    for a in range(seg.shape[1]):
        for b in range(a + 1, seg.shape[1]):
            f.append(float(sl[a] / (abs(sl[b]) + 1e-12)))
    f.append(float(t) / 10000.0)                    # absolute age
    return np.array(f, float)


# --------------------------------------------------------------------------
# family B: sequence model over the resampled trajectory
# --------------------------------------------------------------------------

def sequence(dz, t, mu, sd):
    """History up to t resampled to SEQ_LEN steps (shape over life), plus a
    constant age channel so absolute rate is not withheld from the model."""
    seg = (dz[:t] - mu) / sd
    idx = np.linspace(0, len(seg) - 1, SEQ_LEN)
    out = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, j])
                    for j in range(seg.shape[1])], axis=1)
    age = np.full((SEQ_LEN, 1), t / 10000.0)
    return np.concatenate([out, age], axis=1).astype(np.float32)


class MechNet:
    """Local to this gate: a GRU with a 5-way head. Kept out of the library
    until the gate says it earns a place there."""

    def __new__(cls, ch, hidden=64, layers=2, n_out=5):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(ch, hidden, layers, batch_first=True, dropout=0.1)
                self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                          nn.Linear(32, n_out))

            def forward(self, x):
                h, _ = self.gru(x)
                return self.head(h[:, -1])

        return _Net()


# --------------------------------------------------------------------------

def build(catalog, H, ch, base, c, ids, rng):
    """Per split: hand features, sequences and targets over random cuts."""
    Xh, Xs, Y, meta = [], [], [], []
    for eid in ids:
        life, shares, total = mechanism_shares(catalog, eid, H, ch, base)
        dz = c['dev'][eid]
        n = min(len(dz), life)
        if n <= CUT_MIN + 200:
            continue
        for t in rng.integers(CUT_MIN, n, size=CUTS_PER_ENGINE):
            t = int(t)
            if abs(total[t]) < MIN_LOSS_C:
                continue
            h = hand_feats(dz, t)
            if h is None:
                continue
            Xh.append(h)
            Xs.append(sequence(dz, t, c['norm'][0], c['norm'][1]))
            Y.append(shares[t])
            meta.append((eid, t))
    return np.array(Xh), np.array(Xs), np.array(Y), meta


def r2_per_col(y, p, y_train_mean):
    """R^2 against the train-mean predictor, per mechanism -- the honest
    baseline for a share that is nearly constant across the fleet."""
    out = []
    for j in range(y.shape[1]):
        ss_res = float(np.sum((y[:, j] - p[:, j]) ** 2))
        ss_tot = float(np.sum((y[:, j] - y_train_mean[j]) ** 2))
        out.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))
    return out


def fit_hand(params, Xtr, Ytr, Xev):
    from sklearn.ensemble import HistGradientBoostingRegressor
    P = np.zeros((len(Xev), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        reg = HistGradientBoostingRegressor(random_state=0, **params)
        reg.fit(Xtr, Ytr[:, j])
        P[:, j] = reg.predict(Xev)
    return P


def fit_seq(params, Xtr, Ytr, Xev, seeds=(0,)):
    from ehmbrain.ai.models import predict_torch, train_torch
    P = []
    for s in seeds:
        net = train_torch(MechNet(ch=Xtr.shape[2], hidden=params['hidden'],
                                  layers=params['layers']),
                          Xtr, Ytr, epochs=params['epochs'], lr=params['lr'],
                          bs=params['bs'], seed=s)
        P.append(predict_torch(net, Xev))
    return np.mean(P, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = yaml.safe_load((REPO_ROOT / 'conf' / 'fault_catalog.yaml').read_text())
    H, ch, base = load_icm('takeoff_hot')
    c = fleet_cache()
    fleet = c['fleet']

    cache = OUT / 'samples.npz'
    if cache.exists():
        print('== samples from cache ==', flush=True)
        z = np.load(cache)
        Xh_tr, Xs_tr, Y_tr = z['Xh_tr'], z['Xs_tr'], z['Y_tr']
        Xh_va, Xs_va, Y_va = z['Xh_va'], z['Xs_va'], z['Y_va']
        Xh_te, Xs_te, Y_te = z['Xh_te'], z['Xs_te'], z['Y_te']
    else:
        rng = np.random.default_rng(11)
        print('== building samples (deterministic truth replay) ==', flush=True)
        Xh_tr, Xs_tr, Y_tr, _ = build(catalog, H, ch, base, c,
                                      split_ids(fleet, 'train'), rng)
        Xh_va, Xs_va, Y_va, _ = build(catalog, H, ch, base, c,
                                      split_ids(fleet, 'val'), rng)
        Xh_te, Xs_te, Y_te, _ = build(catalog, H, ch, base, c,
                                      split_ids(fleet, 'test'), rng)
        np.savez_compressed(cache, Xh_tr=Xh_tr, Xs_tr=Xs_tr, Y_tr=Y_tr,
                            Xh_va=Xh_va, Xs_va=Xs_va, Y_va=Y_va,
                            Xh_te=Xh_te, Xs_te=Xs_te, Y_te=Y_te)
    print(f'   train {len(Y_tr)} / val {len(Y_va)} / test {len(Y_te)} cuts, '
          f'{len(MECHANISMS)} mechanisms', flush=True)

    ymean = Y_tr.mean(axis=0)
    ymean_va = Y_tr.mean(axis=0)

    # --- symmetric tuning budget: N_TRIALS each, selected on val -------------
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj_hand(trial):
        p = {'max_iter': trial.suggest_int('max_iter', 100, 600),
             'max_depth': trial.suggest_int('max_depth', 3, 10),
             'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
             'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 60)}
        return float(np.nanmean(r2_per_col(Y_va, fit_hand(p, Xh_tr, Y_tr, Xh_va),
                                           ymean_va)))

    def obj_seq(trial):
        p = {'hidden': trial.suggest_categorical('hidden', [32, 64, 96, 128]),
             'layers': trial.suggest_int('layers', 1, 3),
             'lr': trial.suggest_float('lr', 3e-4, 6e-3, log=True),
             'epochs': trial.suggest_int('epochs', 40, 160),
             'bs': trial.suggest_categorical('bs', [64, 128, 256])}
        return float(np.nanmean(r2_per_col(Y_va, fit_seq(p, Xs_tr, Y_tr, Xs_va),
                                           ymean_va)))

    import time
    storage = f'sqlite:///{OUT / "gate1_optuna.db"}'
    t0 = time.time()

    def progress(study, trial):
        print(f'   trial {trial.number + 1:3d}/{N_TRIALS}  '
              f'val R2 {trial.value if trial.value is not None else float("nan"):+.3f}  '
              f'best {study.best_value:+.3f}  '
              f'[{(time.time() - t0) / 60:.1f} min]', flush=True)

    def run(name, objective):
        """Persisted study: a kill resumes here instead of starting over."""
        st = optuna.create_study(direction='maximize', study_name=name,
                                 storage=storage, load_if_exists=True,
                                 sampler=optuna.samplers.TPESampler(seed=7))
        done = len([t for t in st.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        if done < N_TRIALS:
            st.optimize(objective, n_trials=N_TRIALS - done,
                        callbacks=[progress])
        print(f'   best val R2 {st.best_value:.3f}', flush=True)
        return st

    print(f'== family A: physics hand features ({N_TRIALS} trials) ==', flush=True)
    st_h = run('gate1_hand', obj_hand)
    t0 = time.time()
    print(f'== family B: GRU over the trajectory ({N_TRIALS} trials) ==', flush=True)
    st_s = run('gate1_seq', obj_seq)

    # --- confirmatory pass on test with the selected configurations ----------
    seeds = (0, 1, 2)
    Ph = fit_hand(st_h.best_params, Xh_tr, Y_tr, Xh_te)
    Ps_mean = fit_seq(st_s.best_params, Xs_tr, Y_tr, Xs_te, seeds=seeds)
    r2_hand = r2_per_col(Y_te, Ph, ymean)
    r2_seq = r2_per_col(Y_te, Ps_mean, ymean)

    verdict = {
        'setup': {
            'mechanisms': MECHANISMS,
            'n_train_cuts': int(len(Y_tr)), 'n_val_cuts': int(len(Y_va)),
            'n_test_cuts': int(len(Y_te)),
            'n_train_engines': len(split_ids(fleet, 'train')),
            'n_test_engines': len(split_ids(fleet, 'test')),
            'seq_len': SEQ_LEN, 'cut_min_cycles': CUT_MIN,
            'inputs': '4 cockpit deviation channels + age, identical for both families',
            'tuning_trials_per_family': N_TRIALS,
            'best_params_hand': st_h.best_params,
            'best_params_sequence': st_s.best_params,
            'best_val_r2': {'hand': float(st_h.best_value),
                            'sequence': float(st_s.best_value)},
            'seeds_sequence': list(seeds)},
        'truth_stats': {m: {'mean': float(Y_te[:, j].mean()),
                            'std': float(Y_te[:, j].std())}
                        for j, m in enumerate(MECHANISMS)},
        'r2_hand_features': dict(zip(MECHANISMS, r2_hand)),
        'r2_sequence_model': dict(zip(MECHANISMS, r2_seq)),
        'r2_mean': {'hand': float(np.nanmean(r2_hand)),
                    'sequence': float(np.nanmean(r2_seq))},
    }

    # G1.a: is the mechanism mix recoverable AT ALL by either family?
    best = [max(a, b) for a, b in zip(r2_hand, r2_seq)]
    verdict['G1a_mechanism_recoverable'] = {
        'best_r2_per_mechanism': dict(zip(MECHANISMS, best)),
        'n_above_0.3': int(sum(b > 0.30 for b in best)),
        'confirmed': bool(sum(b > 0.30 for b in best) >= 3),
        'note': ('the physics question: does trajectory shape carry mechanism '
                 'information the instantaneous state cannot')}
    # G1.b: does it NEED a sequence model, or do hand features suffice?
    gains = [s - h for s, h in zip(r2_seq, r2_hand)]
    verdict['G1b_sequence_needed'] = {
        'r2_gain_per_mechanism': dict(zip(MECHANISMS, gains)),
        'mean_gain': float(np.nanmean(gains)),
        'n_mechanisms_seq_better_by_0.05': int(sum(g > 0.05 for g in gains)),
        'confirmed': bool(sum(g > 0.05 for g in gains) >= 3),
        'note': ('the AI question: if hand features match the sequence model, '
                 'the finding is physics and there is no AI result')}

    (OUT / 'gate1_verdict.json').write_text(json.dumps(verdict, indent=2))

    print()
    print('  mechanism      truth mean/std     R2 hand    R2 seq     gain')
    for j, m in enumerate(MECHANISMS):
        print(f'  {m:13s}  {Y_te[:, j].mean():5.2f}/{Y_te[:, j].std():4.2f}   '
              f'   {r2_hand[j]:7.3f}   {r2_seq[j]:7.3f}   {gains[j]:+7.3f}')
    print(f'\n  G1a mechanism recoverable at all: '
          f"{verdict['G1a_mechanism_recoverable']['confirmed']} "
          f"({verdict['G1a_mechanism_recoverable']['n_above_0.3']}/5 above R2 0.30)")
    print(f'  G1b sequence model needed:        '
          f"{verdict['G1b_sequence_needed']['confirmed']} "
          f"(mean gain {verdict['G1b_sequence_needed']['mean_gain']:+.3f})")


if __name__ == '__main__':
    main()
