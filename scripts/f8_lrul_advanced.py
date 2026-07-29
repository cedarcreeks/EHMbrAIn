"""F8/L-RUL (prereg-v10; fraction sweep added prereg-v15): does an advanced
(similarity-based) classical prognostic narrow the H3 RUL gap that linear
Theil-Sen extrapolation showed?

Similarity-based prognostics: the health indicator HI(n) = smoothed
takeoff-EGT degradation; a test engine's recent HI window is matched against
the run-to-failure HI curves of the TRAIN engines, and the k best-aligned
matches' remaining lives give the RUL. Unlike Theil-Sen it follows the
nonlinear degradation shape. Compared with the tuned Theil-Sen and the F5 AI.

Disclosure (prereg-v15): the original v10 run adjudicated AI vs the advanced
classical at the 90 % life fraction only, while the headline H3 factor
(2.3-4.4x) is quoted against the *operational* Theil-Sen at all three
fractions. That is an asymmetric comparison. This version pairs AI against the
advanced classical at every evaluated fraction, with a Wilcoxon per fraction
and Holm correction across the three, so the fair margin is stated wherever the
headline margin is. The 90 % adjudication is unchanged and still reported.

Foreground. Output: data/processed/f8/lrul_verdict.json
Usage: uv run python scripts/f8_lrul_advanced.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_f5 import fleet_cache, split_ids                  # noqa: E402
from ehmbrain.trad.pipeline import holt_smooth, theil_sen_rul  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
EVAL_FRACS = (0.5, 0.7, 0.9)
RUL_CAP = 12000.0
W = 1200            # similarity window (cycles)
DS = 25            # subsample stride
K = 8              # nearest matches


def hi_series(c, eid, rul_a=0.08):
    """Health indicator: smoothed takeoff-EGT degradation (rises to failure)."""
    dev = c['dev'][eid][:, 3]
    s, _, _ = holt_smooth(dev, rul_a, rul_a / 3)
    return s          # EGT degradation in K (margin loss)


def build_library(c, fleet, train_ids):
    lib = []
    for eid in train_ids:
        hi = hi_series(c, eid)[::DS]
        life = fleet[eid]['life']
        lib.append({'hi': hi, 'life_ds': len(hi), 'life': life})
    return lib


def similarity_rul(test_hi_window, lib):
    """kNN over aligned HI windows; RUL = inverse-distance weighted remaining life."""
    wlen = len(test_hi_window)
    cands = []
    for e in lib:
        hi = e['hi']
        best_d, best_rem = np.inf, None
        for end in range(wlen, len(hi)):
            d = np.mean((hi[end - wlen:end] - test_hi_window) ** 2)
            if d < best_d:
                best_d = d
                best_rem = (e['life_ds'] - end) * DS      # cycles remaining
        if best_rem is not None:
            cands.append((best_d, best_rem))
    cands.sort(key=lambda t: t[0])
    cands = cands[:K]
    wts = np.array([1.0 / (d + 1e-6) for d, _ in cands])
    rem = np.array([r for _, r in cands])
    return float(np.clip(np.sum(wts * rem) / np.sum(wts), 0, RUL_CAP))


def classical_errors(c, fleet, lib, ids, win, rul_a):
    """Signed RUL error (cycles) per engine and life fraction for both
    classical prognostics. Shared with the UQ reattribution script, which needs
    the same predictors evaluated on the validation split."""
    err = {'theilsen': {}, 'similarity': {}}
    for eid in ids:
        v = fleet[eid]
        hi_full = hi_series(c, eid, rul_a)
        hi_ds = hi_full[::DS]
        for f in EVAL_FRACS:
            i = int(f * v['life'])
            true = min(v['life'] - i, RUL_CAP)
            # Theil-Sen on the EGT margin (85 - HI)
            ts = theil_sen_rul(85.0 - hi_full[:i], window=win)
            ts = min(ts, 25000.0) if ts is not None else RUL_CAP
            err['theilsen'][(eid, f)] = min(ts, RUL_CAP) - true
            # Similarity: recent window ending at cut
            ie = i // DS
            if ie >= W // DS:
                sm = similarity_rul(hi_ds[ie - W // DS:ie], lib)
            else:
                sm = RUL_CAP
            err['similarity'][(eid, f)] = sm - true
    return err


def ai_errors(split, seeds=(0, 1, 2)):
    """Per-engine, per-fraction AI RUL error, averaged over seeds exactly as
    the frozen H3 confirmatory pass does."""
    from tune_f5 import eval_ai
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    per_seed = []
    for s in seeds:
        m = eval_ai(sel_a['rul']['params'], 'train', split, seed=s)
        per_seed.append({(r['engine'], r['frac']): r['err'] for r in m['rul_rows']})
    keys = set(per_seed[0])
    for d in per_seed[1:]:
        keys &= set(d)
    return {k: float(np.mean([d[k] for d in per_seed])) for k in keys}


def rmse(a):
    return float(np.sqrt(np.mean(np.square(list(a)))))


def main():
    c = fleet_cache()
    fleet = c['fleet']
    train_ids = split_ids(fleet, 'train')
    test_ids = split_ids(fleet, 'test')
    lib = build_library(c, fleet, train_ids)

    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    win = sel_t['rul']['params']['rul_win']
    rul_a = sel_t['rul']['params']['rul_a']

    err = classical_errors(c, fleet, lib, test_ids, win, rul_a)
    err['ai'] = ai_errors('test')

    rms = {m: {str(f): rmse(err[m][(e, f)] for e in test_ids if (e, f) in err[m])
               for f in EVAL_FRACS} for m in err}
    ts90, sim90, ai90 = (rms[m]['0.9'] for m in ('theilsen', 'similarity', 'ai'))

    # --- prereg-v15: pair AI against the advanced classical at EVERY fraction --
    paired = {}
    for f in EVAL_FRACS:
        ids = [e for e in test_ids
               if (e, f) in err['similarity'] and (e, f) in err['ai']]
        a_sim = [abs(err['similarity'][(e, f)]) for e in ids]
        a_ai = [abs(err['ai'][(e, f)]) for e in ids]
        w = wilcoxon(a_sim, a_ai, alternative='greater')
        paired[str(f)] = {
            'n_engines': len(ids),
            'rmse_similarity': rms['similarity'][str(f)],
            'rmse_ai': rms['ai'][str(f)],
            'ratio_sim_over_ai': rms['similarity'][str(f)] / rms['ai'][str(f)],
            'ratio_theilsen_over_ai': rms['theilsen'][str(f)] / rms['ai'][str(f)],
            'wilcoxon_p': float(w.pvalue)}
    # Holm across the three fractions
    order = sorted(paired, key=lambda k: paired[k]['wilcoxon_p'])
    for i, k in enumerate(order):
        p_h = min(1.0, paired[k]['wilcoxon_p'] * (len(order) - i))
        paired[k]['p_holm'] = p_h
        paired[k]['ai_wins'] = bool(paired[k]['rmse_ai'] < paired[k]['rmse_similarity']
                                    and p_h < 0.05)

    ratios = [paired[str(f)]['ratio_sim_over_ai'] for f in EVAL_FRACS]
    n_wins = sum(paired[str(f)]['ai_wins'] for f in EVAL_FRACS)

    verdict = {
        'rmse_90': {'theilsen': ts90, 'similarity': sim90, 'ai': ai90},
        'rmse_all': rms,
        'H-RUL.1_advanced_narrows': {
            'confirmed': bool(sim90 < ts90),
            'note': f'similarity {sim90:.0f} vs Theil-Sen {ts90:.0f} at 90% life'},
        'H-RUL.2_ai_still_wins': {
            'ai_rmse90': ai90, 'similarity_rmse90': sim90,
            'wilcoxon_p': paired['0.9']['wilcoxon_p'],
            'confirmed': bool(ai90 < sim90 and paired['0.9']['wilcoxon_p'] < 0.05)},
        'H-RUL.3_fraction_sweep': {
            'per_fraction': paired,
            'fair_margin_range': [min(ratios), max(ratios)],
            'headline_margin_range': [min(paired[str(f)]['ratio_theilsen_over_ai']
                                          for f in EVAL_FRACS),
                                      max(paired[str(f)]['ratio_theilsen_over_ai']
                                          for f in EVAL_FRACS)],
            'fractions_ai_wins': n_wins,
            'uniform': bool(n_wins == len(EVAL_FRACS)),
            'note': ('AI vs the advanced classical at all three fractions, '
                     'Holm-corrected; prereg-v15 disclosure')},
    }
    F8 = F5.parent / 'f8'
    (F8 / 'lrul_verdict.json').write_text(json.dumps(verdict, indent=2))
    (F8 / 'lrul_errors_test.json').write_text(json.dumps(
        {m: {f'{e}|{f}': v for (e, f), v in d.items()} for m, d in err.items()},
        indent=1))
    print(f"RMSE @90%:  Theil-Sen {ts90:.0f}  similarity {sim90:.0f}  AI {ai90:.0f}")
    print(f"H-RUL.1 advanced narrows: {sim90:.0f}<{ts90:.0f} -> {verdict['H-RUL.1_advanced_narrows']['confirmed']}")
    print(f"H-RUL.2 AI still wins @90%: p={paired['0.9']['wilcoxon_p']:.4f} -> {verdict['H-RUL.2_ai_still_wins']['confirmed']}")
    print('H-RUL.3 fraction sweep (AI vs ADVANCED classical):')
    for f in EVAL_FRACS:
        d = paired[str(f)]
        print(f"  {f:.0%} life: sim {d['rmse_similarity']:6.0f}  AI {d['rmse_ai']:6.0f}"
              f"  ratio {d['ratio_sim_over_ai']:.2f}x  (vs Theil-Sen {d['ratio_theilsen_over_ai']:.2f}x)"
              f"  p_holm={d['p_holm']:.4f}  win={d['ai_wins']}")
    print(f"  fair margin {min(ratios):.2f}-{max(ratios):.2f}x vs headline "
          f"{verdict['H-RUL.3_fraction_sweep']['headline_margin_range'][0]:.1f}-"
          f"{verdict['H-RUL.3_fraction_sweep']['headline_margin_range'][1]:.1f}x")


if __name__ == '__main__':
    main()
