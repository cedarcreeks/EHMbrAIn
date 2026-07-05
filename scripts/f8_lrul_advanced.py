"""F8/L-RUL (prereg-v10): does an advanced (similarity-based) classical
prognostic narrow the H3 RUL gap that linear Theil-Sen extrapolation showed?

Similarity-based prognostics: the health indicator HI(n) = smoothed
takeoff-EGT degradation; a test engine's recent HI window is matched against
the run-to-failure HI curves of the TRAIN engines, and the k best-aligned
matches' remaining lives give the RUL. Unlike Theil-Sen it follows the
nonlinear degradation shape. Compared with the tuned Theil-Sen and the F5 AI.

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


def main():
    c = fleet_cache()
    fleet = c['fleet']
    train_ids = split_ids(fleet, 'train')
    test_ids = split_ids(fleet, 'test')
    lib = build_library(c, fleet, train_ids)

    sel_t = json.loads((F5 / 'selected_trad.json').read_text())['selected']
    win = sel_t['rul']['params']['rul_win']
    rul_a = sel_t['rul']['params']['rul_a']

    err = {'theilsen': {f: [] for f in EVAL_FRACS},
           'similarity': {f: [] for f in EVAL_FRACS}}
    per_engine90 = {'theilsen': {}, 'similarity': {}}
    for eid in test_ids:
        v = fleet[eid]
        hi_full = hi_series(c, eid, rul_a)
        hi_ds = hi_full[::DS]
        for f in EVAL_FRACS:
            i = int(f * v['life'])
            true = min(v['life'] - i, RUL_CAP)
            # Theil-Sen on the EGT margin (85 - HI)
            ts = theil_sen_rul(85.0 - hi_full[:i], window=win)
            ts = min(ts, 25000.0) if ts is not None else RUL_CAP
            err['theilsen'][f].append(min(ts, RUL_CAP) - true)
            # Similarity: recent window ending at cut
            ie = i // DS
            if ie >= W // DS:
                w_test = hi_ds[ie - W // DS:ie]
                sm = similarity_rul(w_test, lib)
            else:
                sm = RUL_CAP
            err['similarity'][f].append(sm - true)
            if abs(f - 0.9) < 1e-6:
                per_engine90['theilsen'][eid] = abs(err['theilsen'][f][-1])
                per_engine90['similarity'][eid] = abs(err['similarity'][f][-1])

    def rmse(a):
        return float(np.sqrt(np.mean(np.square(a))))
    rms = {m: {str(f): rmse(err[m][f]) for f in EVAL_FRACS} for m in err}
    ai90 = 858.0     # F5 confirmatory AI 90%-life RMSE (disclosed prior)
    ts90 = rms['theilsen']['0.9']
    sim90 = rms['similarity']['0.9']

    # H-RUL.2: AI vs similarity, per-engine abs error at 90% (AI recomputed for pairing)
    from tune_f5 import eval_ai
    sel_a = json.loads((F5 / 'selected_ai.json').read_text())['selected']
    ma = eval_ai(sel_a['rul']['params'], 'train', 'test', seed=0)
    ai_e90 = {}
    for r in ma['rul_rows']:
        if abs(r['frac'] - 0.9) < 1e-6:
            ai_e90.setdefault(r['engine'], abs(r['err']))
    ids = [e for e in test_ids if e in per_engine90['similarity'] and e in ai_e90]
    w = wilcoxon([per_engine90['similarity'][e] for e in ids],
                 [ai_e90[e] for e in ids], alternative='greater')
    verdict = {
        'rmse_90': {'theilsen': ts90, 'similarity': sim90, 'ai': ai90},
        'rmse_all': rms,
        'H-RUL.1_advanced_narrows': {
            'confirmed': bool(sim90 < ts90),
            'note': f'similarity {sim90:.0f} vs Theil-Sen {ts90:.0f} at 90% life'},
        'H-RUL.2_ai_still_wins': {
            'ai_rmse90': ai90, 'similarity_rmse90': sim90,
            'wilcoxon_p': float(w.pvalue),
            'confirmed': bool(ai90 < sim90 and w.pvalue < 0.05)},
    }
    (F5.parent / 'f8' / 'lrul_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"RMSE @90%:  Theil-Sen {ts90:.0f}  similarity {sim90:.0f}  AI {ai90:.0f}")
    print(f"H-RUL.1 advanced narrows: {sim90:.0f}<{ts90:.0f} -> {verdict['H-RUL.1_advanced_narrows']['confirmed']}")
    print(f"H-RUL.2 AI still wins: AI {ai90:.0f} < sim {sim90:.0f}, Wilcoxon p={w.pvalue:.4f} -> {verdict['H-RUL.2_ai_still_wins']['confirmed']}")


if __name__ == '__main__':
    main()
