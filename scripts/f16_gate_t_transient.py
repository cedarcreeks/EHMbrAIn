"""F16 / Gate T (prereg-v19): would transient response open the confusable angle?

An external proposal (ACTIVE EHM) argued that mechanisms indistinguishable to
steady-state GPA separate during transients, because the ICM is a steady-state
Jacobian while transient response is governed by rotor inertia, heat soak and
volume dynamics. The physics is sound. Before building any transient machinery,
this gate asks the question the proposal never asks: does it open the angle for
the pair the wall is actually made of, eta_HPC against eta_HPT at 1.3 degrees?

A transient contributes two separable things, and only one of them is computable
from what this project owns.

PART 1 -- QUASI-STEADY CONTENT (computable, rigorous). A transient sweeps the
engine through a range of operating points. Each point has its own measured ICM
(this project has six: takeoff, takeoff_hot, climb, cruise, cruise_39k,
cruise_lowpwr). Sampling many points is exactly what F7 showed buys separability
(+79 % from adding one report condition). Gate T computes the LIMIT of that
route: if a transient let you sample the entire measured envelope, at a fixed
total sample budget, how far does the confusable angle open? If even that upper
bound leaves the pair confusable, the quasi-steady route is closed and no
transient schedule can rescue it.

PART 2 -- TRUE DYNAMIC CONTENT (not computable here; stated as a requirement).
Spool time constants and heat-soak lags are genuinely new measurements, not a
resampling of the steady map, and they need a quasi-steady transient deck this
project does not have. Rather than invent one, Gate T inverts the question: given
Part 1's geometry, how large would the DIFFERENCE in dynamic sensitivity between
eta_HPC and eta_HPT have to be, relative to the noise on a time-constant
estimate, for the pair to become separable? That is a specification a future
transient study must meet, derived rather than guessed.

Angles are noise-weighted throughout: signatures are whitened by R^{-1/2} before
any angle is taken, so "separable" means separable relative to sensor noise.

Output: data/processed/f16/gate_t_verdict.json
Usage: uv run python scripts/f16_gate_t_transient.py
"""

import json
from pathlib import Path

import numpy as np

from ehmbrain.datagen.fleet import load_icm
from ehmbrain.perf.icm import HEALTH_PARAMS
from ehmbrain.trad.identifiability import COCKPIT, EXTENDED, SIGMA_PCT

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'data' / 'processed' / 'f16'
POINTS = ['takeoff', 'takeoff_hot', 'climb', 'cruise', 'cruise_39k',
          'cruise_lowpwr']
PAIR = ('hpc.eta', 'hpt.eta')          # the wall
N_SAMPLES = 600                        # total sample budget, held fixed
PRIOR_STD_PCT = 2.0


def whitened_H(point, sensors):
    """R^{-1/2} H at one operating point: rows in units of sigma."""
    H, ch, _ = load_icm(point)
    rows = [ch.index(s) for s in sensors]
    W = np.diag([1.0 / SIGMA_PCT[s] for s in sensors])
    return W @ H[rows]


def angle_deg(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def crb_pair(blocks, weights, sensors):
    """CRB std [%] for the pair, given per-point whitened blocks and the share
    of the sample budget spent at each point."""
    F = np.eye(len(HEALTH_PARAMS)) / PRIOR_STD_PCT ** 2
    for Hw, w in zip(blocks, weights):
        F += (w * N_SAMPLES) * (Hw.T @ Hw)
    S = np.linalg.inv(F)
    idx = [HEALTH_PARAMS.index(p) for p in PAIR]
    return {p: float(np.sqrt(S[i, i])) for p, i in zip(PAIR, idx)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = [HEALTH_PARAMS.index(p) for p in PAIR]
    res = {}

    for sensors, name in ((COCKPIT, 'cockpit'), (EXTENDED, 'extended')):
        blocks = [whitened_H(p, sensors) for p in POINTS]

        # --- per-point angle: what one operating condition can do -------------
        per_point = {}
        for p, Hw in zip(POINTS, blocks):
            per_point[p] = angle_deg(Hw[:, idx[0]], Hw[:, idx[1]])

        # --- stacked over the whole envelope: the transient upper bound -------
        stack = np.vstack(blocks)
        ang_all = angle_deg(stack[:, idx[0]], stack[:, idx[1]])

        # --- CRB at equal budget: one point vs spread over the envelope -------
        one = crb_pair([blocks[POINTS.index('cruise')]], [1.0], sensors)
        spread = crb_pair(blocks, [1.0 / len(POINTS)] * len(POINTS), sensors)

        best_single = max(per_point.values())
        res[name] = {
            'angle_per_point_deg': per_point,
            'angle_best_single_point_deg': best_single,
            'angle_cruise_deg': per_point['cruise'],
            'angle_full_envelope_deg': ang_all,
            'gain_over_cruise': ang_all / per_point['cruise'],
            'crb_pct_cruise_only': one,
            'crb_pct_envelope_spread': spread,
            'crb_ratio': {p: one[p] / spread[p] for p in PAIR},
        }

    # ---- Part 2: what a genuine dynamic measurement would have to deliver ----
    # Add one hypothetical scalar measurement (a time constant) whose whitened
    # sensitivity to the pair is (d_hpc, d_hpt) per unit health. Ask what
    # separation |d_hpc - d_hpt| makes the augmented angle reach a target.
    sensors = COCKPIT
    blocks = [whitened_H(p, sensors) for p in POINTS]
    stack = np.vstack(blocks)
    a, b = stack[:, idx[0]], stack[:, idx[1]]
    req = {}
    for target in (5.0, 10.0, 20.0):
        lo, hi = 0.0, 1e4
        for _ in range(200):                       # bisection on the extra row
            d = 0.5 * (lo + hi)
            aa = np.concatenate([a, [d]])
            bb = np.concatenate([b, [0.0]])
            if angle_deg(aa, bb) < target:
                lo = d
            else:
                hi = d
        req[f'{target:.0f}deg'] = float(0.5 * (lo + hi))
    norm_a = float(np.linalg.norm(a))
    verdict = {
        'question': ('does transient response open the eta_HPC / eta_HPT angle '
                     'enough to matter, before any transient machinery is built'),
        'points': POINTS, 'pair': list(PAIR), 'sample_budget': N_SAMPLES,
        'part1_quasi_steady': res,
        'part2_dynamic_requirement': {
            'note': ('a transient also yields genuinely new measurements (spool '
                     'time constants, heat-soak lags) that are not a resampling '
                     'of the steady map. This states how discriminating such a '
                     'measurement must be, in whitened units, to reach a target '
                     'angle when appended to the full-envelope cockpit stack'),
            'whitened_norm_of_hpc_signature': norm_a,
            'required_differential_sensitivity': req,
            'interpretation': ('a required value >> the steady signature norm '
                               'means the dynamic channel would have to carry '
                               'more information about the pair than the entire '
                               'steady envelope does')},
    }

    ck = res['cockpit']
    verdict['GateT_quasi_steady_opens_angle'] = {
        'angle_cruise_deg': ck['angle_cruise_deg'],
        'angle_full_envelope_deg': ck['angle_full_envelope_deg'],
        'gain': ck['gain_over_cruise'],
        'confirmed': bool(ck['angle_full_envelope_deg'] >= 5.0),
        'criterion': ('full-envelope angle reaches 5 deg on the cockpit set -- '
                      'roughly four times the nominal 1.3 deg, the scale at '
                      'which the confusable pair would stop being confusable')}
    (OUT / 'gate_t_verdict.json').write_text(json.dumps(verdict, indent=2))

    for name in ('cockpit', 'extended'):
        r = res[name]
        print(f'== {name} ==')
        for p, v in r['angle_per_point_deg'].items():
            print(f'   {p:16s} {v:6.2f} deg')
        print(f'   full envelope    {r["angle_full_envelope_deg"]:6.2f} deg  '
              f'({r["gain_over_cruise"]:.2f}x cruise)')
        print(f'   CRB pair cruise-only  '
              + '  '.join(f'{k} {v:.3f}%' for k, v in r['crb_pct_cruise_only'].items()))
        print(f'   CRB pair envelope     '
              + '  '.join(f'{k} {v:.3f}%' for k, v in r['crb_pct_envelope_spread'].items()))
    print(f"\n  Gate T (quasi-steady) confirmed: "
          f"{verdict['GateT_quasi_steady_opens_angle']['confirmed']}")
    print(f"  steady signature norm {norm_a:.1f}; dynamic sensitivity needed for "
          f"5/10/20 deg: " + ', '.join(f'{v:.1f}' for v in req.values()))


if __name__ == '__main__':
    main()
