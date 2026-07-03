"""Health-trajectory library (WP2.1): per-mechanism degradation profiles,
wash/FOD events and their composition into the 10-parameter health vector
x(n) over flight cycles.

All deviations in percent, matching the ICM convention. Vectorized numpy:
a 30 000-cycle engine trajectory costs milliseconds.

Health-parameter order (must match ehmbrain.perf.icm.HEALTH_PARAMS):
    [fan.eta, fan.flow, lpc.eta, lpc.flow, hpc.eta, hpc.flow,
     hpt.eta, hpt.flow, lpt.eta, lpt.flow]
"""

from dataclasses import dataclass, field

import numpy as np

COMPONENTS = ['fan', 'lpc', 'hpc', 'hpt', 'lpt']
PARAM_INDEX = {f'{c}.{k}': 2 * i + j for i, c in enumerate(COMPONENTS)
               for j, k in enumerate(('eta', 'flow'))}
N_PARAMS = len(PARAM_INDEX)


@dataclass
class EngineConfig:
    """Everything random about one engine, drawn once (reproducible by seed)."""
    engine_id: int
    multipliers: dict            # mechanism -> severity multiplier
    wash_cycles: np.ndarray      # cycle numbers of wash events
    wash_recovery: np.ndarray    # recovery fraction per wash
    fod_cycles: np.ndarray       # cycle numbers of FOD events
    fod_component: list          # component hit per event
    fod_step_pct: np.ndarray     # eta step per event (negative)
    drifts: dict = field(default_factory=dict)  # sensor -> (onset_cycle, rate_per_cycle)
    egtm_new_C: float = 85.0
    acute: dict | None = None    # {'param', 'magnitude_pct', 'onset', 'ramp_cycles'}


def sample_engine_config(engine_id, catalog, max_cycles, rng):
    """Draw one engine's degradation personality from the catalog."""
    sigma = catalog['engine_variability']['rate_multiplier_sigma']
    mults = {m: float(rng.lognormal(0.0, sigma)) for m in catalog['mechanisms']}

    ev = catalog['events']
    lo, hi = ev['wash']['interval_cycles']
    interval = rng.uniform(lo, hi)
    washes, c = [], interval
    while c < max_cycles:
        washes.append(c)
        c += interval * rng.uniform(0.9, 1.1)
    washes = np.array(washes)
    rlo, rhi = ev['wash']['recovery_frac']
    recovery = rng.uniform(rlo, rhi, size=len(washes))

    lam = ev['fod']['poisson_rate_per_kcycle'] * max_cycles / 1000.0
    n_fod = rng.poisson(lam)
    fod_cycles = np.sort(rng.uniform(0, max_cycles, size=n_fod))
    fod_comp = list(rng.choice(ev['fod']['components'], size=n_fod))
    slo, shi = ev['fod']['step_eta_pct']
    fod_step = rng.uniform(slo, shi, size=n_fod)

    drifts = {}
    sf = catalog['sensor_faults']['drift']
    for sensor, p in sf['prob_per_engine'].items():
        if rng.uniform() < p:
            onset = rng.uniform(*sf['onset_frac_of_life']) * max_cycles
            rate = rng.uniform(*sf['rate_per_kcycle'][sensor]) / 1000.0
            drifts[sensor] = (float(onset), float(rate))

    var = catalog['engine_variability']
    egtm = float(rng.normal(var['egtm_new_mean_C'], var['egtm_new_sigma_C']))

    acute = None
    af = catalog.get('acute_faults')
    if af and rng.uniform() < af['prob_per_engine']:
        param = str(rng.choice(list(af['targets'])))
        lo, hi = af['targets'][param]
        # onset stored as a fraction of ACTUAL life; the fleet builder resolves
        # it in a second pass once the chronic-only life is known (an absolute
        # draw against max_cycles would push most episodes beyond end of life).
        acute = {'param': param,
                 'magnitude_pct': float(rng.uniform(lo, hi)),
                 'onset_frac': float(rng.uniform(*af['onset_frac_of_life'])),
                 'ramp_cycles': float(rng.uniform(*af['ramp_cycles']))}

    return EngineConfig(engine_id, mults, washes, recovery, fod_cycles,
                        fod_comp, fod_step, drifts, egtm, acute)


def _mechanism_series(mech, spec, mult, n, cycles):
    """One mechanism's contribution: array (n, N_PARAMS). Vectorized."""
    x = np.zeros((n, N_PARAMS))
    profile = spec['profile']
    if profile == 'saturating_exponential':
        shape = 1.0 - np.exp(-cycles / spec['tau_cycles'])
        for comp, (eta, flow) in spec['asymptote_pct'].items():
            x[:, PARAM_INDEX[f'{comp}.eta']] += mult * eta * shape
            x[:, PARAM_INDEX[f'{comp}.flow']] += mult * flow * shape
    elif profile == 'linear':
        for comp, (eta, flow) in spec['rate_pct_per_kcycle'].items():
            x[:, PARAM_INDEX[f'{comp}.eta']] += mult * eta * cycles / 1000.0
            x[:, PARAM_INDEX[f'{comp}.flow']] += mult * flow * cycles / 1000.0
    elif profile == 'bilinear':
        brk = spec['breakin_cycles']
        shape = np.minimum(cycles / brk, 1.0)
        for comp, (eta, flow) in spec['breakin_pct'].items():
            x[:, PARAM_INDEX[f'{comp}.eta']] += mult * eta * shape
            x[:, PARAM_INDEX[f'{comp}.flow']] += mult * flow * shape
        for comp, (eta, flow) in spec['late_rate_pct_per_kcycle'].items():
            late = np.maximum(cycles - brk, 0.0) / 1000.0
            x[:, PARAM_INDEX[f'{comp}.eta']] += mult * eta * late
            x[:, PARAM_INDEX[f'{comp}.flow']] += mult * flow * late
    elif profile == 'linear_accelerating':
        accel = 1.0 + (cycles / 10000.0) ** spec['acceleration']
        for comp, (eta, flow) in spec['rate_pct_per_kcycle'].items():
            x[:, PARAM_INDEX[f'{comp}.eta']] += mult * eta * cycles / 1000.0 * accel
            x[:, PARAM_INDEX[f'{comp}.flow']] += mult * flow * cycles / 1000.0 * accel
    else:
        raise ValueError(f'unknown profile {profile}')
    return x


def _fouling_series(spec, mult, cycles, wash_cycles, wash_recovery):
    """Fouling with wash sawtooth, exact segment-wise solution.

    Between washes the fouling level relaxes exponentially toward its
    asymptote: f(t) = asym + (f0 - asym) * exp(-dt/tau). Each wash multiplies
    the current level by (1 - recovery). This regrows toward the SAME
    asymptote after every wash (a permanent-subtraction shortcut would
    under-predict late-life fouling).
    """
    n = len(cycles)
    tau = spec['tau_cycles']
    x = np.zeros((n, N_PARAMS))
    # normalized level s in [0, 1]; per-parameter magnitude applied at the end
    s = np.zeros(n)
    boundaries = [i for wc in wash_cycles
                  if (i := int(np.searchsorted(cycles, wc))) < n]
    seg_starts = [0] + boundaries
    seg_ends = boundaries + [n]
    level = 0.0
    for k, (a, b) in enumerate(zip(seg_starts, seg_ends)):
        if k > 0:
            level *= (1.0 - wash_recovery[k - 1])
        dt = cycles[a:b] - cycles[a]
        s[a:b] = 1.0 + (level - 1.0) * np.exp(-dt / tau)
        level = s[b - 1] if b > a else level
    for comp, (eta, flow) in spec['asymptote_pct'].items():
        x[:, PARAM_INDEX[f'{comp}.eta']] = mult * eta * s
        x[:, PARAM_INDEX[f'{comp}.flow']] = mult * flow * s
    return x


def health_series(cfg, catalog, n_cycles):
    """Full ground-truth trajectory for one engine.

    Returns (x, contributions, events):
      x: (n_cycles, 10) total health deviations [%]
      contributions: dict mechanism -> (n_cycles, 10)
      events: list of dicts (wash / fod with cycle numbers and magnitudes)
    """
    cycles = np.arange(n_cycles, dtype=float)
    contributions = {}
    for mech, spec in catalog['mechanisms'].items():
        if spec['profile'] == 'saturating_exponential':
            contributions[mech] = _fouling_series(spec, cfg.multipliers[mech],
                                                  cycles, cfg.wash_cycles,
                                                  cfg.wash_recovery)
        else:
            contributions[mech] = _mechanism_series(
                mech, spec, cfg.multipliers[mech], n_cycles, cycles)

    # FOD: permanent eta steps on one component
    x_fod = np.zeros((n_cycles, N_PARAMS))
    for wc, comp, step in zip(cfg.fod_cycles, cfg.fod_component, cfg.fod_step_pct):
        i = np.searchsorted(cycles, wc)
        if i < n_cycles:
            x_fod[i:, PARAM_INDEX[f'{comp}.eta']] += step
    contributions['fod'] = x_fod

    # Acute fault episode: fast ramp on one health parameter, then sustained.
    # Skipped until the fleet builder has resolved 'onset' (second pass).
    x_acute = np.zeros((n_cycles, N_PARAMS))
    if cfg.acute is not None and 'onset' in cfg.acute:
        a = cfg.acute
        ramp = np.clip((cycles - a['onset']) / a['ramp_cycles'], 0.0, 1.0)
        x_acute[:, PARAM_INDEX[a['param']]] = a['magnitude_pct'] * ramp
    contributions['acute'] = x_acute

    x = sum(contributions.values())

    events = ([{'type': 'wash', 'cycle': float(c), 'recovery': float(r)}
               for c, r in zip(cfg.wash_cycles, cfg.wash_recovery) if c < n_cycles]
              + [{'type': 'fod', 'cycle': float(c), 'component': comp,
                  'step_eta_pct': float(s)}
                 for c, comp, s in zip(cfg.fod_cycles, cfg.fod_component,
                                       cfg.fod_step_pct) if c < n_cycles])
    if (cfg.acute is not None and 'onset' in cfg.acute
            and cfg.acute['onset'] < n_cycles):
        events.append({'type': 'acute', 'cycle': cfg.acute['onset'],
                       'param': cfg.acute['param'],
                       'magnitude_pct': cfg.acute['magnitude_pct']})
    events.sort(key=lambda e: e['cycle'])
    return x, contributions, events
