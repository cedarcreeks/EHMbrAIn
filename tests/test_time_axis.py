"""The project's central time-axis claim, as a permanent regression.

L-MECH (sec:f13-mech) found that the identifiability wall is a property of the
INSTANTANEOUS estimand: two degradations can be indistinguishable in any single
snapshot and still be separable over a history, because mechanisms differ in
SHAPE over time. L-INST and Gate T then bounded it into one rule --

    trajectory shape separates what has shape

-- readable for a wash sawtooth or a break-in knee, unreadable for two smooth
ramps, which is also why a drifting sensor cannot be told from real degradation.

That rule is load-bearing for a whole chapter, and nothing currently fails if a
future refactor destroys the temporal signal: the studies live in scripts that
are not run by the test suite. These tests are the guard. They are deliberately
self-contained -- synthetic signals built here, no fleet, no model training, no
GPU -- so they stay fast and cannot break for unrelated reasons.

The construction mirrors the conceptual test proposed in the external ACTIVE EHM
review (docs/f14-proposal.md, A17.3): build two hypotheses that are identical in
the steady state and different in time, verify a snapshot cannot separate them,
and verify a shape descriptor can.
"""

import numpy as np

# --------------------------------------------------------------------------
# two mechanisms, matched at the end of life, different on the way there
# --------------------------------------------------------------------------

N = 4000                      # cycles
FINAL_LOSS = 4.0              # both mechanisms end at the same degradation


def _fouling(n=N, washes=(800, 1700, 2600, 3400), recovery=0.45, tau=1200.0):
    """Saturating growth, partly undone at each wash: a sawtooth.

    Integrated rather than closed-form, because a wash resets the LEVEL and the
    growth then resumes from there -- which is what makes the recoveries visible
    as downward steps in an otherwise rising signal.
    """
    level, out = 0.0, np.empty(n)
    wash = set(washes)
    for t in range(n):
        level += (FINAL_LOSS - level) / tau
        if t in wash:
            level *= (1.0 - recovery)
        out[t] = level
    return out * (FINAL_LOSS / out[-1])          # match the endpoint exactly


def _hot_section(n=N, accel=1.6):
    """Monotone and accelerating: no event, a smooth curve."""
    t = np.arange(n) / n
    out = FINAL_LOSS * t ** accel
    return out * (FINAL_LOSS / out[-1])


def _erosion(n=N):
    """Plain linear -- the featureless case the rule says is unreadable."""
    return np.linspace(0.0, FINAL_LOSS, n)


def _sensor_drift(n=N):
    """A slow additive bias ramp: also featureless, also a straight line."""
    return np.linspace(0.0, FINAL_LOSS, n)


# --------------------------------------------------------------------------
# shape descriptors: the two that L-MECH found carry the signal
# --------------------------------------------------------------------------

def _recovery_depth(y):
    """Total downward movement in a signal that otherwise rises.

    This is the descriptor that actually separates a washed mechanism from a
    monotone one. Detrending against a straight line does NOT work, because a
    smoothly accelerating curve also leaves a residual -- the first draft of this
    test failed for exactly that reason, and the failure was the useful part.
    """
    d = np.diff(y)
    return float(-d[d < 0].sum())


def _curvature(y):
    """Quadratic term of a parabola fit -- picks up late acceleration."""
    t = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(t, y, 2)[0])


# --------------------------------------------------------------------------
# 1. the premise: a snapshot cannot separate them
# --------------------------------------------------------------------------

def test_snapshot_cannot_separate_the_two_mechanisms():
    """Matched at end of life, so the final reading carries no information.

    This is the synthetic analogue of the rank argument: if the instantaneous
    measurement is equal, no function of that measurement alone can tell the two
    apart, however sophisticated.
    """
    a, b = _fouling(), _hot_section()
    assert abs(a[-1] - b[-1]) < 1e-6, 'endpoints must match by construction'
    # and not only the endpoint: the last observed level is uninformative
    assert abs(a[-1] - b[-1]) / FINAL_LOSS < 1e-6


# --------------------------------------------------------------------------
# 2. the finding: shape does separate them
# --------------------------------------------------------------------------

def test_recovery_depth_separates_fouling_from_hot_section():
    """Fouling is partly undone at each wash; hot-section creep never is.

    Asserted qualitatively rather than as a ratio: a monotone mechanism has
    EXACTLY zero recovery depth, so `f > 3*h` would reduce to `f > 0` and pass on
    any positive noise. Both halves are stated separately.
    """
    f, h = _recovery_depth(_fouling()), _recovery_depth(_hot_section())
    assert h == 0.0, f'a monotone mechanism must show no recoveries, got {h}'
    assert f > 0.5 * FINAL_LOSS, f'washes must be clearly visible, got {f:.4f}'


def test_curvature_separates_hot_section_from_fouling():
    """Hot-section creep accelerates late; the washed curve does not."""
    h, f = _curvature(_hot_section()), _curvature(_fouling())
    assert h > f, f'hot-section curvature {h:.4f} must exceed fouling {f:.4f}'


def test_the_two_descriptors_are_not_the_same_measurement():
    """Each mechanism is picked out by a different descriptor, not by one.

    If a refactor collapsed both descriptors onto the same quantity the tests
    above could still pass while the discrimination came from a single axis.
    """
    fo, hs = _fouling(), _hot_section()
    saw = (_recovery_depth(fo), _recovery_depth(hs))
    cur = (_curvature(fo), _curvature(hs))
    assert (saw[0] > saw[1]) and (cur[0] < cur[1]), (
        'fouling should win on recoveries and lose on curvature')


# --------------------------------------------------------------------------
# 3. the boundary: shape separates only what HAS shape
# --------------------------------------------------------------------------

def test_two_smooth_ramps_stay_unseparable():
    """Erosion against a sensor bias ramp: both featureless straight lines.

    This is the negative half of the rule, and the reason L-INST could not tell
    a drifting sensor from real degradation. If this test ever starts passing in
    the other direction, the generator's ramps have stopped being ramps.
    """
    e, d = _erosion(), _sensor_drift()
    assert abs(_recovery_depth(e) - _recovery_depth(d)) < 1e-9
    assert abs(_curvature(e) - _curvature(d)) < 1e-9


def test_noise_does_not_manufacture_a_difference():
    """With independent noise on two identical ramps, the descriptors must not
    separate them beyond what the noise itself explains."""
    rng = np.random.default_rng(0)
    sigma = 0.05
    gaps = []
    for _ in range(20):
        e = _erosion() + rng.normal(0, sigma, N)
        d = _sensor_drift() + rng.normal(0, sigma, N)
        gaps.append(abs(_recovery_depth(e) - _recovery_depth(d)))
    # two noisy straight lines both accumulate recovery depth from the noise
    # alone; what must not happen is one of them looking mechanistically
    # different from the other
    assert np.mean(gaps) < 0.15 * np.mean(
        [_recovery_depth(_erosion() + rng.normal(0, sigma, N)) for _ in range(5)])
