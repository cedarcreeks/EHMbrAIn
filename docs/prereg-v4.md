# Pre-registration v4 — F10 identifiability certificate (frozen 2026-07-05, tag `prereg-v4`)

> **This file was written on 2026-08-02, after the fact, to close a hole in the numbering.**
> It is an index and a record of outcomes, **not** the frozen document. The authoritative
> frozen content is [`docs/f10-proposal.md`](f10-proposal.md) as it stood at the tag —
> recover it with `git show prereg-v4:docs/f10-proposal.md`. Nothing below restates a
> hypothesis that was not already in that file at freeze time; the outcome column is
> obviously later, and is marked as such.
>
> The hole mattered because `v4` is this project's most-promoted result and its
> most-audited one. A missing pre-registration file at exactly that tag is the worst
> possible place for a gap, even when the content exists under another name.

## What was frozen

The first **per-engine, ground-truth-validated identifiability certificate** for gas-path
diagnosis: accumulate Fisher information over an engine's actual flown conditions, invert
it, and obtain a per-direction bound on what any diagnosis of that engine can know. Then —
the part only a synthetic-truth benchmark permits — test that statement against the true
component state.

**Frozen estimator.** Fisher accumulation over each engine's cruise history (subsample
stride 20) in the last 30 % of life; $R = \mathrm{diag}(0.07, 0.5, 0.23)^2\ \%^2$; prior
$P_0 = 4\ \%^2 I$; $\mathrm{CRB} = F^{-1}$. Actual error = mean $|\hat{x} - x|$ over the
last 15 % of life. Coverage = fraction of engines whose true late-life $x$ lies inside the
$\chi^2(10)$ 90 % ellipsoid centred at the estimate.

**Disclosed exploratory exposure at freeze time:** $\rho = 0.70$ ($p = 0.025$), measured
2026-07-05 before the confirmatory pass. Disclosed in the frozen document, as the protocol
requires.

## Frozen hypotheses and what happened

| | frozen decision rule | outcome |
|---|---|---|
| **H10.1** honesty | CRB-predicted precision ranks actual per-direction error, Spearman $\rho \ge 0.6$, $p < 0.05$ | confirmatory $\rho = 0.697$ — **and see below** |
| **H10.2** coverage | $\chi^2(10)$ 90 % region contains the true 10-dim state at 86–94 % | **refuted** (0.85). Post-freeze conformal fix disclosed: radius calibrated on validation engines → 0.95 |
| **H10.3** acquisition | extended sensors shrink median CRB in the three unobservable efficiency directions by $\ge 2\times$, and the certificate predicts which directions each addition rescues | **confirmed** (station probes rescue HPC efficiency $45\times$) |

The frozen risk clause said: *"the phase cannot produce a dishonest positive."* That turned
out to be the one thing in this document that was wrong, and finding out why took four
later experiments.

## What four later controls did to H10.1

H10.1 was pre-registered without a control arm — F10 predates that discipline. Every line
after it has one, and in each case the control changed the verdict. Applied retroactively
here:

- **F22** (`prereg-v25`): column-shuffle the influence matrix, destroying the physics while
  keeping the coupling. Null median 0.242, p95 0.722, **$p = 0.085$**. Real signal well
  above the null median, never significant against it.
- **F21** (`prereg-v24`): on N-CMAPSS, $\rho = 0.842$ — but a shuffled matrix reaches 0.830.
  The correlation there measures the shared matrix, not the physics.
- **F23** (`prereg-v26`): a learned estimator that never sees $\mathbf{H}$ collapses the null
  to 0.006, confirming the coupling is gone. The certificate still ranks at $\rho = 0.455$
  on all ten seeds, at **$p = 0.100$**. The design worked; ten directions is the wall.
- **F24** (`prereg-v27`): the magnitude route fails its own pre-declared gate — the CRB
  varies **1.15 %** between engines, so the powered statistic has nothing to work with.

**Net effect on the frozen claim.** Of $\rho = 0.70$, about 0.24 is the matrix the bound and
the estimator share and about 0.46 survives when that is removed; neither clears
significance. H10.1 as frozen is **under-powered, not refuted** — the certificate is not
shown to be dishonest, it is shown to be unmeasurable on this fleet. H10.3 is untouched: it
is a ratio between sensor configurations, and only absolute magnitude was disqualified.

The one experiment that would settle it is a mission-diverse fleet regeneration —
`docs/TODO.md` §3, and `sec:future-c8` in the report.

## Why this record exists in this form

Writing a retroactive file that *looked* frozen would have been the exact failure this
project spent twenty-seven tags guarding against. The tag is the seal; this file is a map
to it, plus the outcomes the tag could not know.

See [`prereg-index.md`](prereg-index.md) for the same mapping across all 27 tags.
