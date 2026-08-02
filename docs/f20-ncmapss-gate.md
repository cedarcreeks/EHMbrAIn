# N-CMAPSS feasibility gate (path A: decide before downloading)

Run 2026-07-30. **Outcome: feasible, by a method the plan did not consider — and the
by-product is worth more than the original goal.**

The gate was designed to cost ~2 h and to answer one question before committing to a
multi-gigabyte download and a new pipeline: *can the F10 certificate be recomputed on
N-CMAPSS, given that it needs an influence coefficient matrix the dataset does not ship?*

---

## Verified

From the PHM Society 2021 Data Challenge document, written by the dataset's own authors
(Arias Chao, Kulkarni, Goebel, Fink), and from the distribution headers:

| Item | Value | How |
|---|---|---|
| Full archive size | **14.68 GB** (15 760 443 389 B) | `curl -I` on the S3 distribution |
| Per-subset download | **not available** — one zip for DS01–DS08 | same |
| Measured sensors `X_s` | **14**: Wf, Nf, Nc, T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50 | Challenge doc, Table 5 |
| Scenario descriptors `W` | **4**: alt, Mach, TRA, T2 | Table 4 |
| Auxiliary `A` | unit, cycle, Fc (flight class), hs (health state) | Table 6 |
| Failure modes | **7**, over efficiency (E) and/or flow (F) of fan, LPC, HPC, HPT, LPT | Table 2 |
| Units per subset | 10–15; DS01/DS04–DS07 have 10, DS03/DS08a have 15 | Table 2 |
| Underlying model | C-MAPSS (Frederick, DeCastro & Litt, NASA 2007) — a Simulink model | refs [2], [3] |

**Correction to a belief I had flagged as unverified:** I had said "roughly 1.2 GB". It is
**14.68 GB**, twelve times larger. The report's existing justification for substituting FD001
— that the N-CMAPSS distribution "violates the desktop-reproducibility norm" — is therefore
understated rather than overstated, and stands.

**Correction to my own framing:** I claimed N-CMAPSS "ships the true health parameters". True
of the **repository** distribution, which is what DS02 belongs to. The **Challenge** subset
deliberately strips them: its Table 3 lists only `W`, `X_s`, `Y` (RUL) and `A` — no `T`, no
`X_v`. So the claim holds only for the right distribution, and DS02 is explicitly noted as
"included in the data repository but excluded from the Challenge dataset".

## Not verified without the files

- Exact names and count of the `T` group (believed 10: efficiency and flow modifiers for five
  rotating components, consistent with the seven failure modes in Table 2, but not confirmed).
- DS02's unit count and its individual file size.

These are cheap to confirm once downloaded and should not be quoted before then.

## The obstacle, and why it dissolves

**No Jacobian, sensitivity matrix or influence coefficients are published anywhere** — not in
the Challenge document, not in the dataset paper, not in the benchmark literature. The model is
treated as a black box by everyone using it. So the original worry was correct: F10 as written
cannot simply be ported.

But the ICM does not have to come from NASA. **The dataset itself contains everything needed to
estimate one.** Each sample pairs the health state `θ` with the measurements `x_s` at a known
operating condition `w`. Regressing sensor deviations on `θ` at matched `w` gives an
**empirically estimated influence coefficient matrix** — the same object this project computes
from pyCycle, obtained by regression instead of by perturbing a deck.

That is a real method, not a workaround: this project's ICM *is* a Jacobian, and estimating a
Jacobian from paired (input, output) samples is the ordinary way to get one when the model is
closed.

**Identifiability caveat, stated up front.** Degradation within a unit is coordinated across
components, so `θ` columns are correlated and a single subset may not contain enough
independent variation to separate all ten. The seven failure modes are spread *across* subsets
(Table 2: DS01 affects HPT efficiency only; DS05 HPC; DS06 LPC+HPC; DS08a all five), so
pooling several subsets is what buys the variation. This must be checked with a condition
number or a rank test before any estimated ICM is used, and if it fails, that failure is itself
the finding.

## The by-product, which is worth more than the original goal

The plan wanted N-CMAPSS to revalidate F10 and F11 off this project's own generator. An
empirically estimated ICM enables something sharper:

> **Measure the confusable-pair angle on a completely independent engine model.**

The project's central geometric claim — that $\eta_{\mathrm{HPC}}$ and $\eta_{\mathrm{HPT}}$
sit $1.3^{\circ}$ apart in a rank-3 cockpit measurement space, and that this is a property of
the physics rather than of our calibration — has only ever been measured on our own pyCycle
twin. N-CMAPSS is a different model, of a different engine, built by different people, with 14
sensors instead of 3. Computing the same angle there tests whether the wall is a fact about
turbofans or an artifact of one deck.

Both outcomes are publishable and neither is available any other way:

- **Angle comparably small on the shared sensors** → the wall is a property of gas-path
  physics, and this document's central negative generalises beyond its own simulator. This
  would be the strongest answer to the circularity objection the project has.
- **Angle materially different** → our ICM geometry is model-specific, which bounds every
  geometric claim in the document and would have to be said plainly.

Note the sensor sets differ: N-CMAPSS gives 14 measurements where the cockpit set gives 3. The
honest comparison is on the shared subset (Nf/Nc/Wf/T48-ish maps to our N2/WF/EGT), with the
full 14-channel angle reported alongside as the analogue of our *extended* set — which this
project already measures at $25$–$30^{\circ}$ (§sec:gate-t).

## Recommendation

**Proceed, but scoped and declared.**

1. Download once, **outside** the reproducible pipeline. `make all` must not depend on 14.68 GB;
   this becomes a separately-invoked study with its own note, exactly as the FD001 substitution
   is disclosed today.
2. **Estimate the ICM first, and gate on its conditioning.** If the rank test fails, stop and
   report that — cheap, and it kills the rest honestly.
3. **Confusable-angle comparison before F10/F11.** It is the higher-value result, it needs only
   the estimated ICM, and it does not require porting the certificate machinery.
4. Only then consider recomputing the certificate and the floor.

Estimated: download plus HDF5 loader plus quality audit is a day; the angle comparison is hours
once the ICM exists; the certificate port is open-ended and should stay gated behind step 3.
