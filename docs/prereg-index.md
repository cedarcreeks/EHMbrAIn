# Pre-registration index — where each frozen hypothesis actually lives

**Status: this file records an OPEN DOCUMENTATION DEFECT (D3 in `TODO.md`).**
Written 2026-08-02, covering `prereg-v1` … `prereg-v27`.

## The problem, stated plainly

Anyone auditing this project opens `docs/` and finds `prereg-v1.md` … `prereg-v14.md`,
with `v4` missing and nothing after `v14`. The natural reading is that the
pre-registration discipline — the single practice the project's credibility rests on —
was **abandoned at v14, exactly where the adversarial audit begins**.

That reading is wrong, but the repository currently supports it, and that is the defect.
Every tag from `v15` on IS pre-registered; the *medium* changed and nobody wrote that down.

## What actually happened

From `v15` the pre-registration moved into the **driver script's module docstring**,
committed and tagged *before* the run. Those docstrings are not comments — they are
32–49 lines each, and they carry what a pre-registration has to carry:

- the hypothesis and where it came from,
- the design and why that design and not another,
- the **gate declared in advance**, with its numeric threshold,
- and what a positive result *and* a null would each mean.

`f24_crb_scale_magnitude.py` is the clearest example: 49 lines, and it fixes
`MIN_CV = 0.05` as executable code before the run, which is why the significant
within-direction p-values it later produced could be refused as evidence rather than
argued about afterwards. A threshold in a docstring is prose; a threshold that is also
a constant the script branches on cannot be quietly moved.

The **annotated git tag** carries the timestamp and a summary, and is the freeze that
matters — the docstring is content, the tag is the seal.

## The index

| tag | subject | pre-registration lives in |
|---|---|---|
| `v1` | global freeze: hypotheses, dataset hashes, tuning budget, statistics | `docs/prereg-v1.md` |
| `v2` | F7 confirmatory | `docs/prereg-v2.md`, `docs/f7-proposal.md` |
| `v3` | F8/L6 H4-v2 | `docs/prereg-v3.md` |
| **`v4`** | **F10 identifiability certificate** | **`docs/f10-proposal.md`** — no `prereg-v4.md` exists |
| `v5` | F8/L9 PCS validation | `docs/prereg-v5.md` |
| `v6` | F8/L7 drift estimation | `docs/prereg-v6.md` |
| `v7` | F8/L4 recoverable fraction | `docs/prereg-v7.md` |
| `v8` | F8/L5 architecture breadth | `docs/prereg-v8.md` |
| `v9` | L-H2 / L-H2b wall-breaking | `docs/prereg-v9.md` |
| `v10` | F8/L-RUL advanced prognostic | `docs/prereg-v10.md` |
| `v11` | F11 prognostic floor | `docs/prereg-v11.md` |
| `v12` | F-OPS unscheduled→scheduled conversion | `docs/prereg-v12.md` |
| `v13` | L-ICM robustness to calibration | `docs/prereg-v13.md` |
| `v14` | C6 noise axis | `docs/prereg-v14.md` |
| `v15` | re-audit of the AI headlines (post-hoc, disclosed) | tag + `scripts/f_uq_reattribution.py`, `scripts/f8_lrul_advanced.py` |
| `v16` | F13 / L-MECH — mechanism attribution over a history | tag + `scripts/f13_gate1_mechanism.py` (39 ln) |
| `v17` | F14 / L-EXT — external Bi-LSTM replication | `docs/f14-proposal.md` + `scripts/f14_bilstm_replication.py` (43 ln) |
| `v18` | F15 / L-INST — instrument or engine? | tag + `scripts/f15_h158_instrument_vs_engine.py` (36 ln) |
| `v19` | F16 / Gate T — would transients open the confusable angle? | tag + `scripts/f16_gate_t_transient.py` (36 ln) |
| `v20` | F17 / H15.3 — certificate-gated hybrid, confirmatory | tag + `scripts/f17_certificate_hybrid.py` (38 ln) |
| `v21` | F18 / H15.2 — where does bidirectionality earn its keep? | tag + `scripts/f18_bidirectionality.py` (37 ln) |
| `v22` | F19 / H15.11 — certificate information at fixed channel count | tag + `scripts/f19_certificate_isolated.py` (41 ln) |
| `v23` | F20 / L-EXT2 — confusable wall on an external engine | `docs/f20-ncmapss-gate.md` + `scripts/f20_ncmapss_icm.py` (33 ln) |
| `v24` | F21 / L-CERT2 — port certificate and floor to N-CMAPSS | tag + `scripts/f21_ncmapss_cert_floor.py` (37 ln) |
| `v25` | F22 — the control F10 never had | tag + `scripts/f22_f10_shuffle_control.py` (32 ln) |
| `v26` | F23 / L-DECOUP — estimator that never sees **H** | tag + `scripts/f23_decoupled_certificate.py` (36 ln) |
| `v27` | F24 / L-SCALE — calibrate CRB scale, test magnitude | tag + `scripts/f24_crb_scale_magnitude.py` (49 ln) |

Recover any of them with `git show prereg-vN` (annotation) or
`git show prereg-vN:scripts/<driver>.py` (the docstring as frozen at the tag).

## What is still pending

1. **A sentence in `sec:safeguards` or `sec:replication`** stating that from `v15` the
   pre-registration medium is the driver docstring sealed by an annotated tag, and
   pointing here. Without it the report describes a practice the repository appears to
   contradict. *This is the item that matters* — the optics are worse than the defect.
2. **`docs/prereg-v4.md`, or a redirect**, so the numbering has no hole. F10 is the
   project's most-promoted result and the most-audited; a missing pre-registration file
   at exactly that tag is the worst possible place for a gap, even though
   `docs/f10-proposal.md` holds the content.
3. **Decide and record whether the medium change was an improvement.** The honest case
   is that it was: a gate written as `MIN_CV = 0.05` in the file that branches on it is
   harder to move after the fact than the same threshold in prose. That argument belongs
   in the methodology chapter, made explicitly, not left for a reader to reconstruct.

## Rule going forward

A run is pre-registered when **both** exist before it starts: the driver docstring with
its numeric gate, and an annotated tag naming the hypothesis. Add the row here in the
same commit. Nothing about this changes if the result turns out to be null — see
`prereg-v25` and `prereg-v27`, both of which were tagged before results that went
against the project's own headline.
