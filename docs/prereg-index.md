# Pre-registration index — where each frozen hypothesis actually lives

Written 2026-08-02, covering `prereg-v1` … `prereg-v27`. **Defect D3 closed** — see the end.

## The problem this file was written to fix

Anyone auditing this project opened `docs/` and found `prereg-v1.md` … `prereg-v14.md`,
with `v4` missing and nothing after `v14`. The natural reading is that the
pre-registration discipline — the single practice the project's credibility rests on —
was **abandoned at v14, exactly where the adversarial audit begins**.

That reading was wrong, but the repository supported it. Every tag from `v15` on IS
pre-registered; the *medium* changed and nobody had written that down.

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
| `v4` | F10 identifiability certificate | `docs/f10-proposal.md` (frozen content) + `docs/prereg-v4.md` (retroactive record of outcomes) |
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

## How D3 was closed (2026-08-02)

1. **The report now states the medium change.** `sec:safeguards`, under
   *Pre-registration*: tags `v1`–`v14` freeze a prose document, and from `v15` — the point
   at which the project turned its instruments on its own headline results — the
   pre-registration is the driver docstring sealed by an annotated tag. It says why that
   is an improvement rather than a lapse, with `sec:f24-scale` as the worked case:
   `MIN_CV = 0.05` was committed before the run, the fleet returned 1.15 %, and the
   significant $p$-values the test produced were refused as evidence rather than argued
   about afterwards.
2. **`docs/prereg-v4.md` exists**, and is explicit that it was written after the fact as a
   map to the tag plus the outcomes the tag could not know. Writing a retroactive file
   that *looked* frozen would have been the exact failure twenty-seven tags exist to
   prevent.
3. **The index above is the third piece** — every tag resolvable to its content.

Nothing here changed a result. It changed whether a reader can tell that no result was
changed, which is the only thing pre-registration was ever for.

## Rule going forward

A run is pre-registered when **both** exist before it starts: the driver docstring with
its numeric gate, and an annotated tag naming the hypothesis. Add the row here in the
same commit. Nothing about this changes if the result turns out to be null — see
`prereg-v25` and `prereg-v27`, both of which were tagged before results that went
against the project's own headline.
