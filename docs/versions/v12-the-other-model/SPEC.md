# V12 — The Other Model. Implementation translation of the top-level spec

**Written before any V12 result was seen, 2026-08-24.** The scientific program is
[`V12_SPEC.md`](V12_SPEC.md) beside this page (filed from the repository root when V12 closed) and is not narrowed here. This page
records only how that program maps onto code, the constructions chosen where the spec left a
choice, and the floors each card runs at. Deviations after data go to `RESULTS.md` with the
original beside the replacement.

## Layout

| spec object | implementation |
|---|---|
| world, makers, regimes, domains, opportunity records | `ghostscale/validation/soundingline/v12/world.py` |
| self-model measurement, priors, information-matched controls, similarity rulers | `v12/self_other.py` |
| exact posteriors, joint regime inference, exact expected information gain | `v12/exact.py` |
| legacy PyMDP active reader (probe factor, epistemic value, probe cost) | `v12/pymdp_reader.py` |
| choice world with opportunity strength (trunk R) | `v12/opportunities.py` |
| director/distributed and layered/flat emitters (trunks D, F) | `v12/hierarchy.py` |
| posterior → C_AIF → policy bridge (trunk U) | `v12/uptake.py` |
| cards, one module per trunk | `v12/cards/trunk_*.py` |
| manifest, coverage, runtime, schemas | `v12/manifest.py`, `v12/schemas.py`, `results/v12/` |
| pre-specification lock | `ghostscale/prereg_v12.py` → `results/v12/prereg_v12_lock.json` |
| resumable runner and program validator | `runners/run_v12.py`, `runners/validate_v12_program.py` |
| confirmation pass (wave 5) and generated results packet | `runners/run_v12_confirmation.py`, `runners/report_v12.py` |
| verdicts | `results/validation/soundingline/v12/<CARD>.json` (+ `.produced` marker); confirmation-lane verdicts under `confirmation/` |
| tests | `tests/test_v12_gates.py`, `tests/test_v12_metamorphic.py` |

## Constructions chosen

**Worlds.** Discovery world 0 is the default construction and reproduces V11's objects to the bit
(tested). Discovery worlds 1–11 and confirmation worlds 100–111 are independent randomized
constructions drawn from the declared ranges (`world.RANGES`), each seeded from its lineage id.
The two lineages are disjoint by id and asserted by card I06.

**Regime.** Every profile in the family owns one cue slot among the tail features. A bard emits
its own cue, a neutral maker a random one, a concealer the decoy profile's cue. Pair mass, entropy,
length, and effort are identical across regimes by construction (tested), so the only difference
is inferential correspondence.

**Self-model.** A reader is a maker; its self-model is estimated from its own artifacts and scored
by within-artifact continuation (predict the rest of a held-out artifact from its prefix) against
pooled-frequency and population baselines.

**Information-matched generic prior.** Same entropy as the self-first prior, centred on the
population mean profile, found by bisection on temperature; permuted-self and random-local priors
travel alongside. Residual mismatch in expected distance to truth is reported, not assumed away.

**Solvers.** Exact categorical inference is the reference path everywhere. The legacy PyMDP agent
supplies probe choice and stopping; card I03 maps where its mean-field posterior becomes
confidently wrong and every later PyMDP result is read against that map.

**Floors.** Spec §5.1 floors are recorded per card in the manifest; the validator refuses a card
that lowers one without an amendment.

**Lineages.** Discovery worlds 0–11 carry every card in waves 0–4. The confirmation lineage 100–111 is
untouched until the confirmation pass, which re-runs promoted cards there and keeps their verdicts in a
separate directory. Cards that need fresh worlds during discovery (S08, X12) use a third, transfer lineage
200–211, so nothing in the discovery record has seen a confirmation world.

**Smoke pass.** Before the queue started, every card ran once on world 0 with verdicts redirected to a
scratch directory (`GS_V12_WORLD_LIMIT=1`; the runner refuses to start under that cap). The gate
corrections that pass forced are noted in the card modules; the two that changed what a card measures
are recorded in `RESULTS.md`: the T-trunk battery moved off the CREATOR ceiling to the CURATOR tier with
four steps, and F03 gained a readability ladder so the dependency ruler is validated where blocks are
readable before it is read at the floor.

## Wave order

0: I01, I03, I04, I05, I06. 1: S01–S03, Q01–Q02, B01–B02, U01–U02, R01–R02, T01–T02, D01–D02,
F01–F02, I02. 2–3 per the spec. 4: X. 5: confirmation.

## What this translation does not do

It does not change any criterion, add a card, remove a card, or reopen the families the spec
excludes (affect counts, label sweeps, E8, AL-6 until its gates pass).
