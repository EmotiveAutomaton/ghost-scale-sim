# V13 — Common Ground. Implementation translation of the top-level spec

**Written during the build, before any scientific result was read, 2026-08-25.** The scientific
program is [`V13_SPEC.md`](V13_SPEC.md) at the repository root (the curator's immutable
handoff; it files beside this page when the version closes) and is not narrowed here. This page
records the viability assessment the curator asked for, how the program maps onto code, the
constructions chosen where the spec left a choice, and the freezes. Deviations after data go to
`RESULTS.md` with the original beside the replacement.

## Viability assessment

The spec was assessed **fully viable and built in its entirety**: all 132 mandatory cards, the
20-attack matrix, four evidence lanes, the tier-parameterized workload with a discarded runtime
pilot, the anti-shrink validator, the committed completion ledger, and the fresh-clone receipt.
No design decision required the curator; the spec anticipated its own forks (the tier rule for a
forecast outside the envelope, the one-repair rule, downstream-oracle continuation). Four
judgment calls were taken and are recorded here rather than silently:

1. **The 72–96 hour envelope has a gap case.** If some tier forecasts under 72 h and the next
   above 96 h, the rule chosen is: take the largest tier whose forecast is at or under 96 h, then
   instantiate expansion packets in the frozen §7.3 order until the forecast enters the envelope
   (stopping early rather than overshooting 96 h when close). This is the natural completion of
   the spec's two stated boundary rules and is applied mechanically by the pilot.
2. **"Human-shaped goal recovery" (C13)** is implemented as *common-shaped*: attention to the
   common axes of the constructed substrate. The spec's own prohibition table forbids reading
   `z_common` as a human universal, so the wording was taken as a slip, not a target.
3. **Fresh-clone "install from the lockfile"**: `uv` is absent on this machine. The receipt
   records what was done — validation with the pinned interpreter against the clone by default,
   with `--install` attempting a fresh venv from the project pins. The receipt never claims an
   install that did not happen.
4. **One surface fact is reported rather than matched away (I07):** across a source's artifacts,
   the one-sidedness of evidence polarity is itself diagnostic of persuasion- and concealment-like
   goals. Per-artifact surfaces are matched exactly (the mirror construction); the across-artifact
   consistency is a correspondence structure, and the I07 verdict names it instead of pretending
   it does not exist.

## Layout

| spec object | implementation |
|---|---|
| nested world: families, groups, ecologies, individuals, state, episode | `ghostscale/validation/soundingline/v13/world.py` |
| exact reader over the hypothesis grid; channel-factorized likelihoods; EIG | `v13/exact.py` |
| twelve prior routes and joint matching (entropy + expected divergence) | `v13/priors.py` |
| selection and precision attention, policies, no-information nulls | `v13/attention.py` |
| cost vectors, menus, actors, rival causes, weighting families, neglect mimic | `v13/costs.py` |
| communicative goals, source reliability, content, trust dynamics, uptake | `v13/goals_trust.py` |
| role-relative event graphs, seven team ecologies, rewrites, readers | `v13/hierarchy.py` |
| projection correction, evidence types, robust readers, ensembles | `v13/projection.py` |
| legacy PyMDP active reader and probe audits | `v13/pymdp_reader.py` |
| cards, one module per trunk; the unit/reduce contract and the seven-gate battery | `v13/cards/` |
| manifest, tiers, expansions, expected cells, ledgers | `v13/manifest.py`, `results/v13/` |
| pre-specification locks (structural → workload → scientific) | `ghostscale/prereg_v13.py` |
| resumable runner, pilot, forecast; worker accounting | `runners/run_v13.py`, `v13/runtime.py` |
| confirmation, packet, validator, fresh clone | `runners/run_v13_confirmation.py`, `report_v13.py`, `validate_v13_program.py`, `fresh_clone_v13.py` |
| verdicts | `results/validation/soundingline/v13/<CARD>.json`; `transfer/`, `attacks/`, `confirmation/` |
| tests | `tests/test_v13_gates.py`, `test_v13_metamorphic.py`, `test_v13_fresh_clone.py` |

## Constructions chosen

**Families.** Six factorizations (pair, chain, additive, gated; sparse and mixture reserved for
the transfer lineage and expansion E1), each with its own goal-to-feature map, method structure,
action constraints and profile grid. A reader needs the family's structure for the right
likelihood; a label buys nothing (card I04's family-identification check).

**Profiles.** Individuals are label-anchored: a label drawn from the group's distribution over
grid profiles, plus a continuous jitter share (5–25% by world). This keeps the truth of a card
well defined while the population stays continuous. Expertise corruption blurs execution toward
the family's average emission — low competence is undifferentiated execution, not random junk.

**Channels.** Every artifact carries disjoint evidence channels (surface, common structure,
group convention, mechanics, goal consequences, shaping, anomaly, process records; opportunity,
cost and source history attached by their modules). Disjointness makes attention well defined:
neutral weights reproduce the plain posterior bit for bit (I05), and tempering never redefines
the evidence.

**Readers.** The reader's likelihood is its own execution model (its expertise); a small robust
floor (2% uniform) represents execution noise the reader cannot model and is part of the reader,
not the world. Exact inference is the reference everywhere; PyMDP appears only where the reader
acts, against the exact EIG and with its discrepancy surface mapped first (I10).

**Matching.** Routes 1, 2, 7, 8, 9 are matched on entropy exactly (bisection) and on expected
divergence to the truth over the sampled population: optimized for the generic-local centre,
chosen-best for the equal-local reader with the residual and a sensitivity bound (nats of gain a
nat of residual could manufacture) travelling in every C and P verdict.

**The central/shared-brief pair** shares its random streams: the same subordinates, proposals,
corrections and artifacts, differing only in who issues each correction. Twin identity is checked
directly (I09), which is stronger than any classifier.

**Lanes.** Discovery worlds 0–511, confirmation 1000–1095, transfer 2000–2127 (fresh
factorizations, fresh cost ecologies, an extra domain), pilot 9000–9003 (quarantined, ignored by
git). Every seed string carries its lane; I12 checks disjointness and clone determinism.

**Smoke pass.** Before anything scientific, every card ran repeatedly at smoke sizes on one world
with verdicts redirected to scratch. The gate corrections that pass forced are visible in the
git history of the card modules; the ones that changed what a card measures are listed in
`RESULTS.md` if any survive to a deviation.

## Freezes

1. Structural lock before the pilot: cards, criteria, closures, flights, attack relevance,
   generator sources, the expected-cell template, the report interface.
2. The discarded pilot measures twelve heavy cards at two per-world sizes; a power law in
   makers-per-world interpolates the other tiers; the workload lock freezes the tier, the
   expansion packets, the instantiated cell matrix and the forecast before discovery.
3. The scientific lock binds both. `runners/run_v13.py --stage discovery` refuses to start
   without it.

## What this translation does not do

It does not change a criterion, add or remove a card, weaken the shared-brief rival, or reopen
anything V12 closed. V12's record is untouched; its reading is corrected only through I01/I02 and
the C04 comparison the spec ordered.
