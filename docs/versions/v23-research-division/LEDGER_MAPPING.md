# V23 ledger mapping and prediction boundary

This mapping is specified before scientific execution. ToMpathy owns the record
models. Ghost imports its pinned `schema.py` and `ids.py`; it does not redefine
canonical classes. JSONL transport has a first line `{"header": ...}` containing
the canonical schema version/hash, record type, transport version and source
identity. Subsequent lines are canonical model payloads. Headers are transport
metadata, never ledger records. Content is addressed by SHA-256, with a separate
synthetic blob table. Each record has a matching Provenance record.

One case is a synthetic maker, artifact and session, except population worlds,
which preserve distinct makers and their observed population membership. IDs
are opaque deterministic identifiers independent of hidden variables. Times are
synthetic sequential observed-event times, not truth labels. Every query is an
unresolved Decision: `taken=null`, empty `outcome`, and no future correction.
No truth-sidecar path is passed to an engine module. Reader inputs are a whitelist
of the family observation fields below; arbitrary context/outcome keys fail.

| Family | Canonical observed records and reconstruction |
|---|---|
| R1 | Own measurement, claim, selection bits and operation bits are separate binary Decisions, with an Artifact Revision for each produced claim/signature. Their observed channel and sequence are public. Property truth, goal, technique, agenda and law remain sidecar-only. |
| R2 | Each observed prefix channel and flag is a binary Decision. Unavailable initial content/value channels are omitted. No future prefix is emitted. Purpose/value/technique truth remains isolated. |
| R3 | Each offered perturbation is an Option with state/effect/cost features; the chosen perturbation is `taken`. Current menu is unresolved. Set points, cost role and law are evaluator labels. |
| R4 | Observed leaf bits and the root/function cue are binary Decisions attached to revisions at observed node positions. Unseen nodes are absent. The target root and other true states are isolated. |
| N1 | Each maker retains its own offered menus, historical choices and unresolved current-context and changed-context menus. Population membership is observed; culture, types, drift and law are isolated. |
| N2 | Binary Decisions carry unitless latency bins in observed context, class cues, previous mark and any visible opponent payoff. True class and diagnosticity are isolated. |
| N3 | Offered edits are Options. The selected edit is a Decision and, for a nonempty edit, a Correction with before/after content references and parent-linked Revision witnesses. Final choices are separate Decisions; local context is a ContextEvent. Direction and correction class are isolated. |
| N4 | Choice Decisions distinguish abstention from an observed selected option. Need level is a ContextEvent. The forced-choice query has no act or choice outcome. Recipe and vigor truth are isolated. N4b uses history Decisions and an observed contingency-change event. |
| N5 | Four handling options, observed suggestion quality and historical choice are Decisions. The next quality is available before the unresolved query. Personal propensity and quality law are isolated. |
| N6 | Claim/procedure observations are Decisions; the next action is unresolved. Warmth, competence, future action and realized payoff are isolated. |
| N7 | Stated recipe scores are StatedPreferences. Menus and choices are Decisions; the future query is unresolved. Stated preferences are observations, never ground truth. Hidden recipe, aspiration and shift labels are isolated. |
| N8 | Threat is a ContextEvent; public play signal and action are Decisions. True preference and mode are isolated. |
| T1 | Synthetic sessions, prompts, offered suggestions, handling Decisions and subsequent Corrections; aggregate real-shape provenance only, no human text. |
| T2 | Shared game artifacts, move Decisions, clock/opponent ContextEvents; true values, skill, style and mark classes isolated. |
| T3 | Multi-maker artifacts, parent-linked Revisions, edit Decisions, decision-scoped summary StatedGoals and observed revert Decisions; hidden goal trees and honesty isolated. |
| T4 | Shared item artifacts, ordinal-rating Decisions, StatedPreferences and reason StatedGoals; group membership only where declared visible. Hidden reliability and recipes isolated. |

Conformance must cover model validation, cross-record references, chronology,
content hashes, independent sidecar canaries, ID permutation, order-free row
permutation, native/ledger forecast agreement and forbidden future-field rejection.
Synthetic relative order remains visible when sequence matters; incidental wall
clock time is excluded. Real CoAuthor execution belongs to Sounding Line and is a
separate receipt, not implied by synthetic conformance.
