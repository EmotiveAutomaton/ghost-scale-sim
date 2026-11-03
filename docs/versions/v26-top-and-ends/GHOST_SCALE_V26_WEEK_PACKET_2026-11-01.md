# Ghost Scale Sim, V26 week packet (2 to 8 November): the top of the hierarchy and the ends at its end

*Spec sheet for the Ghost agent. Every decision is made here, and section 0.4 says how the agent fills any modelling choice a card leaves open. Return to the curator only on the triggers in section 7.*

**What this week tests.** The curator's walk of 1 November moved the program's hard problem to one place: readers recover the bottom of a maker's hierarchy (who she is, what she's doing locally) and not the top (the higher-order goals and ends the steps serve). His working model:
- a person produces most of her work on an autopilot of compiled expertise, and **stops to plan** (System 2) where the autopilot won't do;
- **global revision** realigns autopilot output with a higher-order goal settled later;
- a reader gets to the top by **slotting in her own expertise**: running her own forward model under the maker's constraints, corrected toward the population;
- the same structure runs **forward** (to act) and **backward** (to read another);
- at the end of the hierarchy are either **terminal values**, set in some way and becoming identity when locked, or **circulating values**, where what matters is how often each one fires over time;
- feelings may be **expertise again**: cached solutions rather than amplifiers of current wants;
- a source a reader **can't reverse-engineer** is taken in like an authority.

Ghost's job is to find, in constructed worlds with exact readers, what each of these can and can't identify, and what Sounding Line should look for in real records.

**Lessons this packet is built on** (V25 review, 30 October):
- V25's thirteen primaries ran in about 2.5 hours, and the week charged 8.6 of 46 CPU-hours. This packet adds a texture lane (amendment F) so the workers stay busy.
- Three V25 designs partly decided their own results: W1 and W6's libraries (V25's tests of reading ends from piece choices, without and with frustration-like signals) could never produce two ends that share every available piece, W7's vetoes (its attention test) let the final text reveal the goals, and two of W13's four shifts (its persistence-or-proxy test) couldn't separate its agent types. Amendment E requires a design pre-mortem before any draw.
- Routes to ToMpathy weren't stated, only bulk copies delivered. Every card here states its route, and every exchange note quotes it.

## 0. How to run this

### 0.1 V26 and its acceptance

- **Open V26** at V25's drain (2026-11-02T06:00Z), or at handoff if that is later. Nothing is charged to V26 before it opens.
  - Folder: `docs/versions/v26-top-and-ends/`.
  - Contract: `ghostscale/validation/soundingline/v26/contract.py`, modelled on V25's, with the window below, two single-thread below-normal workers, and V25's resource and window checks.
  - Status file: `results/v26/CURRENT_STATUS.json`. AGENTS.md's terminal-event and twelve-hour monitor rules apply unchanged.
  - This packet authorizes all three, and the claim ledger rows `G-V26-<card>-<n>`.
- **What the V26 contract holds:** 60 CPU-hours in four pools.

| Pool | CPU-hours | Packet kinds charged to it |
|---|---:|---|
| Studies | 36, split by the family ceilings below | `probe`, `study` |
| Preparation | 3 | `preparation`: protocols, engineering fixtures, known-answer harnesses |
| Follow-ups, healing and texture | 15 | `followup`, `healing`, `texture` (section 0.3) |
| Protected | 6 | verification and closure |

- **Family ceilings** (CPU-hours, studies pool):

| Family | Ceiling | Family | Ceiling |
|---|---:|---|---:|
| H1 | 6 | E2 | 4 |
| H2 | 4 | E3 | 3 |
| H3 | 4 | U1 | 3 |
| E1 | 6 | R1 | 3 |
| | | R4 and R13 | 3 together |

- **If the studies pool would pass 36 hours,** drop whole cards in this order: R13, U1, E3, R4.
- **The 15-hour pool:** healing has first claim on 4 hours until Thursday 5 November, 12:00 PST (20:00Z); follow-ups may use up to 3; texture uses the rest. After Thursday noon, any may use what remains.

### 0.2 The N16 robustness extension

- The fresh-seed extension of N16 (V24's test of choosing questions by expected information against a fixed order), which the owner approved on 30 October, is V25's, not V26's. Its published record stops at the verified 32-block prefix: the adaptive policy P4 over the fixed order P1 at two probes, 0.06512 [0.06228, 0.06796], no negative blocks. Its status file is dated 30 October, 17:49Z.
- **Once the extension has stopped, or at V25's drain if it is still running then,** land whatever it completed: the 48- and 64-block prefixes if they exist, with replay and review, or a closure record saying where it stopped and why. Charge this to the extension's own two protected hours, inside its 24-hour cap under N16's original 64-CPU-hour ceiling, as the extension's protocol says (lines 51 to 53). None of it is charged to V25's or V26's pools.
- **No further N16 work** runs in V26.

### 0.3 Standing amendments (permanent)

The 25 October amendments (A never idle, B automatic healing, C automatic follow-ups one generation deep, D the claim ledger) stay in force. Write amendments E, F and G below into `AGENTS.md` at handoff, in a section headed "Standing amendments, 2 November", and announce the change in the reply that makes it. The old sentences the 25 October section superseded ("An idle machine is a state to report, not an instruction to invent work", the "one bounded repair per family" line, and "the correct response to an idle machine here is to report state, not to invent work") still stand verbatim elsewhere in the file (lines 10 to 11, 338 and 938 at `ada2e349e`); add a pointer beside each to the 25 October section, so a restarted agent doesn't read the old rule as current.

**E. Design pre-mortem before any draw.** Every card's protocol records, before the probe:
1. **The case.** The situation the question is about, stated as a property of worlds (for example: "two top goals that share every subgoal available in some situation").
2. **The check.** A deterministic or enumerated test, run on the declared grid before any study draw, on check-only worlds under a separate seed identity that never enters a summary, that the worlds can produce the case: the share of sampled worlds in which it occurs, which must be at least the card's stated floor (default 25%). If the grid fails the check, the owner revises the grid within the card's declared ranges and reruns the check; if no grid in those ranges passes, the card is recorded as **"design cannot produce the case"**, with the check's output, and is not reported as a null.
3. **By construction.** Every contrast whose sign or size the design fixes is listed and reported for scale only.

**F. A texture lane.** After a family's primary is verified:
- the owner runs **texture batches**: the card's declared variant grid (section 2 lists each card's), one variant per batch, each batch frozen before its draws, priced from measurement and charged to the texture share of the 15-hour pool;
- texture results are descriptive and never upgrade a primary;
- **stopping rule:** a family's texture lane stops after five consecutive batches in which no reported estimate differs from the card's verified primary estimate by more than its practical bar (for a records-needed primary, one sampled record level; for E2's crossover, one distance bin), or when the pool is spent;
- texture batches register no follow-ups.

This answers the curator's standing preference (14 October) for wide sweeps over many variants, with a stopping rule of about five to seven null batches per family.

**G. Routes in words.** Every verified primary states its route to Sounding Line and to ToMpathy in one or two sentences, in its report and in the exchange note that delivers it. A bulk copy without a stated route doesn't count as delivery.

### 0.4 Admission

As V25 (its packet, section 0.4), unchanged:
- the probe is admitted on a complete card, every open modelling choice declared, controls including at least one known-answer check that can fail, and immutable source and dependency hashes;
- the study adds the probe's measured complete replay;
- open modelling choices are the agent's, declared in the protocol before any draw, marked provisional and listed in the daily line;
- known-answer checks are exact statements about likelihoods, posteriors or noise-free responses (tolerance 1e-12 unless a card says otherwise);
- house sequence: protocol, design pre-mortem (amendment E), known-answer controls, cost probe and replay, one reserved batch, full replay and independent reconstruction;
- sixteen independently seeded whole-world blocks per setting; blocks are the inference unit.

### 0.5 Reserve from measurement

As V25 (section 0.5): a cost probe at the cheapest and costliest settings, all passes within 1,800 CPU-seconds; full packets reserve 1.5 times the measured projection per pass; past the family ceiling, shed in the card's order, never controls; up to 1.5 times the ceiling logged; beyond that, trigger 1. Price every review from the probe.

**Precision.** Any detection rule that rounds a probability to a fixed number of decimals uses V25's exact-arithmetic fallback (`population_precision.py`) near the rounding boundary.

### 0.6 Readers, workers, window

- **Readers:** exact finite-library readers, as V25. Scores: true-hypothesis log-score gain over the declared prior, in nats; half-Brier loss; fractional tie accuracy. Exact aliases stay visible. Every report carries "miniature, architecture untested" and states the choice model it assumed.
- **Workers:** at most two single-thread, below-normal workers. Sounding Line keeps priority. No GPU and no environment synchronization.
- **Window** (UTC, since the clocks changed on 1 November):
  - **At handoff:** write the amendments (section 0.3), and land N16 as section 0.2 says.
  - **At V26's opening:** generate the V26 acceptance, then write every card's protocol and design pre-mortem, charged to the preparation pool.
  - **Studies open:** Monday 2 November, 08:00 PST (2026-11-02T16:00Z).
  - **New starts close:** Sunday 8 November, 20:00 PST (2026-11-09T04:00Z).
  - **Drain by:** 22:00 PST (2026-11-09T06:00Z).

## 1. State as of 1 November, 13:36 PST (`ada2e349e`)

| Item | State |
|---|---|
| V25 | Thirteen primaries and eight follow-ups verified; claims in `docs/theory/CLAIMS.json`. Held for theory review |
| N16 | Confirmed at 175 blocks (log evidence 2.998 against log 20). Fresh-seed robustness extension: 32 of 64 blocks published, gain 0.06512 [0.06228, 0.06796] |
| V26 | Not started |

**What V25 found that this packet builds on** (constructed miniatures, exact readers; none is evidence about people):

| Finding | Bearing here |
|---|---|
| One intervention chosen by expected information didn't beat one more passive choice at any overlap; eight helped where pieces serve few ends | H1 asks the same of the hierarchy's top |
| A merged or split piece library cost 1.2 to 1.4 nats on the true ends at 128 records (a missing piece 0.19 to 0.40); one question gained at most 0.04, eight together at most 0.164 | H1's expert-reader arms test library mismatch at the top |
| Where pauses fell added almost nothing beyond the final text; revision content added more; the design partly built the null in | H2 replaces W7 without vetoes; H3 isolates revision |
| A borrowed population prior never improved deviation detection | E2 tests the reader's own model as the prior |
| Within the trained stake range, trades, devaluation and substitution never separated a held end from a habit that carries feeling; only a planted pleasure-in-the-act signal did | E1 asks which observations separate terminal from circulating values; E3 asks what signatures reveal if feelings are cached solutions |
| Modelling a source's ends protected no better than learning from outcomes | U1 formalizes the opacity gate |
| RAGE-like signals helped only with a logged trigger, and misled under a wrong link | E3 |
| W4's records-needed curve fell from 0.77 to 0.36 detection between 512 and 1,024 records, which looks like a rounding artifact | R4 |

## 2. Studies, in order

Every card below states its case, its check (amendment E), its primary with a practical bar, a falsifiable arm where one exists, known-answer checks, its texture grid (amendment F), its shed order and its route (amendment G). R1, R4 and R13 repair V25 cards: each states what differs and inherits the rest from its V25 card; R4 and R13 have no texture lane. Shed orders apply to the studies pool only; texture variants stop under amendment F instead.

### H1. HIERARCHY_TOP. Can the top of a goal hierarchy be read from steps, and does the reader's own expertise help or mislead? (ceiling 6)

**Why.** The curator: "you're able to rigidly reverse engineer the bottom... unable to reverse engineer the top... It is the critical issue", and humans get there by "slotting in your existing expertise into bits and pieces of the larger puzzle."

**Worlds.**
- **A three-level hierarchy over steps.** Top goals: a library of 6. Subgoals: a library of 8. Pieces: a library of 12. Steps: an alphabet of 10.
  - Each top goal is a distribution over subgoals; each subgoal a distribution over pieces; each piece emits a fixed short step string (2 to 4 steps).
  - **Overlap** between top goals' subgoal distributions: none (disjoint supports), partial (two shared subgoals), heavy (four shared subgoals with different weights).
- **A document:** 6 subgoal episodes drawn from the maker's top goal; each episode, 1 to 3 pieces from the subgoal; pieces' steps concatenated. Episode and piece boundaries are hidden; a variant shows episode boundaries (like paragraph breaks).
- **Maker:** one top goal per maker; documents independent given it.
- **Records:** 1, 2, 4, 8 and 16 documents.

**Readers** (all exact, with marginal likelihoods by dynamic programming over hidden segmentations). Every piece-based reader (R-exact, R-expert-k, R-pieces) also holds one unknown piece, used at any piece slot with probability 0.05 whatever the subgoal, which emits 2 to 4 steps (length uniform), each step uniform over the alphabet; every other piece's probability is scaled by 0.95. So no document has zero likelihood under any reader.
- **R-exact (the ceiling):** the true libraries at every level. It is the generator's own model up to the unknown-piece allowance, so it is reported as the ceiling, never as a contrast.
- **R-novice:** the true subgoal and top libraries with an empty piece library, so every piece is unknown; every top goal then has the same likelihood and its gain is exactly 0, the reference for "misleads".
- **R-expert-k:** the reader's own piece library, which agrees with the maker's on k = 100%, 75% or 50% of pieces; its remaining pieces emit different step strings. Subgoal and top libraries are true.
- **R-flat:** top goal to step bigrams (no hierarchy): each top goal's bigram table and first-step distribution are estimated once from 10,000 documents simulated from the true generator under a declared seed, with add-one smoothing, and frozen before any draw. It is a projection of the true model, reported as a baseline.
- **R-pieces:** pieces known; each top goal is read as a distribution over pieces (its subgoal mixture summed out), pieces independent given the top goal, segmentation summed over (no subgoal or episode level).

**Case and check (amendment E).** The case: top goals that a single document can't tell apart, because they share subgoals and no step is near-exclusive to one top goal.
- Check: in every sampled world, no top goal holds more than half of any step type's expected occurrences summed over the six top goals; redraw worlds that fail, up to 1,000 draws per world, after which the grid is revised within its ranges.
- And in at least 25% of partial- and heavy-overlap worlds, R-exact's posterior on the true top goal after one document is below 0.8.

**Primary.** At 4 documents, partial overlap: R-expert-75's true-top log-score minus R-flat's. Under 0.05 nats counts as no practical advantage. Also the documents needed to reach 1 nat, by reader and overlap.

**Falsifiable arm.** R-expert-50's true-top log-score gain over the prior at 4 documents, heavy overlap. Prediction: a half-mismatched expert misleads (a gain more than 0.05 nats below R-novice's exact 0). If it doesn't, expertise mismatch at the piece level doesn't mislead at the top in these worlds. Report R-expert-50 minus R-flat beside it.

**Known-answer checks.**
- R-expert-100 equals R-exact (posteriors within 1e-12).
- With all top goals given identical subgoal distributions (a fixture), every reader's top-goal posterior equals the prior.
- The dynamic-programming likelihood equals brute-force enumeration over segmentations on a fixture of two short documents.

**Also report:** the gain from visible episode boundaries; how the top-goal posterior's error concentrates (which top goals alias).

**Texture grid:** library sizes (top 4/6/10, pieces 8/12/20), steps per piece (2 to 6), documents with 3 or 12 episodes, mixtures of two top goals per maker.

**Shed order:** the 16-document level; R-pieces; heavy overlap except at 4 documents, which the falsifiable arm needs.

**Route.**
- To Sounding Line's T3 (a writer's personal chunk library) and T5 (plans inferred by a language model, checked against her later choices): whether a reader's own library speeds inference at the top, and how much mismatch costs. That sets expectations for reading real writers with a language model's expertise.
- To ToMpathy: the top tier of the expertise-and-habit panel may cite H1 as constructed-world support, with its records-needed figure; its marks stay "untested" on real records.

### H2. PAUSES_DIFFICULTY_DELIBERATION. Do pauses reveal goals when some pauses come from difficulty rather than deliberation? (ceiling 4)

**Why.** This replaces W7, whose vetoes let the final text reveal the goals. It adds the strongest rival from the response-time literature: pause length can track how hard a choice is, not which system made it (Krajbich et al. 2015).

**Worlds.**
- **Units:** 8 types, a document of 32 units.
- **Autopilot:** a habit distribution over the 8 types, from a declared grid of three habits, each with every type between 5% and 25%.
- **Goals:** the maker holds one of 6 top goals. A goal reweights types multiplicatively, with weights drawn log-uniformly in [0.5, 2] and no zero weights; every goal has at least two types with weight below 1 (redraw otherwise), so every goal can deliberate.
- **Deliberation:** each unit is proposed from the habit. When the proposal's goal weight is below 1, the maker deliberates with probability d (0.3, 0.6): she pauses and redraws once from the habit times the goal weights, renormalized; the redraw is final.
- **Difficulty:** two of the 8 types are hard. When the final unit is hard and was not deliberated, it emits a pause with probability q (0, 0.2, 0.5), whatever the goal. Each unit carries one pause flag, set by deliberation or by difficulty.

**Observations:** (a) the final text; (b) plus where the pauses fell; (c) plus what each redrawn unit replaced.

**Readers:** exact posterior over goals and the habit, (i) with the correct pause model, and (ii) misread: every pause treated as deliberation.

**Case and check.** The case: goals the final text alone leaves uncertain.
- Check: at 32 units, reader (a)'s posterior on the true goal is below 0.8 in at least 50% of worlds at the primary setting; otherwise reduce goal weights' range within [0.67, 1.5] and recheck.

**Primary.** Evidence (b) adds over (a), in nats per deliberation pause, at d = 0.6, q = 0.2, 32 units. Under 0.05 nats per deliberation pause counts as none.

**Falsifiable arm.** The misread reader's gain of (b) over (a) at d = 0.6, q = 0.5, 32 units, pooled per block as total gain over total pauses of either kind. Prediction: difficulty pauses mislead (below −0.05 nats per pause).

**Known-answer checks.**
- d = 0 and q = 0: (b) equals (a) exactly.
- All goal weights equal to 1: every goal posterior equals the prior.

**Also report:** (c) over (b); the gain by d and q; which goals stay aliased under (a).

**Texture grid:** units per document (16, 64), hard types (1, 3), goal-weight ranges, habits with a dominant type.

**Shed order:** d = 0.3; (c).

**Route.**
- To Sounding Line's S5 (planning labels after a stop) and S6 (choice conflict against deliberation in pauses): whether pauses can carry goal evidence once difficulty is modelled, and how much a misread pause model costs.
- To ToMpathy: whether current-attention marks drawn from pauses may cite constructed support, and on what condition (a difficulty model).

### H3. GLOBAL_REVISION. Does revision history reveal the goal a writer settled on, and the one she started with? (ceiling 4)

**Why.** The curator: global revision would occur "when you have allowed your expertise to slot in words automatically, and then you need to go back afterwards... and make changes to align those automatic outputs with the higher order functional hierarchy positions."

**Worlds.**
- A maker has one habit (H2's grid) and one provisional goal g0 (one of 6, with H2's weighting), both fixed across her documents. Each document is a 24-unit draft made by H2's process under g0 (deliberation probability 0.6, no difficulty pauses), so the draft carries g0.
- At a discovery point (after unit 12 or after unit 24, equally likely) the document's settled goal g* is drawn: g* = g0 with probability 0.5, otherwise uniformly among the other five. After an early discovery, units 13 to 24 are drafted under g*.
- At the end of the draft she revises once: each unit drafted before the discovery point whose weight under g* is below 1 is redrawn from the habit times g*'s weights with probability r (0.3, 0.6, 0.9), whether or not g* equals g0.

**Observations:** (a) the final text; (b) plus which positions were revised; (c) plus old and new units; (d) plus the discovery point.

**Readers:** exact posterior over (g0, g*, discovery point, r, habit). A record is one document; the records-needed figure on g0 counts one maker's documents, with g* drawn afresh for each.

**Case and check.** The case: a final text that still mixes g0- and g*-consistent units.
- Check: at r = 0.6, reader (a)'s posterior on the true g* is below 0.8 in at least 50% of worlds where g* differs from g0.

**Primary.** Evidence about g* from (c) beyond (a), per revision event, at r = 0.6. Under 0.05 nats per event counts as none. Also: records (documents) needed to reach 1 nat on g0, the goal she started with, from (c).

**Falsifiable arm.** None: (c) carries weakly more information than (a) by construction. The primary's size is the result, and the g0 records-needed figure is the open question.

**Known-answer checks.**
- All goal weights equal to 1: no unit is deliberated or revised, (c) equals (a), and every posterior over (g0, g*) equals the prior.
- r = 0: (b) and (c) carry no revision events, and their posteriors over (g0, g*) equal (a)'s.

**Texture grid:** draft lengths (12, 48), discovery points, partial revisions that move a unit only part way (redraw from a mixture of g0 and g*).

**Shed order:** r = 0.3; (d).

**Route.**
- To Sounding Line's S2 (what gets revised later) and S3 (whether revision moves text toward the finished piece): how much revision history should add over the final text, per event.
- To ToMpathy: whether revised spans deserve a "realigned here" mark in the expertise-and-habit panel, as constructed support.

### E1. TERMINAL_OR_CIRCULATING. Which observations tell terminal values from circulating ones, and from values that lock into identity? (ceiling 6)

**Why.** The curator's two worlds: "there are terminal values that are set in some way. The other option is that they're all bleeding into each other in a kind of circuitous fashion, and it is instead the ratio of how often they fire that we care about over time, both of which I think are extractable from the data that we have." And: "when you have set a value such that it is a terminal output of a utility function, then that is when it becomes an identity."

**Two corrections from the literature, built into the design.**
- A value that has become identity isn't a large utility weight. In the sacred-values literature it refuses trade-offs at any price, and adding a material incentive to a deal can backfire (Tetlock et al. 2000; Ginges et al. 2007). So type C locks in **lexicographically**, and the environments include trade offers.
- One snapshot can't separate a network of mutually reinforcing values from a fixed common cause; the two are equivalent cross-sectionally, and only repeated measures over time tell them apart (van der Maas et al. 2006; Kruis & Maris 2016; Kievit et al. 2017). This card is longitudinal by construction. A snapshot here would be one record, or choice shares with their order discarded (the usage-ratio predictor below); the stationary-only arm keeps the order.

**Worlds.**
- Each choice is among 4 options in a situation; each option serves 3 value dimensions with outcome vectors on {0, 0.5, 1}.
- **Agent types:**
  - **A, terminal:** fixed value weights w on the simplex grid with step 1/4.
  - **B, circulating:** after each choice, w moves toward the outcome vector of the chosen option: w ← normalize((1 − η) w + η · e(chosen)), with η in {0.005, 0.02, 0.1}. Use begets value.
  - **B2, preference learning by counts:** w is proportional to the agent's counts of experienced outcomes plus a prior count of 2 per dimension, the count-based preference learning of Sajid et al. 2021's active-inference agent, here with softmax choice rather than expected free energy. It is B with a learning rate that falls as experience accumulates.
  - **C, identity lock-in:** as B until any weight passes θ (0.6, 0.8), starting from grid weights whose largest weight is below θ, so that lock-in comes only from use. From then on, whenever some option serves the locked value at 1, the agent takes one of those options, choosing among them by its frozen weights, whatever the other options pay (a lexicographic rule, with the 0.02 lapse); otherwise it chooses by its frozen weights.
- **Choice:** softmax with β = 3 and a lapse rate of 0.02.
- **Environments,** mixed in declared proportions. No environment's construction uses the agent's type, weights or history; every random choice below is uniform and independent of the agent, so the menus carry no information about her type:
  - stationary: menus drawn from a fixed distribution in which a declared share of options (0.5 to 1) are specialists, serving one value dimension at 1 and the others at 0, and the rest are drawn uniformly on {0, 0.5, 1}^3;
  - forced: on one record in eight the agent must take an option drawn uniformly from the menu (a constraint, not a choice);
  - devaluation: for 8 consecutive records, a value dimension drawn uniformly at random has every option's outcome on it halved;
  - shift: from a declared record onward, a value dimension drawn uniformly at random becomes costly: options serving it at 1 pay 0 on every other dimension;
  - trade offer: a two-option menu; one option serves a value dimension drawn uniformly at random at 1 and pays 0 elsewhere, and the other pays 0 on that dimension and 0.5, then 1, on both other dimensions across a declared ladder of records.

**Readers:** exact posterior over type, w (grid, and for C the starting points below θ that its generator uses), η and θ, given choices and environments. B and C's weights are deterministic functions of the observed history, so their likelihoods are exact.

**Case and check.** Two cases, each checked on check-only draws (amendment E) before the probe:
- B close to A on stationary records: at the smallest η, at least 25% of B agents hold under 1 nat of evidence against A after 64 stationary records.
- Lock-in by use: at η = 0.1 and θ = 0.6, at least 25% of C agents pass θ between records 8 and 128 on stationary records, from starting weights below θ.
- If a check fails, revise the specialist share of stationary options within 0.5 to 1 and recheck. C cells other than η = 0.1 with θ = 0.6 report their lock-in rate and aren't required to pass.

**By construction (amendment E).** Whenever exactly one offered option serves C's locked value at 1, C takes it with probability at least 0.985, while A at β = 3 puts at most e³/(e³ + 3) ≈ 0.87 on any one of four options; so C separates from A after lock-in on ordinary records, with or without trade offers. Report C against A for scale only. On these outcome levels an A agent with weight 0.5 or more on the offered value never prefers the trade, and C, offered a trade on its locked value, refuses it except by lapse (probability 0.01); offered one on another value, it takes the option that serves its locked value at 1 when there is one, and otherwise follows its frozen weights. η = 0 nests A in B, and θ above 1 nests B in C.

**Primary (records needed).** Records to reach 1 nat between A and B, by environment mix: stationary only, plus forced records, plus devaluation, plus shift. The result can be "not within 256 records".

**Falsifiable arm (stated predictions).**
- Use begets value shows in stationary records alone: at η = 0.1, B separates from A within 256 records in at least half of agents. If it doesn't, the drift is invisible without designed variation, in these worlds.
- Forced records add a trace beyond the drift: at the smallest η, one forced record in eight cuts the median records needed to separate B from A by at least a quarter. If it doesn't, forced use leaves no trace beyond B's own drift, in these worlds.

**Also report:**
- records needed to separate B from B2, and B from C, and which environment does each;
- the posterior on θ after lock-in;
- how often a C agent's locked value is the one it was forced into early (path dependence);
- a usage-ratio predictor: softmax at β = 3 over the offered options, each scored by its outcome vector times the value shares of the agent's last 16 chosen outcome vectors. Report its held-out next-choice log loss minus the exact reader's posterior-predictive log loss, per record, at 64 and 256 records, by type and environment mix, beside the exact reader's type posterior at the same records; under 0.01 nats per record counts as predicting as well. In experience sampling, complex affect-dynamic measures added little beyond the mean and variance in predicting well-being (Dejonckheere et al. 2019); the question here is whether simple usage shares predict as well while the type stays unresolved.

**Known-answer checks.**
- η = 0: B's likelihood equals A's on every record.
- θ > 1: C's likelihood equals B's.
- B2 with an infinite prior count equals a fixed-weight agent at (1/3, 1/3, 1/3), a fixture outside A's grid.
- A forced record contributes no choice likelihood for any type (it isn't a choice).

**Texture grid:** η and θ ranges, 5 options or 4 value dimensions, forced-record rates (1/16, 1/4), devaluation strength.

**Shed order:** θ = 0.8; the shift environment; η = 0.02.

**Route.**
- To Sounding Line's V3 (whether a writer's mix of drive cues recurs) and V4 (paying to say it herself): the real-record analogue of a forced record, such as accepted AI text on a topic followed by her own later writing on it, and how many such events a test would need.
- To ToMpathy: whether the core values panel should show values as fixed weights or as usage ratios that drift, and what evidence would license either.

### E2. ONE_MODEL_TWO_DIRECTIONS. When does reading another through one's own model beat reading through the population's? (ceiling 4)

**Why.** The curator: "the reader in their moment is using their own autopilot, but having been auto-corrected for the general population if they want to... generally they just start with what they're able to do." And: "we can use fear both to reverse engineer someone's painting, or to forward engineer our own survival. Not survival, like persistence." The literature's term for the first is anchoring and adjustment in perspective taking (Epley et al. 2004).

**Worlds.** W1-style piece worlds (6 pieces, 3 ends). Makers' weights are drawn uniformly from the 15 grid points (step 1/4), so the population mean is (1/3, 1/3, 1/3) and the true prior is uniform. For each maker, a reader's weights are drawn uniformly from the grid points in a distance bin (below) chosen in equal shares. Each reader has its own piece library.

**Readers:**
- **S, simulation:** prior on grid weights w proportional to exp(−κ · TV(w, reader's weights)), κ (2, 8), and the reader's own library as the forward model;
- **P, population:** prior proportional to exp(−κ · TV(w, population mean)), the same κ, true library;
- **I, ideal (the ceiling):** the true uniform prior and the true library. It wins by construction and is reported as the ceiling.

**Similarity.** Reader-to-maker distance in weights, total variation in the bins 0, 0.25, 0.5 and 0.75 or more; library overlap: all 6, 4 or 3 of the maker's 6 pieces.

**Case and check.** The case: readers who differ from the maker.
- Check: each distance bin holds at least 20% of reader and maker pairs at each overlap level (the stratified drawing makes it so; the check verifies the implementation).

**Primary.** For each κ, the distance bin at which S's log-score on the maker's ends first falls below P's, at 8 records, with the reader's own library matching 4 of the maker's 6 pieces: the crossover. At full overlap S and P share the likelihood, so where S falls behind is set by where the two priors sit relative to the maker; report that crossover for scale only.

**Falsifiable arm.** Prediction: for near makers (distance 0.25, one grid step), with the reader's library matching 4 of the maker's 6 pieces, S beats P at 8 records by at least 0.05 nats, at each κ. If it doesn't, starting from one's own model doesn't pay even for similar makers once its library is partly wrong, in these worlds. At distance 0 S's prior peaks at the truth by construction; report that bin for scale only.

**Known-answer checks.**
- With S's prior replaced by the true population prior and full library overlap, S equals I (posteriors within 1e-12).
- With the reader's weights set to the population mean (a fixture) and full library overlap, S equals P.
- κ = 0: S's and P's priors both equal the flat prior.

**Also report:** the egocentric error at 32 records (S's posterior mean pulled toward the reader's own weights) by library overlap; a corrected reader that mixes S and P priors 50:50, the "every man kind of filter".

**Texture grid:** κ, number of ends, mismatched libraries in which the reader lacks one of the maker's pieces.

**Shed order:** κ = 8; the 50% overlap level.

**Route.**
- To Sounding Line's T5 and to the engine: when a reader should anchor on its own expertise (a language model's) and when on the population, and how far to correct. W4 already found borrowed population priors didn't help deviation detection.
- To ToMpathy: whether the engine's prior for a new maker should be its own model, the population's, or a mixture.

### E3. FEELINGS_AS_CACHED_SOLUTIONS. What do emotion-like signatures reveal if feelings are cached solutions rather than amplifiers of current wants? (ceiling 3)

**Why.** The curator: emotion-like signals "would just be expertise again. Right? Isn't that what we're treating feelings as?" W6 found RAGE-like signals helped only with a logged trigger and misled under a wrong link.

**Worlds.** W6's choice worlds with logged blocks (W1's pieces at partial and heavy overlap, β = 3). Each record also carries a cue pattern, 4 binary features each present with probability 0.5, drawn without reference to the agent; the record is blocked with a probability set by its cue pattern through a declared map averaging 0.1. Three generators for a RAGE-like signature, each firing only on a blocked record, with probability 0.02 + 0.96 · (1 − exp(−3x)), so no reader gives the true ends probability 0:
- **(i) amplifier:** x is the value the blocked piece was serving, w·e (W6's value-linked generator);
- **(ii) cached appraisal:** x is the mean, over the features present, of the agent's learned block association a(f), which starts at 0 and after every record moves for each feature present by a(f) ← a(f) + λ(B − a(f)), with B equal to 1 on a blocked record and 0 otherwise, λ (0.1, 0.3), whatever her current wants (learned, like expertise);
- **(iii) innate cue:** x is 1 when a declared feature is present and 0 otherwise, whatever her history or wants (phylogenetic).

**Readers:** exact posterior over ends, assuming (i), (ii) or (iii), and a reader that infers which generator holds.

**Case and check.** The case: worlds where (i) and (ii) would fire on different records.
- Check: at least 25% of signature-eligible records differ in firing probability by more than 0.2 between (i) and (ii).

**Primary.** Under truth (ii), the gain on ends from signatures for a reader assuming (i), at 32 records. Prediction: below −0.05 nats per block (the amplifier assumption misleads when feelings are cached solutions).

**Also report:** what a reader assuming (ii) recovers instead: the agent's cue map (which situations she has learned to treat as blocks), scored against the true map; and the generator-inferring reader's accuracy.

**Known-answer checks.**
- In a fixture where (ii)'s x equals w·e on every blocked record, (i) and (ii) give identical likelihoods.
- (iii) with no declared feature fires with probability exactly 0.02 on every blocked record.

**Texture grid:** learning rates for (ii), the share of innate cues, a FEAR-like signature on threatened outcomes.

**Shed order:** (iii); the generator-inferring reader.

**Route.**
- To Sounding Line's V3: drive-channel cues in text may index a writer's cue map, what she has learned to meet with a given system, rather than her ends.
- To ToMpathy: the core values panel's drive channels should say which reading the evidence supports.

### U1. OPACITY_GATE. Does a source the learner can't reverse-engineer get taken in like an authority, whatever its label? (ceiling 3)

**Why.** Labelling a message "AI", "human expert" or nothing changed neither persuasion nor perceived accuracy in a 2025 study, although almost everyone believed the label (Gallegos et al.). The curator's account: neither an expert's nor an AI's text can be reverse-engineered, so both read as equally accurate. In another study, a bulleted list of similar one-sided arguments from the same model moved attitudes less than inline autocomplete did, and its authors credit writing with the suggestions rather than seeing the information (Williams-Ceci et al. 2026).

**Worlds.**
- Two hypotheses, h1 and h0, with prior 0.5 each. On each record the world holds a pool of 8 evidence items, each pointing to h1 with probability 0.7 if h1 is true and 0.3 if h0 is true.
- The source presents one item per record. Unbiased: an item drawn uniformly from the pool. Biased toward h1 with strength s (0.5, 1): with probability s, an item pointing to h1 if the pool holds one; otherwise an item drawn uniformly.
- **Transparent:** the learner knows the source's policy and s, sees only the presented items, and updates by their exact probability under each hypothesis given that policy, summing over the unseen pool: a rational learner treating hypothesis-conditioned samples correctly (Batista & Griffiths 2026).
- **Opaque:** the learner sees only the presented items, treats each as a uniform draw from the pool, and multiplies its log likelihood ratio by a default trust τ (0.5, 0.8, 1).
- **Labels:** "AI" or "expert" map to the same τ unless the learner has a track record of that source type: earlier records (16 by default; the primary varies the number) whose pools were revealed afterwards, from which it infers the type's s.

**Case and check.** The case: biased sources whose bias is visible in their sampling.
- Check: at least 25% of biased sources choose claims whose likelihood ratio differs from the unbiased policy's by more than a factor of 2.

**Primary (records needed).** Capture is the learner's mean posterior on h1 in worlds where h0 is true, at 32 claims. A learner given a track record of the source's type infers s from it (prior uniform on 0, 0.5 and 1), then reads new claims as the transparent learner does, under that posterior. The primary is the track-record length, among 1, 2, 4, 8, 16, 32 and 64 records, at which its capture from a source with s = 0.5 comes within 0.05 of the transparent learner's; the result can be "not within 64 records".

**By construction (amendment E); no falsifiable arm.** In expectation the opaque learner's log odds move by τ · ln(7/3) · (2p − 1) per claim, where p = 0.3 + 0.642 · s is the chance that a presented item points to h1 under h0, so it is captured exactly when s exceeds 0.31, at a speed set by τ; the transparent learner is calibrated. Report opaque minus transparent capture, by s and τ, for scale only. Without track records the label effect (AI minus expert) is exactly zero, the Gallegos pattern.

**Known-answer checks.**
- An unbiased source and τ = 1: the opaque and transparent posteriors are equal on every record (tolerance 1e-12).
- s = 0 (a fixture): the biased source's presentation probabilities equal the unbiased source's, item by item.

**Also report:** a menu condition, where the learner sees several offered claims and which one the source chose (like CoAuthor's menus), between opaque and transparent.

**Texture grid:** τ, bias strength, claims per record.

**Shed order:** the menu condition.

**Route.**
- To Sounding Line's U1 (whether modelling the source protects, on public co-writing logs): the observable a real-record test would need, such as evidence that a writer modelled the source (edits to accepted text) against verbatim acceptance.
- To ToMpathy: none this week.

### R1. W1 REPAIR. Which intervention breaks an alias between ends that share every available piece? (ceiling 3)

**Why.** V25's W1 asked this, but its libraries couldn't produce the case in sampled worlds: the full-menu training situation always offered single-end pieces, so no two ends could share every available piece. That was an error in the V25 packet's design, not a finding.

**Worlds.** W1's design (6 pieces, 3 ends, β, records and the remove, satiate and switch interventions), with new libraries and situations: for one pair of ends, declared per world, pieces 0 to 4 have equal outcome values on those two ends, and piece 5's values on them differ. The four training situations draw only on pieces 0 to 4 (three hold four of them each, one holds all five), so hypotheses that differ only in the split of weight between the pair are exactly aliased on every passive record. Remove and satiate act on the five-piece training situation; each of the two switch situations holds piece 5 and two of pieces 0 to 4.

**Case and check.** Exact aliases of that kind in at least 25% of worlds (the rule makes it 100%; the check verifies the implementation).

**By construction (amendment E).** Remove never separates such a pair; switch always does, through piece 5; satiating one end of the pair separates it exactly when the other end's values vary across pieces 0 to 4, and satiating the third end never does. A habit on piece 5, which no training situation offers, is also exactly aliased with no habit until a switch shows piece 5. These shares are reported for scale only.

**Primary.** For each intervention, among the split-alias pairs at 32 passive records: the true-hypothesis log-score gain in nats from one intervention chosen by expected information, over one more passive record, and the interventions needed to reach 1 nat. Under 0.05 nats counts as no practical advantage.

**Known-answer checks.** W1's own fixtures (the tie, and the tie that breaks under satiation where the shared ends' outcome values differ), plus a new fixture for the case where it doesn't (the unsatiated end's values equal across the probe situation's pieces), which W1's controls lack.

**Texture grid:** overlap structures, β.

**Shed order:** β = 1; half the training situations.

**Route.**
- To Sounding Line: none this week.
- To ToMpathy: the owner-question order its packet planned to take from W1's alias map.

### R4. W4 PRECISION. Is W4's fall in detection between 512 and 1,024 records real? (ceiling 1.5 of the shared 3)

- **The case:** at 1,024 records W4's matched threshold rounds to 1 (tie mass 0.40), so detection there is set by tie credit. Check: arm (a) reproduces that threshold and tie mass.
- Rerun W4's records-needed curve on W4's own worlds and seeds, individual reader, α = 10. Regenerate each world at W4's own record levels (8, 32, 128, 512 and 1,024), so its counts at 128, 512 and 1,024 records are W4's exactly, and make the counts at 192, 256, 384 and 768 records by splitting W4's increments with multivariate hypergeometric draws under a separate declared seed (records within an increment are exchangeable). Score 128, 192, 256, 384, 512, 768 and 1,024 records in three arms:
  - (a) W4's rule unchanged: eight-decimal half-even rounding with the exact-arithmetic fallback (section 0.5), near-mean calibration panel;
  - (b) the same panel, with makers ranked by the exact complementary tail probability (1 minus the flag), computed in log space from integer counts and checked at 50 and 80 digits, without rounding;
  - (c) rule (b), calibrated on the exact-mean panel instead.
- **Primary:** detection at matched 5% false alarms at 1,024 records, arm (b) minus arm (a). The fall is a rounding artifact if arm (b) at 1,024 records is no more than 0.05 below arm (b) at 512.
- **Known-answer:** arm (a) reproduces W4's 512- and 1,024-record values exactly; with the uniform prior in place of the estimated one, the estimated-population flags equal the individual reader's in every arm (W4's own check).
- **Shed order:** arm (c); the 192, 384 and 768 levels.
- **Route:** to ToMpathy, its records-needed figure (its B15 record) is replaced by this curve, or confirmed; to Sounding Line, none this week.

### R13. W13 REPAIR. How much does a misleading cue have to mislead before persistence and proxy come apart? (ceiling 1.5 of the shared 3)

- W13's design, with its two uninformative shifts (restoration without the cue, and restoration delayed after the cue) replaced by shifts in which the cue misleads by degrees: the cue goes with a drop in an internal variable on a fraction m of records (0.1, 0.25, 0.5).
- **Case and check:** in every replacement shift, at every positive learning rate, the learner's trust in the cue changes after at least one record.
- **Known-answer:** W13's own: in training all types give identical choice likelihoods, and at learning rate 0 the learner's likelihood equals the proxy's on every shifted record.
- **Shed order:** m = 0.1.
- **Primary:** shifted records to 1 nat between learner and proxy, by m and learning rate.
- **Route:** to Sounding Line, the episodes a real-record test of "persistence at the top" would need; to ToMpathy, none this week.

## 3. Deferred, no preparation this week

D5's forty worlds (V24's costed successor reader), T1, T3 and T4 (the CoAuthor, Wikipedia and preference-panel twins; T1 and T3 retained as failures), N1 (the population-map triage) and PyMDP (the active-inference library check), as in V25. No V25 card is rerun except through R1, R4 and R13.

## 4. Cross-repository

| To | What | When |
|---|---|---|
| Sounding Line, `docs/exchange/ghost-v26-20261102/` | Each verified primary with its stated route (amendment G): H1 to T3 and T5; H2 to S5 and S6; H3 to S2 and S3; E1 to V3 and V4; E2 to T5; E3 to V3; U1 to U1; R13's episode counts for a test of persistence at the top | As each verifies |
| ToMpathy, `docs/ghost-v26-20261102/` | The same, with the route to each panel stated | As each verifies |
| From Sounding Line | T5's result, for H1's and E2's reports; S-family results, for H2's and H3's reports (no texture variant waits on them). WB1's effects table, if it lands, is held for the next version: no V26 card declares a W5 calibration batch | Optional; nothing waits |

Handing over this packet approves this week's aggregate-only copies into both recipients. Copies stay uncommitted in recipients, as before; the receipt says delivery, not adoption.

## 5. Schedule

| When (PST) | What |
|---|---|
| Handoff, then V26's opening (Sun 1, 22:00) | Amendments at handoff. N16 landed when the extension stops or at the drain. From the opening: V26 acceptance, protocols and every design pre-mortem |
| Mon 2, from 08:00 | H1 and H2: cost probes, then batches |
| Tue 3 | H3 and E1. H1 and H2 replays and reconstructions |
| Wed 4 | E2 and E3. Texture lanes open for H1 and H2 |
| Thu 5 | U1, R1. Healing's first claim on the shared pool ends at noon |
| Fri 6 | R4, R13. Replays and reconstructions for H3 to U1 |
| Sat 7 | Texture lanes and follow-ups |
| Sun 8 | New starts close 20:00; drain 22:00; final note |

## 6. Reporting

- **Daily line** at 20:00 PST: corrections first; verified results with claim ids and control status; every pool's charge against its ceiling; provisional modelling choices; prepared packets; follow-up and healing dispositions; idle worker-hours and their causes; texture batches run and the stopping-rule count per family.
- **Final note** by Sunday 8 November, 23:00 PST: corrections first; for each card its result, its design pre-mortem outcome, its route, and what it can't say; a one-paragraph answer to each of the week's questions (what the top of the hierarchy needs, which observations separate the end-of-hierarchy worlds, when one's own model helps, what the opacity gate predicts).

## 7. Triggers (the only reasons to contact the curator)

1. A card would pass 1.5 times its ceiling after shedding, or the 60-hour contract would be exceeded.
2. A design pre-mortem fails for every grid in a card's declared ranges and the card is the week's H1 or E1 (the two cards the theory leans on most).
3. The V26 acceptance, the N16 landing or the deadline machinery fails.

## 8. Pre-mortem

- **The cards are too easy and finish in hours again.**
  - *Guard:* the texture lane, with its five-batch stopping rule and its own pool.
- **A design decides its own answer, as three of V25's did.**
  - *Guard:* amendment E's case-and-check before any draw, and the "by construction" list in every report.
- **A mismatched expert reader can't parse the maker's steps at all.**
  - *Guard:* the unknown piece in every piece-based H1 reader, and R-novice as the reference for "misleads".
- **Dynamic programming over hidden segmentations is slow at H1's larger texture settings.**
  - *Guard:* the cost probe's costliest setting, and H1's shed order.
- **E1's agent types separate or alias for reasons the design didn't intend** (A's grid can't follow B's drift, a lapse rate swamps it, or C never locks in by use).
  - *Guard:* the known-answer checks at η = 0 and θ > 1, and both of E1's case checks.
- **U1 reads as evidence about people.**
  - *Guard:* it's a constructed mechanism; its report says it licenses design contrasts for real-record tests, not a psychological claim.
- **Routes go unstated again.**
  - *Guard:* amendment G; a copy without a route doesn't count as delivery.

## Sources

Repository records at `ada2e349e`: `docs/versions/v25-pieces-and-ends/FINAL_REPORT.md`, `W1_REPORT.md` to `W13_REPORT.md`, `N16_ROBUSTNESS_PROTOCOL_2026-10-30.md`, `results/v25/n16-robustness-20261030/PREFIX_032.json`, `docs/theory/CLAIMS.json`, `AGENTS.md`.

Literature cited in the cards:
- Gallegos et al. 2025, labelling messages as AI-generated: https://arxiv.org/abs/2504.09865v2 (preprint).
- Williams-Ceci et al. 2026, biased AI writing assistants shift attitudes, *Science Advances*: https://doi.org/10.1126/sciadv.adw5578 (full text via PubMed Central).
- Batista & Griffiths 2026, a rational analysis of sycophantic AI: https://arxiv.org/html/2602.14270v1 (preprint).
- Krajbich et al. 2015, rethinking fast and slow based on a critique of reaction-time reverse inference, *Nature Communications*: https://doi.org/10.1038/ncomms8455.
- Epley et al. 2004, perspective taking as egocentric anchoring and adjustment, *Journal of Personality and Social Psychology*: https://doi.org/10.1037/0022-3514.87.3.327 (not read; cited from secondary sources).
- Tetlock et al. 2000, taboo trade-offs: https://doi.org/10.1037/0022-3514.78.5.853 (secondary).
- Ginges et al. 2007, sacred bounds on rational resolution of violent political conflict, *PNAS*: https://doi.org/10.1073/pnas.0701768104.
- van der Maas et al. 2006, mutualism: https://doi.org/10.1037/0033-295X.113.4.842 (secondary); Kruis & Maris 2016, three representations of the Ising model: https://doi.org/10.1038/srep34175 (abstract); Kievit et al. 2017, mutualistic coupling: https://doi.org/10.1177/0956797617710785 (abstract).
- Sajid et al. 2021, preference learning in reward-free active inference: https://arxiv.org/abs/2106.04316 (preprint).
- Dejonckheere et al. 2019, complex affect dynamics add limited information: https://doi.org/10.1038/s41562-019-0555-0 (abstract).
