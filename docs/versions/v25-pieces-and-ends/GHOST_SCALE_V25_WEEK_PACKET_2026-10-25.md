# Ghost Scale Sim, V25 week packet (25 October to 1 November): pieces, ends, attention and capture

> *Checked three times by an independent reviewer and fixed after each round. The third round's two design fixes follow the reviewer's own recommendations.*


*Spec sheet for the Ghost agent. It supersedes the 25 October "wing" draft. W0 runs at handoff; the studies run from Monday 26 October, 08:00 PDT (15:00Z), to Sunday 1 November, 22:00 PST (2026-11-02T06:00Z). Every decision is made here, and section 0.4 says how the agent fills any modelling choice a card leaves open. Return to the curator only on the triggers in section 7.*

**What this week tests.** The curator's working model:
- a person reuses **pieces** (preset action sequences with expected outcomes) that serve **ends**;
- most production runs on autopilot, and **attention** goes where the autopilot departs from a higher-order goal;
- emotion systems leave **signatures** that carry information about wants;
- **trust** rests on a current projection of someone's transition map and how similar their ends are to yours;
- **capture** happens by repetition, and filtering works by declining to model someone.

Ghost's job is to find, in constructed worlds with exact readers, what each of these can and can't identify, and what Sounding Line should look for in real records. Thirteen study families, plus N16's powered confirmation once its production runner is built.

**What each primary is.** A correctly specified exact reader never loses information in expectation, and several worlds fix a sign by construction. So every primary is one of three things: a magnitude with a stated minimum meaningful size; a records-needed figure with a stated size; or a falsifiable arm, whose result can go against its stated prediction. W5's primary is descriptive. Where a card's sign is fixed by construction, the card says so and reports that contrast for scale only.

## 0. How to run this

### 0.1 V25 and its acceptance

- **The V24 pilot acceptance can't host this work.** It admits pilots only, and only until tonight, 22:00 PDT (`v24/contract.py`, `admission`). It closes as declared, and its remainder doesn't transfer.
- **Open V25.**
  - Folder: `docs/versions/v25-pieces-and-ends/`.
  - Contract: `ghostscale/validation/soundingline/v25/contract.py`, modelled on V24's, with its window from the times in the header, two single-thread below-normal workers, and the same resource and window checks. It admits the packet kinds in the pool table below; V24's pilot-only kind check doesn't carry over.
  - Status file: `results/v25/CURRENT_STATUS.json`. AGENTS.md's single terminal-event and twelve-hour monitor rule applies to V25 unchanged.
  - This packet authorizes all three.
- **What the V25 contract holds:** 46 CPU-hours in four pools.

| Pool | CPU-hours | Packet kinds charged to it |
|---|---:|---|
| Studies | 32, split by the family ceilings below | `probe`, `study` |
| Preparation | 2 | `preparation`: protocols, engineering fixtures, known-answer harnesses |
| Follow-ups and healing | 6 | `followup`, `healing` (section 0.3) |
| Protected | 6 | verification and closure |

- **Family ceilings** (CPU-hours, studies pool):

| Family | Ceiling | Family | Ceiling |
|---|---:|---|---:|
| W1 | 4 | W8 | 2 |
| W2 | 2 | W9 | 1 |
| W3 | 4 | W10 | 2 |
| W4 | 3 | W11 | 2 |
| W5 | 3 | W12 | 2 |
| W6 | 2 | W13 | 2 |
| W7 | 3 | | |

- **If the studies pool would pass 32 hours,** drop whole cards in this order: W13, W11, W5, W4, W9.
- **Healing has first claim on the follow-up and healing pool.** Until Saturday 31 October, 12:00 PDT (19:00Z), 3 of its 6 hours are held for healing; follow-ups may use only the other 3. After that, either may use what remains.
- **A heal larger than what remains of the pool** is charged to the failed card's own 1.5 times allowance in the studies pool (section 0.5). Beyond that, trigger 1.
- **Otherwise, follow-ups and healing never displace this packet's cards,** and never draw on the studies pool.

### 0.2 N16's powered confirmation (V24 production), after a bounded build

- **Why.** N16's pilot was positive in 4 of 4 blocks, and ToMpathy's scheduler (its B1) waits on a powered answer.
- **What doesn't exist yet.** The pilot runner hardcodes four pilot blocks and "does not admit production at any time" (`PROTOCOL.md`). The status table still lists a frozen production order, thresholds, power and cost calibration, and an uncertainty assessment as unbuilt (`IMPLEMENTATION_STATUS.md`).
- **The bounded build** (up to 8 builder-hours, after W1's and W2's probes are admitted; preparation CPU is charged to the production contract, not to V25):
  1. **Freeze the confirmation design** from the retained calibration (`results/v24/preparation/N16_DESIGN_CALIBRATION.json`: about 260 complete balanced blocks, at an assumed α of 0.1, a first e-LOND γ of 0.5 and a log threshold of log 20). The freeze fixes the order, α and γ before any confirmation block is drawn. The calibration itself says its threshold "is not a frozen V24 confirmation order".
  2. **Build a production runner** with block checkpoints and resume, yielding its worker slot at block boundaries, and the sequential e-value ledger.
  3. **Run the controls before data.** N16's own known-answer controls (informative choice, fully aliased, posterior arithmetic, support and selection probability, as in the pilot report), which the protocol requires before data; the evidence instrument's controls (the exhaustive fair-sign null and the positive fixture); and a full replay on one excluded block.
- **Contract.** `ghostscale/validation/soundingline/v24/production.py`, for N16 alone: 64 CPU-hours including replay and reconstruction (the protocol's feasibility bar), one numerical thread, below-normal priority (the repository rule; V24's admission code rejects any other), window from the build's acceptance to the drain, 2026-11-02T06:00Z. This packet authorizes it.
- **Start condition.** The confirmation starts when the runner passes its controls and replay, not at a fixed hour.
- **Sharing.** It uses one of the two V25 workers. At each block boundary it yields to a W card waiting for the slot. Sounding Line keeps priority over both.
- **Budget.** The measured pilot cost (2,182.7 CPU-seconds per 4 blocks, original plus replay plus review) scales to about 39.4 CPU-hours for 260 blocks, and 1.5 times that is 59.1, inside 64. The README's 53.68-hour forecast would give 80.5, outside it. If the projection at the first checkpoint passes 64 hours, stop at the next block boundary and report the e-value at that block. The predictable-betting e-value stays valid at that stopping time; a run that hasn't crossed its threshold reports "not confirmed within budget".
- **Throughput and the drain.**
  - At the first checkpoint after 24 wall-hours, project completion from measured CPU-seconds per wall-hour.
  - The last block may start no later than the drain minus twice the measured wall time per block.
  - If completion would fall after the drain, keep running to that last block and report the e-value at the last complete block before the drain. Predictable betting allows stopping there; a run that hasn't crossed its threshold by then reports "not confirmed by the drain", with the e-value as descriptive evidence.

### 0.3 Standing amendments (permanent)

**Authority.** The curator asked for these on 25 October, by dictation: "healing passes and follow-ups need to just automatically be queued if there are errors ... permanent change to all their respective skills and claw.markdown" ("claw.markdown" is the transcription of CLAUDE.md). That authorizes changing Ghost's operating contract. CLAUDE.md points to AGENTS.md and says new shared instructions belong there, so write these into `AGENTS.md` at handoff, in a section headed "Standing amendments, 25 October", and announce the change in the reply that makes it (AGENTS.md requires that).

**What they supersede, quoted.**
- "An idle machine is a state to report, not an instruction to invent work" (AGENTS.md, lines 10 to 11), and "the correct response to an idle machine here is to report state, not to invent work" (lines 865 to 871). **Amended:** commissioned work waiting in the queue is not invented work. An idle machine with commissioned work remaining is a defect to fix and record; with none remaining, it is still a state to report, and nothing is invented.
- "with one bounded repair per family" (line 271). **Amended:** up to two healing packets per family per week, under B.
- Unchanged: T1 and T3 stay retained and are not retried; closed versions stay closed; nothing is rerun to occupy the queue.

**A. Never idle, on commissioned work only.**
- Keep at least two prepared packets queued while commissioned work remains.
- When the queue empties with commissioned work remaining, build and admit the next packet before acknowledging the depletion (AGENTS.md already requires this). Failure events are always acknowledged first.

**B. Healing passes run automatically.**
1. **What gets healed:** engineering and resource failures (dependency metadata, an exhausted pass allowance, an I/O error, a numerical tolerance), and failed known-answer checks. A failed known-answer check is an instrument failure, as AGENTS.md's "validate the ruler before the signal" and the V24 protocol treat it: it blocks acceptance and is repaired, never published as a finding.
2. **How:** the failed packet stays retained and charged. Without waiting to be asked, the owner diagnoses the cause, writes a one-paragraph repair note, and admits a `healing` packet with its own reservation, priced from measurement. It is charged to the follow-up and healing pool, where healing has first claim, and past that pool to the card's own 1.5 times allowance (section 0.1). It is a new packet, not a silent retry.
3. **Limit:** at most two healing packets per family per week. A repair that changes a scientific setting is a new frozen card, not a heal.
4. **What isn't healed:** a verified result that contradicts a substantive expectation outside the known-answer set is a finding. It is reported, not healed.

**C. Follow-ups run automatically, one generation deep.** After each verified result, register up to three `followup` packets, frozen before outcomes, from this menu. The 23 October follow-ups the curator had to request were among the week's most valuable results; this makes them routine.

| Parent's result | Follow-ups |
|---|---|
| Positive | (a) a stress map: the same contrast under misspecification, omitted truth, or a second generator; (b) the records or interventions needed to reach the effect at half and at twice the size; (c) the mechanism: which setting drives it |
| Null or tied | (d) an identifiability check: is the null an exact alias, with the intervention that would break it? |
| Surprising | (e) one exploratory packet, labelled exploratory |

- **Pool and caps.** Follow-ups draw only on the follow-up share of the 6-hour pool (3 hours until Saturday 31 October, 12:00 PDT, then whatever remains), after the parent's primary is verified. Each is capped at half its parent's family ceiling and at what remains of that share. A follow-up that doesn't fit isn't run, and the daily line says so.
- **One generation.** A follow-up's result doesn't register further follow-ups.
- **Rationale.** Each carries one line on why. The curator may veto at the next note; nothing waits for him.

**D. A claim ledger, version-independent.**
- **The file.** `docs/theory/CLAIMS.json`. `docs/theory/READING_INTENT.md` stays the hypothesis store, and its lines cite ledger ids, so there is one convention, not two.
- **One row per reportable claim:**
  - id, in the form `G-<version>-<card>-<n>`;
  - allowed wording;
  - the estimate;
  - control status: passes, biased reference, invalid or untested;
  - method or mechanism (AGENTS.md requires every result to say which);
  - "miniature, architecture untested";
  - the assumed choice model and the reader;
  - source file and line;
  - supersedes / superseded by;
  - the correction class when it changes.
- **Backfill at handoff:** V23's R4, N5 and N7 (ToMpathy's preview quotes them), N16's pilot, and every claim in section 1's tables. W0's correction is the first changed row.
- **Quoting.** Every exchange note quotes claims by id. Corrections come first in every note.

### 0.4 Admission

Each W card is a descriptive constructed study.
- **The probe** is admitted on:
  - a complete card, with every open modelling choice declared (below);
  - controls, including at least one known-answer check that can fail (AGENTS.md: "Every measure ships with a null that can fail it, written before the run");
  - immutable source and dependency hashes.
- **The study** is admitted on the same, plus the probe's measured complete replay, which meets the protocol's replay condition.

**Open modelling choices.** Where a card leaves a setting open (a grid, a choice rule's constants, a seed rule), the agent chooses it, declares it in the card's protocol before any draw, marks it provisional, and lists it in the daily line. That isn't a trigger; AGENTS.md already lets the owner authorize reasonable analyst choices, and this packet does.

**Known-answer checks are exact.** Each is a statement about likelihoods, posteriors or noise-free responses, under the fixture conditions the card declares, with a numerical tolerance (1e-12 unless a card says otherwise). None is a per-block path property of random draws or a sampling tolerance.

**House sequence:**
1. declared protocol before sampling;
2. known-answer controls;
3. cost probe and replay;
4. one reserved batch;
5. full replay and independent reconstruction.

Sixteen independently seeded whole-world blocks per setting. Blocks are the inference unit; cases and prefixes aren't extra evidence.

### 0.5 Reserve from measurement

- **Why.** T1 and T3 together hold about 70% of everything V24 charged, for zero worlds. Each reservation was sized before its cost was measured, and T3 failed at its largest setting.
- **The probe.** Each card's first packet is a cost probe: one whole-world block at the cheapest setting (every factor at its cheapest level) and one at the costliest (every factor at its costliest level). Its original, replay and review passes together must fit in 1,800 CPU-seconds.
- **The study.** The **projection** is the probe's measured cost per pass scaled to the full grid. The full packet **reserves** 1.5 times the projection for each pass (original, replay, review).
  - Price the review from the probe, never as a fixed share. N16's review cost about twice its original pass.
- **If the projection passes the family ceiling:** shed in the card's own shed order, never controls. Cards marked "no sheddable factors" skip this step.
  - A card may then run up to 1.5 times its ceiling, logged.
  - Past that, trigger 1.
- **Wall-clock.** If the first daily line's measured throughput projects that the remaining cards can't finish by the drain, shed whole cards in section 0.1's order and say so in the line. This isn't a trigger.

### 0.6 Readers, workers, window

- **Readers.**
  - Exact finite-library readers: enumerated posteriors over a declared hypothesis library. They are not PyMDP or learned readers (both have run in earlier versions); say which kind wherever a result is quoted.
  - **Scores:**
    - true-hypothesis log-score gain over the declared prior, in nats;
    - half-Brier loss;
    - fractional tie accuracy.
  - Exact aliases stay visible.
  - Every report carries "miniature, architecture untested", the wording V24 uses, and states the choice model it assumed. Any reading of values rests on an assumed choice model, which no data supplies (Armstrong & Mindermann 2018).
- **Workers.**
  - At most two single-thread, below-normal workers. One is shared with N16's confirmation once it starts (section 0.2).
  - Sounding Line keeps priority.
  - No GPU and no environment synchronization.
- **Window.** Daylight saving ends on 1 November, so the contract uses UTC.
  - **At handoff:** generate the V25 acceptance and do W0. Protocols and engineering fixtures may follow, charged to the preparation pool. Nothing scientific is drawn before Monday.
  - **Studies open:** Monday 26 October, 08:00 PDT (2026-10-26T15:00Z).
  - **New starts close:** Sunday 1 November, 20:00 PST (2026-11-02T04:00Z).
  - **Drain by:** 22:00 PST (2026-11-02T06:00Z).

## 1. State as of 23 October, 13:28 PDT (`8165fb817`)

Charges after that commit aren't included.

| Item | State |
|---|---|
| Pilot acceptance | 7.12 of 40 CPU-hours charged; 32.88 remain, 8 of them protected. Closes 22:00 PDT, 25 October |
| N16 pilot | +0.0666 nats over the fixed order, positive in 4 of 4 blocks. A powered confirmation: about 39.4 CPU-hours by linear extrapolation of the measured pilot, 53.68 by a design forecast the README calls optimistic |
| D5 cost pilot | Verified. Forty worlds would cost about 18.8 CPU-hours |
| T1 and T3 | Zero worlds. Retained as apparatus and resource failures, not retried |
| Eight follow-ups | Verified, at 314 CPU-seconds of execution, replay and review (preparation, monitoring and landing checks were charged separately) |
| Production | Nothing prepared: no production runner, frozen order or thresholds |

**What the follow-ups found** (all miniatures, architecture untested):

| Study | Finding | Bearing on this week |
|---|---|---|
| Gallery controls | Shared context alone identifies 5.76% (chance 2.22%), against a matched within-context reference of 6.67%; reusing the same sample gives 100%. An identity-excluding reference has mean (1 minus accuracy)/(N minus 1). Ghost calls this "a methods warning, not a diagnosis of Sounding Line's failed controls" | Sounding Line replaces its control rule this week |
| Needs and drives | 61.59% overall. The need behind a drive is readable wherever the actions' public consequences differ by need: 98.18% in those cells, and 100% for all four mechanisms with no satiation and no consequence swap. Where consequences don't differ, every mechanism stays exactly aliased (25% tie credit), with or without satiation. Satiation wasn't needed in this miniature | W1's exact-tie class is the uninformative case. The miniature couldn't test whether satiation breaks it, because its uninformative rows were uniform for every mechanism; W1 tests that. W6 |
| Set points | 100%, because the set point was a public input | W2 hides it |
| Higher-goal editing | 81.25%. Goals stay aliased where drafts don't conflict with them or the goal can't be seen | W1's alias map; W7 |
| Context convergence | Missing support leaves 0.150 predictive error, against 0.025 with full support | W5 |
| Disturbance | 98.96%, but it classified supplied policies; it didn't reconstruct an unknown controlled variable | W2 reconstructs the controlled variable |
| N16 stress | In 6 of 48 truth-omitted cells, all from source N4d, the chooser fell below the correct-model fixed order, to −0.044 nats | W3's omission arm; section 0.2 |

## 2. Studies, in order

### W0. Correct the stress-map sentence (no CPU; at handoff)

- **The sentence.** "The true-mechanism-omitted candidate receives the fixed lower-bound score by construction, log(0.1)" (`FOLLOWUP_RESULTS_2026-10-23.md`, scheduler stress interpretation).
- **What the saved cells show.** The 48 truth-omitted cells don't get log(0.1). Their means run from −0.044 to +0.087 nats, with a lowest block of −0.057. log(0.1) is the protocol's analytic lower bound, not an assigned score; the ε-mixture takes the comparator's support, as the result's own limitation line says.
- **Do.** Add a dated correction beside the sentence, keep the original, and state what the score measures. It is the first changed row of the claim ledger.

### W1. PIECES_ENDS. Can ends be read from piece choices, and which intervention separates ends that share pieces? (ceiling 4)

**Worlds.**
- **Library.** K = 6 pieces. Each piece has an expected-outcome vector over M = 3 ends, with entries in {0, 0.5, 1}, fixed per world.
- **Overlap:**
  - none: each piece serves one end;
  - partial: two pieces serve two ends each;
  - heavy: four pieces serve two or three ends.
- **Situations.** Four training situations, each making 4 to 6 pieces available, plus two held-out "switch" situations, declared in the protocol, whose available sets each differ from every training situation by at least two pieces.
- **Maker.**
  - End weights w on the simplex grid with step 1/4 (15 points).
  - Rationality β in {1, 3}.
  - **Habit variant:** a fixed bonus h = 1 on one piece, whatever the ends: a means that has become an end in itself (functional autonomy).
  - **Choice** among available pieces is in proportion to exp(β · (w·e + h·[the habit piece])). The bonus sits inside β.
- **Records:** 8, 32, 128 and 512 passive choices; a record is one choice.

**Reader.** The exact posterior over all 210 hypotheses: 15 weights × 2 rationalities × 7 habit states (none, or one of six pieces).

**Interventions,** one record each:
- **remove:** take one piece out of the available set before choice;
- **satiate:** set one end's weight to zero for that record (devaluation);
- **switch:** present one of the two held-out situations.

**Primary.** At 32 passive records, the true-hypothesis log-score gain from one intervention chosen by expected information, over one more passive record, by overlap level. This can come out negative, because a passive record can carry more information. A gain under 0.05 nats counts as no practical advantage.

**Secondary:**
- **The alias map:**
  - every pair of hypotheses with identical passive choice distributions in every training situation (exact aliases);
  - every pair with symmetric KL below 0.01 nats per record (near aliases);
  - for each pair, the intervention type that separates it, and by how much;
  - for each intervention type, the share of alias pairs it separates. ToMpathy orders its owner questions by this.
- **The passive identification curve:** the mean posterior mass on the true hypothesis and the accuracy, by record count and overlap, reported per block without a bar.

**Known-answer checks (exact):**
- **Exact tie.** Two hypotheses that differ only in the split of weight between two ends sharing every available piece have identical passive likelihoods on every record.
- **The tie breaks.** In a fixture where the shared ends' outcome values differ across the probe situation's available pieces, one satiation probe on either end gives the two hypotheses different likelihoods. (Where those values are equal across the available pieces, the probe leaves the tie in place, and the alias map records that.)
- **Posterior arithmetic.** On frozen fixture records, the posterior agrees with an independent reconstruction.
- **Habit against end, as a control.** Satiation leaves the habit bonus unchanged, so one satiation probe separates a habit piece from a piece that serves the satiated end. This is true by construction and was already shown in V22's B2, V23's N11 devaluation probes and V24's selective-satiation known answer; it is a control here, not a finding.

**Shed order:**
1. β = 1;
2. the 512-record level;
3. half the training situations.

**Route.** The alias map goes to Sounding Line and to ToMpathy by Thursday 29 October, 12:00 PDT (19:00Z), after its replay and reconstruction pass.

### W2. HIDDEN_SETPOINTS. Can a hidden set point and the controlled variable be found, and what does a designed disturbance add while the record is short? (ceiling 2)

**Why.** The V24 set-point pilot saturated because the set point was public, and the disturbance pilot classified supplied policies. This study hides both. A calculation of this card's first draft found that, with the plant known, as here, passive states identify the controller within about 16 records in most cells, so the informative range is the first few records.

**Worlds.**
- **Dynamics.** Two observed variables: x_i(t+1) = 0.9 · x_i(t) + a_i(t) + d_i(t), for i = 1, 2, starting at x = (0, 0). A record is one transition.
- **The maker** controls one candidate perception, p in {x1, x2, x1 + x2}, toward a hidden reference r in {−2, −1, 0, 1, 2}, with gain g in {0.5, 0.9}. With error e = r − p(x):
  - **p = x1:** a1 = g · e plus noise, and a2 is noise only;
  - **p = x2:** the mirror image;
  - **p = x1 + x2:** a1 = a2 = g · e / 2, each plus noise.
  
  Action noise is N(0, 0.1²).
- **Natural disturbances:** d_i ~ N(0, σ_i²), with σ1 and σ2 each in {0.25, 1}, crossed.
- **Observed:** x at every step. The actions stay hidden, as in real logs.
- **Plant-misspecified arm:** the true open-loop decay is 0.8 or 0.95, while the reader assumes 0.9.

**Reader.** The exact posterior over all 30 hypotheses (3 perceptions × 5 references × 2 gains), from the Gaussian likelihood of each observed state given the previous one, with action noise and disturbance folded together.

**A passive readout.** For each candidate perception, its observed variance over its variance with control switched off, computed from σ and the assumed 0.9 decay. This is the minimal intervention principle's signature (Todorov & Jordan 2002). It needs the open-loop variance, which Sounding Line's WC4 doesn't have.

**Interventions.** A designed disturbance of +2 on x1 or x2, known to the reader, one per record, chosen by expected information. This is the test for the controlled variable (Powers 1973; Marken 2014).

**Primary.** After 4 and after 8 natural records, the true (perception, reference) log-score gain from 4 designed disturbances over 4 more natural records. This can come out negative. A gain under 0.1 nats counts as no practical advantage. The gain is skewed, so report the median and the share of cases above 0.1 nats beside the mean.

**Secondary.**
- The posterior curve on the true (perception, reference) at 2, 4, 8, 16 and 32 natural records, by σ cell.
- How often the passive readout names the controlled perception, beside the exact reader, and the settings where it fails (for example an uncontrolled variable with σ = 0.25).
- **The plant-misspecified arm:** the same primary and readout when the reader's decay is wrong. Either can fail here.

**Known-answer check (exact, noise-free).** After a disturbance δ, the controlled perception's deviation follows δ(0.9 − g)^k and an uncontrolled variable's follows δ · 0.9^k. For p = x1 + x2, the sum follows δ(0.9 − g)^k while x1's own deviation, after a disturbance on x1, is (δ/2)(0.9^k + (0.9 − g)^k).

**Shed order:**
1. the plant-misspecified arm's 0.95 decay;
2. g = 0.5.

The σ1 = σ2 = 1 cell, the least saturated, is never shed.

**Route.** The readout's failure rate goes to Sounding Line by Thursday 29 October, 12:00 PDT (19:00Z), for its WC4, labelled "with the open-loop variance known; a lower bound on failure without it", with the plant-misspecified arm beside it.

### W6. EMOTION_SIGNATURES. What do frustration signatures add, which aliases can they break, and what happens when they track something other than value? (ceiling 2)

**Why.** The curator's objection to "ends need designed variation": emotion systems leave signatures that fit only certain outcomes; "anger only fires when you want something". The research pass supports a weak form (an emotion is evidence about a want when the triggering event is known) and contradicts the strong form: no appraisal component is necessary or sufficient for anger (Kuppens et al. 2003), and aversive events generate aggressive inclinations to the degree that they arouse negative affect (Berkowitz 1989). This study measures what a value-linked signature adds, which aliases it can break, and what happens when the signature is really linked to something else.

**Worlds.**
- **Base.** W1's maker at partial and heavy overlap, β = 3.
- **Outcome blocks.** On each record, the world blocks the chosen piece's outcome with probability b in {0, 0.1, 0.3}. Nothing is designed. (An outcome block happens after choice. It differs from W1's "remove", which takes a piece away before choice.)
- **Two generators for the frustration signal,** each world drawing one:
  - **value-linked** (the curator's): on an outcome block, a signal fires with probability 1 − exp(−κ · w·e_k), where w·e_k is the value the blocked piece was serving, κ in {1, 3};
  - **rival:** on an outcome block, a signal fires with probability 1 − exp(−κ · (h·[the habit piece] + v0)), with v0 = 0.2: it tracks a blocked habit and the violated expectation itself, not the value of the ends.
- **Misspecified variant:** a temperament base rate of 0.2 on every record, block or not.

**Readers:**
1. choices only;
2. choices plus outcome blocks;
3. choices plus outcome blocks plus frustration signals, inferring which generator holds (both are in its library);
4. choices plus a count of frustration signals per 8 records, with no log of outcome blocks, so the reader can't tell whether or where a block happened: a fully unlogged trigger;
5. reader 3 with the value link assumed, not inferred (the misspecified reader).

**Primary (magnitude).** Under the value-linked generator, at heavy overlap and b = 0.1, the true-ends evidence per outcome block that reader 3 gains over reader 2, in nats, and the blocks needed to reach 1 nat. Under 0.02 nats per block counts as no practical advantage.

**Falsifiable arm.** Under the rival generator, reader 5's true-ends log-score minus reader 2's, at 32 records. Below −0.05 nats, an assumed value link misleads; above it, it doesn't. Report it beside reader 3's, which infers the link.

**Also report:**
- **Which aliases a value-linked signature can break, and which it can't.** It depends on w·e_k, the same quantity choice depends on, so it can't separate W1's shared-piece aliases. It can break level-shifted aliases (choices are unchanged by adding a constant to every available piece's value) and confusions between habit or β and weights. Report the share of each alias class broken.
- reader 4 against reader 3: what an unlogged trigger costs;
- how many outcome blocks match one designed satiation probe from W1;
- b = 0, where the signature can't fire;
- the temperament variant.

**Known-answer checks (exact):**
- For two hypotheses that differ only within W1's shared-piece alias class, the frustration likelihoods are identical on every record.
- With b = 0, readers 2 and 3 give identical posteriors.

**Shed order:**
1. κ = 1;
2. b = 0.3.

**Route.**
- To Sounding Line's FR1 (frustration windows): which conditions carry information, and the outcome blocks needed. Not effect sizes; Sounding Line's estimands are in different units.
- To ToMpathy (TE6): frustration-like events count as evidence only where outcome blocks are logged, and only against level-shifted or habit aliases. The falsifiable arm's result goes with it.

### W7. ATTENTION_DIVERGENCE. How much do attention events reveal about higher-order goals beyond the final text? (ceiling 3)

**Why.** The curator's process idea: production runs on autopilot, and where the writer departs from it (a pause, a revision, a "second look") marks attention, which may reveal the goals she serves. This study measures what attention events add to a final text, and what happens when the reader misreads where attention comes from.

**Worlds.**
- **A process pyramid.** A plan of 4 sections × 4 units (16 units). A record set is two plans: 32 units.
- **An autopilot generator** proposes each unit from a habit distribution over 8 unit types, declared on a grid in the protocol with positive mass on every type.
- **Higher-order goals.** The maker holds a subset of 3 from a library of 6. Each goal vetoes some unit types; examples are "no stereotype", "hedge claims", "stay on topic".
- **Attention.** When a proposal violates a held goal, the maker attends (a pause event) and redraws the unit from the habit distribution restricted to the allowed types, renormalized.
- **Other attention:**
  - planning pauses at section starts, with probability 0.5;
  - random attention at rate ε in {0, 0.1}.

**Observations:**
- (a) the final text only;
- (b) plus where attention events happened;
- (c) plus what each replaced unit was (the revision history).

**Reader.** The exact posterior over the 20 goal subsets and the generator's habit parameters, on the declared grid.

**Primary (magnitude).** At 32 units, the true goal-subset evidence that (b) adds over (a), in nats, divided by the number of violation-triggered attention events (the events the goals cause), pooled across the block (total evidence over total events), and the units needed to reach 1 nat. Under 0.05 nats per violation event counts as no practical advantage. Report the same per attention event of any kind beside it.

**Falsifiable arm.** The reader assumes ε = 0 when ε = 0.1, and treats planning pauses as goal-driven. Report (b)'s gain over (a) under this misreading. Below −0.05 nats, the misread attention model misleads; above it, it doesn't.

**Also report:**
- (c) over (b);
- how planning pauses and random attention degrade (b);
- which goals stay aliased under (a) and come apart under (b).

**Known-answer check (exact, on a fixture).** Generate a world, and fix the reader's habit distribution, at a fixture habit that puts zero mass on every type any goal vetoes. No proposal then violates a goal, every goal subset's normalizer is 1, and (b) and (a) give identical goal-subset posteriors. (With the reader's own habit grid, which keeps positive mass on vetoed types, they don't, so the check must use the fixture.)

**Shed order:**
1. ε = 0.1 in the correctly specified arm;
2. (c).

**Route.**
- To ToMpathy (TE3 and TE4's badge text): ToMpathy may cite W7 as constructed-world support. Its marks stay "untested" on real records.
- To Sounding Line's PP2 and PP4: which conditions carry information, and the units needed.

### W3. SELECTIVE_UPTAKE. When a source's ends conflict with yours, does modelling those ends protect you, and does that survive a wrong model of the source? (ceiling 4)

**Why.** V22's A1 found that source inference beats the three tested fixed trusts. V22's A3 found that withdrawing attention from a flagged maker cut absorbed purpose from 0.43 to 0.04 when procedures didn't carry the goal, and couldn't remove it when they did. This study asks whether modelling the source's ends does better than global trust piece by piece, and whether that holds when the learner's model of the source is wrong.

**Worlds.**
- **Library.** W1's pieces at partial overlap, β = 3.
- **The learner** knows its own end weights w_L, which value one end negatively. It doesn't know the pieces' effects. Its prior is uniform over 8 candidate effect libraries: the true one, and seven that each swap the effect vectors of one pair of pieces, the pairs drawn per world from a declared seed.
- **The source** chooses by its own weights w_S and the true effects.
- **What the learner sees:**
  - every one of the source's choices;
  - for one choice in four, a noisy outcome on the learner's own ends (noise standard deviation 0.5).
- **Alignment:**
  - aligned: w_S = w_L;
  - partly conflicting: the source gives positive weight to the end the learner values negatively;
  - opposed.
- **Bundling:**
  - separable: the pieces most useful to the learner don't advance that end;
  - bundled: they do.
- **Source records:** 8, 32 and 128.

**Learner strategies:**
1. **Adopt all:** copy the source's choice frequencies. This is over-imitation.
2. **Reject all:** plan with its own weights under its prior over effects.
3. **Reliability-weighted:** mix the source's policy into its own with weight equal to the posterior mean of the source's reliability, under a Beta(1, 1) prior. A source choice with an observed outcome counts as a success when that outcome's utility under w_L exceeds the learner's prior expected utility of its own best piece, and as a failure otherwise. This is global trust.
4. **End-modelled, piece by piece:** infer the source's weights and the effects jointly from its choices and the outcomes (exact inverse planning), then plan with its own weights and the inferred effects. It takes the source's knowledge, not its ends.
5. **Outcomes only:** update the same prior over the 8 effect libraries from the observed outcomes alone, ignoring the source's choices and ends, then plan with its own weights.

**Measures,** in held-out situations, against an oracle that knows the effects and chooses by the same rule:
- **Regret:** the oracle's expected utility on the learner's ends, minus the learner's.
- **Capture:** the expected advance of the negatively valued end under the learner's policy, minus the oracle's.

**Primary (falsifiable arm).** In the omission arm (strategy 4's library of source weights leaves out the source's true weight on the conflicting end), capture under strategy 4 minus capture under strategy 5, at 32 source records, in partly conflicting, bundled worlds. It can come out either way. A difference under 0.05 utility units counts as none.

**For scale, fixed by construction.** With the source correctly modelled, strategy 4 plans with the learner's own weights, so it captures less than strategy 3 by construction. Report that gap and the regret it costs, labelled as built in.

**Secondary.** Strategy 4 against strategy 5, on regret and capture: does a model of the source's ends add anything to learning from outcomes alone?

**Known-answer checks (exact):**
- With the learner's prior concentrated on the true library and an aligned source, strategy 4's limiting policy equals the oracle's.
- With the learner's prior concentrated on the true library, strategies 2 and 5 give identical policies at every record count: both plan with the learner's own weights and the true effects.

**Shed order:**
1. 128 source records;
2. the opposed setting.

**Route.**
- **End-modelling cuts capture at no worse regret, and survives omission:** ToMpathy may later show "useful, but carries this maker's end" for a single piece (recorded in its B12).
- **Bundling or omission defeats it:** only a maker-level warning is defensible.

### W10. ABSORPTION. Under which assumptions can repetition install a habit the learner explicitly rejects, and which filter protects it? (ceiling 2)

**Why.** The curator's account of indoctrination: the sense that something "might make sense sometimes", plus repetition until it becomes the default autocomplete in flow. Absorption shows in automatic behaviours you weren't attending to. Sounding Line's TR3 tests capture by suggestions the writer saw; this study gives it a map of what to expect under stated habit rules. Every W10 output is labelled "under the stated habit rule".

**The learner.**
- **Pieces.** K = 6, among them the disliked piece and its preferred alternative.
- **Two layers:**
  - **explicit:** a Bayesian belief about each piece's value to its own ends;
  - **habit:** a propensity per piece, starting at 10 for every piece (in units of η, as if each piece had ten earlier productions). A production adds η to that piece; an attended exposure adds λ·η, regardless of belief. λ is in {0, 0.25, 1}, so exposure alone can matter or not.
- **Choice.** Each choice is deliberate with probability α (attention available) and habitual otherwise; α is in {0.2, 0.8}. A deliberate choice follows the explicit layer. A habitual choice picks a piece in proportion to its propensity.
- **Habitual choice rate:** the probability that a habitual choice picks the disliked piece, conditional on habit mode. With equal starting propensities it is 1/6 at N = 0. Report the unconditional rate, which multiplies it by 1 − α, beside it.

**Growth rules,** each reported:
- linear: each increment adds its full amount;
- saturating: each increment is scaled by (1 − p/30), so a propensity p approaches 30;
- normalized: after each increment, the six propensities are rescaled so their total stays 60.

**Exposure.** A source makes the learner produce a disliked piece N times (forced production, the essay-writing case: each adds η) or merely shows it N times (repetition: each attended showing adds λ·η). N is in {0, 10, 50}, and runs to 200 for the primary's curve.

**Filters:**
1. explicit rejection only;
2. attention withdrawal: exposures are attended at half the rate, which halves exposure's increment by construction;
3. counter-rehearsal: produce the preferred alternative after each exposure;
4. declining to model the source: the learner draws no inference about the source's reasons, and attends to its exposures at half the rate. On the habit layer it ties with filter 2 by construction; the two differ only in the explicit layer.

**Primary (records needed).** For each growth rule and λ, the number of exposures at which the disliked piece's habitual choice rate first exceeds its N = 0 rate (1/6) by 0.1, at α = 0.2, under filter 1, for repetition and for forced production, up to 200. "Not within 200" is a result. With λ = 0, repetition never gets there, by construction.

**Also report:**
- which filter cuts habitual adoption most, under which λ and growth rule, at what cost in learning the source's useful pieces;
- the dose-response across N under each growth rule, for TR3 to compare against.

**Known-answer checks (exact):**
- With η = 0, every propensity keeps its starting value, so habitual choice rates don't change under any filter.
- With λ = 0, repetition never raises the disliked piece's habitual choice rate under any filter. Under filter 3 the rate falls, because the preferred alternative's propensity grows.

**Shed order:** the normalized growth rule.

**Pre-mortem.** The habit rules build in absorption whenever propensity grows regardless of belief. The output is a map of which assumptions produce absorption and which filters cut it, not evidence that people absorb.

### W8. VOLATILE_WANTS. How should a reader weigh a track record when wants drift, and when does more history hurt? (ceiling 2)

**Why.** The curator's "Heisenberg" problem: confidence about a lifetime trajectory costs confidence about what someone wants now. The formal answer is a learning rate that tracks volatility (Behrens et al. 2007) or explicit change points (Adams & MacKay 2007). V21 already compared static inference with fast and slow filters (hazards 1/8 and 1/64) in a supplied-law miniature. Sounding Line's TR4 tests this on writers.

**Worlds.** W1's maker on the step-0.1 simplex grid (66 points), whose end weights either:
- follow a random walk with per-record step probability s in {0, 0.02, 0.05};
- follow a local linear trend: a level plus a latent continuous slope, the set point's derivative, projected onto the grid; or
- jump at change points with hazard h in {0, 1/64, 1/16}.

**History lengths:** 16, 32, 64, 128 and 512 records, a factor for every reader. Each reader starts from the declared prior at the start of its history.

**Scores.** The log-score of the true current weights is taken on the ε = 0.01 mixture with the uniform prior, so a static reader's score stays finite.

**Readers:**
- whole history, assuming static weights;
- fixed windows of the last 4, 8, 16, 32, 64 and 128 records;
- a two-point slope from the window's first and last quarters;
- volatility-adaptive: the exact posterior over weights jointly with a grid of drift, slope and hazard hypotheses;
- the exact change-point or drift reader with its assumed volatility fixed at 0 (static), one tenth, one quarter, half, the true value and twice the true value.

**The U statistic.** A reader's error at 512 records of history minus its minimum over the five history lengths. It is never negative; a U shape shows as a positive value.

**The prediction, with numbers.** A scalar proxy calculated for this packet (a 19-point Bernoulli rate read by an exact filter, 120 runs per cell; not W8's world; scored without the ε mixture) gave these U statistics, in nats:

| True process | Static | One tenth | One quarter | Half | True | Twice |
|---|---:|---:|---:|---:|---:|---:|
| Hazard 1/64 | above 9 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| Hazard 1/16 | above 9 | 0.21 | 0.07 | 0.01 | 0.00 | 0.00 |
| Walk 0.02 | above 9 | 0.71 | 0.20 | 0.04 | 0.00 | 0.00 |
| Walk 0.05 | above 9 | 0.57 | 0.14 | 0.02 | 0.00 | 0.00 |

Scored the card's way, on the ε = 0.01 mixture, an independent check gave U at one tenth of 0.03, 0.20, 0.33 and 0.53 (average 0.27, still above the 0.1 prediction) and 0.00 at twice the truth. The mixture also caps a static reader's error, at about 8.8 nats on W8's grid, so "above 9" applies to the unmixed proxy only.

So the prediction is:
- with the correct change model, error is non-increasing in history and flattens at a floor set by volatility: U within 0.02 nats of zero at the true value and at twice it;
- U grows as assumed volatility falls below the truth: small at half (0 to 0.04 nats in the proxy), large at one tenth and static;
- over-estimating volatility gives a flat curve at a worse level, not a U;
- windowed readers show a U over window length.

**Primary (falsifiable).** U at one tenth of the true volatility minus U at twice it, averaged over the four nonzero drift settings (s 0.02 and 0.05; h 1/64 and 1/16). The prediction is at least 0.1 nats (0.39 in the proxy). It fails if the difference is below 0.1 nats, or if U at twice the true value exceeds 0.02 nats in any setting.

**Also report:**
- the adaptive reader minus the best single fixed window, chosen once across all drift settings, at 512 records (for scale; non-negative in expectation when the adaptive reader's model is right);
- precision about the lifetime average against precision about the current want, by reader, window and drift.

**Known-answer check (exact).** With drift and hazard fixed at zero, the adaptive reader reproduces the whole-history posterior.

**Shed order:**
1. the 512-record history for the windowed readers;
2. the local linear trend.

**Route.** If the U appears only under an under-estimated change model, the U diagnoses a misspecified reader, and Sounding Line's TR4 can look for it in writers (digest reading; no card waits).

### W12. SACRED_STRUCTURES. Which observations separate an end from a habit, an instrumental value and a sacred value, when habits can carry affect? (ceiling 2)

**Why.** The curator's "dearly held structures": a learned structure, such as honor, that served some deeper end and has become an end in itself. The research pass proposed that only three markers separate an end from a means: it survives devaluation, acting on it carries affect of its own, and it refuses a cheaper substitute. V23's N4d already showed that fixed stakes alias weighted and lexicographic rules and varying stakes break that alias, so this study doesn't repeat it. It asks the part N4d didn't: do devaluation, substitution and affect separate these types when habits can carry affect too?

**Worlds.** An agent trades a structure's payoff (honor) against a deeper payoff (persistence) in small choices. Types are generated from mechanisms, not from markers:
- **instrumental:** honor has value only through its learned contribution to persistence;
- **habit:** a model-free cache learns a value for each honor-serving act at each trained stake, so within the trained range its value scales with the stake as the end's does; outside that range it carries the nearest trained value. In half the worlds the cache also produces distress when the act is blocked (habit disruption);
- **end:** honor has its own terminal weight, which scales with the honor at stake, and acting on it yields affect of its own (means-ends fusion; Kruglanski et al. 2018);
- **sacred:** a very large finite terminal weight plus a taboo-trade aversion, a penalty on accepting any trade offer.

**Affect, declared before data.** Positive affect while acting belongs to the end type only. Distress when the act is blocked appears for habit-with-affect, and for the end type in half the worlds.

**Choice rule.** Softmax with β = 3. With probability 0.02 (the lapse) the choice is instead uniform over the offered options, so no choice has probability zero. The quantity range is set relative to the weight grid in the protocol, before any draw. It covers the trained stakes and at least two untrained stakes on each side.

**Observations:**
- micro trade-offs at varied exchange rates and quantities;
- a devaluation probe, after which honor no longer serves persistence;
- a substitution offer: a cheaper route to the same persistence payoff that skips honor;
- a novel-context probe: honor reachable only through a new act;
- a noisy affect signal during honor-serving acts and when they are blocked.

**Reader.** The exact posterior over type and weight.

**Primary (records needed).** Observations needed to reach 1 nat of true-type evidence between end and habit-with-affect, under each observation mix: trade-offs only; plus devaluation; plus substitution; plus affect; plus the novel-context probe. Two outcomes would count against the research pass:
- its three-marker proposal fails in this model if devaluation, own affect and substitution together leave end and habit-with-affect within 1 nat after 32 observations;
- its claim that size and price don't mark ends fails in this model if trade-offs at varied quantities within the trained range alone reach 1 nat.

Outside the trained range, varied quantities separate end from habit by construction, because the cache can't extrapolate. Report that contrast for scale only, beside the novel-context probe, which tests the same thing.

**Also report:**
- which observations separate which pairs of types, including sacred against a heavy end;
- what a single large sacrifice reveals, against many small ones.

**Known-answer checks (exact):**
- Within the trained stake range, before any probe, with the cache converged so its values equal the end's at every trained stake, matched weights and habit affect off, the end and habit types give identical trade-off likelihoods. (At untrained stakes they differ by construction.)
- Within the declared quantity range, the sacred type's margin over every trade offer exceeds 28/β, so its probability of refusing a two-option offer equals 1 minus half the lapse rate (0.99) within 1e-12.

**Shed order:** the habit-without-affect worlds.

**Route.** To ToMpathy (TE7): which observations separate an end from a habit that carries affect, and how many are needed. If one does so within 32 observations, TE7 records the matching owner question for when drawn tips exist.

### W9. BETRAYAL_TYPES. When does one betrayal collapse trust across a network? (ceiling 1)

**Why.** The curator: betrayal hurts because you learn they are "the kind of person who would do this to you, and would do it again", and "the math converges on you being alone".

**Worlds.**
- **Partner types,** with ε = 0.05 and two stake levels, low and high:
  - **benevolent:** defects at rate ε at either stake;
  - **conditional exploiter:** defects at rate ε at low stakes and 1 − ε at high stakes;
  - **mixed:** defects at rate ε at low stakes and 0.5 at high stakes.
- The learner sees n low-stakes cooperations from the defector, n in {5, 20, 80}, then one high-stakes defection. Every type defects at rate ε at low stakes, so low-stakes trials carry no type information: n moves only the counting reader.
- **Network.** The learner has k close partners, k in {1, 3, 8}, in one group with the defector.
  - Each group member is an exploiter with probability equal to the group's exploiter share; the rest are benevolent or mixed, 3 to 1.
  - Each close partner has m high-stakes trials of its own with the learner, m in {0, 2, 8}, drawn from its type.
  - **The hyperprior** on the group's exploiter share, all with mean 0.2, sets how alike the learner believes group members are: homogeneous, Beta(0.1, 0.4) (groups tend to be all one kind); moderate, Beta(1, 4); heterogeneous, Beta(4, 16).
  - The defector is either drawn from the group (typical) or from outside it (atypical). The hierarchical reader pools it into the group either way, so under an atypical defector any collapse is a mistake.
- Recovery trials, wherever reported, are high-stakes cooperations.

**Readers:**
- **counting:** a beta-binomial on the cooperation rate;
- **type inference:** an exact posterior over types with stake dependence, per partner;
- **hierarchical:** type inference that pools every member's trials, the defection included, into the group's exploiter share.

**Primary (falsifiable).** The rise, from just before to just after one typical betrayal, in the share of close partners whose predicted high-stakes defection is above 0.5, under the hierarchical reader, by hyperprior and m, at k = 1, 3 and 8. The two predictions, as numbers:
- the curator's ("the math converges on you being alone"): a rise of at least one half at k = 8 under the moderate hyperprior with m = 2;
- the research pass's: a rise of at least one half only under the homogeneous hyperprior with m = 0, and below a quarter at k = 8 whenever m is 2 or more.

A calculation for this packet (these types, exact integration over the share, 300 worlds per cell) gave: under the homogeneous hyperprior, a rise of 1.00 at every k with m = 0, and 0.19, 0.01 and 0.00 at k = 1, 3 and 8 with m = 2; under the moderate and heterogeneous hyperpriors, no rise at all. The full card adds the atypical defector, deception and dependence, and its result is the record.
- The homogeneous collapse is sensitive to ε: the post-betrayal prediction crosses 0.5 near ε ≈ 0.085 (0.506 at 0.08, 0.494 at 0.1). Report ε = 0.05 and 0.1 as a sensitivity.
- The homogeneous hyperprior is the curator's claim's most favourable cell and the heterogeneous one its least. The report says which cells carry each conclusion.

**For scale, fixed by construction.** Type inference predicts more future high-stakes defection than counting after a high-stakes defection; report the gap, and the high-stakes cooperations needed before type inference's prediction falls below 0.2.

**Also report:**
- **Deception against defection.** People's trust recovers after consistently good behaviour, unless the untrustworthy acts came with deception (Schweitzer et al. 2006). Add a deception variant, in which the defector earlier announced a benevolent type. The reader treats an announcement as evidence about type with a reliability declared in the protocol, and the variant reports recovery after the announcement is shown false, against plain defection.
- **Dependence, beliefs and decisions separately.** An exact reader's beliefs don't depend on the cost of leaving; only its decision does. Report both, and add a motivated reader that down-weights defection evidence in proportion to its dependence, the formal shape of "betrayal blindness" (Johnson-Freyd & Freyd 2013).
- The expected number of trustworthy partners left in the network, by reader, k and typicality.

**Known-answer check (exact).** A stake-independent rate library whose prior matches the counting prior reproduces counting's predictions.

**Shed order:**
1. n = 80;
2. the heterogeneous hyperprior;
3. the deception variant.

### W4. POPULATION_DEVIATION. How many records until a maker's deviation from her population shows? (ceiling 3)

**Worlds.**
- **Library.** W1's pieces at partial overlap, β = 3.
- **Population.** 64 makers per world.
  - Work in integer tenths throughout, so no comparison depends on floating-point rounding. Each maker's weights are three whole numbers of tenths, each from 1 to 8, summing to 10: 36 grid points.
  - Prior mass is proportional to the Dirichlet density with parameters α × (0.5, 0.3, 0.2) at those points, with α in {3, 10, 30}.
- **Records:** 8, 32, 128, 512 and 1,024.

**Readers:**
- **individual:** uniform prior over the grid;
- **oracle population:** the true prior;
- **estimated population:** the average of the other makers' individual posteriors, leaving the maker out. This is a leave-one-out empirical prior, a simple stand-in for Dimitrakakis & Rothkopf's hierarchical approach.

**Deviation and the flag.** A maker's deviation is its largest absolute deviation from μ over the three ends, a whole number of tenths. The flag is the largest of its six posterior probabilities (each end's weight above μ_m, or below it) against one threshold.

**Two null sets.**
- **Near-mean (primary):** per world and α, 256 held-out makers drawn from the Dirichlet prior restricted to the seven grid points whose deviation is at most one tenth, weighted by the prior. They set thresholds only.
- **Exact-mean (reported):** 256 makers placed exactly at μ.
- Null makers get a leave-one-out prior from a random 63 of the 64 population makers, as population makers do.

**Matched false detection.** For each reader and record count, set the threshold so that 5% of the near-mean null set is flagged. Report the false-alarm rate on the exact-mean set and among near-mean population makers too, and each reader's raw false detection at a 0.95 threshold.

**Primary.** At 128 records, the detection rate for deviations of at least two tenths, at matched 5% false detection, estimated population minus individual, at α = 10. It can come out negative. A difference under 0.05 counts as none.

**Also report:**
- records needed for 80% detection at matched false detection, by reader and α, up to 1,024;
- **use above or below her own optimum:** using W1's habit variant, the records needed to detect a piece used more than her own ends predict. Her ends against the population, and her piece use against her own ends, are reported separately.

**Known-answer check (exact).** With the estimated-population prior replaced by the uniform prior, the estimated-population reader's posteriors and flags equal the individual reader's exactly.

**Descriptive placebo (reported, not a check).** In worlds with no deviant makers (every population maker at most one tenth from μ), report both readers' detection rates at matched false detection.

**Shed order:**
1. α = 3 and α = 30;
2. the 1,024-record level;
3. half the worlds.

**Route.** The records-needed curve goes to Sounding Line and ToMpathy by Saturday 31 October, 12:00 PDT (19:00Z). Sounding Line's CA1 and CA2 compare it with real annotators.

### W5. LIBRARY_MISSPEC. How much does a wrong piece library cost the ends reader? (ceiling 3; runs after W1's checks pass and its primary is computed)

**Worlds.** W1's, at partial overlap.

**Reader libraries:**
- the true library;
- merged: two pieces read as one;
- split: one piece read as two. The true piece's choices go to the two split pieces by a fair coin per choice, and their assumed effects are the true vector plus and minus 0.25 on its largest end, clipped to [0, 1];
- missing: one piece unknown, its choices read as an "other" choice with assumed effects (0.5, 0.5, 0.5);
- calibrated, if Sounding Line's WB1 aggregates arrive by Wednesday 28 October, 12:00 PDT (19:00Z). Otherwise skip it.

**The calibrated setting:**
- K is Sounding Line's number of piece classes, capped at 6 by keeping the six classes that cover the most events.
- The three effect measures with the largest spread between classes stand in for the three ends.
- Each class's standardised mean effect is cut at the terciles across classes, into {0, 0.5, 1}.

**Primary.** At 128 records, the true-ends log-score loss of each wrong library against the true one. Descriptive; no direction is claimed.

**Also report.** The false-confidence rate: posterior above 0.9 on wrong weights.

**Known-answer check (exact).** A "wrong" library that relabels the true one (pieces permuted together with their effects) gives a loss of exactly zero.

**Shed order:**
1. the split library;
2. half the worlds.

### W11. CARE_GROUPS. How far does declining to model the other group keep groups apart, and does it act like lower care? (ceiling 2)

**Why.** The theory has to explain in-group and out-group dynamics, including how caring and bonding can feed ethnocentrism. Kleiman-Weiner, Saxe & Tenenbaum's model of moral learning aligns values toward others' values and toward one's own attachments (2017). W11 makes the adoption rate equal to care, a simplification of that idea.

**Worlds.**
- **Agents.** 8, in 2 groups of 4. Each holds weights over 2 ends. Group A starts at (0.7, 0.3) and group B at (0.3, 0.7), each agent jittered by up to ±0.05.
- **Each round,** an agent observes 4 choices by a partner and infers the partner's weights exactly on a grid. It then moves its own weights toward the inferred ones: w ← w + care × (inferred − w).
- **Care:** 0.3 within group; across groups, in {0, 0.1, 0.3}.
- **Contact:** cross-group partners are sampled at rate c in {0.1, 0.5}.
- **Withheld inference:** with probability q in {0, 0.5}, an agent declines to model a cross-group partner and skips its update on that interaction (the curator's filter by declining to model). Declining to model someone also means not learning from them.
- **Stereotype arm (secondary, labelled):** instead of skipping, the agent moves toward the other group's starting mean. That pulls the groups past each other toward the caricature, so its result is reported separately and says so.

**The gap.** The signed gap: group A's mean weight on the first end minus group B's, 0.4 at the start. Report it signed, never as a distance, so a crossing isn't read as separation.

**Primary (magnitude, with a falsifiable comparison).**
- **Magnitude:** with equal care (0.3 within and across groups), the signed gap at round 50 at q = 0.5 minus at q = 0, by c. In mean field the gap shrinks by a factor of 1 − 2 × care × c × (1 − q) per round, so with noise-free inference the expected gaps at c = 0.1 are about 0.087 at q = 0.5 and 0.018 at q = 0; at c = 0.5 both are near zero. A difference under the tolerance below counts as none.
- **Falsifiable comparison:** q = 0.5 at care 0.3 against q = 0 at care 0.15, which mean field makes equal. The prediction is that skipping acts like lowered care: signed gaps within a tolerance of each other. It fails if inference noise makes them differ by more.
- **Sample size.** With noisy inference, one population's signed gap varies by about 0.13 at c = 0.1 and 0.06 at c = 0.5. So W11 runs 64 populations per cell, not 16, which puts the standard error of a difference near 0.006. The tolerance for both the magnitude bar and the comparison is 0.02 or twice the measured standard error of the difference, whichever is larger.

**For scale, fixed by construction.** Lower cross-group care keeps the gap larger; report the signed gap at each care level, labelled as built in.

**No hate term.** The model has no out-group hostility. The research pass finds in-group love doesn't require out-group hate (Brewer 1999), and cooperative favouritism comes from favouring the in-group, not derogating the out-group (Balliet et al. 2014). If parochial distance appears here, it comes from care-weighted adoption and withheld inference alone.

**Also report:** within-group spread; the share of cross-group inferences that are wrong; how c and q move the signed gap; the stereotype arm.

**Known-answer check (deterministic).** With noise-free inference (an agent reads its partner's weights exactly), the mean-field update in place of random partner sampling, full contact (c = 1, run only for this check), equal care and q = 0, the signed gap's magnitude falls below 0.01 by round 50.

**Shed order:**
1. the stereotype arm;
2. c = 0.5.

**Pre-mortem.** This is an agent population, not a single exact reader. Each agent's inference is exact. The population dynamics are simulated, and the report says so.

### W13. PERSISTENCE_OR_PROXY. How many shifted records does a reader need to tell an agent that uses cues as evidence of persistence from one that values the cue itself? (ceiling 2; idle slots)

**Why.** The curator suspects persistence sits at the top of every value chain, with people as imperfect optimizers of it. The research pass found:
- the base-and-mesa account treats humans as optimizers of proxies, noting that "humans tend not to place explicit value on evolution's objective" (Hubinger et al. 2019);
- homeostatic reinforcement learning makes reward-seeking equivalent to keeping internal states near their set points, for discounted reward (Keramati & Gutkin 2014);
- goal misgeneralization shows up out of distribution (Langosco et al. 2022);
- self-preservation serves almost any goal under broad conditions (Omohundro 2008; Turner et al. 2021), so behaviour that serves persistence says little on its own.

The simulator assigns each agent's type, so this study can't test whether persistence is at the top. It is a design for real data: how much off-distribution evidence a reader needs before the two readings come apart.

**Worlds.**
- **Internal variables.** Two, energy and safety, each in [0, 1] and restored when acted on.
- **Actions** restore one or both.
- **A cue** (a taste, or approval) accompanies restoration in the training environment, with intensity equal to the restoration amount.
- **Agent types:**
  - **homeostatic:** values restoration and senses it directly;
  - **homeostatic learner:** values restoration, but at choice time the cue is its only evidence of restoration. A delayed reading of its internal state, one record later, is used only to update how far it trusts the cue, at a learning rate in {0, 0.05, 0.2} (0 is the check's fixture);
  - **proxy:** values the cue;
  - **mixed:** values restoration and the cue with weights that sum to 1.
- **Choice rule:** softmax with β = 3 and a lapse rate of 0.02.
- **Environments:**
  - training: cue and restoration perfectly correlated;
  - shifted, in four ways: the cue without restoration; restoration without the cue; the cue paired with a drop in an internal variable; restoration delayed after the cue.

**Reader.** The exact posterior over type, learning rate and weights.

**Primary (records needed).** Shifted-environment records needed to reach 1 nat of evidence between the homeostatic learner and the proxy type, by shift and learning rate. Early shifted records alias the two, because the learner still follows the cue. The result can be "not within 128 records", and that is reported as such.

**Also report.** Records needed to separate the homeostatic type from the proxy type, and how a reader with training records only ties.

**Known-answer checks (exact, under the declared conditions):**
- In training, with the cue's intensity equal to the restoration amount, the mixed type's weights summing to 1 and the learner's initial trust in the cue set to 1, all types give identical choice likelihoods.
- With a learning rate of zero and its initial trust in the cue set to 1, the homeostatic learner never updates that trust, its only evidence at choice time, so its likelihood on shifted records equals the proxy type's.

**Shed order:** the delayed-restoration shift.

**Route.** The theory gains a design for testing "persistence at the top" in real records: how much evidence from settings where cues and persistence come apart would be needed.

## 3. Deferred, no preparation this week

D5's forty worlds, T1, T3, T4, N1 and PYMDP. N16's confirmation runs under section 0.2.

## 4. Cross-repository

**Approval for this week's copies.** Handing over this packet approves this week's aggregate-only copies into Sounding Line's `docs/exchange` and into a ToMpathy `docs/` folder named for the study. Copy, hash and record. Don't commit or push in the recipient repository.

| To | What | When |
|---|---|---|
| Sounding Line, via `docs/exchange` | W1's alias map; W2's readout failure rate, labelled "with the open-loop variance known; a lower bound on failure without it" | By Thursday 29 October, 12:00 PDT (19:00Z), after their checks pass |
| Sounding Line | W6's and W7's conditions that carry information and the records needed, for FR1, PP2 and PP4. W8's result by drift, for TR4, and W10's map, for TR3, as digest reading; no card waits on them | As each lands |
| Sounding Line | W4's records-needed curve | By Saturday 31 October, 12:00 PDT (19:00Z) |
| ToMpathy, as copies into `docs/` | W1's alias map and separation shares (TE7). W6's and W7's results (TE6; TE3 and TE4's badge text). W12's result (TE7's fourth question). W3's route (B12). W4's records-needed curve (its B15 record). N16's confirmation state (B1). The corrected stress-map sentence (B1's notes) | As each lands; W1 by Thursday 29 October, 12:00 PDT (19:00Z) |
| From Sounding Line | WB1's piece classes and standardised effects, for W5 | By Wednesday 28 October, 12:00 PDT (19:00Z) |
| From Sounding Line | PP1's pause hierarchy; FR1, WC2 and TR4's results, for calibrating W6 to W8's follow-ups | As each lands |

## 5. Schedule

| When | Work |
|---|---|
| Sun 25, from handoff | V25 acceptance generated. W0. Claim-ledger backfill. Protocols and fixtures for W1, W2, W6 and W7; nothing drawn |
| Mon 26, from 08:00 PDT | W1 and W2: cost probes, then batches. Once their probes are admitted, N16's production build starts |
| Tue 27 | W6 and W7. W1 and W2 replays and reconstructions. N16's confirmation starts when its runner passes controls and replay |
| Wed 28 | W3 and W10. W5's calibrated setting decided at 12:00 PDT (19:00Z) |
| Thu 29 | W1 and W2 posts by 12:00 PDT (19:00Z), after their checks pass. W8 and W12 |
| Fri 30 | W9 and W4. Replays and reconstructions for W3 to W12. N16's first 24-hour throughput projection, if it has started |
| Sat 31 | W5, W11 and, in idle slots, W13. W4 post by 12:00 PDT (19:00Z). Follow-ups under section 0.3 C |
| Sun 1 | Final note by 20:00 PST (04:00Z on 2 November), when new starts close. Drain by 22:00 PST (06:00Z) |

## 6. Reporting

- **Daily:** a status line with corrections first, then charged CPU against each pool and ceiling, each landed card with its claim id and checks, open modelling choices made, and follow-ups and healing packets registered.
- **Final note by Sunday 1 November, 20:00 PST.** Lead with three things in plain words:
  - which ends passive records can't separate and which intervention breaks each alias (W1), and which of those aliases a value-linked signature can break (level-shifted, habit) and can't (shared-piece) (W6);
  - what attention adds to a final text, and what a misread attention model costs (W7);
  - what protects against capture, and under which assumptions (W3, W10).

## 7. Triggers (the only reasons to contact the curator)

1. A study would need more than 1.5 times its ceiling after shedding, or a heal would need more than the follow-up and healing pool plus the card's 1.5 times allowance.
2. A verified result reverses a delivered public claim.
3. The V25 acceptance, N16's production contract or the deadline machinery fails.

## 8. Pre-mortem

- **The exact reader is too kind.** It holds the true library and the true choice rule.
  - *Guard:* W5 measures what a wrong library costs. W3's omission arm measures a wrong model of the source. W2's plant arm, W6's assumed-link reader, W7's misread attention and W8's wrong-hazard arms measure misspecification.
- **Studies build in their answers.** Several signs are fixed by construction or by information never hurting a correct reader.
  - *Guard:* each primary is a magnitude, a records-needed figure, or a misspecified arm that can fail. Built-in contrasts are labelled and reported for scale only.
- **Boltzmann choice is an assumption.** Real choice may be habitual or satisficing.
  - *Guard:* the habit variant, with rationality as a hypothesis dimension, and lapse rates where refusal would otherwise be certain. Every report states its assumed choice model.
- **Pieces are our units.** People may not act in discrete pieces (Botvinick & Plaut 2004).
  - *Guard:* Sounding Line's WB1 tests that on real records.
- **Thirteen families and N16's build crowd the builders.**
  - *Guard:* the order above, the shed orders, N16's build capped at 8 builder-hours and starting only after W1's and W2's probes are admitted, and open modelling choices made by the agent without a trigger.
- **The host is busier than it looks.** Sounding Line runs a counted lease and its own never-idle rule on the same machine.
  - *Guard:* below-normal priority, N16 yielding at block boundaries, and the wall-clock shedding rule in section 0.5.
- **Constructed worlds can't validate a human claim.** Every route says what a result licenses in the engine, and nothing more.

## Sources

- [Todorov & Jordan 2002](https://doi.org/10.1038/nn963)
- Powers 1973, *Behavior: The Control of Perception*
- [Marken 2014](https://doi.org/10.3758/s13414-013-0552-8)
- [Behrens et al. 2007](https://doi.org/10.1038/nn1954)
- [Adams & MacKay 2007](https://arxiv.org/abs/0710.3742)
- [Dimitrakakis & Rothkopf 2012](https://doi.org/10.1007/978-3-642-29946-9_27)
- [Kleiman-Weiner, Saxe & Tenenbaum 2017](https://doi.org/10.1016/j.cognition.2017.03.005)
- [Armstrong & Mindermann 2018](https://arxiv.org/abs/1712.05812)
- [Botvinick & Plaut 2004](https://doi.org/10.1037/0033-295X.111.2.395)
- [Kuppens et al. 2003](https://doi.org/10.1037/1528-3542.3.3.254) (abstract)
- [Berkowitz 1989](https://doi.org/10.1037/0033-2909.106.1.59) (abstract)
- [Schweitzer, Hershey & Bradlow 2006](https://doi.org/10.1016/j.obhdp.2006.05.005)
- [Johnson-Freyd & Freyd 2013](https://doi.org/10.1017/S0140525X12000398)
- [Brewer 1999](https://bcsh.bard.edu/files/2019/06/Brewer-ingroup-love-or-outgroup-hate.pdf)
- [Balliet, Wu & De Dreu 2014](https://doi.org/10.1037/a0037737) (abstract)
- [Dickinson 1985](https://doi.org/10.1098/rstb.1985.0010)
- [Kruglanski et al. 2018](https://doi.org/10.1037/rev0000095) (abstract)
- [Hubinger et al. 2019](https://arxiv.org/abs/1906.01820)
- [Keramati & Gutkin 2014](https://doi.org/10.7554/eLife.04811)
- [Langosco et al. 2022](https://arxiv.org/abs/2105.14111)
- [Omohundro 2008](https://selfawaresystems.com/2007/11/30/paper-on-the-basic-ai-drives/)
- [Turner et al. 2021](https://arxiv.org/abs/1912.01683)
- Prior Ghost results cited: V21 `DRIFT_REPORT.md`; V22 `FINAL_REPORT.md` (A1, A3, B2); V23 `N4D_REPORT.md` and `N11_REPORT.md`; V24 `FOLLOWUP_RESULTS_2026-10-23.md`, `N16_PILOT_REPORT.md` and `results/v24/preparation/N16_DESIGN_CALIBRATION.json`.
