# Ghost Scale Sim, V26 theory day (6 to 8 November): what a feeling's speed is measured against, identity, security, and closing the model

*Spec sheet for the Ghost agent. Every decision is made here, and section 0.4 says how the agent fills any modelling choice a card leaves open. Return to the owner only on the triggers in section 3; the owner is the curator. As of Friday 6 November 2026, against `origin/main` at `cf6778e8f` (V26 published, on the theory-review hold since 11:59 UTC). An independent design check with its own simulations ran on every card before handoff, and its findings are folded in.*

**What these cards test.** The curator's theory thread (3 and 4 November) and his answers of 6 November put five claims where a constructed world can test them:
- **G1. Velocity against the balance of needs (C3).** A channel's trigger is a slope toward or away from a set point, and the speed is "balanced against the strength of all the other needs".
- **G2. Identity by preemption (C5).** A value nothing trades for is a pre-solved block that answers first. When a frame puts another value "front and center", the two "have to compete", and "the first answer you come up with gets precedence".
- **G3. Closing the model (C10).** DISGUST "does follow an inferred desire to harm", and then the reader stops: "They never take one step further once you find a puzzle piece that is truly inefficient."
- **G4. Identity at any price (C5).** Is "nothing trades for it" a distinct type, or the limit of a strong terminal weight? V26's E1 never asked at a price that would buy a compensatory agent.
- **G5. Security as a set point (C4).** On 3 November he wanted to be "as far away from dissolution as possible"; by 4 November he called security a set point, which the thread summarized as "defended from below, spent from above". Can a reader tell a defended set point from a smooth preference for more security?

Ghost's job is V26's: find, in constructed worlds with exact readers, what each claim lets a reader identify and what it doesn't, and what Sounding Line and ToMpathy should look for. Every result is a constructed-world result, "miniature, architecture untested", and says nothing about how people work.

**Pre-checks.** Every registered case check and primary below was pre-checked before handoff. Most were simulated twice, by two independent standalone implementations; where the two differ, both are given. G3's registered values come from an exact programme over counts. G3's contamination quantity, G4's open quantity and G5's information form were computed by one implementation. The numbers given are expectations for scale, not results. Several verdicts are fixed by the design; each card lists them, and the reports must say so. The agent writes its own implementation from this text, under the house sequence.

## 0. How to run this

### 0.1 The hold

- **Handing this packet over is the owner's direction** to lift the theory-review hold for the five cards in section 2 only:
  - their protocols, design pre-mortems and known-answer controls;
  - their probes, studies, replay and review;
  - their texture lanes;
  - healings under amendment B, at most one per card;
  - one terminal-event and twelve-hour monitor for this window, retired when it closes.
- **No follow-ups are commissioned** (amendment C does not apply to these cards).
- **Everything else stays held:**
  - V25's other work;
  - the eight admission-held preparations;
  - the failed originals with no retry;
  - any unexecuted Friday scope.
- **No valid packet is rerun,** and no Friday map is extended.
- **When these cards finish, or the window closes,** retire the monitor and return V26 to `PUBLISHED_AWAITING_THEORY_REVIEW`, with these results published beside the rest. Add a dated paragraph to `AGENTS.md` under the hold section saying which cards this packet lifted, and when they closed.

### 0.2 Window

- **Opens at handoff.**
- **New starts close:** Sunday 8 November, 20:00 PST (2026-11-09T04:00Z), as in the 4 November authorization.
- **Drain by:** 22:00 PST (2026-11-09T06:00Z).
- **Order:** G1, G2, G3, G4, G5 for the registered primaries; texture lanes after each family's primary is verified (amendment F), in the same order.

### 0.3 Budget

- **Total.** The 4 November authorization's 110 CPU-hours stand unchanged.
- **Remaining at publication** (`THEORY_REVIEW_PUBLICATION.json`):

| Pool | Remaining (CPU-hours) |
|---|---:|
| Studies | 43.89 |
| Preparation | 2.11 |
| Follow-ups, healing and texture | 13.00 |
| Protected | 5.88 |

- **Charging.**
  - Each card's probe, study, replay and review go to the studies pool, under the card's studies ceiling.
  - Texture and healing go to the follow-up, healing and texture pool, texture under the card's texture ceiling. A healing takes precedence over texture cells not yet started, if the pool cannot hold both.
  - Protocols, pre-mortems and fixtures go to preparation.
  - The 4 November family ceilings already account for the whole studies pool, so they are frozen at their charges. The ceilings below come from the studies pool's unspent hours outside them.
- **Ceilings** (CPU-hours; the projections include the 1.5 reserve. G1 and G2 are priced at the slower of their two pre-check rates, G3 at its measured rate, and G4 and G5 at V26's measured E1 and W13 rates):

| Card | Studies ceiling | Studies projection | Texture ceiling | Texture projection |
|---|---:|---:|---:|---:|
| G1 | 2.6 | 2.46 | 6.0 | 6.0 |
| G2 | 0.5 | 0.30 | 2.0 | 1.96 |
| G3 | 0.25 | 0.07 | 0.5 | about 0.2 |
| G4 | 1.25 | 1.17 | 1.4 | 1.35 |
| G5 | 1.0 | 0.94 | 0.75 | 0.71 |
| **Total** | **5.6** | **4.94** | **10.65** | **about 10.2** |

- **Shedding.** If a card's studies would pass its studies ceiling, shed in the card's stated order, never primaries or controls; a card over 1.5 times its studies ceiling after shedding is trigger 1. Shed orders cover studies only. Texture cells run in the order each card lists them, out-of-case maps last, and a texture lane stops when it reaches its texture ceiling or amendment F's stopping rule, whichever comes first. It never triggers.

### 0.4 Rules

- **Imported from the V26 week packet, by name only:**
  - amendment E (design pre-mortem);
  - amendment F (texture lane);
  - amendment G (routes in words);
  - section 0.4's admission rules and house sequence, through the day module in 0.5;
  - section 0.5's reserve from measurement;
  - section 0.6's readers and scores.

  V26's one-time handoff steps (writing amendments into `AGENTS.md`, landing N16, generating the V26 acceptance) are not repeated.
- **Who revises.** Where amendments E and F say "the owner" revises a grid or runs texture batches, the agent does so for these cards, within each card's declared revision ranges. Cards that declare no revision ranges are not revised.
- **Open modelling choices** are the agent's. Declare them in the protocol before any draw, mark them provisional, and list them in the daily line.
- **Known-answer checks** are exact (tolerance 1e-12 unless a card says otherwise). Each card's fixtures run as separate tests before its probe.
- **One rule for every bar.** A bar is met when the 95% interval (across blocks, or the block bootstrap for records needed) lies wholly on the passing side. It is not met when the interval lies wholly on the other side. Otherwise it is reported as "unresolved at this sample size", beside the point estimate. No verdict is read from a point estimate alone.
- **Texture cells.**
  - Each texture cell runs its card's case check first, with the cell's own wording where the card gives one.
  - A cell that passes is a texture variant under amendment F.
  - A cell that fails, or that a card declares an **out-of-case map**, runs descriptively. It is exempt from the failed check, not counted by amendment F's stopping rule, and it never upgrades or overturns a primary. Its report says which check it fails and by how much.
- **Workers:** Gear 2, at most two single-thread, below-normal workers. No GPU. No environment synchronization. Sounding Line keeps priority: its theory-days packet runs this weekend on the same machine.
- **Inference unit:** sixteen independently seeded whole-world blocks per setting. Report means with 95% intervals across blocks, and censored medians with block bootstraps for records needed.

### 0.5 Admission module and supplement

- **Module.** Build `ghostscale/validation/soundingline/v26/day.py`, modelled on `friday.py`. The V26 contract rejects families outside H1 to R13, and Friday's admission rejects texture and any start after its 14:00Z review target, so neither can admit these cards. The day module:
  - window: new starts close 2026-11-09T04:00Z and drain by 06:00Z, with no review target;
  - packet kinds: probe, study, texture and healing;
  - families G1 to G5, with the studies and texture ceilings in 0.3;
  - the unchanged 110-hour total, with prior charges carried forward;
  - V26's drain-margin test (`contract.admission`): refuse a start whose projected wall time reaches the drain;
  - Gear 2, below-normal, one numerical thread.

  Its tests run before the first probe, and its build is charged to preparation.
- **Supplement.** Copy this packet to `docs/versions/v26-top-and-ends/day-20261106/PACKET.md`. Record `results/v26/day-20261106/AUTHORIZATION.json` with:
  - the copy's path and SHA-256;
  - the owner's direction, as handing over this packet;
  - the ceilings in 0.3, and the statement that the 4 November family ceilings are frozen at their charges;
  - the pools' remaining hours at publication;
  - the unchanged 110-hour total;
  - the window.

### 0.6 Folders and ids

- Folder: `docs/versions/v26-top-and-ends/day-20261106/`.
- Status: `results/v26/day-20261106/CURRENT_STATUS.json`.
- Claim-ledger rows: `G-V26-DAY-<card>-<n>`.

## 1. State

| Item | State at `cf6778e8f` |
|---|---|
| V26 | 10 primaries, 56 texture variants, 2 follow-ups and 52 Friday claims published; 45.12 of 110 CPU-hours charged; workers, controller and monitor stopped |
| Hold | `PUBLISHED_AWAITING_THEORY_REVIEW`, resumed by the owner only |
| V25 | Held for the owner's theory review |

**What V26 found that these cards build on** (constructed, exact readers):

| Finding | Bearing here |
|---|---|
| E1: offers were capped, so "nothing trades for it" was never tested at a price that would buy a compensatory agent; usage counts read locked identity worst of three value types | G2, G4 |
| E3: a reader that takes cached firings for signals of current wants loses 0.08 to 0.24 nats per maker; nothing in V26 triggers on a rate of change | G1 |
| U1: an opaque learner is captured by a half-biased source (0.87 to 0.92 on the false hypothesis at 32 claims, by construction); about 24 revealed records fix it | G3 reuses U1's world |
| R13 and W13: a homeostatic world with decaying stocks exists, but no card contrasted a set point with a level | G1, G5 |

## 2. Cards

Each card states:
- its case and check (amendment E), and its revision ranges or "no revision";
- what its design fixes;
- its primary, with a practical bar judged by the rule in 0.4;
- its falsifiable arm, or why it has none;
- its known-answer checks;
- its texture grid, with out-of-case maps marked;
- its shed order;
- its route (amendment G).

### G1. VELOCITY_REFERENCE. What is a trigger's speed measured against? (studies 2.6, texture 6.0)

**Why.** C3 holds that triggers are slopes, not levels: the rate of change toward or away from a set point. The curator's answer on 6 November to what that speed is measured against:

> What is velocity measured against? I mean, it can't just be social. Is it just like other needs? Because at a certain point we have to balance the needs against each other. And so it kind of seems like it might be balanced against the strength of one need balanced against the strength of all the other needs, maybe. And the kind of weighting on that, that's one of the things that Solms has suggested is consciousness, for choosing how to weight one versus the other.

> We're talking about you control attention; that's what you're controlling attention on.

And on why mood falls when nothing changes: "all of the needs that you currently have unmet will slowly drift further away from acceptability, and mood decreasing would be normal from that."

*Transcription repairs: "waiting" to "weighting", "wait" to "weight", "Psalms" to "Solms".*

**World.**
- **Two needs** (security and belonging), each a stock on the grid 0 to 20 units; one unit is 0.05 on the 0-to-1 stock scale.
- **Each record, for each need, one hidden outcome:** decay by 1 unit (probability 0.5), a drop of 5 units (probability 0.05), or nothing. A drop replaces that record's decay.
- **Menu:** each need's restoring option is on offer with probability 0.35, independently, and an "other" option is always on offer. The reader sees the menu.
- **Restoring** adds 5 units to its need. Stocks are clipped to 0 to 20.
- **Choice:** softmax over the offered options with precision 1 per unit, and a lapse of 0.02 spread over the offered options. Restoring need i is worth min(5, its deficit in units); "other" is worth 1.
- **Set points:** each need's set point is drawn from {8, 11, 14} units (0.40, 0.55, 0.70), independently. Initial stocks are uniform on 0 to 20.
- **Channels:** each need has a channel that fires on a record with probability 0.02 + 0.96(1 − e^(−g·x)), with x on the 0-to-1 stock scale (a one-unit fall is x = 0.05; g = 8 gives g·x = 0.4). The maker's gain g is drawn from {8, 16, 32}.
- **Records:** 512 per maker.

**Trigger laws.** Deficit D = max(0, set point − stock); "before" and "after" bracket the record.
- **(a) level:** x = D after the record.
- **(b) raw slope:** x = the fall, max(0, stock before − stock after).
- **(c) direction-relative, C3 as stated:** x = max(0, D after − D before). A fall fires only while it moves the stock further below its set point.
- **(d) balance-relative, his 6 November answer:** x = the fall × n × u_i / Σ_j u_j, with urgency u = 0.1 + D after, both in units, and n = 2. With equal urgency the weight is 1, which is law (b).

**Readers** (exact; 108 hypotheses: 4 laws × 9 set-point pairs × 3 gains; uniform priors; initial stocks uniform):
- **Ceiling:** full observation. It sees the stocks, so every decay and drop. Reported as the ceiling, never as a contrast.
- **Hidden-stock readers** see menus, choices and firings, not stocks. They run the forward algorithm over the 21 × 21 stock grid and the hidden outcomes:
  - *both:* choice and firing likelihoods;
  - *firings:* firing likelihood only, with choices treated as known actions;
  - *choices:* choice likelihood only.

**Case and check** (amendment E; check-only seeds). The case: laws whose firing predictions differ on the records the maker actually generates, in worlds that visit both sides of each set point.
- For every ordered pair of laws, pooled over makers, at least 25% of eligible records must have firing probabilities differing by more than 0.2. A record is eligible when either law's trigger is above zero on that channel (E3's denominator). The rival law is held at its best-matching gain, so a difference of scale can't pass the check.
- At least half of the worlds must spend at least 25% of their records on each side of each set point, for both needs.
- **Pre-checks** (240 check-only worlds, 512 records):
  - smallest pair share 0.356 (b against d) in one implementation, 0.336 in the other;
  - c against d 0.404 and 0.451, and 0.463 and 0.397;
  - both sides of the set point, both needs: 0.90 of worlds at 256 records, 0.97 at 512.
- **Two grids failed and are retained as failed grids.** Decay on every record fails on b against c (0.198) and on the time split (0.004 of worlds), because stocks then sit below the set point on 76% of records. Menus always on offer fail b against d (0.071) and the time split (0.000).
- **Revision ranges:** menu availability 0.25 to 0.5, restoring 4 to 5 units, urgency floor 0.1 to 1.

**By construction** (report for scale only):
1. Conditional on law (b), the firings carry no set-point information: a reader whose law set is {b} has a firings-only set-point posterior equal to the prior. The registered reader marginalizes over four laws, so under truth (b) its firings-only posterior departs slightly from the prior (up to 0.08 at 256 records in the pre-check).
2. Choices carry the set points under every law, because the restoring rule depends on them, and they carry nothing about the law. Menus, stocks and choices have the same distribution under every law, so P2's choices-only recovery has the same expectation under all four laws.
3. (d) equals (b) whenever both deficits are equal, including whenever both needs are at or above their set points.
4. (a) and (c) are zero at or above the set point. (b), (c) and (d) sit at the 0.02 floor on any record without a fall.
5. The ceiling dominates the hidden-stock readers in expectation.
6. The arm's statistic under (b) is at most about 0 (items 1 and 2), so only its values under (a), (c) and (d) are informative.

**Primary.**
- **P1, laws.** Records for the hidden-stock *both* reader to hold a sustained 1-nat margin for the true law over the other member of the (c) and (d) pair, with each as truth. This sets C3 as stated against his answer. Within 128 records counts as "a reader can tell them apart". Also report the full truth-by-rival matrix for the ceiling, *both* and *firings* readers.
- **P2, set points.** Mean posterior on each need's true set point at 256 records (chance 0.333), from firings alone, choices alone and both, by law, with the ceiling beside them. A mean of 0.8 or more counts as recovery.

**Falsifiable arm.** Under (c), firings alone put at least 0.05 more posterior on the true set point than choices alone, at 256 records. If they don't, C3's set-point term doesn't make firings a better window on set points than choices, in these worlds.

**Known-answer checks** (tolerance 1e-12 unless stated):
- **Forward algorithm:** it equals brute-force enumeration over initial stocks and hidden outcomes on a 4-record fixture (tolerance 1e-9).
- **Law fixture:** hand-computed x for (a) to (d) in three cases:
  - a fall that crosses the set point, where (c) is only the part below it;
  - a fall entirely above the set point, where (c) is 0 and (b) is the fall;
  - unequal deficits, for (d).

  A law (c) built on a signed deficit collapses into (b) and passes every other check here, so this fixture is the one that catches it.
- **{b}-only reader:** under (b), its firings-only set-point posterior equals the prior.
- **(d) and (b):** they are equal at equal deficits.
- **No drops:** with deterministic decay and a known start, the hidden-stock reader equals the ceiling.
- **Floors:** no likelihood is ever zero (firing floor 0.02, choice lapse 0.02).

**Also report:**
- the posterior on the true law at 128, 256 and 512 records;
- firing and restoring-choice rates by law;
- the hidden-stock cost: records needed by *both*, over the ceiling's.

**Texture grid** (amendment F; run in this order):
- **Menu availability 0.25.** It passes the check at 512 records.
- **Urgency squared.** It passes.
- **(k), a one-sided variant of the drive in homeostatic reinforcement learning** (Keramati & Gutkin 2014), applied to rises in drive, as a fifth law beside (a) to (d):
  - Drive is one-sided, D(H) = (Σ_i max(0, h*_i − h_i)^n)^(1/m), over both needs on the 0-to-1 scale, with n = 4 and m = 3 by default; any change must keep n > m > 1. The paper's drive is two-sided, symmetric about the set point.
  - The channel for need i fires on x_i = max(0, D(need i after, the other need before) − D(before)): the rise in total drive from need i's own change.
  - Under this law the same fall counts for more the bigger that need's own deficit, and for less the bigger the other need's deficit. In the paper, the "inhibitory effect of irrelevant drives" is a property of reward; (k) carries the same property over to the rise in drive. A fall that starts at the set point barely fires it.
  - It runs its own case check against (b), (c) and (d). The pre-check passes narrowly (k against c, 0.26 to 0.27), so the cell may run as an out-of-case map.
- **Out-of-case map, menu availability 0.5.** It fails b against d (0.241) and the time split (0.192). It documents the alias between law (d) and the raw slope when restoring options are often on offer.

Dropped after the pre-check:
- the urgency floor of 1.0 fails b against d (0.149);
- gains {4, 8, 16} fail c against d (0.205);
- C3's direction-relative velocity weighted by urgency share sits too close to (c) to separate (0.188).

**Deferred.** The forewarned-drop arm, which would test the curator's note that forewarning doesn't reduce grief, needs an expectation-relative law and a forewarning cue the reader sees. It wasn't pre-checked, so it isn't commissioned.

**Shed order** (studies): the 512-record level; then the *firings* reader's truth-by-rival matrix, keeping its P2 column and the arm.

**Cost.** The two pre-checks ran at 5.24 and 7.35 to 7.71 ms per maker-record per pass, for all four readers. This card is priced at 7.5 ms, or 22.5 ms over V26's three passes.
- Registered: 4 laws × 128 worlds × 512 records = 262,144 maker-records, 1.64 CPU-hours; 2.46 with reserve.
- Texture: three four-law cells and one five-law cell, each 64 worlds × 512 records, about 557,000 maker-records. The five-law cell is priced at 1.5 times the rate (the second pre-check measured 10.57 against 7.06 ms per maker-record per pass), so the lane comes to 3.99 CPU-hours; 6.0 with reserve.

**Route.**
- *ToMpathy:* the engine reads channels without seeing a writer's stocks. G1 says how many records it needs to tell trigger laws apart, and whether set points can be shown from channel cues, choices or both. The pre-checks expect hidden stocks to leave set points poorly placed, so the core-values panel shouldn't show set points from cues alone.
- *Sounding Line:* a real-record test of trigger laws needs observed context for the stocks (dated events, losses, contacts), not channel cues alone.

**Pre-check expectations** (first implementation 64 makers per law; second 32 to 96):
- **P1, (c) against (d), hidden-stock *both* reader:** 23 and 25.5 records, or 37.5 and 16, with (c) and (d) true. In the menu-0.25 cell, the second implementation needed 115.5 and 41, so P1 isn't settled by the case check.
- **Against all rivals at once:** *both* needed 3, 48.5, 29 and 57 records for (a) to (d), or 2.5, 49, 49 and 40.5. The ceiling needed 1, 18, 19.5 and 25, or 1.5, 24.5, 22 and 30.
- **P2 at 256 records** (firings / choices / both / ceiling; first, then second implementation):
  - (a): 0.533 / 0.415 / 0.528 / 1.000; 0.546 / 0.463 / 0.550 / 1.000;
  - (b): 0.334 / 0.453 / 0.515 / 0.989; 0.330 / 0.425 / 0.465 / 1.000;
  - (c): 0.460 / 0.453 / 0.515 / 1.000; 0.508 / 0.461 / 0.536 / 1.000;
  - (d): 0.426 / 0.449 / 0.541 / 1.000; 0.416 / 0.472 / 0.524 / 1.000.
- **Arm under (c):** +0.007 [−0.030, 0.044] in the first, and +0.038 [+0.011, +0.065] in the second (96 makers). The first interval lies wholly below the bar of 0.05 and the second straddles it. At the registered 128 makers the half-width is about 0.025, so expect "not met" or "unresolved". Under (a) the statistic is +0.12 in the first implementation and +0.083 [+0.028, +0.139] in the second.

### G2. IDENTITY_BY_PREEMPTION. Does a reader need the frame to read identity? (studies 0.5, texture 2.0)

**Why.** C5: identity is terminal; nothing trades for it. The curator on 6 November, on protected values that give way when reframed:

> To an extent, what we're talking about is pre-solved blocks kind of jumping to the wall when something looks familiar. Like, you see a pre-solved block of "I value life", and that provides some of the satisfaction for one of my needs, and I value that high enough that I can't see anything else that would be worth trading off for it. That is an end goal.

> But you could just as easily explain to me something else that could be worth trading for it, and present a circumstance where another one of my values is all of a sudden front and center. And now they have to compete a little bit. Often, I think, you get preemption, where the first answer you come up with gets precedence a little bit, which makes sense. That would just be high-level implementation of synaptic inhibition.

The literature agrees in outline. People "often acquiesce" when a taboo trade-off (a sacred value against a secular one) is reframed as routine or tragic (Tetlock 2003). Taboo trade-offs felt easier than routine ones, and tragic ones (two sacred values) felt hardest (Hanselmann & Tanner 2008). A latency arm, quick refusal on taboo offers and slow choice on tragic ones, is not commissioned.

**World.**
- **Values and menus:** three values; four options per record. Each option's outcome vector is on {0, 0.5, 1}^3, and each option is a one-hot specialist with probability 0.25.
- **Frame:** each record carries a frame drawn independently of the maker: no cue with probability 0.55, and each value cued first with probability 0.15.
- **Records:** 512 per maker.

**Maker types** (choice precision 3; lapse 0.02 over the four options):
- **A, compensatory:** fixed weights w on the simplex grid with step 1/4 (15 points); softmax over e·w. Ignores frames.
- **L, lexicographic identity on value k:** if an option serves k at 1, it chooses among those by e·w; otherwise softmax over e·w. Ignores frames. Weights have w_k as the (weak) maximum.
- **P, preemptive identity on value k:**
  - Precedence goes to the value cued first. With no cue, k counts as cued first.
  - It tries the cued value, then k, and takes the options serving the first of these that has an option at 1, choosing among them by e·w.
  - Otherwise, softmax over e·w. Weights as L.
- **Texture only, Ps, soft precedence ("a little bit"):** softmax over e·w + π·e[cued value], with π in {0.5, 1, 2}. With no cue, k counts as cued.

**Readers** (exact over type, identity value and weights; 51 hypotheses, 69 with Ps). The prior is uniform over types, then uniform within each type, as E1's. Both pre-checks used a prior uniform over hypotheses; under the registered prior, the second implementation's frame-blind figure moved from 17.5 to 19 records.
- **Frame-aware:** sees frames. It is the full-observation reader, reported as the ceiling.
- **Frame-blind:** marginalizes the frame on each record with its known distribution.

**Case and check** (check-only seeds; floor 25% for each).
- **The case:** preemptive makers whose choices, seen without the frame, differ from every compensatory maker's, on menus where the identity value conflicts with a cued competitor.
- **The checks:**
  1. Conflict menus: one option serves the identity value at 1, another serves a different value at 1, and none serves both. Pre-checks 0.334 and 0.316.
  2. A competing value is cued first. The share is twice the per-value cue probability, by rule. Pre-checks 0.307 and 0.300.
  3. Frame-blind P differs from the best-matching compensatory maker (best of the 15 weights over the maker's records) by more than 0.2 in total variation. Pre-checks 0.327 of records with 80% of worlds at or above 0.25, and 0.308 with 73%. For Ps cells, the same check applies to Ps makers.
- **Failed grids.** All-specialist menus fail check 3 (0.03 to 0.05), and fully graded menus fail check 1 (0.214); both are retained as failed grids.
- **Revision ranges:** specialist share 0.15 to 0.5, and cue probability 0.15 to 0.2. Below 0.125, check 2 fails by rule.

**By construction** (report for scale only):
1. With frames, P and L choose differently only on records where a value other than k is cued first and served at 1. One such record gives about 5 nats, so frame-aware P-against-L separation arrives with the first one (pre-check medians 6 and 5.5 records).
2. L takes a k-at-1 option with probability at least 0.98, while A gives it only its softmax share: the sharpness gap E1 used.
3. P equals L when no competing value is ever cued.
4. A and L ignore frames.
5. Lexicographic choices on conflict menus reveal k.
6. Inside the case region, check 3 makes "preemptive identity misread as compensatory" all but impossible by 64 records. The pre-checks found 0% to 1.6%.
7. Inside the case region, the frame-blind reader is expected to need fewer than 64 records. The pre-checks found 12 to 19 records across passing cells, so the registered verdict, "the frame is not needed", is expected by design. "Needed" appears only in the rare-cue out-of-case maps.

**Primary.** For preemptive makers, the records the frame-blind reader needs to hold a sustained 1-nat margin for P over the strongest rival type, with the frame-aware reader's records and the ratio of the two beside them. The bar: a frame-blind median above 64 records *and* more than twice the frame-aware median counts as "a reader needs the frame to read preemptive identity". Both parts are judged by 0.4's rule, with block-bootstrap intervals for the frame-blind median and for the ratio. By item 7, the registered verdict is fixed. The records-needed estimate and the ratio are the result. Also report, for every type, separation from compensatory makers and recovery of the identity value without frames.

**Falsifiable arm.** None. The arm first drafted, "misread as compensatory", is the negation of check 3 (item 6). Report the shares of P makers read as A, and read as L over P, at 64 records, for scale.

**Known-answer checks** (run as separate fixtures):
- With no competing cues, P's likelihood equals L's.
- On a 4-record fixture, the frame-blind sequence likelihood equals the sum, over all 4^4 frame sequences, of each sequence's probability times the frame-aware sequence likelihood.
- On a menu with no option serving any value at 1, L and P equal A with the same weights.
- No likelihood is zero (lapse 0.005 per option).

**Texture grid:**
- specialist share 0.5 (borderline at 0.256; a variant if it passes);
- cue probability 0.25;
- five options;
- **out-of-case maps:**
  - all-specialist menus (fails check 3);
  - cue probability 0.02 and 0.05 (fail check 2 by rule; these are where "needed" can appear);
  - Ps at π 0.5, 1 and 2 (fail check 3 for Ps makers: 0.009, 0.087 and 0.243).

**Shed order** (studies): the 512-record level.

**Cost.** The pre-checks ran at 0.20 to 1.21 ms per maker-record per pass for both readers, and up to 1.77 ms with Ps in the roster. This card is priced at the higher rates.
- Registered: 3 types × 128 worlds × 512 records = 196,608 maker-records, 0.20 CPU-hours over three passes; 0.30 with reserve.
- Texture: 9 cells × 3 types × 64 worlds × 512 records, up to 1.31 CPU-hours; 1.96 with reserve. In the Ps cells, Ps makers replace P makers, so every cell has three types.

**Route.**
- *ToMpathy:* does the engine need to know what the context made salient first, to read identity? This decides whether the core-values panel may mark a value as held as identity from choices alone.
- *Sounding Line:* a real-record test of reframing needs the frame (the prompt, or the first-mentioned consideration) recorded beside the choice.

**Pre-check expectations** (64 worlds per type):
- **Registered cell:** frame-blind 16.5 and 17.5 records under the prior uniform over hypotheses, and 19 under the registered prior (second implementation), against frame-aware 6 and 5.5. The identity value was recovered by 64 records with or without frames.
- **Other passing cells:** cue 0.125 needed 17.5 records, cue 0.2 needed 12, specialist 0.15 needed 16 and specialist 0.5 needed 15.
- **Out-of-case maps:**
  - competing cues rare (0.02): 143 against 52, and 116.5 against 32.5;
  - all-specialist menus: 39 against 3.5;
  - soft precedence at π 1: 44.5 against 16.5.

### G3. CLOSING_THE_MODEL. When does DISGUST's closure protect a reader? (studies 0.25, texture 0.5)

**Why.** C10: DISGUST shuts intake, pulls away and stops running the other's model. The curator on 6 November:

> Disgust would be dislike of inefficiency, whereas anger triggers as soon as you start hurting my copies.

> I mean, disgust does follow an inferred desire to harm. Once you figure out that they have a desire to harm, the inefficiency of that is dismissible. They don't care about why there is a desire to harm. They never take one step further once you find a puzzle piece that is truly inefficient. It cannot be slotted anywhere. There's just no reason to ask why someone is racist, unless you yourself are considering becoming racist, and that's how you would do so.

So the gate fires after a read of intent, not before it. It also fires on a piece that can't be slotted.

**World** (U1's).
- **Hypotheses and evidence:** two hypotheses with prior 0.5. Each record, the world holds a pool of 8 items, each pointing to the true hypothesis with probability 0.7.
- **Sources:** each presents one item per record, over 64 records. Each world holds 32 sources with alternating true hypotheses.
- **Source types:**
  - *H, honest:* a uniformly drawn item.
  - *M, in-library manipulator:* agenda hypothesis 1 at strength 0.5. It presents an item pointing to 1 with probability 0.5, if the pool holds one.
  - *Mx, stationary over-pusher:* agenda 1 at strength 1. Outside the reader's library.
  - *Mc, con:* honest for 16 records, then agenda 1 at strength 1. Outside the library.
  - *HU, honest with unfittable acts:* honest, plus, with probability ρ = 0.05 per record, an act that carries no item and that no model in the library can slot.

**Readers.**
- **T, ceiling:** knows each source's true policy (U1's transparent learner).
- **K, keeps modelling:** the exact posterior over the hypothesis and the source type. Its library holds three source models, each with a prior: honest (0.5); agenda 1 at strength 0.5 (0.25); and agenda 0 at strength 0.5 (0.25). An unfittable act has probability 0.001 under every hypothesis, so it carries no evidence.
- **D, disgust gate:** acts as K until it reads harmful intent (its posterior that the source pushes the false hypothesis reaches θ), or until it meets an unfittable act. Then it closes: no further intake or inference, and its posterior on the hypothesis is frozen after the triggering update. θ cells 0.5, 0.7 and 0.9; 0.7 registered.
- **O, opaque:** U1's opaque learner at trust 1.

**Case and check.** The case: manipulators that the keep-modelling reader can't represent capture it. The check: among sources whose true hypothesis is 0, K's posterior on the false hypothesis exceeds 0.5 at 64 claims for at least 25% of sources, for both Mx and Mc. Exact shares: 1.000 for both Mx and Mc (Mc's mean capture is 0.964). The Mx half holds by the library's geometry (item 4). **Revision ranges:** no revision; a failed check records "design cannot produce the case".

**By construction.** Every source emits Bernoulli items, and every reader is a function of the counts and of the record at which D stops, so all G3 expectations can be computed exactly by dynamic programming over counts. A count-level programme does it in seconds.
1. **Closing costs on average.** Averaged over sources drawn from the reader's own library prior, closing can only cost in expected log score. For a single source type it can gain: on honest sources at θ 0.7, D minus K is +0.002 at 32 claims and +0.008 at 64.
2. **The HU loss is set by the rule.** The gate closes at the first unfittable act, so its loss on HU sources is set by ρ and the horizon: about −0.075 nats at 32 claims; closed by 64 claims in 96 to 98% of sources, median record 14.
3. **O is captured** by any agenda source, and O equals T on H and HU sources. So its win over K on honest sources (−0.025 against −0.314 log score at 32 claims) holds by construction.
4. **The over-pusher never trips the gate.** Its emission rate (0.942) is nearest the library's "agenda 1, hypothesis true" model (0.85), so K puts 0.999 there and almost nothing on harmful intent. It trips the gate with probability 1.9e-10 at θ 0.7. Its capture of K is 0.998 at 32 claims and 0.99999 at 64. K reads it as an agenda source pushing what K takes to be the true hypothesis, so harmful intent never forms.
5. **Floors.** The harmful-intent posterior can first reach 0.5 at record 14, 0.7 at record 28 and 0.9 at record 52, on any path. So nothing can close during the con's honest phase at θ 0.7. At θ 0.9 the gate can fire only in the last 13 records, which sets that cell's result near 0.
6. **The registered results are exact.**
   - Primary (D minus K capture against the con, at 64 claims): −0.4837, −0.0897 and −3e-7 at θ 0.5, 0.7 and 0.9.
   - Arm (honest sources closed by 64 claims): 34.0%, 8.99% and 0.58%.
   - With about 1,024 qualifying con sources per cell, the standard error is near 0.006, so the registered verdicts are known before running: the primary passes and the arm fails.
7. **No pull back.** With θ = 1 and ρ = 0, D equals K.
8. **The θ 0.9 contamination cell is fixed by the floor.** The gate and K's drift (primary (ii)) can act only from record 52, so that cell's answer is "above 0.1" (in the pre-check, D still trailed at δ 0.5).

**Primary.** Two results, both registered:
- **(i) Exact.** The dynamic-programme values in item 6, labelled "fixed by the definitions". The bar: D minus K capture at or below −0.05 counts as "closing protects against the con". The simulation must reproduce the values within its sampling error; that is a known-answer check of D.
- **(ii) Contamination, the card's open question.** It formalizes the curator's "that's how you would do so", as the analyst reads it:
  - The drift, operationally: while K's posterior that a source pushes the false hypothesis is at or above θ, after each record's Bayesian update, add δ to the log-weight of every joint hypothesis whose hypothesis equals the agenda K's posterior favours for that source (the agenda model with more posterior mass). D, once closed, takes no such drift. With δ = 0, K is unchanged. The drift stops itself: it lowers K's own harmful-intent posterior below θ.
  - δ grid: 0.01, 0.02, 0.05 and 0.1.
  - Report D minus K in expected log score on the true hypothesis at 64 claims, by δ, over in-library manipulator (M) sources whose true hypothesis is 0, at θ 0.7, with the 0.5 and 0.9 cells beside it. The answer is the smallest δ at which D's lead is met under 0.4's rule. If a smaller δ is unresolved, report the interval between them ("between 0.05 and 0.1"); if no δ on the grid is met, report "above 0.1"; if 0.01 is met, report "at or below 0.01".
  - Report D minus K by δ on honest sources too, as a second table. There the drift helps K: a false alarm on an honest source attributes to it an agenda that is in fact the true hypothesis. So "the smallest δ at which D leads" is not the summary there.
  - The drift makes K's path matter, so simulate, with the exact programme's δ = 0 values as the known answer.
  - **Pre-check expectation** (one implementation; 20,000 M sources with true hypothesis 0 per cell, 64 claims):

    | θ | D minus K at δ 0.01 / 0.02 / 0.05 / 0.1 | Expected answer under 0.4's rule |
    |---|---|---|
    | 0.5 | −0.035 / +0.010 / +0.125 / +0.218 | between 0.02 and 0.05 |
    | 0.7 | −0.050 / −0.041 / −0.011 / +0.020 | between 0.05 and 0.1, or 0.1 (crossing near 0.07) |
    | 0.9 | −0.017 / −0.015 / −0.015 / −0.012 | above 0.1, by item 8 |

    At δ 0 the simulation reproduced the exact programme (−0.0619 against −0.0633 at θ 0.7). On honest sources at θ 0.7, D minus K fell from +0.009 at δ 0 to −0.017 at δ 0.1. With about 1,024 qualifying M sources per registered cell, the half-widths are about 0.023 at θ 0.5 and 0.012 at θ 0.7, which gives the expected answers in the table: at θ 0.5, δ 0.02 is expected "unresolved"; at θ 0.7, δ 0.05 is.

**Falsifiable arm.** Registered: a read of intent rarely fires on honest sources. At θ 0.7 the gate closes on fewer than 5% of honest sources (no unfittable acts) by 64 claims. Its exact value (8.99%) fails, so it is reported as fixed by the definitions.

**Also report:**
- log score on the true hypothesis for T, K, D and O, by source type;
- learning lost or gained from good sources: D minus K on H and HU.

**Known-answer checks** (run as separate fixtures):
- θ = 1 and ρ = 0: D equals K (1e-12).
- K with a library holding only "honest" equals T on honest sources.
- An unfittable act leaves K's posterior unchanged.
- O's log odds move by ±ln(7/3) per item.
- **The gate:** a hand-computed path on which the harmful-intent posterior first reaches 0.5 at record 14, the earliest possible, with D's frozen posterior equal to K's after that record.
- **The simulation against the exact programme:** primary and arm, within sampling error.

**Texture grid:**
- ρ 0.01 and 0.2;
- the con's switch at record 8 and at 32;
- Mx at strength 0.75;
- 128 claims;
- a "pull away" closure that reverts to the prior instead of freezing;
- **out-of-case maps** (both fail the case by construction):
  - 16 claims: the con captures K in 7.4% of sources;
  - a library that also holds agendas at strength 1: the over-pusher captures K in 9.7%.

**Shed order** (studies): the θ 0.5 cell; then the honest-source half of the δ grid.

**Cost.** The pre-check measured 11 µs per maker-record (five source types, three readers). The registered design is 5 types × 64 worlds × 32 sources × 64 claims × 3 θ cells, about 2 million maker-records, about 65 CPU-seconds over three passes. The δ grid belongs to primary (ii), so to studies: about 0.03 CPU-hours over three passes (the pre-check ran all 48 of its cells in 38 CPU-seconds). Studies come to 0.07 CPU-hours with reserve, and texture to about 0.2.

**Route.**
- *ToMpathy:* should the engine stop modelling a source after an unfittable act or a read of harmful intent? This sets whether a "closed" state belongs in the source panel, and when. Item 4 is the hazard: a steady manipulator the reader's library can't represent is never read as harmful.
- *Sounding Line:* the real-record analogue is the theory-days card TD7, which asks whether a writer stops weighing the AI's offers after an examined rejection.

### G4. IDENTITY_AT_ANY_PRICE. Is "nothing trades for it" a type, or the limit of a strong terminal weight? (studies 1.25, texture 1.4)

**Why.** C5. The curator in the theory thread:

> I was imagining identity as being something that shows up when a heuristic becomes [a] terminal utility value. For example, I think I referenced honor as something you can use to help have a good place in society, or you can decide that honor is all that matters and set that up as your terminal metric for success.

*Theory thread, 3 November 2026, 07:32 PST.*

> An honorable person can be either someone who makes a whole bunch of different honorable decisions or someone who simply optimizes for honor as an outcome. God, is there some way to tell the difference between the two of those things? Yeah, the first one, the person would be willing to sacrifice it for something else, I guess. They'd be willing to nest it as part of a larger function.

*Theory thread, 3 November 2026, 16:35 PST.*

> By definition, you cannot trade a willingness to be worse at this thing for anything else, or it's not an identity.

*Theory thread, 4 November 2026, 17:38 PST.*

His definitions fit two formal types: **terminal weight at the vertex (A1)**, a utility that puts all its weight on one value, and **lexicographic lock-in (C)**, a rule that serves one value first whenever it can. Both are candidate readings of his identity. This card names them by their formal types and never calls either one "identity" in a verdict. E1 capped its trades; this card uncaps them and aims each offer at the value the agent has been choosing.

**World** (E1's, re-implemented).
- **Values and menus:** three values. An ordinary record has 4 options with outcomes on {0, 0.5, 1}³. Each option is a one-value specialist (1 on a random value, 0 elsewhere) with probability 0.5. There is no shift arm.
- **Trade offers.** With probability ρ per record (registered 1/16), the menu is two options instead:
  - KEEP serves the target value at 1 and pays 0 elsewhere;
  - TRADE pays 0 on the target and m on each other value, with m uniform on the uncapped ladder {1, 2, 4, 8}.

  The target is the value the agent's chosen options served at 1 most often over its last 16 records, with ties broken at random.
- **Choice:** softmax with precision 3 on w·outcome, plus a lapse of 0.02 spread uniformly over the menu.

**Types** (one maker per world):
- **A75, A9 and A1, compensatory:** fixed weights with top weight 0.75 (the grid point (0.75, 0.25, 0) and its permutations), 0.9 ((0.9, 0.05, 0.05)) or 1.0 (the vertex), on a random value.
- **C, lexicographic lock-in:** compensatory with weights w0 before lock time τ, with w0 on the grid and w0ₖ = 0.5, the weak maximum. τ is drawn from {0, 32, 64, 128}. From τ on, C is lexicographic on k: whenever the menu has an option serving k at 1, it takes one of those, choosing among them by w0. Otherwise it chooses by w0.

**Reader** (exact, 108 hypotheses):
- A over the 15-point simplex grid (step 1/4) plus the three 0.9 points.
- C over k × w0 (grid points with w0ₖ the weak maximum) × τ ∈ {0, 32, 64, 128, 256}.
- Equal class priors, uniform within each class. Weighting every hypothesis equally would give the C class five times A's prior mass.
- Three likelihoods: full; offers only (trade-offer records); and ordinary only.
- **Two class contrasts:**
  - A against C. For the A9 primary.
  - "Locked by record t" against "not locked". Locked means C with τ ≤ t; not locked means A, or C with τ > t. For on-grid makers it does not escape the plateau in item 5 (item 8).

**Case and check** (check-only seeds). The case: after lock-in, the locked type chooses unlike every compensatory type on ordinary menus, and offers at m = 8 carry the A9-against-C contrast (A9 accepts 0.427, locked C 0.01).
- **Check 1:** the share of post-lock ordinary records on which C's choice distribution differs by more than 0.2 (total variation) from the best-matching A hypothesis is at least 0.25 on average, and at least 0.25 in at least 75% of worlds. Pre-checks 0.471 and 0.479, with every world at or above 0.25 (minimum 0.351).
- **Check 2:** the share of offers with m above 3. It is 0.5, true by rule. The primary's contrast lives only at m = 8, a quarter of offers, also by rule.
- **Revision ranges:** no revision.

**By construction** (report for scale only):
1. On offers aimed at the maker's top value, the thresholds are m > w/(1 − w). A75 is bought at m 4 and 8. A9 never prefers the trade, since its price of 9 sits just past the ladder. A1 and locked C are never bought. The target rule aims some offers elsewhere: in the pre-check, 11.1% of A75's offers, 5.3% of C's, 2.4 to 4.3% of A9's and 1.7% of A1's. Every type takes those, because they pay m on its top value. Report the off-target share.
2. On offers, A1 and locked C differ only through choice noise. A1 accepts with probability 0.98/(1 + e³) + 0.01 = 0.057 at every m; locked C accepts with probability 0.01. That is about 0.05 nats per offer.
3. A9's evidence on offers comes mostly from m 8, about two-thirds of it: about 1.29 nats per on-target offer at m 8, 0.39 at m 4, 0.19 at m 2 and 0.12 at m 1. Against that sits a within-class prior handicap of about 1.6 nats: 24 locked-on-k hypotheses among C's 90, against 1 for A9 among A's 18.
4. **The A9 verdict follows.** Items 1 to 3 set the evidence each offer carries, so the offers needed to separate A9 from C hardly depend on ρ. The pre-checks' sustained crossings took about 12 offers; a drift estimate from item 3 (1 nat plus the 1.6-nat handicap, at about 0.5 nats per on-target offer) gives about 5 offers to a first crossing. Records needed are offers divided by ρ, so the expected value of the records bar is fixed: met when about 12/ρ ≤ 256. ρ only rescales; whether the interval clears the bar depends on the sample size.
5. With τ unknown, the τ-256 C hypotheses copy every on-grid compensatory type until record 256. So an on-grid terminal maker's A-against-C margin plateaus near ln 5 = 1.61 nats until then: a soft ceiling, which nearby hypotheses can lift to about 4. A9 is off the C grid and escapes it.
6. Before τ, a C maker is compensatory, so its records before τ carry no evidence of lock-in.
7. After lock-in, the reader names C within about 6 to 7 records. That is E1's sharpness gap: a locked maker takes a k-at-1 option with probability at least 0.985, while A at precision 3 gives it at most about 0.86. Check 1 requires the gap. On the locked-by-t contrast the delay was 7 to 8 records for lock times 32 to 128, and 15 for lock time 0, so the detection delay is set by the gap too.
8. **On-grid lock-state figures are set by the lock-time prior.** At each lock-time grid point, the C hypothesis with an on-grid maker's own weights joins "locked", carrying a fifth of the maker's prior mass, so the locked-by-t margin drops to about ln 5 and crossings bunch just after the grid points. In the pre-check, 31% of A1's crossings fell between records 257 and 288, and offers saved no records at all for 72% of A1 and 62% of A75 makers. A lock-time grid every 8 records removed the bunching and slowed the read (A1's median 405 records). Off-grid A9 is far less affected.

**Primary.** On trade offers alone, the *offers* (not records) the reader needs to hold a sustained 1-nat A-against-C margin for A9 makers, with the pre-check expectation of about 12 offers beside it.
- Registered cell: ρ 1/16, specialist share 0.5, τ unknown, 64 worlds; censored median.
- Records needed are reported, with the bar "at most 256 records". Its expected value is fixed by item 4; the interval verdict under 0.4's rule depends on the sample size.
- The ρ 1/4 and 1/32 cells are reported beside it, as rescalings.

**The open quantity**, reported with the primary:
- **Records saved by offers, for A9 makers:** on the locked-by-t contrast, the censored median of ordinary-only records needed minus that of full records needed, with a block bootstrap, and the share of makers for whom offers saved nothing. Pre-check (one implementation): 72.5 minus 43, or 29.5 records, with 58% of makers saving nothing.

**Also reported:**
- for every type, the posterior on the true lock state, with its log score, at records 48, 96, 192 and 384, away from the lock-time grid points, from full, offers-only and ordinary-only likelihoods. On-grid makers' figures are labelled "fixed by the definitions" (item 8);
- for C makers, the locked-by-t detection delay, by τ, labelled the same way (item 7).

**Falsifiable arm.** None. The A1-against-C contrast on offers is fixed by choice noise (item 2). Its offers-only margin is reported for scale.

**Known-answer checks:**
- Acceptance at each m matches the closed form, for example:
  - A9 at m 8: 0.98·e^2.4/(e^2.4 + e^2.7) + 0.01 = 0.427;
  - A75 at m 4: 0.676;
  - A1: 0.057 at every m;
  - locked C: 0.01.
- A C hypothesis whose τ lies beyond the horizon gives choice probabilities identical to the A hypothesis with the same weights (1e-12).
- With no records, every class margin is 0.
- The smallest choice probability is 0.02/4 = 0.005, so no record scores minus infinity.

**Texture grid** (run in this order):
- specialist share 0.75, which passes check 1 narrowly (79% of worlds at the floor);
- the ladder {1, 2, 4, 8, 16}, which buys A9 at 16 (check 2's share becomes 0.6, by rule);
- 1,024 records;
- precision 6, which narrowly fails check 1 in the pre-check (73% of worlds at the floor, against 75%), so it will probably run as an out-of-case map.

**Shed order** (studies): the ρ 1/32 cell; then the τ-known cell.

**Cost.** The pre-checks ran at 0.15 ms per maker-record per pass. This card is priced at V26's measured E1 rate of 2.2 ms.
- Registered: ρ 1/16 with all four types; ρ 1/4, ρ 1/32 and the τ-known cell with C, A1 and A9 only; 64 worlds × 512 records per type and cell. That is 425,984 maker-records, 0.78 CPU-hours over three passes; 1.17 with reserve.
- Texture: 4 cells with C, A1 and A9, one at 1,024 records, 491,520 maker-records, 0.90 CPU-hours; 1.35 with reserve.

**Route.**
- *ToMpathy:* whether the values panel needs a "lexicographic lock-in" state distinct from "very strong weight". In these worlds the evidence for lock-in comes from ordinary choices, by item 7, not from refusals.
- *Sounding Line:* a real-record analogue is a writer's refusals of suggestions that trade away a signature commitment. On refusals alone, telling a strong preference from lock-in takes about a dozen well-priced offers, and a preference at the vertex may never separate.

**Pre-check expectations** (64 worlds per type; first, then second implementation):

| Cell | A9 offers-only records | A9 offers needed | A1 offers-only resolved by 512 | C read after lock |
|---|---|---|---|---|
| ρ 1/16 | 222; 199.5 and 198.5 (bootstrap [141, 258] and [157, 249]) | 12 | 39%; 33% | 7; 6 |
| ρ 1/4 | 52.5; 63.5 | 16.5 | 70%; 72% | |
| ρ 1/32 | 349; 376 (14 unresolved) | 8.5, censored | 11%; 22% | |
| τ known | 109; 136 | 8 | 30%; 28% | 5 |

At ρ 1/16 the bar of 256 records sits inside one of the second implementation's two intervals; the other passes. So expect "unresolved" under 0.4's rule, or a narrow pass.

### G5. SECURITY_SET_POINT. Is security defended from below and spent from above, or simply liked? (studies 1.0, texture 0.75)

**Why.** C4 lists security among the set points. The curator in the theory thread:

> Margin from dissolution isn't a set point. We optimally want to be as far away from dissolution as possible. So I guess our set point is as far away as we can get. And if something moves us away suddenly, then we can try to use anger to counteract that quickly.

*Theory thread, 3 November 2026, 18:10 PST.*

> Speaking to the set points, margin isn't a great name for that set point. [...] So what we're talking about is things like security. We're talking about safety and security, I think.

*Theory thread, 4 November 2026, 17:38 PST.*

So on 3 November security was something pushed as far as it can go, which is closer to a smooth preference for more. By 4 November he called it a set point. The thread's 4 November summary describes it as "defended from below, spent from above", which is the thread's phrase, not his. This card asks whether a reader who sees the security stock can tell a defended set point from a smooth preference for more security, from the maker's choices. No V26 card contrasts set-point valuation with level valuation, or tests the asymmetry.

**World** (W13's stock dynamics, `v25/persistence.py`, re-implemented).
- **Stock:** one observed security stock S in [0, 1], updated as S′ = clip(0.95 S + Δs, 0, 1), where Δs is the chosen option's stock change. S₀ is uniform on [0.1, 0.9].
- **Menus.** Every record offers three options:
  - stay: Δs 0, v 0;
  - spend: Δs −d, v +p·d;
  - save: Δs +d, v −p·d.

  Spend and save each draw their own d, uniform on [0.05, 0.25], and their own price p, log-uniform on [0.25, 16]. So every record trades the stock against another value, in both directions.
- **Choice:** softmax with precision 3 on v + V(S′), plus a lapse of 0.02 spread over the three options.

**Types** (one maker per world):
- **SP, set point:** V = A·min(S′ − s*, 0) + B·max(S′ − s*, 0), with A = 10 (defended from below), B = 0.5 (spent from above), and s* from {0.35, 0.45, 0.55, 0.65}.
- **LV, smooth level:** V = 2(1 − e^(−λ(S′ − c)))/λ. The marginal value is 2 at c, rising below c and falling above it. c comes from the same four points, and λ from {4, 8, 16}.
- **HO, W13's homeostatic type:** options are valued by how much they refill, up to the ceiling: V = h·S′, with h from {0.5, 1, 2, 4, 8}.

**Reader** (exact, 148 hypotheses):
- SP over s* ∈ {0.25, 0.30, …, 0.75} × A ∈ {5, 10, 20} × B ∈ {0.25, 0.5, 1}: 99 hypotheses.
- LV over c on the same grid × λ ∈ {2, 4, 8, 16}: 44.
- HO over the five values of h: 5.
- Equal class priors. The stock is observed, so the likelihood needs no hidden-state filter.

**Case and check** (check-only seeds). The case: set-point agents spend real time on both sides of their set point, with trades offered on both sides, and their choices carry enough evidence against the best-matching smooth preference. Two checks:
1. **Time split:** at least 75% of SP agents spend at least 25% of their 256 records on each side of s*. Every record offers both a spend and a save, by design. Pre-checks: every agent passes; below s*, means 0.419 and 0.421 (minimum 0.266); above, 0.581 and 0.579.
2. **Information form:** the summed expected log-likelihood ratio of the true SP hypothesis over its best-matching LV hypothesis reaches 3 nats by 256 records in at least 25% of SP agents. Pre-check: median 19.2 nats at precision 3 (10th percentile 13.8), with every agent above 3.

Also report, as a limitation, E3's form: the share of records on which SP's choice distribution differs from the best-matching LV hypothesis by more than 0.2. The pre-checks found 0.086 and 0.088 (minimum 0.031). The kink shows through many small differences, not a few large ones.

**Revision ranges:** no revision.

**By construction** (report for scale only):
1. HO's value is linear, so its choices depend on prices alone except at the floor and the ceiling. It is read in about 10 records.
2. In the registered cell every SP maker's s* and slopes lie in the reader's family, so placing s* is identification within the family.
3. The stock is observed, so no evidence is lost to state uncertainty.
4. Far below s*, LV's marginal value explodes while SP's stays at A, and both save. The evidence therefore comes from the band near s* and from spending above it.
5. A set-point agent with strong defence and a high value for its surplus need not spend 25% of its time on each side, which is why the off-grid cell filters on the case.

**Primary.** Records the reader needs to hold a sustained 1-nat margin for the true class: SP over the best other class, for SP makers; LV over the best other class, for LV makers.
- Registered cell: precision 3, 256 records, 64 worlds per type; censored medians.
- Both medians at most 64 records counts as "a set point is told from a smooth preference for more security, in practice", judged by 0.4's rule.

**Falsifiable arm** (can it place s*?).
- The off-grid cell: SP makers draw s* uniformly on [0.3, 0.7], off the reader's grid, and slopes from {5, 20} × {0.25, 1}, with 960 SP worlds. About 300 are expected to pass the time split, which narrows the arm's interval to about ±0.05.
- Among the makers that pass the time split, the reader puts at least 0.9 of SP's posterior (marginal over slopes) within ±0.05 of s* by 256 records, in at least 75% of them.
- Report placement among all off-grid makers beside it, since the filter selects the harder ones.

**Also report:**
- s* placement in the registered cell at 64 and 128 records;
- the time split, by type;
- the off-grid cell's SP class margin;
- the best-matching LV (c, λ) for each SP maker;
- **the engine's hazard:** the off-grid result again, with the reader's slopes fixed at 10 and 0.5. Compute it from the registered reader's per-hypothesis log-likelihoods, restricted to those slopes and re-priored with equal class priors; it needs no extra pass.

**Known-answer checks:**
- With A = B, SP is linear, and its choices equal HO's with h = A (1e-12).
- With A ≠ B, a fixture confirms the marginal value is A below s* and B above it. This is the check that catches swapped slopes, which the A = B fixture can't see.
- LV with λ → 0 approaches HO with h = 2 (at λ = 1e-6, within 1e-6).
- With no records, every class margin is 0.
- The smallest choice probability is 0.02/3, so no record scores minus infinity.

**Texture grid:**
- precision 1.5 and 1;
- 512 records;
- prices log-uniform on [0.5, 8]. 6.3% of agents fail the time split, so the cell passes the 75% floor.

**Shed order** (studies): the off-grid cell's SP worlds beyond 192.

**Cost.** The pre-checks ran at 0.09 to 0.11 ms per maker-record per pass. This card is priced at W13's V26 rate of at most 2.3 ms.
- Registered: 49,152 maker-records for the registered cell, plus 278,528 for the off-grid cell (960 SP, 64 LV and 64 HO worlds), 0.63 CPU-hours over three passes; 0.94 with reserve.
- Texture: about 245,800 maker-records, 0.47 CPU-hours; 0.71 with reserve.

**Route.**
- *ToMpathy:* whether the needs panel should show a set point with an asymmetric band (defend below, spend above) or a level, how many records the engine must watch before it claims one, and that the defence and surplus slopes must be inferred, not assumed.
- *Sounding Line:* a real-record analogue is a writer's reserve of something she guards, such as a buffer of finished pages or of time, spent freely above a level and defended hard below it. That needs records of the reserve, not only of choices.

**Pre-check expectations** (64 worlds per type, 256 records; first implementation, then second):
- **Registered primary:** SP 58 and 47 to 50; LV 60 and 52.5 to 82.5 over four replicates of 64 worlds; HO 11 and 13. The LV median straddles the bar, so expect "unresolved" under 0.4's rule.
- **s* placement in the registered cell:** 97% and 95% by 64 records, and 100% by 128.
- **Precision 1.5, registered reader:** SP 73.5, LV 127. At precision 1: SP 107.5, LV 140.
- **Off-grid arm:** 65% of 20 case-passing makers placed by 256 records in the first implementation. In the second, 73.3% of 60 and 74.2% of 62 at the old size, and 79.0% of 310 and 79.8% of 307 in two runs at the registered 960 worlds, with block-bootstrap intervals [0.755, 0.823] and [0.752, 0.839]: both met, narrowly. Pooled over the second implementation's four runs, 78.5% [75.5%, 81.4%]. Among all off-grid makers, 84 to 88% were placed. Expect "unresolved" or a narrow "met".
- **The engine's hazard:** with the reader's slopes fixed, 63 of 64 and 96 of 96 off-grid set-point makers were never read as set points by 256 records. In the second implementation, the fixed-slope reader put most of its mass on the smooth preference for 92 of the 96.
- **What the world does:** it keeps set-point agents above their set point 58% of the time. They spend down to it, and decay pulls them under.

## 3. Triggers to return to the owner

1. A card's studies charge would pass 1.5 times its studies ceiling after shedding. Texture lanes stop at their ceilings and don't trigger.
2. A registered case check fails for every grid in the card's declared revision ranges, or fails at all in a card that declares none (G3, G4, G5). Record the card as "design cannot produce the case", with the check's output; this is not a null. Return to the owner only if two or more cards end this way.
3. The day admission module, the window or the deadline machinery fails.

## 4. Pre-mortem: how this day goes wrong

- **Cards too easy, done in hours.** This was V26's first failure, and it recurred on Friday. The projected work here is about 15 CPU-hours with reserve, texture included, which two workers finish in about eight hours. *Guard:* the texture lanes are real variants and declared out-of-case maps. Never inflate sample counts to fill the clock (4 November commission).
- **A verdict fixed by the design gets reported as a finding.** This applies to G2's registered verdict; G3's primary and arm (computed exactly in advance) and its θ 0.9 contamination cell (fixed by the record-52 floor); G4's A9 offers needed and its records bar's expected value (fixed by the acceptance curve), its A1 contrast (fixed by choice noise), and its on-grid lock-state figures and detection delay (fixed by the lock-time prior and the sharpness gap). *Guard:* each is listed under its card's "By construction", and the reports label them "fixed by the definitions". The open quantities are G1's records needed and set-point recovery, G2's records-needed figures and ratio, G3's contamination threshold at θ 0.5 and 0.7, A9's records saved by offers in G4, and G5's primary and arm.
- **A bar read from a point estimate.** V26's H1 arm was "met on the mean only". *Guard:* 0.4's one rule, applied to every bar.
- **G1's low set-point recovery read as "set points are unreadable".** The hidden-stock readers placed set points at 0.42 to 0.55 at 256 records, across sources, laws and both implementations, apart from the raw slope's firings-only figure, which sits near the prior (0.334 and 0.330; item 1). Chance is 0.33, while seeing the stocks gave 1.00. That says what not seeing the stocks costs in this world, with two needs and menus on offer 35% of the time. *Guard:* the report says so, and quotes the ceiling beside every hidden-stock number.
- **G1's laws are miscoded.** A signed deficit turns (c) into (b) unnoticed. *Guard:* the law fixture.
- **G2's frame-blind reader knows the frame distribution.** Real readers don't. *Guard:* the report names this as the reader's advantage.
- **G3's gate is one design among several, and its threshold drives the result** (−0.48 to 0.00 across the declared cells, exactly). *Guard:* report every θ cell, never the best one alone.
- **G4's on-grid figures are set by the prior.** The lock-time grid lets locked hypotheses copy on-grid makers, on the A-against-C contrast and the locked-by-t contrast alike (item 8). *Guard:* the only open G4 quantity is A9's, and on-grid figures are labelled "fixed by the definitions".
- **G5 can't see "spent from above" if makers rarely sit above the set point,** and its kink shows only through many small differences. *Guard:* the time split and the information form, with E3's form reported beside every G5 result.
- **Routes read as licences.** *Guard:* every route says what a real-record test would need; none licenses an engine display by itself.

## 5. Reporting

- Each card's report carries:
  - "miniature, architecture untested";
  - its assumed choice model;
  - its by-construction list, with every verdict fixed by the definitions labelled so;
  - its route (amendment G);
  - the pre-check expectations beside the results, so surprises are visible.
- Out-of-case maps are reported separately from the card's variants.
- **Daily line:** cards admitted, running and verified; charges by pool; anything shed; any healing.
- **At the close:** publish with replay and independent review, as V26's snapshot was. Delivery of these results to Sounding Line or ToMpathy waits for the owner's uptake. Then restore the hold, as section 0.1 says.

## Sources

- [Keramati & Gutkin 2014](https://doi.org/10.7554/eLife.04811), homeostatic reinforcement learning (G1 texture).
- [Tetlock 2003](https://doi.org/10.1016/S1364-6613(03)00135-9), taboo, tragic and routine trade-offs; [Hanselmann & Tanner 2008](https://doi.org/10.1017/S1930297500000164), taboo trade-offs felt easier and tragic ones hardest (G2, G4).
- [Miller, Gordon & Buddie 1999](https://doi.org/10.1207/s15327957pspr0303_8), explaining harmdoing and condoning (G3's contamination question).
