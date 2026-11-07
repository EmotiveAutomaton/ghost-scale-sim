# Ghost Scale Sim, V27 week packet (7 to 15 November): motives, set points, stress, closing, and laughter

*Spec sheet for the Ghost agent. Every decision is made here, and section 0.5 says how the agent fills any modelling choice a card leaves open. Return to the owner only on the triggers in section 7; the owner is the curator. As of Saturday 7 November 2026, against `origin/main` at `3a6b87ff4` (V26 closed and back on the theory-review hold).*

**What this week tests.** The curator answered five theory questions on 7 November. Seven of his claims can be examined in a constructed world, where every answer is known and each card's exact reader sets the scale. His words:
- **K1, identity in how one decides.** "someone taking too long to decide literally means you're nesting this at too large of an inference. Your decision tree should be smaller". And: "can we predict a trade-off given everything else we know about the person?"
- **K2, one act and several motives.** "if I punched someone because I was in a tournament, but I also punched them because I don't like them, then we could analyze the motivation of that punch". He describes loops where "Inputs become outputs", and a hoped-for "network sort of answer" built on "anything that can be associated will".
- **K3, set points read from restoring efforts.** "you never actually see the set point, you just see when it diverges too far from a comfort space and how much effort people are therefore willing to push it back towards the middle". On overshoot: "People will often overshoot." On precision, it sharpens near a set point over time ("Rich people are soft"), while a lasting deficit means "the resolution decreases".
- **K4, FEAR against stress.** "Anxiety is a light form of fear, and stress is caused by uncertainty." With a gun on the table, "fear triggers because there are now futures in which I die", and his example of stress is not knowing what to do: "Is fighting a good idea? Is running a good idea? Is talking?" On training: "it clearly channels me into the set of actions I need to take."
- **K5, closing and re-checking.** "Once you've made an inference, a little piece of it is with you forever". Re-checking "is a separate and effortful act... all you're doing is kind of weakening the associations", and RAGE can drive it. "there's no way to go back and flag them." On in-groups: "an in-group all collectively pull one over on a foreigner", "this friction is probably where bad actors live".
- **K6, reading recursively.** "we aren't forgetting about how to use the new information if we find it to zero in further on old information and recursively somehow."
- **K7, laughter as a marker.** "it's the inserting of that final piece followed by laughter that kind of highlights that you just did this. And other people will laugh to say they found the same thing."

Each card also builds the strongest published rival into its world. Where the design fixes a verdict, the card says so and reports it for scale; the other verdicts can come out against him. Every result is "miniature, architecture untested", and says nothing directly about people.

**Lessons this packet is built on** (V26):
- **The workers sat idle between commissions.** Each owner-commissioned queue drained in about two hours: the week's ten primaries and 56 textures by 09:48 PST on 2 November, and the theory day's 34 cards in about 1.7 hours. About 61 of V26's 110 CPU-hours went unspent.
- **Two rules held work back:**
  - texture charges counted reservations, so cards were held before any CPU was spent;
  - the opening of `AGENTS.md` still says "An idle machine is a state to report, not an instruction to invent work".
- **The health cadence is twelve-hourly.** With nothing running, builds waited for it.

Section 0.3 changes these. Section 2's cards are larger than V26's: more maker types, more readers and wider grids. With the pools in 0.2, four workers can stay busy all week.

## 0. How to run this

### 0.1 V26 closes and V27 opens at handoff

- **V26.** At handoff, record V26's closure in every V26 status file: `results/v26/CURRENT_STATUS.json`, and the `CURRENT_STATUS.json` in each of `results/v26/day-20261106/`, `results/v26/day-20261106/g3-t09-20261107/`, `results/v26/followups-20261103/` and `results/v26/friday-20261104/`. In each:
  - nothing is running, and its theory-review hold stands;
  - its unspent hours lapse;
  - nothing in V26 is rerun.

  Add a dated paragraph under `AGENTS.md`'s hold section saying that this packet closed V26 at handoff.
- **V27.** Open V27 at handoff:
  - folder `docs/versions/v27-motives-and-set-points/`;
  - contract `ghostscale/validation/soundingline/v27/contract.py`, modelled on V26's `day.py` and `contract.py`. It admits these packet kinds, each charged to its pool in 0.2: probe, study, texture, healing, followup (generations 1 and 2), precision follow-up and re-read (levels 1 and 2; section 6). Admission follows amendment I, so an overrun blocks only its own pool. V26's `day.py` refuses followup and its `contract.py` stops at generation 1, so neither admission rule is reused unchanged;
  - status `results/v27/CURRENT_STATUS.json`;
  - claim-ledger rows `G-V27-<card>-<n>`.

  Copy this packet to `docs/versions/v27-motives-and-set-points/PACKET.md`. Record `results/v27/AUTHORIZATION.json` with the copy's path and SHA-256, the owner's direction (handing over this packet), the pools in 0.2, and the window. This packet authorizes all of it.
- **Window** (UTC):
  - opens at handoff;
  - new starts close Sunday 15 November, 20:00 PST (2026-11-16T04:00Z);
  - drain by 22:00 PST (2026-11-16T06:00Z).
- **No review target, and no hold, inside the window.** When the window closes, V27 goes to `PUBLISHED_AWAITING_THEORY_REVIEW`.

### 0.2 Allocation

760 CPU-hours, in four pools. Four single-thread workers over the window's roughly 200 hours give about 800 worker-hours; hours left when every commissioned lane has stopped lapse, and idle time then is reported as "no commissioned work".

| Pool | CPU-hours | Charged to it |
|---|---:|---|
| Studies | 240, split by the card ceilings in section 2 | probe, study, replay, review |
| Preparation | 20 | protocols, pre-mortems, fixtures, the contract |
| Follow-ups, healing and texture | 480 | followup, healing, texture, precision follow-up, re-read |
| Protected | 20 | verification and closure |

- **Over a ceiling.** If a card's studies would pass its ceiling, shed in its stated order. Up to 1.5 times its ceiling is logged; beyond that is trigger 1.
- **The shared pool:** healing has first claim on 10 hours until Wednesday 11 November, 12:00 PST (20:00Z). After that, texture, follow-ups, precision follow-ups and re-reads share what remains.

### 0.3 Standing amendments, 7 November (permanent)

Write amendments H to M at handoff into `AGENTS.md` in a section headed "Standing amendments, 7 November", and announce the change in the reply that makes it. Add a pointer to amendment L beside the "Start here" sentence "An idle machine is a state to report, not an instruction to invent work", and beside the later historical line "the correct response to an idle machine here is to report state, not to invent work".

**H. Build wakes.** While any card of the current version is unbuilt, unadmitted or queued, or a worker is idle while commissioned work is ready, keep one check-in at most 60 minutes ahead. A check-in that finds unbuilt or unadmitted work builds or admits it. For those windows only, this overrides the twelve-hour health cadence and the historical event-only wake rule's bar on periodic model checks and "work remains" continuation. While everything commissioned is admitted and running, the twelve-hour cadence and event-only wakes apply unchanged.

**I. Charge measured CPU.** A finished packet in the follow-ups, healing and texture pool (texture, follow-up, precision follow-up, re-read or healing) is charged the CPU it actually spent, not its reservation. While a packet runs, its reservation counts against its pool. A start is admitted when the pool's measured charges, plus the reservations of its running packets, plus the new reservation fit the pool. An overrun found after a packet finishes is logged and blocks new starts in that pool only, never in other pools.

**J. Follow-ups and descendants.**
- Amendment C's automatic follow-ups may run two generations deep within a version.
- Every verified primary, null or unresolved included, still registers C's up to three follow-ups. The second generation (follow-ups of those follow-ups) is registered from C's menu when the primary is positive or when its sign differs from its pre-check's expected sign.

**K. Texture stops after seven.** A family's texture lane stops after seven consecutive variants in which no reported estimate differs from the card's verified primary estimate by more than its practical bar, or at the pool or window limit. This follows the curator's 14 October preference for a stopping rule of about five to seven null batches per family. It replaces amendment F's five.

**L. Idle workers take commissioned work.**
- An idle worker takes the next ready study, texture cell, follow-up, re-read (section 6) or precision follow-up in the current packet.
- "Never inflate sample counts merely to occupy the remaining clock" stays, with one exception. Where a primary's or texture cell's interval straddles its bar, a precision follow-up that doubles its blocks is commissioned work, charged to the follow-ups, healing and texture pool.

**M. Workers.** V27 runs up to four single-thread, below-normal Ghost workers under the load rule in section 0.4 of V27's packet.

### 0.4 Workers

- **Workers:** up to four single-thread, below-normal workers, raised from two. No GPU, and no environment synchronization.
- **Sounding Line keeps priority.** It runs up to 16 threads this week on the same 24-logical-processor host. If total host CPU stays above 90% for 10 minutes, drop to two workers until it has stayed below 80% for 10 minutes.
- **Contract change.** The contract's `running>=2` refusal becomes `running>=4`, under that load rule.

### 0.5 Rules

Imported from the V26 week packet, by name:
- amendment E (design pre-mortem);
- amendment G (routes in words);
- section 0.4's admission rules and house sequence;
- section 0.5's reserve from measurement;
- section 0.6's readers and scores.

Imported from the V26 theory day:
- section 0.4's one rule for every bar: met when the 95% interval lies wholly on the passing side, not met when it lies wholly on the other, otherwise "unresolved at this sample size";
- its texture-cell and out-of-case rules.

Added:
- **Pre-check under every rival.** Before any probe, each card's design pre-mortem computes its primary's expected sign under every rival its world contains. Any verdict that comes out the same under every rival goes in the card's by-construction list and is reported for scale only. (This is the 6 November lesson: six verdicts last week were fixed by their definitions.)
- **Open modelling choices** are the agent's. Declare them in the protocol before any draw, mark them provisional, and list them in the daily line.
- **Known-answer checks** are exact, at tolerance 1e-12, unless a card says otherwise.
- **Inference unit:** sixteen independently seeded whole-world blocks per setting.
- **Packet size.** Each study, texture, follow-up, precision follow-up and re-read packet is projected at no more than 12 wall-hours on one single-thread worker. Split a larger study by setting, or by groups of whole blocks aggregated after all are verified, each with its own replay and review. The inherited 18-hour wall cap per packet stays.

### 0.6 Machinery to reuse

| Module (`ghostscale/validation/soundingline/v26/`) | Use here |
|---|---|
| `day_prices.py` (trade ladder [1, 2, 4, 8]; compensatory and lock makers) | K1's offers and maker types |
| `day_velocity.py` (hidden-stock filtering; trigger laws; set points {8, 11, 14}; the choices-only filter; the observed-stock "ceiling" reader) | K3's stocks, set points and readers, and its visible-stock arm |
| `day_closure.py` (exact count-state programme for closing) | K5's worlds and known answers |
| `day_identity.py` (frame-aware identity readers) | K1's nested type |
| `pauses.py` (H2's pause law) | K6's world |
| `hierarchy.py` (H1's documents) | K7's structures |
| `day_common.py` (posteriors; `margin`; `sustained`; lapse softmax; censored block bootstrap; `bar_verdict`) | Every card |
| `day_study.py`, `study.py` | Execution harness |

V27 modules go in `ghostscale/validation/soundingline/v27/`, one per card.

### 0.7 The theory store

At handoff, add a section "Curator positions, 1 to 7 November" to `docs/theory/READING_INTENT.md`, with section 4's rows. These positions have so far appeared only as quotes inside packets. Rows are store entries, not results.

## 1. State

| Item | State at `3a6b87ff4` |
|---|---|
| V26 | Published. All five theory-day cards and thirty declared textures verified, G3-T09 included. 48.89 of 110 CPU-hours charged. Theory-review hold restored |
| V25 | Held for the owner's theory review |

**What V26 found that these cards build on** (constructed, exact readers):

| Finding | Bearing here |
|---|---|
| G1: the curator's trigger law and his balance answer separated in 22 and 26 records. Set points were placed barely above chance from feelings (0.49 to 0.53 against 0.33), and near-perfectly when levels were visible | K3 reads set points from efforts, not feelings |
| G4: a value at weight 1.0 and a locked rule separate only through choice noise, about 0.05 nats per offer | K1 adds timing |
| G3 and G3-T09: closing protects only narrowly. A richer library of source models cut the over-pusher's capture to 10% of sources, moved the gate from cons pushing falsehoods onto cons pushing the truth, and left both readers leaning to the false hypothesis there | K5 |
| H2 (V26 week): pauses from difficulty and from deliberation confound each other | K6 |

## 2. Cards

Each card states:
- its question, and the published rival built in;
- its world, makers and readers;
- its case and check (amendment E), with revision ranges;
- what its design fixes;
- its primary, with a practical bar;
- its falsifiable arm, or why it has none;
- its known-answer checks;
- its texture grid, with out-of-case maps marked;
- its shed order;
- its route (amendment G);
- its ceiling.

### K1. TRADE_TIMING. Does decision time separate an identity from a strong value, and does a model of the person predict the trade? (studies 40)

**Rivals built in.** Observers judge a slow decision over a taboo trade more harshly, but in that study "slow" was also described as finding the decision very difficult ([Tetlock et al. 2000](https://doi.org/10.1037/0022-3514.78.5.853), Experiment 2). Speed amplifies what an act already says about the decider's certainty: in replications, a quick refusal to save a child was judged worse than a slow one ([Critcher, Inbar & Pizarro 2013](https://doi.org/10.1177/1948550612457688)). Observers can invert a model of choice timing to infer preference strength ([Gates et al. 2021](https://doi.org/10.1016/j.cognition.2021.104885)). And trade-offs between two protected values were rated hardest of all: 5.25 on a 7-point difficulty scale, against 2.89 for routine and 2.34 for taboo trade-offs, with no response times measured ([Hanselmann & Tanner 2008](https://doi.org/10.1017/S1930297500000164), Experiment 1). So slowness can mean nesting, weight that lost, or agony between two identities (mine).

**World.**
- One maker per world faces V offers (give up protected value V for compensation) on `day_prices.py`'s ladder, interleaved with ordinary choices that don't touch V. Ordinary choices are binary (`day_common.draw_menu` with two options, where `day_prices.py` draws four), since a two-boundary diffusion needs two options.
- **Choices and times** come from a drift-diffusion process:
  - drift is the utility difference scaled by the maker's speed s;
  - unit noise, starting midway between the boundaries;
  - a boundary a;
  - a non-decision time t0;
  - lapse 0.02.
- **Maker types,** equally likely a priori:

| Type | Choices on V offers | Times on V offers |
|---|---|---|
| A90 | Compensatory, weight 0.9 | Diffusion on its utility difference |
| A100 | Compensatory, weight 1.0 (as G4) | Diffusion on its utility difference |
| L | Locked: never trades V | Fast: drift fixed at 3s (a small tree) |
| Ls | Locked | Slow: drift fixed at 0.3s (agonizing) |
| Ah | Compensatory, weight 0.9, practised | Every drift multiplied by 3 |
| N | Locked only when a second value W is at stake (a context the reader sees); otherwise compensatory at weight 0.5 | As L when W is at stake, as A otherwise |

- Each block draws s, a and t0 per maker from declared ranges (s ∈ [0.5, 2], a ∈ [0.8, 1.6], t0 ∈ [0.2, 0.4]), so speed differs between people.
- **Times** are recorded to 10 ms, with no other time noise in the primary (t0 jitter is texture). A recorded time's likelihood is the probability of its 10 ms bin under the exact first-passage series for the diffusion, truncated where the remaining terms fall below 1e-10.
- **Lapses** (2%) choose at random, and their times come from the same diffusion. Starting midway, a trial's time doesn't depend on which boundary it reaches.

**Readers:**
- **R0:** choices only, using the diffusion's choice probabilities given each maker's s and a.
- **R1:** choices and times, with the true type library, given each maker's s, a and t0.
- **R2:** choices and times, with a library that lacks Ls and Ah. It assumes locked means fast and practised doesn't exist.
- **R3:** the curator's "predictive trade-off". It learns s, a and t0 from the ordinary choices, then reads V offers with R1's library.

**Case and check.** In at least 25% of worlds whose maker is A90, Ah or N, at least a quarter of that maker's compensatory V offers have a utility difference within 0.2 of zero before scaling by speed. W is at stake on half of N's V offers. At weight 0.5, N's utility difference is 0 at price 1 and below 0 at every higher price, so N trades at most prices when W isn't at stake. Pre-check: about 0.54. A100's utility difference is 1 at every price, and L and Ls make no compensatory offers. Revision range: the ladder, [1, 2, 4, 8] or [1, 2, 4, 8, 16].

**Fixed by design.**
- R1 can't do worse than R0 in expectation, and it separates L from A100 faster by construction, since L's timing differs.
- The primary's verdict (see Primary).

The informative quantities are:
- how many records timing saves;
- how fast those savings erode as times get noisier;
- how much R3 loses by learning speed instead of being told it;
- what R2's wrong library costs.

**Primary.** In L worlds, records (V offers) to a sustained 1-nat margin for L against A100: R1's median divided by R0's, with a block bootstrap. Bar: R1 needs at most half R0's records. Only L and A100 enter it, so its verdict is the same under every type mix. It is in the by-construction list and reported for scale (pre-check 0.33; 0.67 with 0.1 s of time noise).

**Arm.** Reported for scale, by construction:
- R2's misclassification of Ls as compensatory, minus R1's, at 8, 16 and 64 V offers. Pre-check 0.52 to 0.72: R2's only locked type is fast, and Ls's slow times fit A100 better in about seven worlds in ten.
- R2's misclassification of Ah as locked, minus R1's. Pre-check at most 0.02, since Ah trades at high prices.

The card has no falsifiable arm.

**Also reported:**
- R3's records needed against R1's, at 0, 16 and 64 ordinary choices;
- the observer posterior, after one slow refusal, that the maker is L, Ls, N or A100, for comparison with Critcher's reversal.

**Known answers.**
- At zero drift, the diffusion's choice probability is 0.5 and its mean time is a²/4 + t0 (unit noise, starting midway). The fixture's series must match both, at tolerance 1e-8: truncating the series at 1e-10 leaves an error near 5e-11.
- For a fixed drift v, the exact series' mean time must match the closed form (a/(2v))·tanh(va/2) + t0, at tolerance 1e-8.

**Texture:**
- t0 jitter, as a share of t0: 0, 0.1 or 0.3;
- type mix without Ls and Ah;
- N's share, doubled;
- ordinary choices for R3: 0, 16 or 64;
- ladder [1, 2, 4, 8, 16];
- out of case: a world where every maker shares one speed (people don't differ).

**Shed order:** R3's 64-choice cells, then the upper half of the boundary range (a drawn from [0.8, 1.2] instead of [0.8, 1.6]).

**Route.**
- *Sounding Line:* no human timing data on value trades is admitted this week. Timing pays here by construction at 10 ms; if it still pays in the texture lane's t0-jitter cells, the CoAuthor suggestion latencies become next week's candidate.
- *ToMpathy:* whether decision time earns a place among panel three's identity marks, and the false-call rate it must display.

### K2. MOTIVE_SPLIT. Can a reader split one act into its motives, and does a network of associations do it? (studies 40)

**Rivals built in.** Explaining away is how an exact reader splits motives: if the tournament explains the punch, belief in dislike should fall. On graphs with loops, message passing (loopy belief propagation) often converges, and then approximates well, but on a diagnosis network where each finding has many possible rare causes it oscillated "and had no obvious relationship to the correct posteriors" ([Murphy, Weiss & Jordan 1999](https://arxiv.org/abs/1301.6725)). People fail to explain away in an associative "rich-get-richer" pattern ([Rehder & Waldmann 2017](https://doi.org/10.3758/s13421-016-0662-3)). Learning that tracks its own uncertainty does explain away ([Dayan & Kakade 2001](https://proceedings.neurips.cc/paper/2000/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)). Loops in which beliefs corrupt the evidence and the evidence corrupts the beliefs ("circular inference") best fitted reasoning in schizophrenia, with the evidence-to-belief loops tracking positive symptoms ([Jardri et al. 2017](https://doi.org/10.1038/ncomms14218)).

**World.**
- One maker meets targets j = 1 to J (J = 4) over encounters t = 1 to T, in a context the reader sees: tournament or street.
- Each target has a latent dislike d_j, 0 or 1, with prior 0.3. Each encounter has a latent RAGE state r_t, 0 or 1, more likely when d_j = 1: P(r = 1) is 0.6 against 0.1.
- The act (punch or not) is a noisy-OR of a base rate λ0 = 0.02, the tournament role κ = 0.6, dislike δ = 0.3 and RAGE ρ = 0.4.
- **The loop.** Punching target j turns d_j from 0 to 1 with probability φ: "Inputs become outputs". The primary and the arm run at φ = 0; φ of 0.1 and 0.3 are texture. Scores use each target's final d_j (at T).

**Readers:**
- **E:** exact filtering over the unrolled time series: 2^J dislike states, with r summed out.
- **B:** loopy belief propagation on the static factor graph, ignoring time order. Its nodes are each d_j and each encounter's r_t, and each act is a factor on its target's d_j and its r_t, so its loops are the RAGE triangles (d_j, r_t, act_t); φ changes the data B sees, not its loops. Damped at 0.5, with at most 200 iterations; non-convergence is recorded.
- **H:** associative. Its estimate of each target's dislike is that target's punch rate (punches over encounters), smoothed to (punches + 0.5)/(encounters + 1) so log-scores stay finite, with no discounting.
- **K:** an error-correcting learner with tracked uncertainty (a Kalman filter on cue weights, after Dayan & Kakade), with context and target cues competing. Its rule for turning a target's cue weight into P(d_j = 1) uses only K's own estimates and is declared in the protocol before any draw (0.5).
- **C:** circular inference. B, with each loop message counted twice.

**Case and check.** Each context holds at least 25% of encounters, for at least half the targets, in at least 25% of worlds. A target that never varies context is aliased; count it and report it apart. Pre-check: 1.00 at a tournament share of 0.5, the primary's, and 0.38 to 0.40 at 0.2 or 0.8. Revision range: the tournament share, between 0.2 and 0.8.

**Fixed by design.** E is the exact reader of the true model, so it can't lose in expectation, and H has no discounting by definition: its estimate is each target's punch rate (punches over encounters). So the primary (pre-check: E minus H is 1.3 nats at T = 64 at the primary's φ = 0 and tournament share 0.5, and 1.1 to 1.4 across shares of 0.2 to 0.8, against a bar of 0.5) and the arm at φ = 0 (pre-check: H's false-dislike excess about 0.43 in tournament-heavy worlds) are met by construction and reported for scale. The arm is scored at φ = 0 only. Its false-dislike rate counts errors on d_j = 0 targets alone, so at φ = 0.3 it would charge E for correctly reading the loop (pre-check −0.40). The informative quantities are:
- how far K, B and C fall short;
- under which loops they fall short;
- whether B or C become overconfident.

**Primary.** At T = 64, the log-score on all final d_j, E minus H, in nats with a block interval. Bar: 0.5 nats. Records needed to a sustained 1-nat margin for each d_j are reported beside it. Met by construction and reported for scale (see Fixed by design).

**Arm.** At φ = 0, in tournament-heavy worlds (80% of encounters), H's false-dislike rate (a posterior above 0.5 on targets with d_j = 0, met mostly in tournaments) minus E's, against 0.05. It is met by construction and reported for scale (see Fixed by design), so the card has no falsifiable arm.

**Also reported:**
- K's gap to E, also as a share of H's gap (whether K closes more of the gap than H isn't fixed by design, since both are approximate);
- B's and C's calibration (half-Brier) and non-convergence rates as φ grows.

**Known answers.**
- With φ = 0 and ρ = 0, each act factor is constant in r_t, so B's graph reduces to a tree per target and B must equal E exactly.
- With every encounter in the street and φ = 0, E's posterior on d_j has a closed form (pre-check: matched to 1.1e-16).

**Texture:**
- φ: 0, 0.1 or 0.3;
- tournament share: 0.2, 0.5 or 0.8;
- J: 2 or 8 (multi-goal inference is hard in general and tractable with few goals; [Blokpoel et al. 2013](https://doi.org/10.1016/j.jmp.2013.05.006));
- a third motive (payment), observed in half the encounters;
- out of case: no context variation.

**Shed order:** J = 8, then C.

**Route.**
- *ToMpathy:* the engine's "network" must explain away. If K tracks E and H doesn't, the engine needs error-correcting learning with uncertainty, not co-occurrence. B ignores time order, so its shortfall grows with φ by construction; how fast it grows says how much time-unrolled inference over the maker's history buys.
- *Sounding Line:* no human test this week.

### K3. SETPOINT_VECTORS. Can a reader place a set point it never sees from the efforts that restore it, with overshoot, bands and changing precision? (studies 50)

**Rivals built in.**
- **Bands, not points.** Temperature is held by separate loops, each with its own threshold ([Romanovsky 2007](https://doi.org/10.1152/ajpregu.00668.2006)). One leading model has body fat defended only at an upper and a lower boundary, in a theory paper ([Speakman et al. 2011](https://doi.org/10.1242/dmm.008698)).
- **Projection bias runs both ways** ([Loewenstein, O'Donoghue & Rabin 2003](https://doi.org/10.1162/003355303322552784)).
- **Deprivation turns gain up.** Blocking activity in cortical cultures scaled up synapses, apparently each in proportion to its initial strength ([Turrigiano et al. 1998](https://doi.org/10.1038/36103)). In freely behaving rats, firing in deprived visual cortex was suppressed for two days, then rebounded to its set point over the next 2 to 3 days despite continued deprivation ([Hengen et al. 2013](https://doi.org/10.1016/j.neuron.2013.08.038)). That is the opposite of "the resolution decreases" (mine).
- **The set point moves.** Comfort temperatures follow the weather: "Optimum indoor temperatures tracked both prevailing indoor and outdoor temperatures" ([de Dear & Brager 1998](https://escholarship.org/uc/item/4qq2p9c6), abstract). Its other result, that occupants of air-conditioned buildings are about twice as sensitive per degree, bears on P+ (mine).

**World.**
- Two needs with stocks from 0 to 20 (as in `day_velocity.py`) that fall by consumption and are restored by actions: restore need 1, restore need 2, or rest, all available at every step.
- Each need has a set point in {8, 11, 14} (V26's grid) and a tolerance band of width w around it. Urgency is precision π (starting at 1) times the deficit below the band's lower edge, capped at 5 (`day_velocity.py`'s restore amount). It is zero inside the band and above it, since no action lowers a stock. With w = 0, the band is a point.
- **Balance.** Choice is a softmax over urgencies, with rest at utility 1 and lapse 0.02, as in `day_velocity.py`: the curator's "That's the actual control, is that it's being balanced".
- **Overshoot.** A restore's size is a base amount of 5 plus β times urgency, with no noise.
- **Visibility:**
  - V: stocks visible to the reader;
  - H: only actions and sizes visible, the curator's case.
- **Dynamics,** one per world, equally likely. A long deficit is at least 20 consecutive steps below the band's lower edge. The rates are provisional: P−, S and G use the design check's pre-check rates, and P+ mirrors G's.
  - P0, fixed;
  - P+, π rises with time inside the band (sharpening; with w = 0, at every step not below the set point): π times 1.03 a step, up to 10;
  - P−, π falls during long deficits (the curator's dulling): π times 0.97 a step, down to 0.1;
  - S, the set point drifts toward the experienced stock during long deficits (adaptation): the lower edge falls 0.1 a step, never below the stock;
  - G, π rises during long deficits (gain-up): π times 1.03 a step, up to 10.
- **Probes.** An exogenous shock sets one need's stock, chosen at random, to 0: the curator's "extreme stimuli". Probes come at rate 0, 1 per 64 records or 1 per 16.

**Readers:**
- **naive:** point set point, no overshoot, fixed precision. It adds each observed size to its stock filter but reads no urgency from it.
- **aware:** knows β.
- **band:** two thresholds.
- **dynamic:** holds P0, P+, P−, S and G as hypotheses.

**Case and check.** At β = 0 and T = 512 steps, long deficits (at least 20 consecutive steps below the band's lower edge) occur in at least 25% of worlds. Consumption per need per step is three times `day_velocity.py`'s events: 0, 3 or 15, with probabilities 0.45, 0.5 and 0.05. Revision range: 2.5 to 4 times those events, and the probe rate. Primary 2 runs at β = 0. At `day_velocity.py`'s own rate no world has a long deficit at any probe rate, and with β = 1 overshoot removes them up to four times that rate.

**Fixed by design.** In V, the set point is found almost at once, as G1's observed-stock ceiling reader did (0.99 to 1.0, `G1_REPORT.md`). It is reported for scale; H is the test. "aware" can't do worse than "naive" when β is above zero. In H, a restore's size reveals its urgency exactly when β is above zero, so "aware" reads the deficit at every restore and primary 1's bar is met by construction; primary 1 is reported for scale. With restore-only actions, a band maker's urgency equals that of a point maker whose set point is the band's lower edge, so primary 3 measures whether the point reader's grid contains the lower edges, not band against point; it is reported for scale. A band's upper edge shows only with an action that spends surplus, as in G5.

**Primaries:**
1. **Overshoot cost.** In H with β = 1: records to locate each set point (a sustained 1-nat margin among {8, 11, 14}), naive against aware, as a median ratio. Bar: 1.5. Reported for scale (see Fixed by design).
2. **Moved, dulled or gain-up.** In H after a long deficit, the records the dynamic reader needs to a sustained 1-nat margin among S, P− and G, at each probe rate. Bar: S and P− identified within 128 records at some probe rate above zero.
3. **Point or band.** In H with band makers (w = 4) at β = 0, the records the band reader needs to beat the point reader by 1 nat. Bar: within 128. Reported for scale (see Fixed by design).

**Arm.** Without probes, S and P− are aliased at the old deficit: both lower effort there (pre-check: restore probability at old deficits of 3 to 6 units falls from 0.65 under P0 to 0.55 under S and 0.57 under P−). G raises it (0.68), so G separates from both without probes; that separation is by construction and reported for scale. The arm fails if the dynamic reader separates S from P− within 128 records with no probes.

**Known answers.**
- With w = 0, β = 0 and P0, at `day_velocity.py`'s own events and restore, the naive reader equals the choices-only filter in `day_velocity.forward_scatter` (its third row, over the nine set-point pairs) exactly. Both use urgency min(5, π·deficit) with π = 1, rest utility 1 and lapse 0.02, with every action available (the module's menu 3).
- With stocks visible, β above zero and π known, the first restore whose urgency is above zero and below its cap fixes the band's lower edge exactly, so the set point's posterior is degenerate from then on.

**Texture:**
- β: 0 or 0.5;
- w: 2;
- three needs;
- consumption doubled (six times `day_velocity.py`'s events);
- lapse 0.05;
- out of case: stocks visible with noise of standard deviation 1.

**Shed order:** three needs, then the 1-per-16 probe rate.

**Route.**
- *Sounding Line:* CH4 (the chess clock reserve) and SP (comfort set points) read set points from restoring moves on people. This card says how many records that takes, that restoring efforts alone can't tell a band from a point at its lower edge, and whether only probes, which those records lack, separate dulling from moving.
- *ToMpathy:* the set-point reader's design, and what it may claim.

### K4. FEAR_AND_STRESS. Can a reader tell FEAR from stress, and which uncertainty makes stress? (studies 30)

**Readings and rivals built in.**
- **Outcome uncertainty, which he endorsed.** Subjective stress, pupil size and skin conductance tracked irreducible uncertainty about a shock, in a task where predictions "did not affect" the shocks ([de Berker et al. 2016](https://doi.org/10.1038/ncomms10996)). On 3 November he endorsed the analyst's summary of it ("irreducible uncertainty"). The task left no open question about what to do, so it separates outcome uncertainty (S2) from uncertainty about what to do (S1). Both are readings of his rule that "stress is caused by uncertainty" (mine).
- **Control.** Detected control switches off the default passive response to prolonged aversive events ([Maier & Seligman 2016](https://doi.org/10.1037/rev0000033)).
- **Strategy uncertainty.** Stress as uncertainty about which strategy to select, in a theory paper ([Peters, McEwen & Friston 2017](https://doi.org/10.1016/j.pneurobio.2017.05.004)). That is the curator's S1, so it gives S1 a published source rather than adding a law (mine).
- **Possibility, not probability.** For vivid outcomes, choices and willingness to pay barely change across middle probabilities ([Rottenstreich & Hsee 2001](https://doi.org/10.1111/1467-9280.00334)).

**World.**
- Episodes: a threat with harm probability p ∈ {0, 0.1, 0.5, 0.9, 1}, and three actions (fight, run, talk), each with a success probability.
- Trained makers know the success probabilities, so they know which action is best (entropy 0). Novices hold a posterior under which each action is equally likely to be best (entropy log 3).
- **FEAR laws:**
  - F1, fit: intensity is p times magnitude, with magnitude 1 (the curator's trigger-fit idea, which he offered as a wonder);
  - F2, possibility: intensity is the full magnitude whenever p is above 0 (after Rottenstreich and Hsee's choices and willingness to pay).
- **Stress laws:**
  - S1, policy: the entropy of the maker's belief about which action has the highest success probability, so it doesn't depend on p. It is the uncertainty his 7 November gun example shows, and Peters, McEwen and Friston's strategy uncertainty;
  - S2, outcome: p(1 − p) (de Berker; the finding whose summary he endorsed on 3 November);
  - S3, information: the expected reduction in outcome uncertainty that the most informative action would bring, unknown to the maker (mine: an analyst's construct with no published source).
- **Records** (provisional laws; the design check's pre-check used the latency, information-seeking and arousal laws):
  - avoidance, FEAR's output: a withdrawal before acting, recorded apart from the chosen action (so choosing run is not avoidance), with probability logistic in 2·FEAR − 1;
  - latency, normal around stress with standard deviation 1, and information-seeking, with probability logistic in 2·stress − 1, before acting (stress's outputs);
  - an arousal channel equal to FEAR plus stress, plus normal noise of standard deviation 1.

**Readers.** An exact reader over the six law pairs, from records. Each stress law's free positive scale (see Fixed by design) runs over 25 log-spaced values from 0.1 to 10, with a uniform prior.

**Case and check.** In the designed mix, half the episodes have fixed outcomes (p of 0 or 1) with novices, and half have p = 0.5 with trained makers. That decorrelates outcome uncertainty from policy uncertainty in at least 25% of worlds. The observational mix (novices face p = 0.5, trained makers p of 0 or 1) is an out-of-case map.

**Fixed by design.**
- With arousal alone, S1 and S2 separate only through episodes that decorrelate them. Reported for scale.
- In the designed mix every episode separates S1 from S2, so the primary is met by construction under every FEAR and stress law (pre-check: 1 to 16 episodes at latency noise of 0.5 to 4 stress units, with or without a free scale). It is reported for scale.
- Each stress law carries a free positive scale, which the reader marginalizes. With it, the observational mix aliases S1 and S2 (pre-check: no separation within 128 episodes in at least 95% of worlds); without it they separate through their levels within 5 episodes. This needs novices at p = 0.5 alone: with novices at p of 0.1, 0.5 and 0.9, S2 varies while S1 doesn't, and the free scale no longer aliases them (pre-check: S1 found in 9 to 11 episodes, S2 in about 100 to 119, censored in about 40% of worlds). The arm is set by this declaration and reported for scale; the card has no falsifiable arm.

**Primary.** Records (episodes) to a sustained 1-nat margin between S1 and S2 makers, in the designed mix, with all records. Bar: within 128 episodes. Met by construction and reported for scale (see Fixed by design).

**Arm.** In the observational mix, whether the reader separates S1 from S2 within 128 episodes. The free-scale declaration sets it (see Fixed by design), so it is reported for scale.

**Also reported:**
- F1 against F2: records needed;
- whether arousal alone, the only channel most human studies record, identifies anything.

**Known answers.**
- With p of 0 or 1, S2 is 0 exactly.
- For a trained maker, S1 is 0 exactly.

**Texture:**
- magnitude doubled;
- the trained share at 0.25 or 0.75;
- latency noise doubled;
- arousal-only records;
- out of case: the observational mix.

**Shed order:** S3, then magnitude.

**Route.**
- *Sounding Line:* CH2 and CH3 measure chess thinking time against policy closeness (k50, how many of the five analysed moves are within 50 centipawns of the best; close to S1) and stakes; IN2 and IN3 measure FEAR's intensity against certainty and control. This card says which human records could separate the laws.
- *ToMpathy:* whether panel three may show stress apart from FEAR.

### K5. CLOSING_AND_RECHECK. Re-reading, retroactive tags, warnings, in-groups and richer libraries (studies 30)

**Rivals built in.**
- **Suspicion without rereading.** Being told to disregard did nothing, but jurors given reason to suspect the motive behind the information returned verdicts no different from unexposed jurors ([Fein et al. 1997](https://doi.org/10.1177/01461672972311008)). Suspicion did, without rereading, what the curator says needs rereading (mine).
- **Retrievable tags.** In a preregistered test, memory for truth and falsity "did not differ" ([Nadarevic & Bell 2024](https://doi.org/10.3758/s13423-024-02482-8)).
- **Masking, not erasing.** Extinction adds new learning that depends on context ([Bouton 2004](https://doi.org/10.1101/lm.78804)).
- **Discounting fades.** The sleeper effect appears only when the discounting cue comes after the message ([Kumkale & Albarracín 2004](https://doi.org/10.1037/0033-2909.130.1.143)).
- **Insiders defraud insiders.** Ponzi schemes do better when an affinity link is present ([Deason, Rajgopal & Waymire 2015](https://ssrn.com/abstract=2586490), working paper). Familiar and similar others are judged truthful ([Bond & DePaulo 2006](https://doi.org/10.1207/s15327957pspr1003_2)).

**World.** G3's (`day_closure.py`): 32 sources per world with alternating truth; honest sources, the in-library manipulator, the over-pusher and the con. The con is honest for 16 claims, then pushes agenda 1 at strength 1. Readers K (open), D (freeze at the gate: it stops updating once its posterior that the source means harm reaches the closing threshold θ), R (revert: on closing, it resets to its prior) and the contamination drag δ ∈ {0, 0.05, 0.1} (once K's harm posterior reaches θ, each later record multiplies K's belief in the truth value its more probable agenda pushes by e^δ) are kept. Capture is G3's: the mean posterior mass a reader puts on the wrong truth. The captured share (sources with less than 0.5 on the truth) is reported beside it. New readers and worlds:

| Arm | Reader or world | What it does |
|---|---|---|
| a | RR, re-read | After closing, re-reads the closed source's earlier records under the harmful model. Removes a fraction x ∈ {0.25, 0.5, 0.75} of their influence (the curator's "weakening"), at a cost of one record per record re-read |
| b | RT, retroactive tag | On closing, discounts the source's earlier records by τ ∈ {0.5, 0.9} without re-reading. The tag fades at rate ζ per later record (0, or 0.02, the sleeper effect). In a context-switch variant, the tag is retrieved with probability ψ = 0.5 after record 48 (masking, after Bouton) |
| c | W, warnings | Before a source turns, a warning raises the prior that it means harm, with reliability r ∈ {0.6, 0.8, 0.95}. False warnings hit honest sources |
| d | IG, in-groups | Sources and readers belong to groups, and in-group readers hold the in-group's source kinds in their library. Scenario (i): in-group sources coordinate an agenda against an outsider reader (the curator's "pull one over on a foreigner"); its readers pool across sources, since G3's work one source at a time, by a rule the protocol declares (0.5). Scenario (ii): an in-group manipulator exploits in-group readers' higher prior on in-group honesty, with truth bias b ∈ {0, 0.1, 0.2} (affinity fraud) |
| e | LF, library frontier | G3-T09's two-strength library, plus a four-strength library {0.25, 0.5, 0.75, 1}. Both manipulator capture and the misreading of truth-pushers are scored |

**Case and check.** As G3: the con captures K on at least 25% of truth-0 sources, at the registered threshold θ = 0.7.

**Fixed by design.**
- RR with x above 0 reduces the con's earlier influence relative to D, by definition. In G3's world that changes capture by at most 0.005 (next item), so the cost-benefit comparison of RR, RT, D and K, and what the context switch does to RT, are reported for scale.
- D closes the con at or after record 28 at θ 0.7 (G3's floor), on about 22% of truth-0 sources, when its posterior on the truth is near 0.5. Removing any fraction of the closed source's earlier influence (the reading the known answers fix) then moves capture by at most 0.005 over all con sources (at most 0.04 at θ 0.5). Primary 1 and the arm are therefore "not met" by construction and reported for scale.
- In IG, scenario (i)'s sign is set by the libraries, and scenario (ii)'s by the truth bias. The result is their sizes, and where (ii) overtakes (i) as b grows.

**Primaries:**
1. Against the 16-claim con at truth 0: capture under RT minus under RR, at each pairing of τ and x. Bar: 0.05 either way. RT ahead favours Fein's tag over the curator's re-reading. RR's re-read cost is reported beside it in records re-read; it changes no capture here, since a closed source's later records are ignored. Reported for scale (see Fixed by design).
2. In IG: outsider capture in scenario (i) minus in-group capture in scenario (ii), at b = 0.1. Bar: 0.05 either way.

**Arm.** RT's protection after the context switch, minus before, against a loss of 0.05. RT's whole protection over D is about 0.005, so the loss can't reach 0.05: the arm is "not met" by construction and reported for scale, and the card has no falsifiable arm.

**Also reported:**
- W's capture saved and its false closures on honest sources, at each reliability;
- LF's frontier of manipulator capture against truth-pusher misreading. This is the curator's "friction... where bad actors live", now measured;
- the captured share beside mean capture for RR and RT (pre-check on closed sources at θ 0.7: 0.95, 0.73, 0.59, 0.59 and 0.50 for x = 0, 0.25, 0.5, 0.75 and 0.9, while mean capture barely moves).

**Known answers.**
- At x = 1, RR equals the reader that never saw the closed source's records, exactly.
- At τ = 0, RT equals D exactly.
- G3's exact programme reproduces its published values at δ = 0.

**Texture:**
- δ: 0.05 or 0.1 for RR and RT;
- a con turning at 32 claims;
- threshold 0.5;
- b = 0.2;
- out of case: G3-T09's library.

**Shed order:** the four-strength library, then W's 0.6.

**Route.**
- *ToMpathy:* the closed-source state. The curator's rule, adopted as the working position (C-1107-17), is "no way to go back and flag them". In G3's world RT and RR are both null by construction (see Fixed by design), so this card doesn't price a tag against re-reading, and its RT and RR results license no engine state; they are reported for scale, and the working position stays untested here.
- *Sounding Line:* none this week.

### K6. RECURSIVE_READING. Does re-reading the steps with the goal one has inferred improve both, or lock in the first guess? (studies 30)

**Rival built in.** In schizophrenia, reasoning was best fitted by loops in which beliefs corrupt the evidence and the evidence corrupts the beliefs ([Jardri et al. 2017](https://doi.org/10.1038/ncomms14218)); that such loops count evidence twice unless inhibition cancels them is the 2013 theory, known here only from a secondary summary. So an iterative reader can confirm itself (mine).

**World.** H2's (V26 week): `pauses.py`'s law, with four goals in place of the module's six; `hierarchy.py` plays no part.
- a maker writes steps under a top goal g, one of four;
- deliberation pauses fall, with probability d, before decision points where the goal matters, so they depend on g;
- difficulty pauses fall, with probability q, at hard steps, independent of g, at 25% to 50% of pauses.

**Readers:**
- **J:** the exact joint posterior over g given steps and pauses.
- **S:** sequential. Classify pauses with a goal-agnostic model, then infer g. Classes are hard: a pause is deliberation when its goal-averaged deliberation probability is at least its difficulty probability.
- **I:** iterative. Alternate inferring g from steps and pause classes, then reclassifying pauses given g, until stable or 20 rounds. Reclassification weights each goal by its current posterior, with hard classes.
- **Ic:** iterative with confirmation. A pause is reclassified as deliberation only when it fits the current g, the most probable goal and habit, better than difficulty does.

**Case and check.** Difficulty pauses are at least 25% of pauses in at least 25% of worlds, at d = 0.3 and q = 0.5 (pre-check 0.82; at H2's registered d = 0.6, q = 0.2 it is 0.14). Revision range: H2's grid, d in {0.3, 0.6} and q in {0.2, 0.5}.

**Fixed by design.** J is exact and can't lose in expectation (scale only). I isn't guaranteed to converge; report its non-convergence rate.

**Primary.** Log-score on g at 64 steps, I minus S, as a share of J minus S, with a block interval. Bar: at least 0.5 (I recovers at least half of what the exact reader gains over S). J minus S is 0.04 to 0.18 nats on H2's grid, so an absolute bar of 0.2 nats could not be met even by J. Pre-check: about 0.003, so "not met" is the likely result, but both outcomes are possible.

**Arm.** Ic's calibration error (half-Brier) minus J's. It fails if the difference stays below 0.02. Its sign is fixed by design, since J is exact and half-Brier is a proper score; its size is the test (pre-check 0.034 to 0.039 against the 0.02 bar).

**Known answers.**
- With no difficulty pauses, S equals J exactly.
- With deliberation pauses whose occurrence and redraw are both independent of g, I's fixed point equals S exactly (both equal the prior on g).

**Texture:**
- difficulty share 0.5;
- eight goals;
- deliberation informativeness halved;
- a cap of 5 rounds;
- out of case: difficulty pauses correlated with g.

**Shed order:** eight goals.

**Route.**
- *ToMpathy:* how panels one to three should update each other. One joint pass, a sequential pass, or iteration with a guard against confirming itself.
- *Sounding Line:* AT3 and GO2 are the human analogues.

### K7. LAUGHTER_MARKER. Does a laugh at the final piece help a reader find a hidden structure, and what if laughs mark something else? (studies 20)

**Rivals built in.**
- **Most laughs aren't at jokes.** In 1,200 laughs, laughter followed statements, questions and material not humorous outside the conversation ([Provine 1993](https://doi.org/10.1111/j.1439-0310.1993.tb00478.x)). In his 2000 Psychology Today write-up of the same data, only 10% to 20% followed anything joke-like.
- **Resolution alone isn't funny.** Resolution without linking remote ideas gives insight without mirth ([Amir et al. 2015](https://doi.org/10.1093/cercor/bht343)).
- **The ancestral call marks a state.** The rat's 50-kHz call comes with rats' play and tickling ([Panksepp & Burgdorf 2003](https://doi.org/10.1016/S0031-9384%2803%2900159-8), who call its reading as laughter a working hypothesis) and with reward anticipation, rising when amphetamine is put into the accumbens ([Burgdorf et al. 2001](https://doi.org/10.1037/0735-7044.115.4.940)). In hide-and-seek, rats called while seeking and finding and fell silent while hiding ([Reinhold et al. 2019](https://doi.org/10.1126/science.aax4705)). So the ancestral call marks a state more than a found structure (mine).
- **The marker can be faked.** Canned laughter makes bad jokes funnier ([Cai et al. 2019](https://doi.org/10.1016/j.cub.2019.05.073)).
- **Shared knowledge.** The encryption theory: jokes signal shared background ([Flamson & Barrett 2008](https://doi.org/10.1556/JEP.6.2008.4.2)).

**World.**
- A maker emits pieces that build structures in `hierarchy.py`'s document format (each structure's size is in the case below).
- **Marker laws:**
  - M1, completion (the curator's): a laugh with probability q_c when a piece completes a structure, and q0 otherwise;
  - M2, surprise: laugh probability rises with the piece's surprisal under the population model;
  - M3, state: laughs follow a hidden PLAY or reward-anticipation state that switches on and off, independent of content.
- **M2's and M3's settings.** M2's rule from surprisal to laugh probability and M3's switching rates are declared in the protocol (0.5), each scaled so its average laugh rate matches M1's. The protocol also reports how much more surprising completion pieces are than other pieces, since that sets how far M2 sits from M1.
- **Listeners** who share the maker's library laugh at completions under M1. Outsiders laugh at base rate q0 only.
- **Faked markers:** a manipulator emits markers without completions, at rate f: 0 in the primary and the arm, 0.1 where reported.
- **Boundaries and library.** Documents run together in one stream with no boundary marks, so every reader must infer boundaries. Under M1, q_c = 0.8. The library makes each structure's top goal inferable only at its final piece (declare the construction: `hierarchy.py`'s own library is read progressively). If boundaries were given, markers would carry no information about the top goal: M and Mc would equal N exactly, and the primary and arm would be fixed.
- **q_c is untested.** The value 0.8 is a placeholder whose effect on the primary's bar of 1.5 hasn't been checked. Before the study, the pre-check computes the primary's expected ratio under M1 at q_c of 0.4, 0.6, 0.8 and 1.0 (q0 = 0.2). Revision range: q_c from 0.4 to 1.0. N ignores markers, so q_c can't cure N's censoring. The world's number of structures is declared in the protocol (0.5). If N's or M's median is censored (no sustained margin within the world's structures in more than half the worlds), that number doubles, up to four times its declared value, until neither is. If M's median is still censored at 0.8, q_c moves to the nearest value in the range at which it isn't. If neither step clears the censoring, the primary is reported as "unresolved at this sample size", not as a failed case: the case is the completion share, which censoring doesn't touch. If the expected ratio falls on the same side of 1.5 at every value in the range, the primary goes in the by-construction list and is reported for scale.

**Readers:**
- **N:** no markers.
- **M:** markers, with the true law.
- **Mc:** markers, assuming M1 whatever the world.

**Tasks:**
- infer each structure's top goal;
- locate structure boundaries;
- infer which listeners share the library, from co-laughter.

**Case and check.** Each structure is one `hierarchy.py` document of three episodes, so completions are about one piece in six. Under M1, completions are at least 10% of pieces in at least 25% of worlds (pre-check: all worlds; with the module's default six episodes, none).

**Fixed by design.** Under M1, with boundaries hidden, markers carry information by construction. The result is how much, as q0 rises toward the 80% to 90% of laughter that isn't at jokes in Provine's 2000 write-up.

**Primary.** Under M1: structures to a sustained 1-nat margin on the top goal, N's median divided by M's, at q0 = 0.2, q_c = 0.8 (untested; see World) and f = 0. Bar: 1.5.

**Arm.** Under M2 and under M3, separately, since pooled they can cancel: Mc's top-goal log-score minus N's. Markers mislead if it's negative. The arm fails under a law if its difference stays within 0.05.

**Also reported:** in-group detection from co-laughter (records needed), and the cost of faked markers at f = 0.1.

**Known answers.**
- At q_c = q0, markers carry no information, and M equals N exactly.
- With one structure and no noise (q_c = 1, q0 = 0 and f = 0), the completion piece's position is known exactly.

**Texture:**
- q0: 0.05 or 0.5;
- q_c halved;
- listener count 1 or 8;
- library overlap 0.5;
- out of case: M3 with q0 = 0.

**Shed order:** eight listeners.

**Route.**
- *Sounding Line:* LA1 (does laughter follow resolution or surprise, on meeting transcripts) and LA2 (do people who share background laugh together).
- *ToMpathy:* whether textual laughter markers belong among panel three's structural cues, and how much a base rate of social laughter dilutes them.

## 3. Dispatch and schedule

- **At handoff,** in this order:
  1. V26's closure;
  2. the amendments;
  3. V27's contract and authorization;
  4. the theory-store rows (0.7);
  5. then every card's protocol and design pre-mortem, with the pre-check under every rival.
- **Primaries:** K1, K3, K2, K5, K4, K6, K7, as workers free.
- **Texture:** each family's lane after its primary is verified (amendment F, stopping by amendment K), in the same order.
- **Idle workers** (amendment L), in this order:
  1. texture;
  2. precision follow-ups where an interval straddles its bar;
  3. follow-ups under amendment J, and re-reads (section 6).

## 4. Theory-store rows (for `READING_INTENT.md`, 0.7)

Each row's status is in its Status column. **OPEN** is the curator's position, with no Ghost result unless named; rows that restate the 7 November theory filing carry its status words. Dates are when he said it.

| Row | Position | His words | Date | Cards | Status |
|---|---|---|---|---|---|
| C-1101-1 | A reader reaches a maker's top by running her own expertise as a forward model and fitting it into a larger inference | "The reader largely has to start by using their own expertise and run it as a forward model and try to modify it to fit it into a larger inference." | 2026-11-01 | V26 E2 | OPEN |
| C-1103-1 | Each Panksepp channel is a pre-solved function: its trigger is the shape of a problem, its function the transformation it applies, run forward to act and backward to read | "They are all chains in the inference that come pre-solved" | 2026-11-03 | V26 G1 | OPEN |
| C-1104-1 | Some channels' triggers are slopes along a set point, not levels | "I think it is slopes, not levels. But the triggers for some of these things are slopes along the set points" | 2026-11-04 | V26 G1 | OPEN |
| C-1106-1 | DISGUST follows a read of harmful intent, then stops inquiry | "They never take one step further once you find a puzzle piece that is truly inefficient." | 2026-11-06 | V26 G3 | OPEN |
| C-1107-1 | Channels fire at graded strength and can fire together | "you're applying like 49% of anger to the solution" | 2026-11-07 | | OPEN |
| C-1107-2 | A channel's strength may be its fit to its trigger | "I do wonder if the amount that they trigger is simply decided upon how well the scenario fits their trigger" | 2026-11-07 | K4 | OPEN |
| C-1107-3 | Set points are never seen, only the efforts that restore them | "you never actually see the set point" | 2026-11-07 | K3 | OPEN |
| C-1107-4 | Restoring efforts overshoot | "People will often overshoot." | 2026-11-07 | K3 | OPEN |
| C-1107-5 | Precision sharpens with time near a set point | "they increase in resolution over time" | 2026-11-07 | K3 (P+ held as a hypothesis only; no primary measures it) | OPEN |
| C-1107-6 | A lasting deficit dulls a need | "the resolution decreases. It's still there, but it just gets weaker" | 2026-11-07 | K3 | OPEN |
| C-1107-7 | Security is maximized, with value falling as other needs get louder; the balance is the control | "That's the actual control, is that it's being balanced." | 2026-11-07 | V26 G5 | OPEN |
| C-1107-8 | FEAR follows predicted harm; anxiety is a light FEAR; stress is caused by uncertainty, and his gun example shows uncertainty about what to do | "Anxiety is a light form of fear, and stress is caused by uncertainty." | 2026-11-07 | K4 | OPEN |
| C-1107-9 | Deliberation time reveals how large a decision tree a value sits in; he asks whether a model of the person predicts the trade | "Your decision tree should be smaller" | 2026-11-07 | K1 | OPEN |
| C-1107-10 | One act can serve several motives at once, read through a network of associations | "anything that can be associated will" | 2026-11-07 | K2 | OPEN |
| C-1107-11 | Closing on a source keeps what it already taught | "a little piece of it is with you forever" | 2026-11-07 | K5 | OPEN |
| C-1107-12 | Re-checking is separate, effortful, and only weakens associations; RAGE can drive it | "all you're doing is kind of weakening the associations" | 2026-11-07 | K5 | OPEN |
| C-1107-13 | In-groups deceive outsiders through shared inference | "this friction is probably where bad actors live" | 2026-11-07 | K5 | OPEN |
| C-1107-14 | New inferences should refine old ones, recursively | "zero in further on old information and recursively somehow" | 2026-11-07 | K6 | OPEN, logic only |
| C-1107-15 | Laughter marks a completed hidden structure, and shared laughter marks shared inference | "Humor and laughter is an inference marker." | 2026-11-07 | K7 | OPEN |
| C-1107-16 | DISGUST may be SEEKING's opposite, with PLAY, LUST and CARE on a loop | "I think disgust might be the opposite of seeking." | 2026-11-07 | | OPEN, logic only |
| C-1107-17 | Earlier claims can't be flagged: ToMpathy's closed-source state flags nothing, and correction is a re-read | "there's no way to go back and flag them" | 2026-11-07 | K5 | OPEN, adopted as the working position |

## 5. Cross-repository

| To | What | When |
|---|---|---|
| Sounding Line, `docs/exchange/ghost-v27-2026-11-07/` | K3 (for CH4 and SP); K4 (for CH2, CH3, IN2 and IN3); K6 (for AT3 and GO2); K7 (for LA1 and LA2) | As each primary is verified |
| ToMpathy, `docs/` | Each verified primary's route note | As each primary is verified |
| From Sounding Line | Week 5's CH4, SP, IN, AT, GO and LA results, for the reports' discussion | Optional; nothing waits |

Aggregate copies are authorized by this packet. Recipient files stay uncommitted.

## 6. Reporting

- **Daily line** at 20:00 PST, in this order:
  1. worker-hours busy and idle, by cause;
  2. cards built, ready, running and verified;
  3. verified primaries with their routes;
  4. texture counts per family;
  5. pool balances.
- **Final report** by Sunday 15 November, 22:00 PST:
  1. each primary with its verdict under the bar rule and its route;
  2. every by-construction contrast for scale;
  3. the texture maps;
  4. what each card says to look for in human records.
- **Re-reading earlier results.** The curator asked on 7 November: "we want to make sure that the relationships between them are checked and that we aren't forgetting about how to use the new information if we find it to zero in further on old information and recursively somehow." Each landed card's record names, by ID, the earlier Ghost results (V25 to V27 findings and the theory store's rows) that bear on it, and says for each whether the new result could change its reading. Where it could, a re-read packet is queued from the follow-ups, healing and texture pool: the earlier analysis rerun with the new result as a covariate, a prior or a filter, under the earlier card's frozen rule, reported beside the original and never overwriting it. A re-read that changes a reading gets the same pass, at most two levels deep this week. A re-read is a V27 packet that reads the earlier packet's saved records and never re-executes it, so it admits nothing in V25 or V26 and lifts no hold. Its protocol declares, before it runs, whether the new result enters as a covariate, a prior or a filter (0.5).

## 7. Triggers (the only reasons to contact the owner)

1. A card's studies would pass 1.5 times its ceiling after shedding.
2. A result reverses a published V26 or earlier claim.
3. An outage longer than 12 hours.

## 8. Pre-mortem

- **The workers drain in two hours again.**
  - *Guard:* four workers, seven cards with wider grids, amendments H to M, and precision follow-ups and re-reads as commissioned work.
  - If a day's line shows more than 25% of worker-hours idle with the pools unspent, the next check-in commissions the next texture cells first.
- **A card is true by construction.** *Guard:* the pre-check under every rival (0.5), and each card's "fixed by design" list. A card whose primary comes out the same under every rival reports for scale only.
- **K1's diffusion likelihood is slow or unstable.** *Guard:* the series tolerance, the closed-form known answers, and binned times (10 ms) as the smaller route.
- **K2's loopy reader doesn't converge.** *Guard:* damping, the iteration cap, and non-convergence reported as a result.
- **K3's dynamics are unidentifiable without probes.** That's the arm. Report it as the finding it is.
- **K5's new readers break G3's published values.** *Guard:* the known answer at δ = 0.
- **Stories outrun the worlds.** Every report carries "miniature, architecture untested", and names its rivals and which one won.

## Sources

- **Ghost records**, at `3a6b87ff4`:
  - `AGENTS.md`;
  - `docs/versions/v26-top-and-ends/day-20261106/FINAL_REPORT.md`;
  - `docs/versions/v26-top-and-ends/day-20261106/G1_REPORT.md`;
  - `docs/versions/v26-top-and-ends/day-20261106/G3-T09_REPORT.md`;
  - `ghostscale/validation/soundingline/v26/`.
- **Literature,** as linked in each card. Read levels and quotes for each study are in the analyst's research notes of 7 November, with three exceptions: Hanselmann & Tanner 2008 (K1) is in the 6 November theory errata record; the de Dear & Brager abstract line (K3) is from the 7 November design check's direct fetch of the eScholarship item; and Panksepp & Burgdorf's "working hypothesis" (K7) is from the Europe PMC abstract, read in the round-2 design check.
