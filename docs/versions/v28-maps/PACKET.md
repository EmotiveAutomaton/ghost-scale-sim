# Ghost Scale Sim, V28 (11 to 15 November): one study, two planning tables, one page

*Spec sheet for the Ghost agent. `AGENTS.md` says "A subsequent production campaign requires a separately costed owner decision" and "New activity needs an explicitly admitted scope and the same scientific gates." This packet, handed over by the curator, is that decision: the scope is the four items below and nothing else, at the cost in section 0. Every decision is made here. As of Wednesday 11 November 2026, written against `origin/main` at `d27ff94d4`.*

**What V28 is for.**
1. **One study that bears on the engine (G1).** V27's K6 found that reading goal and habit in separate passes recovered none of the joint reader's gain (primary 0, with a joint gain of 0.227 reported descriptively). The theory thread's Round 12 handoff (11 November, supplied with this packet) turned that into an engine note: "infer goal and habit jointly. Do not pick a channel and then re-read." G1 asks what that costs and what replaces it: how much of the joint reader's gain comes from knowing habits at all, how much from a maker's own history, and when a second pass helps or hurts. K6's loop passed hard class labels between passes. G1's loops pass probabilities, which is why they can differ from K6.
2. **Two planning tables (G2, G3).** They say which of the handoff's set-point rules could ever show in a record, and how many records it takes. They are scale reports, and they are labelled that way.
3. **One page that re-reads V27 the same way (G0).**

**What V28 is not.** There is no texture lane, no follow-up lane and no precision lane. A grid cell is a row in a map, not a card. V27 completed 720 packets and its studies used 1.085 of their 240 CPU-hours. V28 writes three reports, one page and one return, and registers no claim. Once V28's own items are done, Ghost's workers will be idle, and that is reported as idle: "An idle machine is a state to report, not an instruction to invent work."

**Lessons this packet is built on** (the first three from this repository's own records):
- **Met by construction.** K1, K2 and K4's primaries were met, and each was fixed by design and reported for scale. A verdict on a row whose answer the design fixes tells the theory nothing. The pre-flight below found the same of the one claim this packet first meant to register, so V28 registers none. Every row is descriptive or scale only.
- **Hidden stocks need particles, and the particles failed.** K3's dynamic cards are held after three healings; the redesign failed calibration on the first fixture it completed (28 of 28 comparisons, margin error up to 5.15 nats against a limit of 0.25). G2 asks K3's question with a visible stock, where the reader is exact.
- **Preparation cost more than science.** Preparation used 20.35 of its 20 CPU-hours while studies used about one. V28's worlds are small enough to state completely on this page.
- **Cards drafted without simulation fail their own case checks.** The analyst's 7 November drafts did, before delivery. So the analyst ran every world below in numpy before writing this, and a second, independent implementation written from this packet's text alone reproduced the cells it tried (section 5). Those numbers size the grids and seed the known-answer checks. They are not results.

## 0. How to run this

- **Names.** Version folder `docs/versions/v28-maps/`, with this packet as `PACKET.md`. Results under `results/v28/`. Modules under `ghostscale/validation/soundingline/v28/`. Finding ids `G-V28-<family>-<n>`.
- **Clock.** Opens at handoff. No new starts after Sunday 15 November, 20:00 PST. Drain by 22:00.
- **CPU.** One pool of 12 CPU-hours: G1 6, G2 2, G3 1, and 3 held for a rerun after a failed check. Everything is charged at measured CPU, preparation included, as amendment I does for its pools. The pre-flight ran every grid in minutes, so the pool is mostly headroom for this repository's replay and reconstruction.
- **Workers.** Up to four single-thread, below-normal workers, as V27. Sounding Line keeps priority. No GPU. Amendment L applies to V28's own items: an idle worker takes the next one.
- **Wakes.** Work inline from handoff. For V28, amendment H's sixty-minute check-in is replaced by this: while a run is in flight, keep one wake at its expected end; if a session ends with a V28 item unbuilt, keep one check-in at most four hours ahead. When V28 is filed, report completion and retire wakes (`AGENTS.md` lines 297 to 298). Tokens are short this week, so finish in as few sessions as the work takes.
- **Verification, once per family, not per cell:**
  - the original run;
  - one replay from seeds;
  - one independent reconstruction of the whole grid by separately written code, as `study.py` does;
  - the family's known-answer checks.
- **Intervals.** The worlds of a block share one library and one set of goals, so every interval and standard error in G1 is over blocks, not worlds. G2 and G3 worlds are independent.
- **Pre-flight comparison.** After your own implementation passes its known-answer checks, and not before, open `preflight/PREFLIGHT_SUMMARY.md` (supplied) and compare the cells it lists. Tolerances: in G1, 0.3 nats at K = 0 and 0.1 nats elsewhere; for a median number of records, a quarter of the value; for a share, 0.05. Beyond a tolerance, stop that family and report both numbers. Don't tune either side. Don't read the pre-flight scripts before your own code runs; independent implementations agreeing is the point.
- **Records.** One report per family G1 to G3 under `docs/versions/v28-maps/reports/`. Rows are grid cells, with columns: setting, measure, estimate, 95% interval, scale only (true or false). One FINDINGS entry and one READING_INTENT row per family. CLAIMS.json gains at most three rows, all `descriptive`.
- **K3.** Held stays held. The running diagnostic (K3-RD1-DIAG1) finishes and is recorded. No further healing, ladder, gate change or redesign is commissioned. V27's other open administrative items stay as they are; this packet decides none of them.
- **Exchange.** Copy G1's report to ToMpathy and to Sounding Line, and G2's to Sounding Line, uncommitted in the recipient, as V27 did.

## 1. G0: V27 on one page

Write `docs/versions/v28-maps/V27_MAP.md`, under 600 words, with no new computation and no new claims. One table, one row per family K1 to K7:
- the question;
- what the design fixed (scale only);
- what was open, and what it returned;
- one sentence a theory reader can use;
- pointers to the report and finding ids.

Mark K7-T036's minus-infinity gain as an artefact of exact zero model support, not a result. List K3's dynamic cards as held, not null.

## 2. G1: reading goal and habit together, with and without a maker's history

**The question.** A page mixes what its maker always does (habit) with what this page is for (goal). A reader wants the goal, and wants to mark which tokens were departures from habit. What does the reader need to know about the maker's habits, and where can it get that?

**Small world.**
- 12 moves. A library of 8 habit types, each one draw from Dirichlet(0.3) over the moves. 4 goals. The library and the goals are drawn once per block of 50 worlds.
- Goal g's distribution t_g puts 0.9 on its three target moves, equally, and spreads 0.1 over all twelve.
- **Overlap.** *Low:* a random permutation of the moves cut into four disjoint triples. *High:* disjoint triples, with goal g taking habit type g's three most frequent moves not already taken by goals before it. So in the high condition a goal's signature looks like some maker's habit, while goals stay apart from each other.
- A maker has one habit type i, drawn uniformly. A page has one goal g, drawn uniformly, and 64 tokens. Each token comes from t_g with probability rho, and otherwise from the habit h_i. Keep each token's true source, for the marks. Write m(i, g) = (1 - rho) h_i + rho t_g for the token distribution.
- History: K earlier documents by the same maker, each with its own goal drawn uniformly, the same length and the same rho. No reader is told a history document's goal.

**Larger world.** 200 moves (Dirichlet(0.3) over them), 32 habit types, 8 goals with 10 target moves each (0.9 on the ten, 0.1 over all), 128 tokens. Low overlap: disjoint random sets. High: goal g takes type g's ten most frequent moves not already taken. Otherwise the same.

**Readers.** Every reader knows rho and the goal distributions. They differ in what they know about habits. Each returns a distribution over the page's goal.

| Reader | What it knows about habits | How it reads |
|---|---|---|
| J | the block's library | Exact. Type posterior from history: p(i) proportional to the product over history documents of the average over goals of that document's likelihood under m(i, g). Goal: p(g) proportional to the sum over i of p(i) times the page's likelihood under m(i, g) |
| Ps | the library | Pass 1: q(i) proportional to the likelihood of every page and history token under h_i, as if all were habit. Pass 2: p(g) proportional to the sum over i of q(i) times the page's likelihood under m(i, g) |
| I(1) | the library | Start from Ps's p(g). Re-estimate the type: q(i) proportional to J's history posterior p(i) times the sum over g of p(g) times the page's likelihood under m(i, g). Then read the goal again as in pass 2 |
| B(k), k = 1, 2 | the library | J restricted to the k types with the largest p(i) times the page's likelihood under h_i, with p(i) renormalised over them |
| Pp | one typical habit: the mean of the library's types | One pass: p(g) proportional to the page's likelihood under (1 - rho) times the typical habit plus rho t_g. It ignores history |
| Pe | only the maker's own tokens | Habit estimate: counts of the page's and the history's tokens, plus 0.5 each, normalised. One pass as Pp with that estimate |
| PeL(n), n = 1, 4 | the same | Start from Pe's estimate. Each round, for every document (the page and each history document): compute that document's goal posterior under the current estimate; a token of move v counts toward habit with weight equal to the average, over that posterior, of (1 - rho) h(v) divided by [(1 - rho) h(v) + rho t_g(v)]. The new estimate is those weighted counts plus 0.5, normalised. After n rounds, read the page's goal once more |
| PpS(a), a = 32, 128 | the typical habit as a prior worth a tokens | Habit estimate: a times the typical habit plus the maker's token counts, normalised. One pass |
| PpSL(a) | the same | PpS's estimate, then four rounds as PeL with the prior in place of the 0.5 |

**Measures.**
- Mean log-score on the true goal, and each reader's gap to J in nats.
- The share of worlds in which a reader gives the true goal less than 1%. The tail carries the mean, so both are reported.
- Accuracy, and calibration (mean top probability minus accuracy).
- **Departure marks.** For each page token, the reader's probability that it came from the goal. For J: the average over its joint posterior of rho t_g(v) divided by m(i, g)(v). For Pp, Pe and PeL(4): with the reader's most probable goal and its habit estimate h, rho t_g(v) divided by [(1 - rho) h(v) + rho t_g(v)]. AUC against the true source, pooled over a cell's tokens, ties at mid-rank.

**Grid.**
- Small world: K in {0, 1, 2, 4, 8, 16}; rho in {0.2, 0.3, 0.4}; both overlaps; 8,000 worlds per cell (160 blocks).
- Larger world: K in {0, 1, 4, 16}; rho in {0.05, 0.10, 0.20}; both overlaps; 2,000 worlds per cell (40 blocks).

**No registered claim.** The statement the analyst first meant to register, that on one page a second pass over the maker's own tokens scores at least 0.2 nats worse than one pass, sat about ten standard errors past its bar in the independent check. It could not fail, so it is a row of the map.

**What the report's first table holds.**
- PeL(4)'s log-score minus Pe's at K = 0, by rho and overlap (the one-page cost or gain of a second pass).
- The K at which PeL(4) first beats Pe.
- The K at which Pe comes within 0.05 nats of J.
- The rho at which Ps falls 0.1 nats behind J at K = 0, and whether history closes it.
- Pp's gap at K = 16.
- The mark AUC of Pe at K = 0 and at K = 16, beside J's and Pp's, and whether J's moves with K.

**Fixed by construction (scale only).**
- J is at least as good as every reader in expected log-score, because it is exact.
- B(8) equals J.
- Pp ignores history, so its row is flat in K.

**Known-answer checks.**
- B(8) equals J to machine precision.
- J's calibration is within three standard errors of zero.
- At rho = 0.97, every reader's gap to J is under 0.005 nats.
- Small world, K = 16, rho = 0.4, low overlap: PeL(4) is within 0.02 nats of J.

**What the pre-flight saw** (for sizing; small world, 4,000 worlds per cell; gap to J in nats at K = 0, 1, 4, 16):

| Reader | low overlap, rho 0.3 | high overlap, rho 0.3 |
|---|---|---|
| I(1) | 0.00, 0.00, 0.00, 0.00 | 0.05, 0.00, 0.00, 0.00 |
| Ps | 0.26, 0.22, 0.11, 0.06 | 0.53, 0.32, 0.17, 0.08 |
| Pe | 0.25, 0.13, 0.05, 0.02 | 0.60, 0.27, 0.11, 0.04 |
| PeL(4) | 0.96, 0.18, 0.01, 0.00 | 2.39, 0.63, 0.03, 0.00 |

- Pp sat 2.6 to 4.2 nats behind at rho 0.2 and 0.4 to 1.1 behind at rho 0.4, at every K.
- In the larger world at rho 0.05 and 0.10, Ps and I(1) matched J, Pp stayed 0.6 to 6.4 nats behind, and Pe closed from 0.2 to 1.5 at K = 0 to 0.02 to 0.21 at K = 16. Mark AUC: J 0.97 to 0.98; Pe 0.81 to 0.95 at K = 0 and 0.95 to 0.97 at K = 16; Pp 0.61 to 0.89.
- One 4,000-world cell took 1 to 3 CPU-seconds.

**What the map is for** (the analyst's reading of the pre-flight, to be confirmed or overturned by the study):
- Knowing this maker's habits matters more than how the reading is ordered. A reader with only a typical habit never improves with history, and was the worst of the one-pass readers in every pre-flight cell where it was run.
- On one page, a loop helps a reader that has the library (I(1)). For a reader with only the page's own tokens it hurt throughout the small world, and in the larger world wherever goals looked like habits.
- A maker's own history replaces the library after about four earlier documents in the small world. It takes more in the larger one when the goal's share of the page is small.
- Passes fall behind the joint reader when much of the page is goal-driven, and history doesn't fully close that.

These bear on the handoff's engine note (its test 9), on its line "habit is a lossy, compressed record of previous attention", and on Sounding Line's question of whose history predicts a writer's pauses. They also bear on the objection the theory thread raised to the curator's wish for an AI detector: "A detector built on your theory inherits those false alarms unless it can tell one person's autopilot from everyone's average." The typical-habit reader (Pp) against the own-history readers (Pe, PeL) is that difference, in a world where the truth is known. ToMpathy's packet of the same date tests it on real keystrokes and on model-written sentences inside human pages.

**Ceiling:** 6 CPU-hours.

## 3. G2: set points with a visible stock (planning table; every row scale only)

**The question.** The handoff's set-point rules say a reference is a band that is often defended on one side (rule 2), moves with the person's own recent level (rule 5), and is held more precisely near it and less under long deficit (rule 7). Before anyone looks for these in people, which of them could show in a record of the stock and the person's moves, in how many records, and under which events?

**World.**
- A visible stock s from 0 to 24, starting at 12. Steps are numbered from 0. Each step, in this order:
  1. the maker sees s and picks hold, restore or spend by softmax (inverse temperature 1) over values: hold 0; restore = pi_t × max(0, lo_t - s) - 3; spend = pi_t × max(0, s - hi_t) - 3. Types that never spend have no spend action;
  2. restore adds 3; spend removes 3;
  3. the stock falls by d, with d = 0, 1, 2 at probabilities 0.5, 0.35, 0.15;
  4. the stock is clipped to the range, once, here.
- **Types** (pi in {0.5, 1, 2} for all; theta in {8, 10, 12, 14, 16, 18, 20}):
  - ONE(theta): lo = theta, never spends. A one-sided defender;
  - POINT(theta): lo = hi = theta;
  - BAND(theta, w): lo = theta, hi = theta + w, w in {4, 8};
  - MOVE(theta0, lam): as ONE, with lo starting at theta0 and, from step 1, following the recent stock: lo_t = lo_(t-1) + lam × (s_(t-1) - lo_(t-1)), clipped to 2 to 20, lam in {0.02, 0.1};
  - DULL(theta, kap): as ONE, with pi_t = pi × exp(-kap × n_t), where n_t counts consecutive steps, the current one included, with s below theta - 2, kap in {0.02, 0.1};
  - SHARP(theta, kap): as ONE, with pi_t = pi × min(3, 1 + kap × m_t), where m_t counts consecutive steps, the current one included, with s within 2 of theta, kap in {0.02, 0.1}.
- **Regimes** of 256 steps: N as above; D, a long deficit, in which a chosen restore adds nothing on steps 64 to 127; W, a windfall, in which the stock gains 2 extra on steps 64 to 127, after the disturbance and before the clip. One N cell runs 1,024 steps. The reader knows the regime.
- **Reader.** Exact Bayes over type and parameters, uniform within each type's grid, then over the six types.

**Tables to produce.**
1. **Records to separate.** For each ordered pair (truth, rival) and regime: the median, over all worlds, of one plus the last step at which the log posterior odds of the truth's type against the rival's type (each summed over its grid) are below 1 nat. A world that never separates counts as infinite; "aliased" when more than half never do. Truth settings: theta 12, pi 1, w 4, lam 0.1, kap 0.1; then lam and kap at 0.02. 400 worlds each.
2. **Against the rival's 0.1 member.** DULL contains ONE as its rate goes to zero, and its 0.02 grid point nearly equals ONE, so a ONE truth can't reach 1 nat against DULL as a type. For every pair whose rival is DULL, SHARP or MOVE, also report the odds against the rival's 0.1 member alone, and read that row.
3. **Which event does the telling.** For each pair: resolved under N; only under D; only under W; under nothing.
4. **What a simple reader says.** Fit POINT (theta every integer from 4 to 22, pi in {0.25, 0.5, 1, 2, 4}) by maximum likelihood to records from each truth under N. Report the fitted reference, the fitted precision, and the loss in nats per step against the true model. Sounding Line's chess and comfort cards fit simple readers to people; this table is the list of symptoms to look for when the truth is richer.

**Fixed by construction (all of this family is scale only; these are the parts to say out loud).**
- One spend rules out every type that never spends.
- DULL acts exactly as ONE whenever s is at or above theta - 2, and SHARP whenever s is at or above theta. A rule that acts only under deficit can only show under deficit.
- The disturbance has the same likelihood under every type, so only actions carry evidence.
- A ONE truth against DULL as a type is aliased by the grid (table 2).

**Why the action cost is 3.** At cost 1 the pre-flight's maker restored 27% of the time with no deficit, the stock sat near 20, and deficits occurred on 0.1 to 0.4% of steps, so nothing about deficits could show. At cost 3 deficits occurred on 20% of a one-sided defender's steps.

**Known-answer checks.**
- A ONE truth never spends.
- Truth ONE(12, 1) under N, with the reader given only ONE's grid: the posterior on theta = 12 at step 256 averages at least 0.9 over worlds (pre-flight and the independent check: 0.95).
- A step on which two types give identical action probabilities moves their odds by exactly zero.
- Truth POINT(12, 1): table 4 returns theta 12 and pi 1 in at least 90% of worlds (pre-flight and the independent check: 99%).

**Pre-flight, for sizing** (cost 3, regime N).
- DULL against ONE separated in a median 141 records with 21% never separated; SHARP against ONE in 85; only D separated ONE from DULL as the truth. The independent check had ONE against MOVE at 87 records and ONE against SHARP at 221 with 32% never separated. W resolved nothing that N or D did not at cost 1, and is kept here to see whether that holds at cost 3.
- Table 4, 200 worlds per truth, all with a true reference of 12 and precision 1: a one-sided defender was read as a reference of 14 held at half the precision; a moving reference as 9 at the lowest precision on the grid, in every world; a dulling one anywhere from 4 to 14 at the lowest precision in 59% of worlds; a sharpening one as 15 at half; a band as 12 to 14 at about the right precision. So three different rules look alike to a simple reader, as a loosely held reference, and differ in which way the reference is displaced.

**Ceiling:** 2 CPU-hours.

## 4. G3: a tool or a target (planning table; every row scale only)

**The question.** The handoff's rule 10: "A rung treated as endlessly exchangeable becomes a target in itself", with money as the example. What record could show that someone holds a rung as a target and not as a tool or a reserve?

**World.**
- A visible resource M from 0 to 30, starting at a uniform draw from 0 to 30. Two visible deficits, dA and dB, from 0 to 3, starting at 0. Each step, in this order:
  1. M rises by 1 with probability q_in;
  2. each deficit rises by 1 with probability q_need, up to 3;
  3. the maker picks hold, spend on A or spend on B by softmax (inverse temperature 1): hold 0; spending on X is worth wX × dX - keep(M), and is unavailable when M = 0 or dX = 0. It lowers dX and M by 1.
- **Types.** TOOL(r) and SAVER(R) are the same rule with different reserves: keep(M) = 3 when M is at or below the reserve and 0 above it, with r in {2, 4} and R in {10, 15, 20}. CAPTURED(k): keep(M) = k at every M, k in {3, 5}. That makes seven classes. Weights (wA, wB) in {(2, 2), (3, 1), (1, 3)}, drawn uniformly per world.
- **Reader.** Exact Bayes, uniform over the seven classes and the three weight pairs.
- **Grid.** q_in 0.3 with q_need in {0.10, 0.15, 0.20}; q_in 0.5 with q_need in {0.20, 0.25, 0.30}; two anchors, plentiful (0.8, 0.15) and scarce (0.3, 0.3). Truths CAPTURED(3), CAPTURED(5), SAVER(10), SAVER(20), TOOL(2). 400 worlds, 512 steps.

**Tables to produce.**
1. **Records to separate** CAPTURED(3) from each SAVER, by setting: the median, over all worlds, of one plus the last step at which the log posterior odds of the true class against the rival class (each summed over the weight pairs) are below 1 nat. Never separated counts as infinite; "aliased" above one half.
2. **How often the telling moment occurs,** for each SAVER reserve R: the share of worlds, and of steps, in which M is above R while a need is unmet.
3. **The number of such moments seen before separation,** for each R.
4. One sanity line for the weights: records to separate (3, 1) from (1, 3).

**Fixed by construction (so every row is scale only).** A step with no unmet need changes no likelihood. CAPTURED(3) is a SAVER whose reserve is never reached. So the only evidence between CAPTURED(3) and SAVER(R) is a choice made with M above R and a need unmet. In the pre-flight, CAPTURED(3) against SAVER(20) needed a median of 1 to 4 such choices wherever it separated. The table's content is how often a world offers the moment.

**Known-answer checks.** A step with no unmet need moves no posterior. keep under CAPTURED(3) equals keep under SAVER(R) wherever M is at or below R. Over 2,000 steps the posterior on the true weights reaches at least 0.99.

**Pre-flight, for sizing** (M starting at 5): at (0.3, 0.15), CAPTURED(3) separated from SAVER(10) in a median 101 records, from SAVER(15) in 331, and was aliased with SAVER(20); at (0.5, 0.3), 328 records, then aliased twice; in plentiful settings, under 90 records with none censored; at (0.3, 0.3) M passed 20 in at most 1% of worlds, so the moment never came. The independent check agreed in the three cells it ran: 106 records against 101 at (0.3, 0.15), 322 against 328 at (0.5, 0.3), and 23 against 26 at (0.5, 0.15).

**What the table is for** (the analyst's reading). A rung held as a target can only be told from a reserve in choices made in surplus with a need unmet. Records of people who are short, or roughly in balance, can't show it. That tells the theory thread which kind of people-data rule 10 needs before anyone designs a test.

**Not run here: the mirror case.** The theory thread's test for whether Learning and External action are set points of their own, and not only Stress's two hands, is "can it be hungry while stress is fine?" That needs a move the maker can make when no need is unmet. This world has none, and one was not pre-flighted, so it is a line under "not run" in the return, not a table.

**Ceiling:** 1 CPU-hour.

## 5. Pre-flight files

Supplied as `preflight/` (a zip): `PREFLIGHT_SUMMARY.md`, the numpy scripts and their printed outputs. Section 0's rule governs when to open them.

An independent implementation, written from an earlier draft of this packet's text alone, reproduced the pre-flight in every cell it tried: G1's four readers at K = 0 and 4, G2's deficit share and both known answers, and three G3 cells. Its one difference above tolerance, 0.15 nats for PeL(4) at K = 0, went away at 40,000 worlds. The places where it had to guess are now spelled out above.

Three things the pre-flight changed in these designs, so they aren't rediscovered:
- G1's first high-overlap condition let goals overlap each other. The condition above keeps them disjoint. At rho 0.1 nothing separated in the small world, so it isn't run there; in the larger world at 256 tokens with rho 0.2 or more, almost nothing separated, so it runs at 128 tokens with small rho.
- G2 at action cost 1 was decided by the cost, not by the types (section 3).
- G3's weights study was trivial, and at a fixed starting M of 5 all evidence between two TOOL classes came from one climb, so M now starts at random.

One known difference to expect: the pre-flight broke ties in the mark AUC arbitrarily, and this packet asks for mid-ranks.

## 6. Return to the theory thread

Write `docs/versions/v28-maps/THEORY_RETURN.md`. For each family: the map; one sentence on what the design fixed; one on what was open and what it returned; and the line of the 11 November handoff it bears on (G1: test 9 and the habit line; G2: set-point rules 2, 5 and 7; G3: rules 9 and 10). It names what was found and stops. The theory thread does the reading.

## 7. Pre-mortem

- **Your numbers differ from the pre-flight.** Section 0's rule: stop that family, report both, tune neither.
- **A reader is implemented a different way.** The reader table is the definition. If a line of it can be read two ways, say which reading you took in the report's first paragraph.
- **The reports multiply.** Three reports, one page, one return. A new question that comes up is one line under "not run" in the return.
- **K3 pulls again.** It stays held.
- **The machine sits idle.** It is reported as idle.

## 8. Triggers

The standing ones only.

## Sources

- **This repository, at `d27ff94d4`:** `AGENTS.md`; `docs/versions/v27-motives-and-set-points/FINAL_REPORT_2026-11-09.md`, `PACKET.md`, `reports/`; `results/v27/FINAL_VALIDITY_2026-11-09.json`; `docs/versions/v27-motives-and-set-points/k3-redesign-20261110/FINAL_REPORT_2026-11-10.md`, `DIAGNOSTIC_CHECKPOINT_2026-11-11.md`.
- **The theory handoff:** `claude/THEORY_HANDOFF_2026-11-11.md` (Round 12), and the theory thread's Round 12 messages to the curator.
- **The analyst's pre-flight of 11 November** (`preflight/`), and the independent check of the same day.
