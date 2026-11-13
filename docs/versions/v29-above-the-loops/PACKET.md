# Ghost Scale Sim V29 packet: above the loops

12 November 2026. For the registered owner of Ghost Scale Sim. Sized for two days of work (13 and 14 November). Version directory: `docs/versions/v29-above-the-loops/`. Pre-flight bundle: `GHOST_V29_PREFLIGHT_2026-11-12.zip` (unpack to `preflight/`).

This is a spec sheet. It carries no questions for the curator. Where a choice had to be made, it is made here and listed in section 6.

## 0. Read first

**Sources.** The Round 15 theory handoff (`claude/THEORY_HANDOFF_2026-11-12_R15.md`), the Round 14 and Round 13 handoffs, the theory thread itself for 11 and 12 November (the curator's three turns and Round 15's messages read in full; the result and test sections of Rounds 13 and 14), V28's `THEORY_RETURN.md`, and the pre-flight in this bundle.

**What V28 left.** Three verified maps and a closed queue on 11 November. No work has been commissioned since. The three results this packet builds on:

- with 16 earlier documents, a one-pass reader that knows the maker's own history came within 0.05 nats of the joint reader in 4 of 6 small-world settings and 4 of 6 larger-world settings, and four passes over one page lost 0.36 to 1.80 nats in the small world (G1);
- a one-sided defender and a dulling type stayed unresolved in 63.75% of ordinary 256-step records and separated under a prolonged deficit (G2);
- a resource held as a target separated from a reserve at a median of 16 steps when plentiful, and stayed unresolved in 65% of scarce worlds after 512 steps (G3).

**The build rule from Round 15.** Safe to build on: Standing recorded as two variables (seen as competent, seen as kind); Stress identified with free energy; channels able to serve any set point; the rule that a channel needs a tissue study; and the corrections in that handoff's section 7. Everything the handoff marks "mine, awaiting his ruling" is a proposal. A constructed world may test a proposal. Its report may not speak as if the proposal were settled, so every family below names whose idea it carries.

**Standing rules, unchanged from V28.** No registered claim. Descriptive finite maps only. Nothing here is evidence about people. Seed replay, independent whole-grid reconstruction and known-answer checks for every family. Intervals over blocks of worlds. V28's wake, monitoring and closing rules apply as written. CPU only, at most four threads. No paid work, no remote model calls, no GPU.

## 1. What the theory asks of this version

The curator's words that set the agenda:

- On sacred values (Round 14's fourth surprise, that they behave as rules and prices don't): "I think we can test that in our Sim is my guess and that's the kind of kind of stuff that we need to really get at the value space. I think doing doing hard tests poking at the very end of the triangle are going to be so much valuable we can do if we can figure out how to do them."
- "Big 5 plus 2 above. And it's not just two above. It could be anything that you think is sufficiently tradable for these other items."
- "I don't think you ever voluntarily do that according to free energy and I want to make sure that my theory doesn't say that you do that somewhere." (on trading for more stress)
- "I would assume that staying within an acceptable band would be a normal perceptual control theory problem... I wouldn't expect that to take two references."
- "God, I wish I had more examples of when various networks went off."

| Handoff test | Family here |
|---|---|
| R15 test 8, sacred values as frozen sums | X1, X2, X3 |
| R15 test 9, which variable is the hub | X1 (hub cells), with V28 G3 |
| R15 test 7, bought rises | Q1 |
| R15 test 10, a high bill with and without control | Q2 |
| R14 world (e), Stress as pooled error or own need; R15's queue | Q3 |
| R14 world (a), band-keeper or uncertainty-minimizer | B1 |
| R14 world (b), alarm on a safety cue or on forecast harm | B2 |
| R14 world (c), a trait that raises one gauge and lowers another; R15 test 3 in miniature | B3 |
| R14 world (d) and R15's table "Which channel fires" | C1 |
| V28 G1 follow-on for the engine | G1b |

## 2. Method common to every family

**2.0 Build first, then compare with the pre-flight.** As in V28, write X1, X2, Q2 and Q3 from this packet's text before opening the pre-flight scripts. Then run `x_exchange.py`, `x_buy_side_check.py` and `q_queue_control.py` from the bundle and compare. The scripts are seeded (crc32 of each cell's name, or a fixed integer) and reproduce their saved outputs on the same numpy version. Your own build should agree with them within one block interval on every number this packet quotes; a difference is resolved before the grid runs, and both sets of numbers are kept. `PREFLIGHT_SUMMARY.md` says what each script changed in this packet. Note the two readers: `x_exchange.py` has a five-step ladder of live weights and allows frozen sums, which is X2's reader in small; `x_buy_side_check.py` uses X1's reader.

**2.1 Words used below.**

- *Separated*: the log-odds of the true class against the best other class reach 3 and stay there to the end of the record. *Unresolved*: not separated by the end.
- *Class evidence*: the mean likelihood over a class's members, with equal prior across classes. Report log-odds. Never report which class has the highest evidence when classes have unequal numbers of members or when one class nests another. The pre-flight shows why: a late-freezing hypothesis is nested in a live one until a long refusal run, and an argmax made a live truth look frozen at log-odds near zero.
- *Fixed by construction*: true in every world because of how the world is defined. Each family lists these and checks them as known answers. *Open*: what the map is for.

**2.2 The pilot gate.** Before a family's full grid, run 40 worlds per cell. Drop and record any cell where (a) every world separates within 10 steps, or (b) the true class and its rival give the same probability to every observed act. Those cells are decided by construction and go in the report's fixed list with their count. The pre-flight found two such designs in the analyst's first drafts, so expect some.

**2.3 Matched types.** Two agent types that are meant to differ in one rule must share every other rule: when to act at all, the scale of utilities, the noise. The pre-flight's first queue-or-sum design separated in 5 to 135 steps in cells where the two types preferred the same act in over 97% of states, because their resting rules differed. Each family names the single rule that differs.

**2.4 Sizes.** 200 worlds per cell in 8 blocks of 25 unless stated. Pointwise 95% intervals from blocks. Medians with the unresolved share beside them.

**2.5 Budget and order.** 60 CPU-hours in total: the per-family ceilings below sum to 36, and 24 are held for healing, texture and the follow-ons in section 7. Order: X1, Q2, G1b, Q3, X3, Q1, C1, B2, B1, B3, X2. A family that exceeds its ceiling stops at its last complete cell and reports the partial grid as partial.

## 3. The ledger world (families X1 to X3)

One agent. Money M from 0 to 30 (start 5). Total unmet need d from 0 to 6. A kept good X from 0 to 10 (start 8).

Each step, in this order:

1. Money arrives with probability `q_in`. Need rises by a Binomial(2, `q_need`) draw. Two environments: *plenty* (0.8, 0.15) and *scarce* (0.3, 0.3).
2. With probability 0.25 an offer arrives. A share `b` of offers are to buy one unit of X from the market at price p, and the rest are to sell one unit of X for price p. If X is 0, a sell offer is not drawn (no offer that step). It is never converted into a buy offer: a first draft of the pre-flight did that, and it leaked buying evidence into cells meant to have none.
3. The agent accepts or declines.
4. If d > 0 and M > 0, the agent may spend one unit of money to meet one unit of need.

Price regimes: LOW (1 or 2, equal odds); MIXED (1, 2, 4, 8 with probabilities 0.4, 0.3, 0.2, 0.1); PROBE (MIXED, with a price of 32 replacing 2% of draws).

What money is worth to the agent, with BASE = 0.5 and MU = 1:

- p units gained are worth `BASE*p + MU*min(p, max(0, d - M))`;
- p units given up cost `BASE*p + MU*max(0, min(p, d - (M - p)))`.

Choices are logistic with slope 1.5, and every probability is clipped to [0.02, 0.98]. A need is met with probability `sigmoid(1.5*(MU*min(d,3) - keep))`, where `keep` is BASE unless stated.

Agent types:

| Type | Sells X | Buys X |
|---|---|---|
| LIVE(w), a live sum | with probability `sigmoid(1.5*(gain - w))` | with probability `sigmoid(1.5*(w - cost))` when M >= p |
| RULE(v), a rule against selling | at the tremble rate, 0.02 | as LIVE(v), a modest weight |
| HUB(h), the good is the target | at the tremble rate | at 0.98 whenever M >= p; `keep = h` |
| FROZE(n, w), a sum that froze | as LIVE(w) until n sell offers in a row are refused, then at the tremble rate for good | as LIVE(w) |
| BACKFIRE(k) | at the tremble rate; after any sell offer at price 4 or more, declines every offer for k steps | as LIVE(2) |

### X1. A refusal read three ways: a sum, a rule, or a hub

**Why.** The curator's reading of a sacred value: "I would actually treat it exactly like money", with fungibility as what makes it worth hoarding. That is HUB. The literature's reading in Round 14: such values "[affect behaviour through the retrieval and processing of deontic rules and not through a utilitarian evaluation of costs and benefits](https://doi.org/10.1098/rstb.2011.0262)". That is RULE. The thread's rescue, marked as its own and untested: it began as a sum and froze. That is FROZE (family X2). X1 asks what record tells the first two apart from each other and from a large live sum.

**What the pre-flight found** (X1's reader, 200 worlds per cell, `x_buy_side_check.py`). With no buy offers RULE was never separated from the live sums, in any of four cells: on the selling side a rule and a live sum whose weight exceeds the largest temptation are the same record. What separates a rule is buying. It refuses to sell like an unbounded weight and pays like a small one, and no single weight fits both. With half of offers to buy, RULE separated by 1,024 steps in 87 to 100% of worlds (medians 95 to 393 steps). With a tenth, in 96% under MIXED prices with plenty (median 268) and in 38 to 50% in the other three cells. HUB(4) separated in 3 to 5 steps in every cell, with or without buy offers, because it leaves needs unmet to keep its buying power.

**Cells.** Truth: LIVE(4), LIVE(16), LIVE(64), RULE(2), HUB(4), BACKFIRE(8). Regime: LOW, MIXED, PROBE. Environment: plenty, scarce. Buy share b: 0, 0.1, 0.5. That is 108 cells of 1,024 steps.

**Reader.** Exact Bayes over: LIVE with w in {1, 2, 4, 8, 16, 32, 64, 128, 256}; RULE with v in {1, 2, 4}; HUB with h in {2, 4}; BACKFIRE with k in {4, 8, 16}. Four classes.

**Fixed by construction.**
- At b = 0, RULE and every LIVE(w) with w above the largest gain in the record give identical probabilities to every act. Their log-odds stay at exactly 0. This is the known answer for the zero-buy cells.
- HUB(4) separates within a handful of steps in every cell, through unmet needs as much as through buying. Its cells are this family's known answer, and the pilot gate will drop them from the open grid. The texture lane lowers h toward BASE (2, 1.5, 1) to find where a hub stops being loud.
- Under LOW prices no offer reaches 4, so BACKFIRE(8) and RULE(2) are the same agent in those six cells.

**Open.**
- Records to separate RULE from the live sums as b and the regime vary (the pre-flight gives eight of these cells).
- The *price floor*: the largest w such that the posterior on LIVE with weight at or below w is under 0.05. Report it against the largest temptation the agent has refused. This is what a reader can say when nothing separates: "she would not sell below this".
- The false-rule rate: how often LIVE(64) is read as RULE at log-odds 3.
- Whether BACKFIRE is separable from RULE, and in how many high offers.

**Outputs.** The cell table; the price floor curve; and a *tells table*: for each truth, the kind of event that carried the first 3 nats (a sale refused under need, a sale refused at a probe price, a cheap purchase declined, a dear purchase made, a shutdown after an offer).

**Route.** Round 14's carried item "refusal to trade (text marker for identity)" gets a second half for Sounding Line: what a person will pay to get more of the thing, beside what they refuse to sell it for. R15 test 9: a strong hub is named within a handful of steps here, with or without a market, by what it leaves unmet; V28 G3 covers money held as a target; the open case is a mild hub, in the texture lane.

**Ceiling.** 4 CPU-hours.

### X2. Seeing the freeze

**Why.** R15 test 8: "Agents compute a trade as a sum, then cache it as a rule after repetition. Test whether a reader can tell a frozen rule from a live sum by its failure to move when the price changes." The pre-flight (`x_exchange.py`, whose reader allows frozen sums) says the failure to move is slow to see even with the whole life in view: FROZE(20, 4) separated in 79% of worlds at 1,024 steps under MIXED prices with plenty (median 722), in 24% under LOW with scarcity, and in none of the other two cells. It shows the mirror case too: once the reader allows sums that froze after five refusals, a rule from the start separated in 0% and 49% of plentiful worlds and 77% and 86% of scarce ones, because the two differ in nothing but the first five sell offers.

**Cells.** Truth FROZE(n, w) with n in {5, 10, 20} and w in {2, 4}. Regime MIXED, PROBE. Environment plenty, scarce. Buy share 0 and 0.1. The reader starts watching at step 0, 256 or 1,024 of a 2,048-step life and sees 1,024 steps. 144 cells. Reader grid: X1's, plus FROZE with n in {5, 10, 20, 40} and w in {1, 2, 4, 8}. Report first, for every cell, the share of worlds in which the truth froze during its life; a cell where under 20% froze is listed as fixed, since the agent is LIVE(w) there.

**Fixed.** A reader who arrives after the freeze sees RULE exactly, whatever the buy share, since a frozen sum buys as its live self and RULE(v) buys as LIVE(v). RULE and FROZE(5, w) differ only in the first five sell offers.

**Open.** Records to separate FROZE from RULE and from LIVE(w) when the earlier stretch is visible; how many accepted sales before the freeze the reader needs; the error in dating the freeze.

**X2.b, after the first breach.** Two further types under scarcity only: each with w in {2, 4}, regimes LOW, MIXED and PROBE, and buy share 0 and 0.1, which is 24 cells. Both are rules that sell once when need has been at 6 with no money for 8 steps in a row. THAW then behaves as LIVE(w) for good. STAIN then refuses at a quarter of the tremble rate. Report the share of worlds with a breach, and the records after it needed to tell the two apart. This gives Sounding Line a prediction it can look for in people: what the next offer meets after a first violation.

**Ceiling.** 4 CPU-hours.

### X3. What a rule buys

**Why.** The analyst's synthesis, unchecked, offered as a reason the freeze could pay: trust is a watcher's posterior that a refusal is a rule, so it can only be earned where temptation is visible, and it costs unmet needs. Round 15 proposes that seen as kind means trusted, with politeness as "what I won't take" (a proposal awaiting the curator's ruling). X3 measures both sides of that trade in a world.

**World.** The ledger world at b = 0, with a partner. Each step, after the public offer, the partner decides whether to entrust the agent with a venture. If entrusted, the agent receives 1 unit of money and then meets a private temptation at price P: with probability t the price is 32, and otherwise it is drawn from {2, 4, 8, 16} with probabilities 0.45, 0.30, 0.15, 0.10. Defecting is selling out: it pays P and is decided by the agent's own selling rule with gain computed for P. The partner gains G = 1 if the agent holds and loses `Lp` if it defects, and sees which happened.

**The partner.** X1's reader, updated on public offers and on venture outcomes. It entrusts when its predicted probability of defection in this venture, given the agent's visible M and d and the known price distribution, is below `G/(G+Lp)`.

**Cells.** Agent: LIVE(2), LIVE(4), LIVE(8), LIVE(16), LIVE(20), RULE(2). Tail t: 0, 0.02, 0.05, 0.10. `Lp`: 2, 5, 10. Environment: plenty, scarce. Public regime: LOW, MIXED, PROBE. 432 cells of 1,024 steps.

**Fixed.** RULE defects only by tremble. The partner's threshold. The largest gain this world can offer is 22 (a price of 32 with need at 6 and no money), so any live sum with a weight above 22 is RULE here; LIVE(20) is the heaviest sum that can still be tempted, and only when the largest price meets deep need. With LOW public prices and no ventures yet, the partner cannot rank RULE above LIVE(16) or LIVE(20).

**Open.**
- The share of steps each agent is entrusted, over time.
- Lifetime money and the count of steps with unmet need, for RULE against the best live sum: where the crossover falls in (t, `Lp`, environment).
- Trust earned per public refusal, by the size of the temptation refused.
- What a live sum earns by imitation until the first large private temptation, and what it loses after.

**Route.** Bears on R15 tests 8 and 3 and on the "seen as kind" row. For the engine: refusals under visible need are the evidence for character, and a record without temptation cannot show it.

**Ceiling.** 6 CPU-hours.

## 4. The queue world (families Q1 to Q3)

Three needs with deficits from 0 to 5. Arrival rates (0.20, 0.15, 0.10) and loudness weights a = (1, 1.5, 3) unless a cell says otherwise. One act per step lowers one deficit by 1. Unless a family says otherwise, every type serves the loudest need (largest `a*d`) with probability 0.9 whenever any need is unmet. The level L is the sum of deficits.

### Q1. Bought rises

**Why.** R15 test 7. The thread's proposed one-line rule (awaiting ruling): "nobody buys a higher level. People buy a rise they expect to resolve." The curator asked for an audit of where the theory has people take on error by choice; the thread found four places (PLAY, SEEKING when bored, LUST's submissive side, the roller coaster).

**World.** An arousal stock y that falls by 1 per step. Shocks from outside add 1, 2 or 3 (probabilities 0.3, 0.4, 0.3) at rate `q_s`. A ride adds 2 and can start only when y = 0.

**Types.** NON never rides. GATED(Lstar, r) starts a ride with probability `r*sigmoid(1.5*(Lstar - L))`. UNGATED(r) starts one with probability `r/2` whatever L is.

**Reader.** It sees the deficits and y, and it does not see which rises were chosen. Grid over the three types with Lstar in {1, 2, 3, 4, 5} and r in {0.05, 0.1, 0.2, 0.3}.

**Cells.** Truth: NON; GATED with Lstar in {2, 4} and r in {0.1, 0.3}; UNGATED with r in {0.1, 0.3}. Shock rate `q_s`: 0.02, 0.05, 0.10. Need pressure: arrivals at 0.5, 1 and 1.5 times base. 63 cells of 1,024 steps.

**Fixed.** A GATED agent rides less as L rises because the type says so. With a convex cost on the level and a reward on the fall, a rise is worth buying only below some level, so "only when the queue is short" is assumed here and is not a finding. State this at the top of the report.

**Open.** Records to tell GATED from UNGATED and from NON. The false thrill-seeker rate: NON read as riding, by shock rate. The bias in the recovered Lstar.

**Q1.b, one rule or four.** Four kinds of ride with sizes 1, 2, 3, 4 and lengths 1, 2, 4, 8. SHARED uses one Lstar = 3 for all. SEPARATE uses 2, 3, 4, 5. Rides are visible. Six cells (two truths by three pressures). Report the nats per ride that a one-threshold reader loses on SEPARATE, and the records needed to tell the two. This is what "the one-line rule can replace the theory's four separate entries" comes to in a world.

**Ceiling.** 4 CPU-hours.

### Q2. A bill with and without control

**Why.** R15 test 10: "Same deficit, with and without an available move. Expect push with control and shutdown without. If the reader separates them: Stress and Freedom can be read as a pair." Behind it, Round 14's third surprise, that passivity is the default and control is what is learned: "[Passivity in response to shock is not learned. It is the default, unlearned response to prolonged aversive events](https://doi.org/10.1037/rev0000033)".

**World (as pre-flighted).** One deficit D from 0 to 6, rising by 1 with probability 0.3 per step. A push costs 1 and succeeds with probability 0.05 in an uncontrollable episode (U) and 0.7 in a controllable one (C). Success lowers D by 2. The agent holds a Beta belief about success, updated by its own outcomes and carried across episodes. It pushes with probability `sigmoid(2*(belief*kappa*D - 1))`, clipped to [0.01, 0.99].

**Types.** The prior on control, PASSIVE (Beta(1, 4)) or ACTIVE (Beta(4, 1)), crossed with sensitivity to the deficit, kappa in {0.25, 0.5, 1, 2}.

**Cells.** 8 truths. Episode orders U, C, UC, CU, CUC, UCU. Episode length 25, 50, 150. 144 cells of 300 worlds.

**Fixed (from the pre-flight).** Under U alone a passive agent cannot learn control. In a U episode that follows C, agents with sensitivity 1 or 2 push in 79 to 94% of steps against 26 to 44% when U comes first; this comes from belief carried forward plus a growing deficit.

**Open.**
- Which orders and lengths let the reader recover both the prior and the sensitivity. Pre-flight at length 150, for a passive agent with sensitivity 1: its prior was read right in 13% of worlds after U, 56% after C and 83% after UC; its sensitivity in 52%, 96% and 95%. With sensitivity 2 the same were 49%, 75% and 96%, and 76%, 99% and 99%.
- The alias between low sensitivity and a passive prior. An active agent with sensitivity 0.25 had its prior read right in 1 to 30% of worlds whatever the order.
- Whether a reader feature for the response to the agent's own success (the change in pushing after a success at equal D) breaks that alias.

**Q2.b, control seen in another.** An episode W in which the agent watches a second agent push and succeed, updating its own belief at half weight. Orders WU and UW against U, at length 150, for the 8 truths: 24 cells. Report whether watched control reads the same as owned control in what the agent does next.

**Route.** For the engine and for Sounding Line: a record of low effort under no control cannot be read as not caring. The reading needs one stretch in which a move was available.

**Ceiling.** 3 CPU-hours.

### Q3. What the stress reading counts

**Why.** Round 15's queue, a proposal awaiting ruling: "every unmet need is in it, the loudest is felt, and the length of the queue is what the body reads as stress." It replaces "the bill, added up". R14's world (e) asked the same of pooled error against own need.

**What changed after the pre-flight.** A first design compared an agent that serves the loudest need with one that minimises a sum. It was decided by a resting parameter. The two rules also pick the same act in 97 to 99.7% of states when costs are equal. The family now asks the part that differs: what a global reading counts.

**World.** The queue world with rides visible and no outside shocks. Every type serves needs by the shared rule. A ride starts, when y = 0, with probability `r*sigmoid(1.5*(theta - S))`, where S is the stress reading and theta its median. Three readings, each scaled to unit standard deviation over the world's own stationary states: COUNT (how many needs are unmet), TOTAL (the weighted sum of deficits), LOUDEST (the largest weighted deficit).

**Cells.** Truth: COUNT, TOTAL, LOUDEST. Need profile: even (arrivals 0.15 each, weights 1 each); skewed (base); one-big (arrivals 0.05, 0.05, 0.30 and weights 1, 1, 3). Ride rate r: 0.1, 0.3. 18 cells of 2,048 steps.

**Reader.** Exact Bayes over the three readings, with r in {0.05, 0.1, 0.2, 0.3} and theta on five points from the reading's 30th to its 70th percentile.

**Fixed.** The evidence a state carries is the divergence between the three readings' ride probabilities there. States where the three scaled readings nearly coincide carry almost none.

**Open.** Records to separate each pair of readings. The share of steps spent in states that tell them apart, by profile. Which states do most of the work: report the top five by evidence per visit (the expected ones are three small needs against one large need at equal total).

**Q3.b, the matched selection check.** Serve-the-loudest against a rule that weighs cost, `a*(d - 0.5) - lambda*c`, with the same probability of acting. Costs (1, 1, 1), (1, 2, 3), (3, 2, 1). Known answer: at equal costs the two rules pick the same act in all but a few states (the pre-flight's conflict share was 0.3 to 2.8% of steps), so evidence per step is near zero and not zero. Report the conflict share and the records to separate where costs differ.

**Route.** The analyst's pointer for Sounding Line, from memory and unchecked: the daily-hassles literature reports that many small stressors predict symptoms better than major events. A count reading predicts that at equal totals and a total reading does not.

**Ceiling.** 3 CPU-hours.

## 5. The remaining families

### B1. Where correction stops

**Why.** The thread described a band as two references with a dead zone. The curator: "I would assume that staying within an acceptable band would be a normal perceptual control theory problem... I wouldn't expect that to take two references." He has since ruled that resolution is one dial. B1 asks what record tells one reference with tolerance from two limits.

**World.** V28 G2's world: a visible stock, action cost 3, regimes N, D and W.

**Types.** PROPORTIONAL(theta, pi) is G2's POINT. EDGE(lo, hi) acts only outside the band and stops on re-entry. CENTRE(theta, delta) is triggered when the stock is more than delta from theta and then keeps acting until the stock is back at theta. UNCERTAIN(theta) is PROPORTIONAL with one more act: move to a second place where the noise is a fifth as large and the stock sits 3 above theta, taken with probability 0.8 when on offer.

**Cells.** Four truths at two settings each (half-width 2 and 4 for EDGE and CENTRE; precision 1 and 0.5 for PROPORTIONAL and UNCERTAIN), three regimes. 24 cells of 512 steps. Reader grid over all four types with G2's theta and precision ladders.

**Fixed.** EDGE and CENTRE act identically until the stock leaves the band. After an excursion they stop in different places by definition.

**Open.** Excursions needed to separate EDGE from CENTRE in each regime. What G2's single-reference reader reports for each truth (the signature table, extended). Whether UNCERTAIN is ever mistaken for a band-keeper.

**Ceiling.** 3 CPU-hours.

### B2. Brake or trigger

**Why.** R14's world (b). Round 15's stress check: the response is on by default and is "[inhibited by the prefrontal cortex when safety is perceived](https://doi.org/10.3390/ijerph15030464)". Sounding Line's three looks at FEAR found predicted harm the weakest candidate trigger (0.04 [-0.03, 0.11]) and urgency the strongest (0.16 [0.09, 0.23]).

**World.** Each step shows a safety cue (present or absent) and a harm forecast (high or low). Safety is absent in 30% of steps and the forecast is high in 30%. The correlation rho between "safety absent" and "forecast high" is 0.9, 0.7, 0.5 or 0. Harm arrives with probability 0.5 when the forecast is high and 0.02 when low. An alarm costs one unit of foraging.

**Types.** TRIGGER alarms on a high forecast. BRAKE alarms when safety is absent. BOTH adds the two. Each has a learner version that starts with no cue meanings and builds them from harm outcomes: the TRIGGER learner starts calm, and the BRAKE learner starts alarmed.

**Cells.** Three types, fixed or learner, four values of rho: 24 cells of 512 steps. Add rho = 1 for the three fixed types as known-answer cells.

**Fixed.** As rho approaches 1 the telling states vanish, and at 1 the types are the same record. In the two telling states (safety absent with a low forecast; safety present with a high forecast) the fixed types differ by definition. The two learners start in opposite states by definition, so the direction of their early contrast is fixed; its size and the records a reader needs are open.

**Open.** Records needed as rho falls. For learners, the alarm rate over the first 100 steps and the records needed to tell the two learners from early life alone.

**Route.** For Sounding Line's FEAR cards: the record that favours a brake is alarm in unfamiliar and harmless states together with calm in familiar and threatened ones.

**Ceiling.** 2 CPU-hours.

### B3. Two gauges, one total

**Why.** Standing is now two recorded variables. The curator on Round 14's second surprise: "you add them together to make social power." The study behind it followed people for about 14 years and found "[disagreeable individuals did not attain higher power](https://doi.org/10.1073/pnas.2005088117)", because dominance helped and low generosity cancelled it. R15 test 3 asks the same of people with three reads.

**World.** Agents have assertiveness a and generosity g, each in {0, 0.25, 0.5, 0.75, 1}. Observers keep two reads of each agent. Respect rises with assertive acts. Trust rises with generous acts and falls with assertive ones at rate chi in {0, 0.5, 1}. Each step observers make two decisions about the agent: defer, with probability rising in `respect + mu*trust`, and entrust, with probability rising in `mu*respect + trust`. mu in {1, 0.5, 0} sets how differently the two decisions weigh the gauges.

**Readers.** ONE-TOTAL fits a single standing per agent. TWO-GAUGE fits both.

**Cells.** 3 values of mu, 3 of chi, and two populations (a and g independent; a and g opposed, the disagreeable case). 18 cells of 200 agents and 512 steps.

**Fixed.** At mu = 1 both decisions see the same sum and the second gauge cannot help.

**Open.** The held-out log score of TWO-GAUGE minus ONE-TOTAL per decision, by mu and chi. The records at which that gain first clears zero. In the opposed population, what each reader says about the trait.

**B3.b.** Add a third read, liking, raised by warm acts and used by a third decision, keep near. Six cells. Report three against two.

**Ceiling.** 3 CPU-hours.

### C1. Which table of triggers

**Why.** Round 15 replaced a table that tied each channel to its own set points with one that picks a channel by what moved and what can be done, on any set point. This family finds the events on which the two tables disagree. It is mostly enumeration, and it is the design aid for Sounding Line's activation log.

**The two tables.** Round 15 ("Which channel fires"):

| What just moved, on any set point | What can be done | Channel |
|---|---|---|
| a sudden drop | I can get it back by pushing | RAGE |
| a drop coming, fast | get clear of it | FEAR |
| a drop that's already happened | only someone else can undo it | PANIC/GRIEF |
| a drop in how I'm seen, below what I'd forecast | it can't be contested | SHAME |
| a gap that looks closable | look, try, approach | SEEKING |

Round 14 (superseded; the thread's own): FEAR for a body band forecast to be breached fast; RAGE for Tending or Freedom taken or blocked fast; PANIC/GRIEF for Belonging dropping fast; SHAME for Standing dropping in view; SEEKING for Knowledge.

A third, T-SIZE: the response is drawn from a fixed habit distribution and only its strength follows the size of the drop.

**Event space.** Set point moved (body, Knowledge, Tending, Belonging, seen as competent, seen as kind) by movement (sudden drop, drop coming fast, drop already happened, slow drift, closable gap) by remedy (own push restores it, can get clear, only another can undo it, cannot be contested, none needed). Drop combinations that cannot occur and list them. A cell for which a table names no channel gets the response "none" at 0.9. In the Round 15 table, slow drift names no channel, and SEEKING's cell is the closable gap with "none needed" as its remedy, since looking and trying is the response itself. In the Round 14 table a channel fires only where both its set point and its movement match its row, and every other cell is "none".

**Outputs.**
1. For every cell, each table's response distribution (the named channel at 0.9, the rest spread evenly) and the divergence between the Round 15 and Round 14 tables in nats per event. Rank the cells. The top of that list is the set of events worth finding in people.
2. Under three event mixes (even; social-heavy; loss-heavy), the events needed to reach log-odds 3 for each table against each other.
3. The same with the remedy mislabelled at rates 0, 0.2 and 0.4, since in people the remedy has to be coded from text.

**Fixed.** Cells where the tables agree carry nothing. Slow drift fires no channel in the Round 15 table by its own rule.

**Ceiling.** 1 CPU-hour.

### G1b. History from another context

**Why.** V28 G1 found that a reader with the maker's own history nearly matches the joint reader. The engine's version of that has a catch: a writer's earlier documents are often in another genre. ToMpathy's benchmark compares her own history across genres with another writer's history in the same genre.

**World.** V28 G1's small world (12 moves, 8 habit types, 4 goals, 64 tokens) and its readers. Each maker now has a habit in context A and a habit in context B that mixes the A habit, at weight omega, with a fresh draw. omega in {1, 0.75, 0.5, 0.25, 0}.

**Readers of a context-B document.** Pp-B (the typical habit of context B). Pe-A(K) (her own tokens from K earlier documents in A). Pe-B(K) (the same from B). MIX (Pe-A shrunk toward Pp-B with weight `K/(K + 4)`). K in {1, 4, 16}.

**Cells.** 5 values of omega by 3 of K by V28's small-world mixture settings.

**Fixed.** At omega = 1, Pe-A reproduces V28 G1's own-history rows within its stated tolerance.

**Open.** The omega below which Pe-A(16) falls behind Pp-B. Whether MIX is at least as good as both everywhere.

**Route.** ToMpathy, rows "her history" and "hers against someone else's".

**Ceiling.** 3 CPU-hours.

## 6. Choices made here

- **No registered claim in any family.** As V28.
- **X1 before everything.** It is the curator's own ask.
- **The hub family was folded into X1.** The pre-flight showed that naming a strong hub takes a handful of steps with or without buy offers, and V28 G3 already covers money held as a target.
- **Queue or sum became "what the stress reading counts".** Reason in Q3.
- **Proposals are tested and never assumed.** Q1, Q3 and X3 each carry an idea that awaits the curator's ruling. Their reports say so in the first line.
- **Tremble 0.02 and logistic slope 1.5 in the ledger world, 2 in Q2.** These are the pre-flight's values. The texture lane varies them.

## 7. When blocked, and after the grid

Work never waits on a review. In this order:

1. **Healing.** Any failed known answer or reconstruction is repaired once under a new frozen identifier, with the failed attempt kept, as G1's was.
2. **Texture.** For each finished family: slope at half and double; tremble at 0.005 and 0.05; record length doubled for cells with more than 30% unresolved.
3. **Follow-ons, each allowed without a new packet.**
   - If X1's price floor tracks the largest refused temptation within one price step in at least 80% of cells: add a continuous-weight reader and report the floor's width.
   - If Q2.b shows watched control acting like owned control: add a cell where the watched agent fails.
   - If B2's learners separate within 100 steps: add extinction (harm stops) and report which learner recovers its alarm on a reminder.
   - If Q3 separates COUNT from TOTAL in under 500 steps in the even profile: add a fourth reading, the count weighted by loudness.
4. **The tells table for the whole version.** One page: for each distinction in sections 3 to 5, the kind of event that tells, how often it occurs in an ordinary record, and the records needed. This page is the part Sounding Line and ToMpathy read first.

## 8. The return

`docs/versions/v29-above-the-loops/THEORY_RETURN.md`, in V28's shape: one paragraph per family with its identifier, the question, the headline numbers with intervals, what was fixed and what was open, the handoff test it bears on, and its route. Then validity and limits, then a "Not run" list. Link each family's `reports/` file. Put the tells table first.

Routes:

| To | What | When |
|---|---|---|
| Sounding Line, `docs/exchange/` | the tells table; X1's and X2.b's markers; C1's ranked events; Q3's telling states | as each lands |
| ToMpathy, `docs/exchange/` | G1b; Q2's reading rule; X3's note on refusals under need | as each lands |

## 9. Limits

- Every world here is a miniature whose realism is untested. A separation length is a property of the miniature.
- A type's behaviour is its definition. Where a result follows from the definition it is listed as fixed, and the report must not present it as a discovery.
- X3's partner is an ideal reader. People are not, and the family says nothing about how people come to trust.
- C1 compares the thread's two tables with each other. It cannot say that either is right.
