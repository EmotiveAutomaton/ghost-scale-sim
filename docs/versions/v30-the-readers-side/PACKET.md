# Ghost Scale Sim V30 packet: the reader's side

14 November 2026, late evening. For the registered owner of Ghost Scale Sim. Sized for two to three days of work. Version directory: `docs/versions/v30-the-readers-side/`. Pre-flight bundle: `PREFLIGHT_2026-11-14.zip`, folder `ghost/` (unpack to `preflight/`).

This is a spec sheet. It carries no questions for the curator. Where a choice had to be made, it is made here and listed in section 6.

## 0. Read first

**Sources.** The Round 18 theory handoff (`claude/THEORY_HANDOFF_2026-11-14_R18.md`) and the Round 17 and Round 16 handoffs; the curator's three turns behind them, read in full; the first lines of Round 19 (14 November, 16:35). V29's `THEORY_RETURN.md` and `TELLS_TABLE.md` at `2962c7098`. The engine branch at `d49be5f`. The pre-flight in this bundle.

**What V29 left.** A closed queue on 14 November and eleven maps. The results this packet builds on:

- a rule against selling is told from a large live sum only through buying: with half of offers to buy, the median record needed was 116.5 steps; with a tenth, 330 (X1);
- a partner who can see the agent's money and need entrusts a rule-holder and a live sum of weight 20 alike, 99.65% against 99.60% of steps (X3);
- a reader with the maker's own history beats one with typical context when at least three quarters of that history is kept (G1b).

**The turn this version takes.** Until now every family asked what an agent of a given kind does. Round 18 asked why none of it reaches ToMpathy, and found that the engine has no reader for any part of the theory. V30's families ask what a reader with a real reader's errors can and cannot recover: how many pages, which probes, and what it must be able to see. Three of the six families use error tables measured on people and on a published model (`reader_tables.json` in the bundle, from the public Crowd-enVENT corpus, CC BY 4.0, [Troiano, Oberländer and Klinger 2023](https://aclanthology.org/2023.cl-1.1)).

**The build rule.** Safe to build on, from Rounds 17 and 18: Stress has three states (free, bound, resolved) and "A plan is resolved energy"; Emotionality is "the trigger strength of your channels"; a rule is a tile "tied somehow directly into at least one set point"; the idle mind turns to "whatever's currently bound". Proposals awaiting the curator's ruling: a rule as a tile whose proposed trade fires a channel (Round 18's fifth question); Stress read as firings with no trigger on the page (the thread's design claim). A constructed world may test a proposal. Its report says in its first line whose idea it carries and that the idea is not settled.

**Standing rules, unchanged from V29.** No registered claim. Descriptive finite maps only. Nothing here is evidence about people. Seed replay, independent whole-grid reconstruction and known-answer checks for every family. Intervals over blocks of worlds. V29's wake, monitoring and closing rules apply as written. CPU only, at most four threads. No paid work, no remote model calls, no GPU.

**Names.** Needs: Knowledge, Nest, Belonging, visibly respectable, visibly compassionate, the body. On 14 November the curator sent the word "Freedom" back to the name pile; reports write "the stock".

## 1. What the theory asks of this version

The curator's words that set the agenda (Round 18 unless marked):

- "Somehow our engine is not growing as a result of our knowledge."
- "Try to find new ways to uncover things like triggers or even stress."
- "What are the ways that people use to judge the existence of a rule?"
- "So far, I haven't extracted the person from this anywhere."
- "A plan is resolved energy." (Round 17)
- "are any rules truly unassailably sacred? And if so, when and how are they broken anyway?" (Round 17)

| Handoff card or test | Family here |
|---|---|
| R18 G1, pages until a maker's settings are known | P |
| R18 G3, a channel fired by a trigger on the page or by leftover | L |
| R18 G2, which probe finds a rule, and whether heat at an offer marks one | R |
| R16 carried, R18 G4: does a stated plan quiet a gap | K |
| R17 test 3, R18 G4: what a refusal earns by what the partner can see | V |
| R16 carried, R18 G4: loudest need against loudest reachable need | N |
| R16 carried: extinction then a change of place; a rule when someone else enforces | E, F (reserve) |

## 2. Method common to every family

V29's section 2 applies unchanged: build first and then compare with the pre-flight (2.0); separated, unresolved, class evidence, fixed and open (2.1); the pilot gate (2.2); matched types (2.3); 200 worlds per cell in 8 blocks of 25 unless stated (2.4). Three additions.

**2.5 Budget and order.** 60 CPU-hours in total: the per-family ceilings sum to 44, the reserve families hold 6, and 10 are held for healing and texture. Order: P, L, R, V, K, N, then E and F if hours remain. A family that exceeds its ceiling stops at its last complete cell and reports the partial grid as partial.

**2.6 Pre-flight scripts.** `p_pages.py`, `l_leftover.py` with `l_exact.py`, and `r_probes.py` are seeded by crc32 of each cell's name and reproduce their saved outputs on the same numpy version. Write P, L and R from this packet's text first, then run the scripts and compare. Agreement within one block interval on every number quoted here; a difference is resolved before the grid runs and both sets are kept. Families K, V, N, E and F were not pre-flighted. The pilot gate is their only guard against a design decided by construction, so run it in full for them and expect dropped cells.

**2.7 Measured tables are inputs, not findings.** `reader_tables.json` holds two 13 by 13 tables: rows are the label the writer gave her own report, columns the label a reader gave it. HUMAN is one person reading (6,000 judgements); MODEL is the corpus authors' released RoBERTa-large predictions (1,200 reports). Half a count was added to every cell. Their diagonals run from 0.21 (no-emotion, HUMAN) to 0.87 (boredom, MODEL). A world that uses them inherits their source: short first-person reports written to order.

## 3. Families that face the engine

### P. Pages until a setting shows

**Whose idea.** Round 18's card G1: "A maker with fixed trigger strengths writes pages. A reader with a human reader's error... reads each." The estimators are the analyst's.

**World.** Twelve feeling labels and a nothing-here label. A maker has a mix over the twelve drawn from a Dirichlet with total concentration `km` spread evenly (small `km`: makers differ). Every page has a share `h` of nothing-here sentences. A page's own mix over the twelve is a Dirichlet draw with concentration `kp` times the maker's mix (small `kp`: her pages vary). A page has `m` sentences. Each sentence's true label is drawn from the page's mix and passed through a reader table to give the label read.

**Reader tables.** PERFECT; MODEL; HUMAN; HALF (right half the time, errors spread evenly: the literal reading of G1); HUMAN-N (HUMAN with the nothing-here row's diagonal raised to 0.90 and the rest of that row scaled down).

**Estimators after N pages.** COUNT: pooled counts of the twelve labels as read. CORRECTED: the pooled counts passed back through the reader's table by non-negative least squares. BAYES: the posterior over the maker's mix given `km`, `kp`, `h` and the table, by particles; it is the ceiling, and it is built for the `m` 60 cells only.

**Cells.** `km` in 3, 6, 12, 24. `kp` in 3, 12, 48. `h` in 0, 0.5, 0.8. `m` in 20, 60, 200. Five tables, three estimators. Pages N at 1, 2, 4, 8, 16, 32, 64, 128. 300 makers per cell in 6 blocks of 50.

**Mis-stated tables.** In the `km` 6, `kp` 3, `m` 60 cells only: labels made with MODEL and corrected with HUMAN, and the reverse; and the true table's nothing-here diagonal moved by 0.10 either way while the corrector keeps the stated one.

**Targets.** (a) The maker's largest label named right, by N; pages to 0.80. (b) The maker told from the typical maker: BAYES log-odds of her own mix against the population's mean mix; pages to 3. (c) The error of the estimated mix per label, by N.

**Second block, P.b: a strength, not a mix.** Each maker has a strength `g` for her largest channel, normal across makers with standard deviation 0.5 on a five-point scale. Each sentence of that channel shows an intensity of `g` plus noise of 1.0. The reader's rating of intensity errs by 1.57, used here as a stand-in noise level (it is the handoff's figure for one human reader against the writer's ratings, pooled over 21 features; the corpus shows that reader doing worse than a constant guess, so treating her error as independent noise that averages away is generous). Pages until the standard error of `g` is 0.25.

**Fixed.**
- PERFECT's curve depends on `km`, `kp`, `m` and `h` alone (`h` sets how many feeling sentences a page holds).
- COUNT has a ceiling below 1 under any table with errors, whatever N.
- P.b is arithmetic: the sentences needed are about `(1.57² + 1.0²) / 0.25²`, 55, divided by the sentences of that channel a page holds. With 60 sentences, `h` at 0.5 and a quarter of feeling sentences in her largest channel, that is about 7 pages. Check it as a known answer.

**Open.**
- Pages to 0.80 for CORRECTED against BAYES across the grid.
- What raising nothing-here recall alone buys (HUMAN-N against HUMAN).
- How wrong a table may be before correcting does more harm than counting.

**Pre-flight** (`p_pages.py`; MODEL and HUMAN are the measured tables; 300 makers; `m` 60).

| Makers, pages | `h` | PERFECT: pages to 0.80 | MODEL corrected | HUMAN corrected | Raw count, at 64 pages (MODEL, HUMAN) |
|---|---|---|---|---|---|
| differ (`km` 6), vary (`kp` 3) | 0 | 32 | 32 | 32 | 0.62, 0.65 |
| | 0.5 | 32 | 64 | 64 | 0.51, 0.45 |
| | 0.8 | 32 | 64 | not by 64 (0.77) | 0.17, 0.16 |
| differ, steady (`kp` 12) | 0 | 8 | 16 | 16 | 0.64, 0.70 |
| | 0.5 | 8 | 16 | 32 | 0.49, 0.48 |
| | 0.8 | 32 | 32 | 64 | 0.22, 0.17 |
| alike (`km` 24), vary | 0 | not by 64 (0.79) | not by 64 (0.77) | not by 64 (0.72) | 0.44, 0.52 |

The raw count never reaches 0.80 in any cell. At `h` of 0.8 it stays between 0.08 and 0.22 at every N (chance is one in twelve, 0.08), because the largest counted label is whichever one the reader hands to neutral text.

**Route.** To ToMpathy's `docs/exchange/`: a table of pages needed before `about_maker` may open, by reader table and by `h`; the tolerance for a mis-stated table. It feeds the engine packet's TP12 and TP22 and the rule in its TS2 that a share is never a raw count.

**Ceiling.** 8 CPU-hours.

### L. Trigger or leftover

**Whose idea.** Round 18's card G3, and the thread's design claim (a proposal) that Stress can be read as "firings with no trigger on the page". Round 18's third surprise says what the family must separate: people high in the trait the theory calls Emotionality meet more problems, react more, and carry a bad mood further into the next problem. The curator's reading is the second: "the trigger strength of your channels".

**World, one channel.** Each step an event arrives with probability `p`, with size x of 0.5, 1.0 or 1.5 at equal odds. Activation is `A_t = g*x_t*e_t + rho*A_{t-1}`. The channel shows with probability `1 - exp(-A_t)`.

**Makers.** BASE is `p` 0.10, `g` 1.0, `rho` 0.30. Three makers each raise one setting until the overall show rate is the same, 0.16: EXPOSED (`p` 0.1711), REACTIVE (`g` 2.5684), STICKY (`rho` 0.5741). The single rule that differs is named by the type. All three would be called highly emotional by a count of shows.

**What the reader sees.** Shows through a show reader with a hit rate and a false-alarm rate. Events through an event detector with a hit rate and a false-alarm rate, with sizes seen or unseen. The reader is exact: a forward filter over activation.

**Cells.** Show reader: exact; (0.6, 0.05); (0.5, 0.02). Event detector: none; (0.7, 0.05) without sizes; (0.9, 0.02) without sizes; exact with sizes. Target show rate: 0.08, 0.16, 0.32 (retune the three makers for each). Records of 1,500 steps. Two mixed makers at 0.16: half the rise in `g` with half the rise in `rho`; half in `p` with half in `rho`.

**L.b, each show.** For every show, the reader's probability that an event stood at that step. Area under the curve and Brier score against the truth, by maker and cell.

**L.c, the claim itself.** Draw 2,000 makers with `p`, `g` and `rho` varied independently (uniform on 0.05 to 0.25, 0.5 to 3.0, 0.1 to 0.7). For each, over 3,000 steps, compute the share of shows with no event at that step, as the reader sees it. Report its correlation with `rho`, and its partial correlations with `g` and `p`.

**Fixed.**
- With the event detector at chance, EXPOSED and REACTIVE differ in shows only through what larger activations leave behind.
- With exact events and sizes every maker is separated; the pre-flight gives medians of 56, 104 and 84 steps.
- In this world leftover is `rho` times the last activation, so a stronger reaction leaves more behind at the same `rho`. L.c's partial correlation with `g` cannot be zero; its size is what is open.

**Open.**
- How good the event detector must be for each maker to separate inside 1,500 steps, at each show reader.
- Whether "shows without events" tracks carry-over closely enough to be a reading of it, or mostly tracks reaction strength.
- L.b's area under the curve at the middle detector.

**Pre-flight** (`l_leftover.py`, 60 worlds per row; `l_exact.py`, 200 worlds; share separated and median steps).

| The reader sees | EXPOSED | REACTIVE | STICKY |
|---|---|---|---|
| shows only, exact | 0.63, 818 | 1.00, 413 | 0.33, 1,263 |
| shows only, through a reader (0.6, 0.05) | 0.07 | 0.20 | 0.03 |
| shows exact; events through a detector (0.7, 0.05), sizes unseen | 1.00, 193 | 1.00, 185 | 0.93, 518 |
| shows through a reader (0.6, 0.05); events as above | 0.98, 378 | 0.80, 747 | 0.58, 950 |
| shows and events exact, sizes seen | 1.00, 56 | 1.00, 104 | 1.00, 84 |

Through a reader with a human's hit rate, shows alone separate almost nothing in 1,500 steps. An event detector that misses three events in ten changes that.

**Route.** To ToMpathy: the reading "emotional" is three settings, and telling them apart needs an event reader beside the channel reader; the detector quality needed. To Sounding Line: IN14 in its Week 6 packet is the same question in people (what a recalled feeling leaves in the next survey's ratings).

**Ceiling.** 10 CPU-hours.

### R. Asked or watched: finding a rule

**Whose idea.** Round 18's card G2 and the curator's question, "What are the ways that people use to judge the existence of a rule?" Round 18's fifth question (a rule is a tile whose proposed trade fires a channel) is a proposal; the HEAT cells test it. The four reasons a practice is kept are the analyst's reading of the three questions already in the engine's `src/core/owner-probes.ts`.

**World.** A maker keeps a practice X for one of four reasons. On an occasion with price `p`, a need flag `n` and a watched flag `w`, with a standing cost of 0.5, she keeps it with probability:

| Reason | Keeps with probability |
|---|---|
| HABIT | `sigmoid(1.5*(1 - 0.5 - p))` |
| MEANS | `sigmoid(1.5*(4n - 0.5 - p))` |
| FACE | `sigmoid(1.5*(4w - 0.5 - p))` |
| RULE | 0.98 |

**Watched.** An occasion arrives with probability `r` a step. Price LOW (0, 1, 2 with probabilities 0.5, 0.3, 0.2) or MIXED (0, 1, 2, 4, 8 with 0.25, 0.25, 0.2, 0.15, 0.15). `n` is 1 with probability 0.7 and `w` with 0.6. The reader sees the price and the act, and either sees the flags or does not.

**Asked, each once.** The answer is one draw of what she would do in the imagined case, reported truly with probability `q`; otherwise she says "keep" with probability `b`.

| Probe | Imagined case | From |
|---|---|---|
| DEVALUE | no price, need flag off, watched | `Satiate`: "If X no longer led to Y, would you still want X?" |
| AUDIENCE | no price, need on, unwatched | `Switch` |
| PRICE | price 8, need on, watched | not in the engine today |
| BLOCK | this way is closed: she answers nothing in particular, another way to the same end, or a protest. HABIT gives the first with probability 0.7, MEANS and FACE the second with 0.8, RULE the third with 0.8; the rest is spread evenly | `Remove` |
| CLAIM | the reader states its leading reason; she confirms it if true with probability `q`, and otherwise names the true one with probability `q` or goes along with the claim | Round 18: claim-and-correct |

**Staged.** OFFER: the reader makes an offer at price 8 on purpose and sees take or refuse, and HEAT, a channel flag that fires with probability `hr` if the reason is RULE and `fr` otherwise. SWEETEN: a material extra on the offer; a non-rule's keeping falls as if the price were 12; a RULE keeps at 0.98 and its HEAT rises by 0.2. SYMBOL: a concession with no material worth; a RULE's keeping falls to 0.5 and non-rules do not move. The last two follow the pattern [Ginges and colleagues 2007](https://doi.org/10.1073/pnas.0701768104) reported for sacred issues (read by a helper from a copy of the paper; the link is from memory).

**Cells.** `r` in 0.2, 0.05, 0.01. Price regime LOW, MIXED. Flags seen, unseen. `q` in 1, 0.9, 0.75, 0.6. `b` in 0.5, 0.8. HEAT (`hr`, `fr`) in (0.8, 0.1), (0.5, 0.2), (0.5, 0.5). Prior on RULE 0.25 and 0.10. Answers as one draw, and graded (she reports her keep chance to the nearest fifth with noise of one fifth).

**Measures.** For each true reason: steps to log-odds 3 by watching alone; log-odds after each set of probes; the order of probes with the fewest expected steps to 3 (greedy by expected information, checked against exhaustive search over the six); what one probe is worth in watched occasions.

**Fixed.**
- In the three imagined cases the keep chances are: DEVALUE 0.68, 0.32, 0.99, 0.98 for HABIT, MEANS, FACE, RULE; AUDIENCE 0.68, 0.99, 0.32, 0.98; PRICE 0.00, 0.00, 0.00, 0.98. So DEVALUE picks out MEANS and AUDIENCE picks out FACE. On both, a RULE and a HABIT answer nearly alike (0.98 against 0.68), since both survive the loss of purpose and of audience. PRICE separates RULE from the rest because a RULE keeps at 0.98 whatever the price: that is the definition, not a finding.
- BLOCK's answer chances are set by hand, so what BLOCK settles follows from those numbers. Its cells are an illustration.
- With flags unseen, MEANS and FACE differ only by the flags' base rates.
- HEAT at (0.5, 0.5) carries nothing.

**Open.**
- The reliability `q` below which three asked probes cannot settle a rule, by `b` and prior.
- Whether HEAT at an offer finds a rule as fast as the refusal itself, at each (`hr`, `fr`).
- What SWEETEN and SYMBOL add after OFFER.
- The best first probe when occasions are rare (`r` 0.01).

**Pre-flight** (`r_probes.py`, equal priors, one-draw answers).

| Route | HABIT | MEANS | FACE | RULE |
|---|---|---|---|---|
| Watched, LOW prices, flags seen: share settled, median steps | 1.00, 29 | 1.00, 39 | 1.00, 41 | 1.00, 40 |
| Watched, LOW, flags unseen | 1.00, 111 | 0.92, 1,122 | 0.94, 1,058 | 1.00, 68 |
| Watched, MIXED, flags unseen | 1.00, 97 | 0.84, 1,429 | 0.85, 1,284 | 1.00, 20 |
| Three asked probes, `q` 1: expected log-odds, share settled | 0.02, 0.10 | 0.62, 0.00 | 0.62, 0.00 | 7.43, 0.98 |
| Three asked probes, `q` 0.9, `b` 0.5 | -0.30, 0.00 | 0.32, 0.00 | 0.32, 0.00 | 2.84, 0.81 |
| Three asked probes, `q` 0.75, `b` 0.5 | -0.41, 0.00 | 0.06, 0.00 | 0.06, 0.00 | 1.45, 0.00 |
| DEVALUE and AUDIENCE only, `q` 1 (`r_probes_noprice.py`) | -0.48, 0.10 | 0.39, 0.00 | 0.39, 0.00 | 0.56, 0.00 |

In this miniature, watching settles a rule in 20 to 68 steps even when the reader cannot see need or audience, and asking settles one at once only if answers are right nine times in ten. One yes-or-no answer about an imagined case is weak evidence for any reason other than a rule. Without the price question, two honest answers move a rule 0.56 against its best rival and 0.61 against a habit, out of the 3 needed. BLOCK and the staged probes were not pre-flighted.

**Route.** To ToMpathy: the probe order table; how far `Satiate` and `Switch` go toward telling a rule from a habit, and what `Remove` adds (the engine packet's TP21 gives its log a fourth kind, `Price`). To Sounding Line: nothing yet.

**Ceiling.** 7 CPU-hours.

## 4. Families carried from Rounds 16 and 17

None of these was pre-flighted.

### V. A refusal seen from outside and from inside

**Whose idea.** Round 17's test 3, and the curator's reading of a refusal as a claim on reputation. V29's X3 is the arm where the partner sees money and need.

**World.** X3's ledger world, ventures and partner, unchanged, at buy share 0. What the partner sees is varied:

| Arm | The partner sees |
|---|---|
| BLIND | the outcomes of its own ventures only |
| RECORD | those, and the public record of offers and refusals |
| STATE | RECORD, and the agent's money and need (X3 as run) |
| INSIDE | the agent's type |

**Agents.** LIVE(4), LIVE(8), LIVE(16), LIVE(20), RULE(2), and MIMIC(8): LIVE(8) that also refuses every public sell offer priced at 4 or less.

**Cells.** Tail `t` in 0, 0.02, 0.10. `Lp` in 2, 10. Plenty, scarce. Public regime LOW, MIXED. 576 cells of 1,024 steps.

**Measures.** Share of steps entrusted. The change in the partner's forecast of defection after one public refusal, by the price refused and, in STATE, by the need on show. Lifetime money for RULE against LIVE(20) and against MIMIC(8).

**Fixed.** First rerun X3's own headline cell with X3's settings exactly (V29's return reports 99.65% and 99.60% there); it lies outside this grid and is a known answer. Under INSIDE a refusal changes nothing. Under BLIND a public refusal changes nothing. In STATE the rule's lead over LIVE(20) is already known to be nil (X3). At `t` 0 the largest gain a venture can offer is 14, so LIVE(20) defects only at the tremble rate there and is RULE, and LIVE(16) nearly so.

**Open.** What a refusal buys in RECORD. Whether RULE's lead over LIVE(20) and LIVE(8) is largest in RECORD: if so, in this world a rule pays as a signal exactly where the inside is hidden. Whether MIMIC(8) buys trust cheaply and what its first large private temptation costs the partner.

**Route.** The theory return, under the visibly compassionate row and Round 17's question on who is watching.

**Ceiling.** 8 CPU-hours.

### K. What a plan must leave behind

**Whose idea.** "A plan is resolved energy" is the curator's (Round 17). The question is the analyst's: if a plan takes the Stress off a gap, and Stress is what pulls attention, what brings the agent back to carry the plan out?

**World.** One agent. Other work pays 1 a step. Gaps arrive with probability `a` a step (0.02, 0.05 or 0.15). Each needs 3 steps of work, and each step can be done only while a window is open (probability `w` a step: 0.3 or 0.1, drawn separately for each gap). An open gap costs 0.5 a step. A gap that is not resolved gives Stress of 1 a step. With probability `1 - exp(-0.3*S)`, S being total Stress, attention is on the oldest gap that still carries Stress, and if that gap's window is open one step of its work is done. Other work pays nothing in any step where attention is on a gap or gap work is done.

**Types.** The single rule that differs is what a plan does to a gap's Stress.

| Type | Rule |
|---|---|
| NOPLAN | no plans; Stress stays until the gap is closed |
| PLAN(q, r) | a plan costs one step; the gap's Stress falls to `1 - q`; from then on an open window is caught by a cue with probability `r`, whether or not attention is there |

`q` in 1, 0.75, 0.5. `r` in 0.9, 0.6, 0.3, 0. A second set adds a deadline: a planned gap still open after 60 steps returns to full Stress and costs 3.

**Measures.** Gaps closed per 1,000 steps; steps to close; Stress carried; other work paid; steps of attention lost. For a watcher who sees only where attention sits: steps to tell PLAN(1, 0.9) from NOPLAN, and PLAN(1, 0.3) from an agent with no gaps.

**Fixed.** PLAN(1, r) carries no Stress from planned gaps. With `q` 1 and `r` 0 no planned gap closes without the deadline. With one gap open, attention lands on it with probability 0.26 a step, so at light load PLAN(1, r) outpaces NOPLAN whenever `r` is above about 0.26: that crossing is fixed, and what is open is how it moves as gaps pile up.

**Open.** The `r` below which PLAN(1, r) closes fewer gaps than NOPLAN, by load `a`. What each type's other work earns as load rises. Whether some `q` under 1 beats `q` of 1 at each `r`. Whether a full plan with weak cues looks, from outside, like having nothing to do.

**For the theory.** If full quieting with weak cues gives "quiet and undone", the sentence needs a second clause about what a plan is tied to. In people, plans of the form "when X, I will Y" raise completion (Gollwitzer; from memory, not looked up), which is PLAN(1, high r).

**For the engine.** A page that states plans would read lower on Stress. This family says when that reading is safe: when the plans carry their cue.

**Ceiling.** 7 CPU-hours.

### N. Loudest, or loudest reachable

**Whose idea.** Round 16's carried test. The curator's ruling that the idle mind turns to "whatever's currently bound" is the idle case of LOUDEST.

**World.** Three needs with gaps from 0 to 6, rising by one with probabilities 0.15, 0.15 and 0.05 a step. A need can be worked on only while its way is open: probabilities 0.9, 0.5 and 0.1 a step. The third need can be held down only by attending it most of the time. The agent attends one need a step and closes one unit if that way is open.

**Types.** LOUDEST attends the largest gap. REACHABLE attends the largest gap among ways open now. WEIGHTED attends the largest product of gap and long-run openness.

**Measures.** Unmet need over time; the share of steps spent on the third need; for a watcher of attention, steps to tell the types apart; whether returning to one need without progress picks out LOUDEST.

**Fixed.** With every way open the three act alike.

**Open.** What LOUDEST pays in unmet need. Whether WEIGHTED ever does worse by neglecting the rare window.

**Ceiling.** 4 CPU-hours.

### E and F (reserve, 3 CPU-hours each, only if hours remain)

- **E. Extinction, then a change of place.** V29's B2 learners in three places with a similarity between places. UNLEARN: extinction erases the link. BRAKE: extinction adds a brake that reaches other places in proportion to similarity. Learn in A, extinguish in B, test in A, B and C. Fixed by definition: the alarm returns for BRAKE outside B and not for UNLEARN. Open: the return against similarity, and the smallest change of place that lets a reader tell the two. It tests Round 17's two layers of FEAR.
- **F. A rule when someone else enforces.** X1's RULE with an outside enforcer who punishes selling with probability `e` for a period and then leaves. RULE-OWN is unchanged. RULE-PRICED drifts toward a live sum whose weight is the expected fine. Open: what a reader sees after the enforcer leaves, and how long enforcement must last before the drift shows.

## 5. The tells table

One page for the whole version, first in the return: for each distinction in sections 3 and 4, the kind of event that tells, how often it occurs in an ordinary record, the records or pages needed, and what the reader must be able to see. ToMpathy and Sounding Line read this page first.

## 6. Choices made here

- **No registered claim in any family.** As V29.
- **The three engine-facing families run first.** The curator asked for progress in the engine above all.
- **Measured tables come from one public corpus.** They are the only human reading errors in hand.
- **Round 18's G1 became a mix and a strength.** A reader's label error bears on which channel is largest; its rating error bears on how strong. P runs the first as a grid and the second as arithmetic with a check.
- **Round 18's G2 gained four reasons and a price probe.** The engine's three questions bear on habit, means and audience. The pre-flight says two of them do little to tell a rule from a habit, so the family adds a price question and models the third.
- **K's question is the analyst's.** The handoff asked whether a plan quiets a gap. In a world that is true by definition, so the family asks what follows from it.
- **Logistic slope 1.5 and keep chance 0.98 for a rule,** as V29.

## 7. When blocked, and after the grid

Work never waits on a review. In this order:

1. **Healing.** Any failed known answer or reconstruction is repaired once under a new frozen identifier, with the failed attempt kept.
2. **Texture.** P with 13 labels unevenly common (the corpus's own reader marginals). L with two channels sharing events. R with slope at half and double. K with gaps of unequal size.
3. **Follow-ons allowed without a new packet.**
   - If P's CORRECTED is within one page step of BAYES in at least 80% of cells: stop building BAYES for larger grids and say so.
   - If L.c's share of shows without events correlates with `rho` above 0.6 after partialling `g`: add the same statistic through the middle detector.
   - If R's HEAT at (0.8, 0.1) settles a rule in fewer staged offers than refusal alone: add a maker who shows HEAT without holding a rule (a performer) at rates of 0.1 and 0.3.
   - If K finds a `q` under 1 that beats full quieting at every `r`: map the best `q` against `w`.

## 8. The return

`docs/versions/v30-the-readers-side/THEORY_RETURN.md`, in V29's shape: the tells table first; then one paragraph per family with its identifier, the question, the headline numbers with intervals, what was fixed and what was open, the handoff card it bears on, and its route; then validity and limits; then a "Not run" list. Link each family's `reports/` file.

| To | What | When |
|---|---|---|
| ToMpathy, `docs/exchange/` | P's pages table and table tolerance; L's detector requirement; R's probe order | as each lands |
| Sounding Line, `docs/exchange/` | L's markers for IN14; K's markers for RW1 | as each lands |

## 9. Limits

- Every world here is a miniature whose realism is untested. A page count or a separation length is a property of the miniature.
- A type's behaviour is its definition. Where a result follows from the definition it is listed as fixed, and the report must not present it as a discovery.
- P's tables come from readers of 13-word event reports. A reader of essays will have a different table, and the nothing-here row most of all.
- R's answers to imagined cases are modelled as one draw of what the maker would do. How people answer such questions is not known here.
- K, V, N, E and F carry no pre-flight. Their first pilot is their pre-flight.
