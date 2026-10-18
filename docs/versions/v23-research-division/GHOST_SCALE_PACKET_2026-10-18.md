# Ghost Scale Sim packet, 18 October: directed probing before the deadline

*Spec sheet for the Ghost agent. It runs from handoff to Sunday 25 October, 22:00 PDT. Every decision is made here. Return to the curator only on the triggers in section 7.*

## 0. How to run this

- **The new studies are V23 conditional studies.** All eight conditional studies of the 17 October packet are complete and verified, and the queue is idle. N16 to N18 are conditional work, charged to the `conditional` group of the 17 October supplement, under its acceptance window, scientific criteria, protected reserve and deadline. Add each to `CONDITIONAL_MODULES` as the earlier studies were.
- **Each study follows the house sequence:**
  - declared protocol before sampling;
  - known-answer controls;
  - cost pilot and replay;
  - frozen fits;
  - one reserved batch;
  - full replay and independent law-block intervals.
  
  Resample laws, not cases.
- **Budget.**
  - 94 of 285 CPU-hours are charged. The `conditional` group's ceiling is 24 CPU-hours, enforced in code, and each packet is at most 4 CPU-hours. The eight finished studies used little of it, about 4.4 hours together with N8 and D10. Read the exact remainder from the resource ledger before admitting anything.
  - **Caps within that remainder:**
    - N16 12 CPU-hours, partitioned into packets of at most 4;
    - N16b 3;
    - N17 3;
    - N18 1.5.
    
    That is 19.5 in all, which fits if the remainder is at least that.
  - D5 is charged to repairs, at 1 to 2 hours, as planned. The protected reserve of 18 is untouched.
  - **If a cost pilot projects past its cap:** shed cells in the order each study lists, never controls.
  - **If the remainder is short after shedding:** write a new hash-bound supplement in the 17 October form. It adds a `directed` group of 20 CPU-hours inside the 285-hour total, with the matching `expected_caps` change in `resource_policy.py`, and is recorded like the earlier supplements. This packet authorizes it.
- **Purpose.** These studies answer engineering questions for ToMpathy:
  - which probe to pose next;
  - whether a probed falsifier has power;
  - whether a lean can be told from taste;
  - when playful records should count more.
  
  Each report states its route to the engine.

## 1. State on 18 October, 09:16 PDT (`0d3c867f3`)

**Conditional studies complete:** N4b, N4c, N14, N9, N13, N10, N11, N12, N15 and N4d.

**N13** (power of the falsifier, FX1):
- Power fails: 2 of 16 law blocks clear zero at 81 records, against 13 required.
- False support: 0 blocks in habit-only worlds.

**Probe studies separate mechanisms in constructed worlds** (gain in true-mechanism log loss):

| Study | What it probes | Gain (nats) | Limit |
|---|---|---|---|
| N12 | Overlays when recommendations conflict | 0.77 to 3.57 | Worse for the unchanged author-final mechanism |
| N4d | Conflict probes | 0.10 to 0.99 | |
| N11 | Reset probes | 0.27 to 0.72 | |
| N15 | Failures of both sources | 0.009 to 0.54 | |

**ToMpathy's tag is out.** The optional dependency `certified-fit-1` now exists on ToMpathy's remote, pointing at `5f3ad0a` (created 09:45 Sunday). The window runs to Monday 19 October, 12:00.

**Sounding Line's matching real-data finding** (PMAP, 45 writers):
- writers are identified 98% of the time from logged rhythm and 47% from style;
- 16% from how they handle suggestions, seven times chance;
- the content channel is uninterpretable: its control is invalid, and the taste instruments are blind.

The person shows far more in production than in selection. Passive choices carry little beyond a lean. That is why probes matter.

## 2. Studies, in priority order

### D5. The Stage 15 profile witness on the certified fit (existing plan)

- The tag is present, and the successor was built at 10:01 (`284549a28`). Run it first, as the 17 October packet specified, before Monday 12:00:
  - matched baseline;
  - donor comparison;
  - null precision fixed in advance on 40 null worlds;
  - 16 decisions per writer.
- Charge it to repairs, at 1 to 2 hours.
- T1's own successor stays in V24.

### N16. Directed probing: which probe next? (cap 12 CPU-hours, packets of at most 4)

**Question.** With a small probe budget, does choosing each next probe by expected information beat the frozen fixed probe order? Does that survive a misspecified model?

**Sources.** The frozen generators, mechanism sets, reliability laws, probe menus and exact readers of four studies:

| Study | Choice the policy makes |
|---|---|
| N4d | Stake orderings |
| N9 | Which scene, and whether to offer |
| N11 | Pressure, context shift or devaluation |
| N15 | Source A or B failure, at low or high stakes |

N12 has no probe choice; it is excluded.

**Design.**
- Reveal probes one at a time. After each, the exact reader updates its posterior over mechanism and reliability.
- Budgets: k = 1, 2, 3, 4 and 6 probes.
- **Policies:**
  - **P1:** the study's frozen fixed order;
  - **P2:** uniformly random probe choice;
  - **P3:** greedy expected information gain about the mechanism, computed exactly over the reader's finite posterior and each probe's outcome distribution;
  - **P4:** P3 with 20% of picks at random.
- **Misspecification arm.** The policy's posterior (not the scoring reader) either omits one true mechanism (each in turn) or uses only the four highest-reliability laws. The final scoring uses the study's exact reader.
- **Mitigation arm (P5).** P3 under the same misspecification with a tempered likelihood, every response treated as noisier than the policy believes (exponent 0.5). Inflating assumed noise removed active-learning bias in [Sloman et al. 2022](https://arxiv.org/abs/2205.13698).
- **Selection probabilities.** Log each probe's selection probability under every policy, so later estimates can be reweighted.

**Primary.** True-mechanism log-loss gain of P4 over P1 at k = 2, averaged over the four studies' settings, with law-block intervals. Direction: positive.

**Secondary:**
- the smallest k at which P4 matches P1 at k = 6;
- P3 against P2 under each misspecification. If P3 falls below random, that is the active-learning-bias failure;
- P5 against P3 under misspecification.

**Known-answer check.** In a world where one probe type alone separates two mechanisms, P3's first pick is that probe.

**Shed order if over cap:**
1. k = 3;
2. the reliability misspecification;
3. half the settings of the largest study.

**Route:**
- **P4 beats P1:** ToMpathy's scheduler and owner probe card choose by expected information, with this study's random share.
- **P3 falls below random under misspecification:** the random share, a tempered likelihood (if P5 repairs it) and a mechanism ensemble are mandatory.
- **Either way:** report the information value of each probe type, in nats per probe. ToMpathy's owner probe card uses those values to choose questions.

### N16b. Does probing give the falsifier power? (cap 3 CPU-hours)

**Question.** FX1 fails power on passive records. Does adding k directed conflict probes per maker (N4d's probe form, chosen by N16's P4) reach the frozen power requirement at 81 records?

**Design.**
- N13's exact design, laws, margin rule and verdict counts, with k = 2, 4 and 8 probe responses entering the goal reader's input.
- The records-only reader receives the same responses as ordinary records.

**Primary.** Blocks with a positive lower bound, out of 16, at 81 records and strength 1m. Required: 13.

**Gate.** Habit-only worlds must still give zero false SUPPORTED.

**Route:**
- **Powered:** V24's FX1 redesign includes designed probes.
- **Not powered:** FX1 stays INCONCLUSIVE; record the shortfall.

### N17. Lean, sensitivity and taste (cap 3 CPU-hours)

**Question.** In menus like CoAuthor's, can a reader separate three things?
- **Criterion:** a maker's lean, how readily she accepts.
- **Sensitivity:** how well her accepts track offer quality.
- **Taste:** a direction over a feature unrelated to quality.

And can a stakes probe and a base-rate probe tell a lean that comes from payoffs (a preference) from a lean that comes from beliefs?

**Worlds.**
- **Menus:** five offers, each with a latent quality and a feature value. Actions: accept one, or dismiss.
- **Makers:** each has a criterion c, a sensitivity d′ and a taste weight w.
- **Calibration to Sounding Line:**
  - within-writer agreement 42.18% against cross-writer 36.45% on matched menus (V5), with chance near 41%, so report the targets with chance beside them;
  - V5.d's Cohen's kappa;
  - acceptance intraclass correlation about 0.09 per decision (V3);
  - acceptance split-half reliability 0.78 across sessions (D12's canonical rerun).
- **Records per maker:** 20, 50, 100 and 250.
- **Lean mechanisms.** Two, crossed with the rest:
  - **payoff:** the lean is set by the costs of a miss against a bad accept;
  - **belief:** the lean is set by the believed base rate of good offers.
- **Probes:**
  - **Stakes probe:** the cost of a bad accept doubles. It moves a payoff lean.
  - **Base-rate probe:** offer quality is announced. It moves a belief lean.

**Readers:** lean only; criterion and sensitivity; criterion, sensitivity and taste; oracle. Directions are estimated on half the records and tested on the other half, which removes double-dipping. Sounding Line's V4.c showed that a split alone did not fix its false alarms, so the no-taste gate below decides.

**Primary.** Held-out log-loss gain of the taste reader over the criterion-and-sensitivity reader when w ≠ 0, at 100 records.

**Gate.** False taste detection at most 5% when w = 0, at every record count.

**Secondary.** Mechanism log-loss gain from one stakes probe plus one base-rate probe, payoff lean against belief lean.

**Shed order:** 250 records, then half the laws.

**Route:**
- **Criterion and sensitivity separate at about 100 records, with the gate passing:** Sounding Line's SDT1 runs.
- **Taste is also detected at 100 records:** report it. SDT1 still estimates only criterion and sensitivity.
- **The gate fails:** ToMpathy shows only the lean, labelled as a lean.
- **The probes separate the two leans:** ToMpathy's owner probe card gains the stakes question.

### N18. Play as a weak situation (cap 1.5 CPU-hours)

**Question.** Records made under weak constraint, such as play or fiction, may show the maker's own policy more clearly. But playful statements can invert their literal meaning. How accurate must a play detector be before playful records should count more than constrained ones?

**Worlds.**
- **Situational strength:** records come from contexts with strength s of 0.2 (play), 0.5 or 0.8. Behaviour mixes the norm with weight s and the maker's own parameter with weight 1 − s, plus noise.
- **Play inversion:** in play contexts, half the records invert their literal reading.
- **The detector** catches the inversion with accuracy a of 0.6, 0.8, 0.95 or 1.0.

**Readers:** ignores frames; uses the detector's output; oracle frames.

**Primary.** The break-even accuracy a* at which a play record carries more information about the maker's parameter than a strength-0.8 record. Report information per record for each s and a.

**Route.** ToMpathy weights playful passages up only when its play detector's measured accuracy exceeds a*. N8's play-reversal cost stays the warning.

## 3. The V23 report (Wednesday)

- Admissions close at 20:04:15 Wednesday. The report runs as planned, with N16 to N18 included as complete or as explicitly incomplete with their receipts.
- Lead the report with what the probes buy, in plain words: which probe types separated which mechanisms, by how much, and where aliases remain.

## 4. After the deadline (Thursday 22 to Sunday 25)

**V24 design is authorized,** with pilots up to 40 CPU-hours:
- the 17 October packet's V24 directions;
- **promote N16 to a V24 family** with a powered confirmation (ε-mixture readers, adaptive betting, sized worlds) if its primary is positive;
- **FX1's redesign** follows N16b's route.

**Limits:**
- Pilots only until Monday 26 October, 08:00 PDT. V24 production batches may start after that.
- Sounding Line's post-talk campaign shares the machine; keep at most two Ghost workers until Sunday 22:00.

## 5. Cross-repository

| To | What |
|---|---|
| ToMpathy, via `docs/exchange` | N16's per-probe information values and random share; N17's gate and probe result; N18's a* |
| Sounding Line | N16's information values for NP1; N17's result for SDT1 |
| Both | Keep `v23-module-gate-witnesses.json` current. Add N16, mapped to the engine's scheduler, and N17, mapped to the lean display |

## 6. Schedule (PDT)

| When | Work |
|---|---|
| Sun 18 | D5 now; protocols for N16, N17 and N18; known-answer checks; cost pilots |
| Mon 19 | N16 batch; N17 batch. D5 window closes at 12:00 |
| Tue 20 | N16b; N18; replays and intervals |
| Wed 21 | Last admissions before 20:04:15; the V23 report |
| Thu 22 to Sun 25 | V24 design and pilots |

## 7. Triggers (the only reasons to contact the curator)

1. A study would need more than its cap plus 50% after shedding.
2. A verified result reverses a delivered public claim.
3. The deadline machinery fails.

## 8. Pre-mortem

- **The policy only re-finds the fixed design.** The fixed orders may already be near-optimal for six probes. That is why the primary is at k = 2, where order matters most.
- **The exact reader is too kind.** Real engines are misspecified. That is why the misspecification arm decides whether a random share is mandatory.
- **N17 is calibrated to one aggregate per quantity.** Its record counts transfer only roughly to CoAuthor. Report the calibration targets beside every number.
- **N18 builds in its own answer.** If behaviour is defined as norm plus own, weak situations show more by construction. The useful number is a*, the detector accuracy that makes play worth reading. Say so in the report.
