# Ghost Scale kickoff: raise the limits, run everything that fits, most interesting first

16 October 2026, revision 2, after the curator's 14:17 answers. **For the Ghost Scale agent, from the curator.** Handing this document over is the curator's authorization for section 2. Record his handoff message verbatim, with its date and this document's SHA-256, as the authority. Read against Ghost `d84e023a2` (14:19 PDT). Nothing here changes V23's scientific criteria, deadline, retained failures or confirmation rules. It replaces the 13:00 analyst addendum.

## 1. What changed since 13:00

- **Total compute now binds.** At 21:16 UTC: 43.53 CPU-hours charged, 23.08 reserved, 18 protected, **5.39 ordinary unreserved**, against the 90-hour total.
- **R3's mandatory second burn-in** needs 7.64 CPU-hours (27,512 seconds), so it's held at the global reserve.
- **N1's completion profile** hit its 3,600 CPU-second allowance inside the first of 240 cells. That's 1,994 of 10,000 makers, and the family is resource-deferred.
- **Still held:**
  - N4's main grid: its repaired pilot hit its cap;
  - N7: it projects 18.3 CPU-hours per complete batch before fitting, against a 7-hour cap.
- **Twins:** no measured references yet. Sounding Line's kickoff makes card S0 its first job and delivers T1 and T4 today.
- **Workers:** at most 3 scientific workers, with 2 running at the snapshot.

The 13:00 addendum said total compute wasn't binding. That was true at 13:00 and stopped being true at 14:19. The limits are policy on the curator's own machine, and raising them costs nothing.

## 2. Resource supplement (in force on handoff)

**Why not a fourth reallocation file.** The 90-hour total and the three-worker limit live in V23's immutable `ACCEPTANCE.json`, the `LIMITS` table and the admission code. `admit_v23_repair.budget_limits()` raises on any `RESOURCE_REALLOCATION*.json` whose total isn't 90, whose workers aren't 3, or whose authority isn't the 15 October one. A fourth file in that format would block every admission, including N2's, N3's and R3's successors.

**Do this instead:**
1. **Write the supplement.** Create `ACCEPTANCE_SUPPLEMENT_2026-10-16.json` beside `ACCEPTANCE.json`. Never edit the original. The supplement carries the curator's handoff words, the date, this document's SHA-256, and the numbers below.
2. **Read it, bound by hash, for new admissions only.** Change the runtime, queue, stocking and admission code so new admissions read the supplement's total, worker limits and group caps. Packets already admitted keep their original limits and bindings.
3. **Keep the old checks.** Existing `RESOURCE_REALLOCATION` files must still validate exactly as before.
4. **Amend `AGENTS.md`.** Point its Gear 2 lines (three workers; the CPU ceiling) at the supplement.
5. **Test before any admission under it:**
   - the pending N2, N3 and R3 successors still admit;
   - the old reallocation files still validate;
   - a supplement with a changed hash refuses.

**The numbers:**

| Item | Value | Basis |
|---|---|---|
| Total | **285 CPU-hours** | Charged 43.5, reserved 23.1, R3 second batch 7.6, N7 45, N4 44, twins 40, confirmation 40, protected 18, and about 20 for setup, failures and reviews: 281 in all |
| Protected | 18 | Unchanged |
| N7 | 45 | Its fits admission reserves two complete batches (about 36.6) plus fits |
| N4 | 44 | A completion pilot first, because the repaired pilot capped before a complete profile existed. Then the main grid, sized from that profile. N4c (section 4) counts here |
| Twins | 40 | T1 to T4 once references land |
| Repairs | 48 | R3's second batch (7.64 hours), with margin |
| Confirmation | 40 | Only for confirmations whose sizes were frozen before any discovery result; sizes don't change |
| N1 | unchanged (16) | It stays resource-deferred (section 3) |
| Workers | 8 from 08:00 to 22:00 PDT, 10 overnight; **3 all day Wednesday 21 October** | Below-normal, one numerical thread each. Sounding Line keeps priority, and the 5% interference rule and memory floors still govern. Wednesday belongs to ToMpathy's rehearsal |
| Packet maximum | unchanged (4 hours) | Large batches already run in bounded partitions |

**Idle watchdog.** After 60 minutes with admissible work waiting and a free worker slot, admit the next admissible successor in the section 3 order, and log the event. The twelve-hour health review stays.

**Unchanged:** the deadline, the protected reserve, scientific criteria, retained failures, partition and replay rules, and the closure of V15, V21 and V22.

**Wall-clock.** About 240 CPU-hours remain under the new total, and mandatory work is due Tuesday 22:00 PDT, about 104 hours from now. That's under 3 workers on average; at 8 to 10 workers it fits with room to spare.

## 3. Order: most interesting first

Running packets keep their owners: N2's second batch and N3's repaired batch. When workers are saturated, new admissions follow this order:

1. **N4c,** the devaluation probe (section 4). The curator adopted this test today.
2. **N4.** The completion pilot, then the main grid. This is sacrifice ordering, his question 2: which goal is on top shows in what is sacrificed for it.
3. **N7.** Stated against revealed values. It bears on the essays and on stated goals as root cues.
4. **T1 and T4** as soon as S0 lands, then T3 and T2.
5. **R3's second burn-in.** It's mandatory and cheap.
6. **N8** (optional), then confirmation in the frozen order.

**N1 stays resource-deferred in V23.** Its fits admission reserves both complete-map burn-ins. Each burn-in is three times the sum, over cells, of measured cell cost times that cell's worlds. At the measured largest cell, that's thousands of CPU-hours: more than the machine can deliver before the deadline. Its deferral is already a terminal disposition, so don't re-admit it. N1 moves to V24 with a redesign (section 6).

**Twin inputs:**
- **T4.** Use Sounding Line's `T4.json`, built over all 123 released DICES raters. Its `T4_104.json` (the paper's 104 raters after quality exclusions) is a frozen sensitivity reference; run it only if capacity allows.
- **T2.** Fix `twin_chess.py` to read `values`, the key Ghost's own shape code emits, and add a regression test. Sounding Line delivers both `values` and an identical `seconds` array, so T2 works either way.

**Deadline rule.** Every mandatory family must reach a terminal disposition by Tuesday 20 October at 22:00 PDT, to leave Wednesday for confirmation. If the projection says otherwise, mandatory work moves ahead of N8 and the twins.

## 4. N4c: the devaluation probe

**Why.** The curator adopted animal learning's test today: a solution has become an end when it survives the need it served being met another way.

**What N4b showed.** One changed contingency separates the transition mechanism (a mistaken belief) from the intrinsic recipe (a learned end), with gains of 0.149 to 0.179 nats. Habit stays aliased.

**N4c adds a second probe:**
- **Probe A: the outcome arrives regardless of the action** (contingency degradation). A goal-directed solution stops; a habit and a learned end both persist.
- **Probe B: disturb or satisfy the candidate end itself.** A learned end is defended when disturbed and stops when satisfied. A habit is neither.
- **Prediction:** with both probes, all three mechanisms separate, and habit becomes identifiable.

**A fourth arm, if cheap: habitual goal selection.** People can pick a goal by habit and then plan toward it ([Cushman & Morris 2015](https://doi.org/10.1073/pnas.1506367112)). The signature: devaluing the goal's outcome stops the means, but the goal is still chosen when offered. That's "anything can be a set point" at the top of the tree.

**Rules:**
- Declare N4c before any run, under the same rules as N4b (12 settings), outside confirmation.
- If V23's frozen rules can't admit a new conditional study, N4c is the first V24 study.

## 5. Gear 3 for Ghost: no

- **V23 fits locally** once the limits rise. N1, the one family that doesn't fit, needs a redesign rather than money.
- **Cloud hosts break replay.** They break the platform match that byte-identical replay depends on. V21's rules already refuse Windows receipts on Linux hosts, and a port costs about a day.
- **If the machine saturates anyway,** families are resource-deferred with their measured costs, as V23 already does.

## 6. V24 design notes (not V23 work)

**N1, redesigned.** The full design needs thousands of CPU-hours. Search the 240 cells instead of enumerating them:
- **Successive halving.** Give every cell a few worlds, keep the most informative third, and triple the worlds for the survivors.
- **History matching.**
  - Fit an emulator per outcome on a space-filling design.
  - Drop cells whose implausibility, I(x)² = (E[f(x)] - z)² / (emulator variance + discrepancy + observation variance), is 3 or more.
  - Simulate only where cells survive.

**Other search methods:**
- **Simulation-based calibration for the direct readers.** Rank the truth among 100 posterior draws.
  - Uniform ranks mean calibrated.
  - A ∪ shape means overconfident.
  - A ∩ shape means overdispersed.

  Readers must emit samples or quantiles.
- **Growing seeds when promising.** When an interim result is promising, add worlds, up to twice the plan.
- **pymdp 1.0.0.** It's JAX-first and batch-oriented, which allows thousands of makers per batch. Its corrected message passing breaks replay identity with V23.

**New families from the curator's answers today** (recorded in `claude/THEORY_RECORD_2026-10-16_PM.md`):
1. **Needs and drives.** Set points (needs) are separate from drives, the pre-made solutions Panksepp named. One drive can serve two needs, and readers must tell which need a given use serves.
2. **Decaying set points.** A set point on a decaying state, such as a bond, produces steady action without the act being an end.
3. **Play narrows error bars.**
   - A PLAY drive acts to reduce uncertainty about the agent's own solutions in unfamiliar states, mainly when other deviations are small (animals play when fed, healthy and unstressed).
   - Shared play narrows each agent's error bars about the other.
   - Readers must tell play from pragmatic action by its dependence on uncertainty and on the other deviations.
4. **Editing protects higher-order goals.**
   - A generator with its own preferences fills in lower decisions.
   - The editor corrects only where higher-order goals are subverted.
   - Readers infer higher goals from corrections, and lower preferences from the generator.
5. **Readers converge with context.**
   - Two readers with different priors disagree less as shared evidence grows, provided their priors allow the same possibilities: their predictions of what comes next merge ([Blackwell & Dubins 1962](https://doi.org/10.1214/aoms/1177704456)).
   - Disagreement persists where one reader can't represent the maker's move.
6. **Controlled variables with disturbance.** N3's half-spaces, with disturbances added.
7. **Sacrifice ordering.** N4's main grid is the V23 version.
8. **Habitual goal selection,** carried over from N4c.

## 7. Errata

- **To the 13:00 addendum.**
  - "Total compute isn't binding" was true at 13:00 and false at 14:19.
  - Its amendment wording would have failed `budget_limits()`; section 2 replaces it.
- **To the fast-lane plan (G3-C).** The 200 to 500 cloud CPU-hours overstated the cloud need for everything except N1, and N1 needs a redesign, not a cloud.
- **To the twin plan.** The T1 to T4 references are a Sounding Line deliverable that no Sounding Line document listed. The Sounding Line kickoff's card S0 fixes that.

## Sources

**Ghost Scale `d84e023a2`:**
- `results/v23/CURRENT_STATUS.json`;
- `results/v23/n1-profile-failure-20261016/` (`CAPACITY.json`, `RESOURCE_CARD.json`, `PROFILE_COVERAGE.json`);
- `results/v23/reserve-boundary-20261016/`;
- `runners/admit_v23_repair.py` (`LIMITS`, `budget_limits`);
- `ghostscale/validation/soundingline/v23/extension_contract.py` (`cost_estimate`), `twin_contract.py`, `twin_writing.py` and `twin_chess.py`;
- `AGENTS.md` (Gear 2);
- `docs/versions/v23-research-division/`: `N1_RESOURCE_REPORT.md`, `RESERVE_CHECKPOINT_2026-10-16.md`, `RESOURCE_REALLOCATION*.json`, `TWIN_INPUT_CONTRACT.md`, `PREPARATION_REPORT.md` (N7) and `N4B_REPORT.md`.

**Methods and theory:**
- Habitual goal selection: [Cushman & Morris 2015](https://doi.org/10.1073/pnas.1506367112).
- Outcome devaluation: [de Wit & Dickinson 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2694930/).
- Merging of opinions: [Blackwell & Dubins 1962](https://doi.org/10.1214/aoms/1177704456).
- History matching: [Vernon, Goldstein & Bower 2010](https://durham-repository.worktribe.com/OutputFile/1510260); [Andrianakis et al. 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003968).
- Successive halving: [Li et al. 2018](https://jmlr.org/papers/volume18/16-558/16-558.pdf).
- Simulation-based calibration: [Talts et al. 2018](https://arxiv.org/pdf/1804.06788).
- [pymdp release notes](https://pymdp-rtd.readthedocs.io/en/latest/development/release-notes/).
