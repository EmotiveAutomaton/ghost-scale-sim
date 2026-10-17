# Ghost Scale packet, 17 October: clear the boundary, then eight small worlds

17 October 2026. **For the Ghost Scale agent, from the curator.** Handing this packet over is the curator's authorization for everything in it, within section 9. It decides every item the twin-profile review left at its boundary, and commissions eight conditional studies in the N4c form. It amends the October 16 revision-2 addendum where it says so; everything else in it stays. Nothing here waits on a decision: where options were close, the cheaper one that loses nothing runs, with its reason. Record the curator's handoff message verbatim, with its date and this packet's SHA-256.

## 0. How to run this

1. **Write the dispositions and the supplement first** (section 2, items D1 to D7). That clears the mandatory set, so N8 becomes admissible and the discovery freeze can follow.
2. **Admit N8,** then build and admit the conditional studies in section 4's order, one after another, inline. Ghost's subagent rule stands: one at a time, for search only.
3. **Each queue-depletion review admits the next commissioned study** before it acknowledges anything. An idle machine with commissioned work is not a terminal state.
4. **Freeze discovery at the first moment N8 is terminal and no packet is running** (`freeze_v23_discovery.py` refuses while any packet is unfinished), and no later than Tuesday 20 October, 22:00 PDT. Conditional studies can still be admitted after the freeze, since admission checks only the acceptance window, so freeze in the first gap between them. Then bound confirmation's expected evidence before admitting it (D8). As the arithmetic in section 1 stands, the bound records confirmation infeasible rather than spending 40 CPU-hours.
5. **Publish each study** as N4c was: report, READING_INTENT row, FINDINGS, exchange note. The last admission is before Wednesday's `report_start`.

## 1. State read (17 October, 10:10 PDT, `6162cc1af`)

- **V23 is active and idle.** 0 workers, all 113 packets terminal. Charged: 89.52 of 285 CPU-hours, of which 18 are protected, leaving about 177.5 ordinary hours. Actual concurrency 3.
- **Nominated:** N5 (0.949 nats), R4 (0.643), N7 (0.250), R2 (0.174), R1 (0.031).
- **Terminal non-nominations:**
  - stopped below margin: N2, R3;
  - primary unavailable: N3, N6.
- **Conditional studies complete:** N4b, N4c. N4c's combined probes gain 1.127 to 1.252 nats. With actions alone, the goal-directed and habitual-goal rules are exactly aliased.
- **At the boundary:**
  - **T1:** its writing twin's pinned `Backbone.fit` reports ABNORMAL on a converged ridge fit. The gradient is 1.16e-5, the objective gap is at most 6.8e-12, and the minimum Hessian eigenvalue is 20.
  - **T3 and T4:** hit CPU caps.
  - **T2:** has no input.
  - **N1 and N4's main grid:** capped.
  - **N8:** not admitted.
- **Confirmation as frozen cannot confirm.** The betting e-value uses a fixed fraction 0.5/(0.02 − L). L is each family's certified lower bound on a world's gain, and it is very negative (R4: −661). If confirmation block means equal discovery means, the expected log e-values are 0.003 to 0.54 against thresholds of 3.0 to 5.7. The code says so itself: "Conservative support can make confirmation underpowered."
- **Deadlines:**
  - dispositions target Tuesday 20 October, 22:00 PDT;
  - science cutoff and admission close Wednesday 21 October, 20:04:15 PDT.

## 2. Decisions on every open item

| Item | Decision | Why |
|---|---|---|
| **D1. T1 numerical successor** | Record **INSTRUMENT_FAILED** in V23, citing the diagnosis: a converged fit, failed by the optimizer's flag. ToMpathy's packet makes certified convergence its first fix. T1's successor runs in V24, pinned to that fix | Even after a fix, T1's batch reserves about 32 to 53 CPU-hours against the 40-hour twins cap, and its disposition must be written by Tuesday |
| **D2. T3 and T4** | Record **RESOURCE_DEFERRED**, with the measured partial costs (T3: 703 of 5,000 makers in 914.5 s; T4: capped before engine fitting at 960.4 s). The speedups go to V24 | A full world needs a 5 to 20 times speedup to fit the cap; no reduced grid |
| **D3. T2** | Record **DEPENDENCY_BLOCKED_AT_FREEZE**. Sounding Line's packet puts the Lichess parser last | The clock never resets; waiting buys nothing |
| **D4. T4's 104-rater sensitivity** | **Drop it** and record why: the released exclusion roster doesn't exist, and author contact is barred | It can never run |
| **D5. Stage 15 profile witness** | Stays **FAIL** in V23. At each health review, look for the tag `certified-fit-1` on ToMpathy's engine branch: its certified-convergence fix, which ends every fit with Newton steps to a gradient below 1e-10 and tests that permuting option order moves predictions by less than 1e-9. If the tag is there with its tests passing before Monday 19 October 12:00 PDT, run the successor the review specified: matched baseline, donor comparison, null precision fixed in advance on 40 null worlds, and 16 decisions per writer in place of four. Charge it to repairs. Otherwise it moves to V24 | It is cheap (the original took 2,221.9 s). The 4-in-20 option-order failures (6.83e-6 against a 1e-7 tolerance) look like fits that stopped at the optimizer's default gradient tolerance, which the Newton finish removes |
| **D6. N4's main grid** | Record **RESOURCE_DEFERRED**. Sacrifice ordering, the curator's question 2, runs as the N4d miniature (section 4) | The 13 profiled cells alone would cost about 530 CPU-hours |
| **D7. Mandatory dispositions** | Write `MANDATORY_DISPOSITIONS.json` with N1 RESOURCE_DEFERRED, N4 RESOURCE_DEFERRED, T1 INSTRUMENT_FAILED, T2 DEPENDENCY_BLOCKED_AT_FREEZE, T3 RESOURCE_DEFERRED and T4 RESOURCE_DEFERRED, with the reviewed families as they stand and evidence hashes. Then admit **N8** under a new hash-bound supplement that raises its group cap from 2 to 8 CPU-hours. If its cost pilot projects past 8, record N8 RESOURCE_DEFERRED | N8 (play) is built and small. Its 2-hour cap is a legacy limit the 16 October supplement never revisited |
| **D8. Freeze and confirmation** | Freeze discovery as section 0, step 4 says. Before admitting any confirmation, bound each nominated family's expected log e-value, as arithmetic only, with no worlds drawn. The bet is log e = Σ over blocks of log(1 + λ(X̄ − m)), with m = 0.02, λ = 0.5/(m − L) and X̄ a 16-law block mean. By Jensen's inequality its expectation is at most (blocks) × log(1 + λ(μ − m)), with μ the family's discovery mean and blocks its frozen world count ÷ 16. Compare the bound with the family's e-LOND threshold computed as if every earlier family in the frozen order confirmed, the lowest threshold it could face. A family whose bound reaches that threshold runs confirmation as frozen. A family whose bound falls short is recorded **CONFIRMATION_INFEASIBLE_AS_FROZEN** in `CONFIRMATION_ARITHMETIC.json` beside the discovery order (a confirmation record, not a discovery disposition); its nomination stands as a discovery-only result. The V24 redesign is section 6 | Running a confirmation whose expected evidence is a hundredth of its threshold spends 40 CPU-hours to learn nothing; recording that is the honest result |
| **D9. Worker expansion** | **Stays at 3.** No interference contract this round | The budget binds, not the workers: about 250 worker-hours to Tuesday against 177.5 CPU-hours, and this packet's work is about 20 |
| **D10. N2's pooling** | Commission a **descriptive robust reanalysis** of the retained N2 data. Use a variance floor for the pooled inverse-variance weights (the largest of the per-setting variance and the median variance across settings), an overflow check, and a partial-conjunction count: the share of the 120 nonstructural settings whose gain over a generically tempered reader (V19's fixed-tempering form, exponent chosen on validation) exceeds 0.02 nats. It doesn't reopen N2's disposition | 48 of 120 settings showed gains above 0.02 while standard errors as small as 1e-158 dominated the pool. Whether timing helps in some worlds deserves a number |
| **D11. New studies** | Commission the eight conditional studies in section 4 | They are the curator's 17 October positions, run where truth is known |

## 3. Operating notes

- **Supplement.** One new hash-bound supplement, `ACCEPTANCE_SUPPLEMENT_2026-10-17.json`, inside the 285 CPU-hour total:
  - a new group `conditional` with a 24 CPU-hour ceiling for the section 4 studies;
  - N8's cap raised to 8 CPU-hours;
  - the D10 reanalysis charged to n2, or to `conditional` if n2 has less than 1.5 CPU-hours left;
  - the D5 successor charged to repairs;
  - every other field identical to the 16 October supplement.
  - Existing packets keep their bindings.
- **Code edits, authorized here** as unambiguous engineering, because today's code refuses any supplement but the 16 October one:
  - `resource_policy.py`: accept both supplement files, each with its own exact group-cap dictionary (16 October: as now; 17 October: the same plus `conditional=24` and `n8=8`). Each packet keeps the binding it was admitted under; new admissions bind the 17 October file through `ACTIVE_RESOURCE_POLICY.json`. `LIMITS` keeps n8 = 2 for legacy resolution and gains `conditional = 24`, so every resolution knows the group; the 17 October group caps raise n8 to 8 for packets bound to that file.
  - `extension_contract.py`: in `validate`, add each new family to the conditional allowlist (today `('N4b','N4c')`) with its module tuple; in `cost_estimate`, map each new family to its own module (today N4c gets `conditional_devaluation` and everything else `conditional_values`, so a new family would be costed against the wrong variants).
  - `queue_v23_extensions.py`: the new families charge the `conditional` group; N4b and N4c stay on n4.
  - Keep the new families out of `registry.ORDER`. They are extensions, like N4b and N4c, so the freeze manifest doesn't need dispositions for them.
- **Allowlist.** Extend `extension_contract.validate` and `admit_v23_extension.py` to each new family name, in the N4c pattern: conditional pilot (controls, full-grid cost, fresh-process replay), conditional fits (the batch reserved here), conditional batch (16 law blocks × 64 cases per setting). No conditional study is ever eligible for confirmation.
- **Wakes.** Queue depletion, failures and the twelve-hourly health keep driving reviews. A queue-depletion review with commissioned work left admits the next study, or starts building it, before acknowledging.
- **Inline work.** Ghost's rule stands: subagents one at a time, for search and research only.
- **Sounding Line keeps priority.** It may run up to four workers this campaign; Ghost stays at three, below-normal, one thread, no GPU.

## 4. Eight conditional studies

**Common form, from N4c.** Each protocol is declared before sampling, citing this packet. Then:
- **Settings and law blocks.** Twelve settings, unless a study says otherwise. Sixteen independent law blocks of 64 cases per setting, with reliability balanced from 0.60 to 0.95 across blocks.
- **Paired randomness.** Common random numbers pair histories and probe orders across mechanisms.
- **Response law.** Reliability mass goes on the rule's response; the remainder spreads uniformly over the joint responses. This is declared noise.
- **Exact reader.** It integrates over all mechanisms and reliability values.
- **Primary readout.** The reduction in true-mechanism log loss from baseline evidence to all probes, unless the study names its own (N10, N12, N13 and N14 do). Ablated readouts name which observations separate which rules. Six fixed and fresh direct rivals forecast the same held-out responses.
- **Controls before any production case:**
  - exact baseline aliasing where the design says it exists;
  - changed-probe information; for a study without probes (N10, N12, N13, N14), a planted-recovery control in its place: at the study's strongest setting, the exact reader's posterior on the planted mechanism beats its prior in every law block;
  - a high-reliability recovery control;
  - probability-space against log-space agreement;
  - ledger round trip;
  - exclusion of evaluator labels.
- **Pilot and reservation.** A complete cost pilot and its fresh-process replay; one complete batch reserved before fitting.
- **Inference.** Bootstrap intervals resample law blocks, never cases.
- **Labels.** These are constructed mechanisms in miniature. They test identifiability, not people.

Order is the rank below. Each study costs about 0.2 to 0.7 CPU-hours by N4c's measure, plus the agent's build time.

### N14. Pervasive or local: the curator's rule against the restated rule (rank 1)

**Source.** The curator, 17 October: "The biggest thing to tell a branch that has been autocompleted is the pervasiveness of similar-looking decisions. Other characters being fearful imply that this character being fearful is about the author, not about the character."

The analyst's research pass located this in Kelley's covariation model:
- **consensus:** do others do it here?
- **distinctiveness:** does she do it only here?
- **consistency:** does she do it every time?

It found two limits:
- without consensus, genre conventions pose as authors;
- covariation shows where a cause sits, not whether a choice was automatic. A deliberate global policy is pervasive and attended, and one-shot fear is automatic and can stay local.

The restated rule: an autocompleted choice is person-located (low consensus, low distinctiveness, high consistency) **and** insensitive to what it should track (fast, unrevised, flat against the character's traits).

**World.** An author writes works; each work has four characters facing scenes of three types at low or high threat. Each response is bold or fearful, with two process traces: latency (fast or slow) and revision (yes or no). A population of other authors supplies genre rates (consensus).

**Mechanisms:**
1. **attended:** responses follow each character's trait, within a per-work attention budget. When the budget runs out, responses fall back to the genre default;
2. **scene-cued habit:** an author-level response to a scene type, strengthening with repetition, insensitive to character traits, fast and rarely revised;
3. **one-shot fear:** after a single aversive event in one scene type, fearful responses generalize along scene similarity, across characters; fast;
4. **deliberate global policy:** the author makes characters fearful on purpose; pervasive, slow and revised;
5. **genre convention:** fear set by the genre and shared by its authors (high consensus).

**Settings (12):** genre convention strength {0, strong} × attention budget {tight, loose} × records {1, 2, 3 works}.

**Readouts,** each the exact reader restricted to an observation subset:
- **(a) the curator's rule:** distinctiveness across characters only;
- **(b) full covariation:** adds consensus and consistency;
- **(c) the restated rule:** covariation plus process traces and trait slopes.

**Primary:** the true-mechanism log-loss reduction of (c) over (a). Also report each subset's confusion matrix.

**Predictions:**
- (a) labels the deliberate policy as automatic, misses one-shot fear when it stays within one scene type, and flags genre conventions as authors;
- (b) fixes the genre case;
- (c) separates policy from habit and fear.

**Route:**
- **To ToMpathy:** which statistics its attribution display needs. If (c) adds a lot over (a), pervasiveness alone shouldn't drive a "habit" label.
- **To Sounding Line:** whether D30's and D31's designs can separate these mechanisms at their record counts. Use Sounding Line's delivered D30 marker counts for realism if they arrive before the protocol is declared. Otherwise use the declared defaults, and say so.

### N9. The branch never drawn: pruned, disliked, or local (rank 2)

**Source.** The curator: a branch can be absent because it was never considered, or because it was considered and declined. His case of the first: "You can have sudden expertise that appears through trauma," which pattern-matches where it shouldn't, so that "she can't even see that her characters could fight back." Fear is "an incredibly highly weighted one-shot experience" that blurs, while expertise sharpens.

Research names:
- **consideration sets against preference** ([Masatlioglu, Nakajima & Ozbay 2012](https://doi.org/10.1257/aer.102.5.2183));
- **pruning of decision trees after aversive outcomes** ([Huys et al. 2012](https://doi.org/10.1371/journal.pcbi.1002410));
- **fear generalization gradients** that widen with threat intensity ([Dunsmoor et al. 2017](https://doi.org/10.1037/bne0000186)).

**Mechanisms:**
1. **baseline:** every option is considered; the bold option has its ordinary value;
2. **fear-pruned:** after a one-shot aversive event in one scene type, bold options leave the consideration set in scenes within generalization width w, with their value unchanged;
3. **dislike:** the bold option is valued lower everywhere, considered and declined;
4. **expert-narrow:** pruning is confined to the conditioning scene type;
5. **attended-local:** a bounded attention budget leaves the bold option unconsidered at random, independent of scene.

**Probes,** six per case:
- **offered branch:** the bold option is presented directly, forcing consideration;
- **timing:** an unconsidered option is never evaluated, so the decision is fast; a declined one is slow.

**Settings (12):** width {narrow, wide} × records {0, 8, 32} × timing {hidden, observed}.

**Readouts:** baseline to all probes, plus ablations: offers only, timing only, actions only.

**Predictions:**
- actions alone alias fear-pruned with dislike, as N4c's actions aliased goal-directed with habitual-goal;
- offers separate them: pruned makers take the offered branch at baseline rates, disliking makers decline;
- timing separates them without offers;
- width separates fear from expertise only with enough scene types.

**Route:**
- **To Sounding Line:** D32's examination split is this study's timing probe. CoAuthor's menu is its offer probe.
- **To ToMpathy:** "offer the branch" becomes a line in "what would move this estimate".

### N13. Validate the falsifier (rank 3)

**Source.** Sounding Line's FX1, frozen today. Its precondition P3 needs exactly this study.

**World:**
- **goal-structured makers:** choices depend on a latent goal that varies across declared contexts;
- **habit-only makers:** matched to the same self-consistency, meaning the same agreement between repeated decisions.

**Readers:**
- **G:** goal-latent;
- **N:** records only.

Both readers are compared against a generically tempered rival. FX1's verdicts apply, with margin m = 0.10 × (Brier(N) − the oracle floor).

**Settings (12):** goal strength {0, m, 2m} × records per maker {32, 81, 250} (81 is CoAuthor's median), plus three cells at goal strength m with lower consistency (0.70).

**Readouts:**
- the frequencies of FALSIFIED, SUPPORTED and INCONCLUSIVE in each setting;
- power at goal strength m with 81 records;
- the false SUPPORTED rate in habit-only worlds.

**Primary, and frozen requirements,** in the conditional-study form (16 law blocks of 64 cases per setting), with each block's verdict computed from its own cases:
- unbiased: 0 of 16 law blocks at goal strength 0 (habit-only) with 81 records return SUPPORTED;
- powered: G's lower bound clears 0 in at least 13 of 16 law blocks at goal strength m with 81 records.

The primary is that pair of rates. FX1's precondition P3 states the same pair. Report the other settings beside it.

If either fails, Sounding Line is told which, and FX1 waits for more records or a larger margin. If Sounding Line's V5 consistency values arrive before the protocol is declared, use them for the habit-only matching.

### N10. The hidden teardrop (rank 4)

**Source.** The curator on teardrops: goals emerge mid-work, editors overlay theirs, and a correction may serve a goal nobody named.

This is the successor to N3, whose hard half-space reader emptied its support in 96 of 96 settings. Robotics' answer: treat each correction as a vote with slack, weighted by a per-correction confidence that drops when the correction is poorly explained ([Bobu et al. 2020](https://arxiv.org/abs/2002.00941)). Bobu et al. also warn that this can't tell a missing goal from a misread human.

**Mechanisms:**
- corrections serve catalog goals only;
- or catalog goals plus an out-of-catalog goal with share s and correlation ρ with the catalog goals.

**Readers:**
- the soft-correction reader, with and without a residual-structure test (do unexplained corrections point the same way across contexts more than shuffled ones do?);
- N3's hard reader, as the negative comparator;
- an oracle that knows the hidden goal, as the upper bound.

**Settings (12):** s {0, 0.1, 0.3} × ρ {0, 0.5} × correction noise {low, high}. If Sounding Line's unexplained-correction rate arrives before declaration, it sets the noise levels.

**Readouts:**
- the share of cases with empty support (soft: 0);
- detection AUC for the hidden goal;
- direction recovery (cosine);
- predictive log loss.

**Primary:** the true-mechanism log-loss reduction (hidden goal present or not) of soft-plus-residual over soft alone.

**Predictions:**
- soft never empties support;
- the residual test detects s ≥ 0.1 at ρ = 0;
- at ρ = 0.5 it is confidently wrong: it credits a correlated catalog goal.

**Route:**
- calibrates Sounding Line's D34 verdicts;
- ToMpathy's corrections become soft votes with misspecification flags.

### N11. Attention as precision: System 2 trains System 1 (rank 5)

**Source.** The curator:
- "Attention is deciding to run an inference on whether to trust in a given specific setting."
- "What you have attention over is System 2, and System 2 creates System 1 over time."
- Habits are "ghosts of repetitions" that attention squashes.

In active inference:
- a policy combines a habit prior E with goal-directed value through a precision γ;
- effort is the divergence between the context-sensitive policy and the habit: "effort is exerted to overcome a mental habit" ([Parr, Holmes, Friston & Pezzulo 2023](https://doi.org/10.1016/j.neuropsychologia.2023.108562)).

Capture errors are the habit taking over an intended sequence that shares its opening ([Norman 1981](https://doi.org/10.1037/0033-295X.88.1.1)).

**Mechanisms,** with P(a) ∝ E(a) exp(γ G(a)):
1. **goal-dominated:** high γ, and E doesn't learn;
2. **habit-dominated:** low γ, with a strong prior E;
3. **automatizing:** E grows toward repeatedly chosen actions (E ← E + η × the chosen action);
4. **capture-prone:** strong E on an old action, which intrudes after the context shifts.

**Probes:**
- **time pressure:** γ is scaled down;
- **context shift:** the goal-directed best action changes;
- **devaluation:** continuous with N4c.

**Settings (12):** four mechanisms × {0, 8, 32} records, as in N4c.

**Readouts:**
- baseline to all probes;
- ablations by probe;
- the effort trajectory: KL(P ‖ E) for the same choice over repetitions.

**Predictions:**
- long histories alias habit-dominated with automatizing makers;
- time pressure separates goal from habit;
- context shift reveals capture;
- effort falls with repetition only under automatization.

**Route:**
- ToMpathy's attention and habits modules (precision, and effort as divergence from habit);
- Sounding Line's D33 and D35.

### N12. Goals emerge, overlays, and the last pass (rank 6)

**Source.** The curator:
- goals emerge and can replace the starting goal;
- the top goal belongs to "the last person to pass judgment over the whole thing with capabilities of changing it";
- "there's no top... different lenses... overlays";
- "Not every creation needs a hierarchy inherently."

Research is split:
- US joint-authorship law ties authorship to control (the "master mind");
- people's felt and given credit track contribution and influence, not final say ([Draxler et al. 2024](https://doi.org/10.1145/3637875)).

**Mechanisms:**
1. **emergent:** an affordance appears mid-artifact, and goal B is adopted beside A;
2. **replacement:** B replaces A;
3. **overlay, author final:** an editor adds goal C, and the author holds the last pass;
4. **overlay, editor final:** the same, with the editor holding it.

**Settings (12):** four mechanisms × conflict rate between goals {0, 0.2, 0.5}.

**Readers:**
- single-top (one root, nested subgoals);
- overlay (concurrent goals, no forced nesting);
- change-point;
- exact.

**Readouts:**
- true-mechanism log loss;
- adoption-time error;
- who held final authority.

**Primary:** the overlay reader's true-mechanism log-loss reduction over the single-top reader.

**Predictions:**
- single-top readers misattribute when goals don't nest;
- the overlay reader recovers both;
- final authority is identifiable only where goals conflict, and aliased at conflict rate 0.

**Route:** ToMpathy's goal display shows overlays when goals don't nest, and labels the goal tree as a functional reading.

### N4d. Sacrifice ordering, in miniature (rank 7)

**Source.** The curator's question 2: which goal is on top shows in "what are you willing to sacrifice for the one above it".

**Mechanisms:**
1. **lexicographic:** the higher goal always wins a conflict;
2. **weighted:** the higher goal wins only when its stakes are larger;
3. **reversed order;**
4. **stakes-blind random.**

**Probes:** forced conflicts, where both goals can't be met.

**Settings (12):** four mechanisms × {0, 2, 6} conflict probes per case.

**Readouts:**
- baseline to probes;
- an ablation with stakes hidden.

**Predictions:**
- order is identified only from conflicts;
- lexicographic and weighted separate only when stakes vary.

**Route:** Sounding Line's D18, and the engine's goal ordering.

### N15. Attention follows trust (rank 8)

**Source.** The curator's attention as trust-setting, with the research term precision weighting ([Feldman & Friston 2010](https://doi.org/10.3389/fnhum.2010.00215)), and trust calibration in people using automation ([Lee & See 2004](https://doi.org/10.1518/hfes.46.1.50_30392)).

**Mechanisms:**
1. **fixed precision** on a source;
2. **precision tracking:** each source's precision is learned from its failures, and checking (sampling more before accepting) rises as precision falls;
3. **global vigilance:** any failure raises checking everywhere;
4. **stakes only:** checking ignores sources.

**Probes:**
- a failure of the source;
- a failure of a second source.

**Settings (12):** four mechanisms × {0, 8, 32} records.

**Readouts:**
- baseline to probes;
- an ablation with only same-source failures.

**Prediction:** precision tracking and global vigilance alias unless the second source fails.

**Route:** Sounding Line's D39 compares model failures with own failures, which is exactly the second-source probe. This study says whether that contrast can separate the two.

## 5. Order and budget

| Order | Work | Group | Estimated CPU |
|---|---|---|---|
| 1 | D1 to D7: dispositions and the supplement | setup | minutes |
| 2 | N8: pilot, fits, batch | n8 (8-hour cap) | up to 8 hours |
| 3 | N14, N9, N13, N10, N11, N12, N4d, N15, in that order | conditional (24-hour cap) | about 4 to 8 hours in total |
| 4 | D10: N2's reanalysis | n2 | about 1 hour |
| 5 | D5: the Stage 15 successor, only if ToMpathy's fix lands by Monday 12:00 | repairs | 1 to 2 hours |
| 6 | Freeze in the first gap after N8 is terminal (it may fall among rows 3 to 5; no later than Tuesday 22:00), then D8's arithmetic | setup | minutes |
| 7 | The V23 report | | |

Total: about 15 to 20 CPU-hours of the 177.5 left. Agent build time, not CPU, sets the pace: the N4c precedent took about 20 minutes from declaration to publication once coded.

## 6. V24, decided directions (designs, not V23 jobs)

1. **Confirmation, designed before any world is drawn.**
   - **Bound the loss.** Make each confirmed reader an ε-mixture with its comparator. Then a case's log-score gain can't fall below log ε (about −4.6 nats at ε = 0.01), in place of bounds like −661. Choose ε per family so the discovery gain drops by under 2%.
   - **Bet adaptively.** Use adaptive betting fractions ([Waudby-Smith & Ramdas 2024](https://doi.org/10.1093/jrsssb/qkad009)).
   - **Size the sample.** Choose world counts for an expected log e of at least 1.5 times each threshold.
   - **Fallback.** A family that can't reach that within 64 CPU-hours gets a fixed-sample test outside e-LOND, reported as such.
2. **T1's successor,** pinned to ToMpathy's certified-convergence fit. **T3 and T4 speedups,** vectorized exports, sized to Sounding Line's measured aggregates.
3. **N1** by successive halving (η = 3) over its 240 cells, or history matching with implausibility of 3 or more as the cut.
4. **Seeds grow** up to twice their plan when an interim result is promising.
5. **pymdp 1.0.0** only in V24, since it breaks replay identity with V23.
6. **Families from the 16 October addendum stay queued:**
   - needs against drives;
   - decaying set points;
   - play narrowing error bars;
   - editing protecting higher-order goals;
   - readers converging with context;
   - controlled variables under disturbance.
7. **Promotion.** Any conditional study here that identifies its mechanisms strongly becomes a V24 family with a powered confirmation.

## 7. Cross-repository

**From Sounding Line,** as aggregates at its digests:
- V5's consistency and headroom curves (for N13), at the first digest after V5 lands;
- content corrections per writer (for N10), and the unexplained-correction rate (for N10) only if its D34.v1 runs;
- D30's marker counts (for N14).

None of these blocks a study: each protocol names its defaults and records which source it used.

**From ToMpathy:**
- the certified-convergence fit (D5 now if in time; T1 in V24);
- a revision-purpose target, still `TARGET_UNAVAILABLE` in the twin contract (V24).

**To ToMpathy,** a module gate list naming which constructed study witnesses which engine module, as `ghost_record` entries:

Module names are the engine's own keys (`module_gates` in `engine/tompathy_engine/interface.py`):

| Engine module | Witnessing studies |
|---|---|
| `habits` | N11, N14 |
| `attention` | N11, N15 |
| `source-trust` | N15 |
| `corrections` | N10 |
| `goal-tree` | N12, N4d; R4, as a constructed-world value only |
| `expertise` | N9 (its expert-narrow mechanism), N11 (automatization) |
| no module yet | N7 (stated purpose); N9's offered-branch reading |

**To Sounding Line:**
- N13 for FX1's precondition P3;
- N14 for D30 and D31;
- N9 for D32;
- N15 for D39.

## 8. Schedule (PDT)

| When | Work |
|---|---|
| **Sat 17** | D1 to D7, the supplement, N8 admitted. Build N14, then N9 |
| **Sun 18** | N13, N10, N11. Publish as each completes |
| **Mon 19** | N12, N4d, N15, D10. D5 if ToMpathy's fix is in by 12:00 |
| **Tue 20** | Freeze discovery by 22:00 at the latest (earlier, in the first gap after N8 is terminal); D8's arithmetic right after. Dispositions done by 22:00 |
| **Wed 21** | Last admission before `report_start` (20:04:15). The V23 report. Three workers all day; Sounding Line and ToMpathy's rehearsal have the machine's priority |

## 9. Authorization (in force at handoff)

1. **Dispositions:** D1 to D10 as decided in section 2, including the new supplement (a conditional group at 24 CPU-hours, N8 at 8) inside the 285-hour total, and the code edits section 3 lists.
2. **Studies:** the eight conditional studies in section 4, each declared before sampling in the N4c form, never eligible for confirmation.
3. **Confirmation:** as D8 decides. The arithmetic runs first, and only families whose expected evidence reaches their threshold run.
4. **Return triggers:** none, except a stone rule blocking a step this packet requires. Record it, skip the step, and note it in the next health review. Nothing waits on the curator.

## 10. Pre-mortem

| Failure | Guard |
|---|---|
| Studies built in sequence run out of time before Wednesday | Rank order. N14, N9 and N13 matter most and go first. Anything not admitted by `report_start` moves to V24 as a design |
| A study confirms its own design, because the mechanisms were written to be separable | Ablations name which observations separate which rules. Aliasing controls must pass where aliasing is designed in |
| N14's mechanisms caricature the curator's claim | The protocol quotes the claim and the restatement. (a) is his rule exactly as stated; (c) is the restatement; both are scored |
| Recording confirmation infeasible looks like giving up | It is arithmetic on frozen inputs, published with the numbers. The V24 redesign follows from it |
| The supplement reads as a cap raise to rescue work | It funds commissioned new studies and a built optional family, not repeats of capped work |
| Sounding Line's numbers arrive late | Each protocol names its default and records its source |

## Sources

- **Repository:** Ghost Scale `6162cc1af`:
  - `TWIN_PROFILE_REPORT.md`;
  - `N4C_PROTOCOL.md`;
  - `ADDENDUM_IMPLEMENTATION_2026-10-16.md`;
  - `results/v23/twin-profile-review-20261017/T1_FAILED_OPTIMIZER.json`;
  - `ghostscale/validation/soundingline/v23/evidence.py` and `extension_contract.py`;
  - `AGENTS.md` (subagent rationing).
- **Consideration sets:** [Masatlioglu, Nakajima & Ozbay 2012](https://doi.org/10.1257/aer.102.5.2183)
- **Pruning:** [Huys et al. 2012](https://doi.org/10.1371/journal.pcbi.1002410)
- **Fear generalization:** [Dunsmoor et al. 2017](https://doi.org/10.1037/bne0000186)
- **Corrections under misspecification:** [Bobu et al. 2020](https://arxiv.org/abs/2002.00941)
- **Effort as divergence from habit:** [Parr, Holmes, Friston & Pezzulo 2023](https://doi.org/10.1016/j.neuropsychologia.2023.108562)
- **Capture errors:** [Norman 1981](https://doi.org/10.1037/0033-295X.88.1.1)
- **Credit in co-writing:** [Draxler et al. 2024](https://doi.org/10.1145/3637875)
- **Attention as precision:** [Feldman & Friston 2010](https://doi.org/10.3389/fnhum.2010.00215)
- **Trust in automation:** [Lee & See 2004](https://doi.org/10.1518/hfes.46.1.50_30392)
- **Betting confidence sequences:** [Waudby-Smith & Ramdas 2024](https://doi.org/10.1093/jrsssb/qkad009)
