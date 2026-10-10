# Errata: Ghost Scale research program of 10 October 2026

**For the coding agent and the planning thread, before dispatch.** Analyst review dated 2026-10-10.

- This file amends `GHOST_SCALE_RESEARCH_PROGRAM_2026-10-10.md`; it does not replace it.
- Nothing here launches work, mutates the repository or authorizes spending.
- Items marked **OWNER DECISION** wait for the curator.
- The companion file is `SOUNDING_LINE_PROGRAM_ERRATA_2026-10-10.md`.

**Pin verified 2026-10-10.** Live `main` is `e347de02aa098b6fb9a905868924cec1b481d5d7`, the program's pin.

**Checked for this file:**
- the program;
- `v20/world.py`, `readers.py`, `studies.py`, `checker.py`, `analysis.py`, `runtime.py`, `integration.py` and `transport.py`;
- the V20 PROTOCOL, OPENING, SHIFT, CONFIRMATION and FINAL reports, and the V19 final report.

Not checked: the workstation, raw V20 packets, and `runners/` (outside my sparse checkout).

**How the numbers below were produced.**
- I ran diagnostics in a scratch copy, outside the repository:
  - `world.py` and `readers.py` are byte-identical copies.
  - The `studies.py` functions `reference`, `inputs`, `posterior` and `training` are copied line for line.
- **The copy reproduces V20's confirmed result.** Settings: confirmation worlds 96–127, 2,048 cases per world, training lineages 0–7, 131,072 labels, four past episodes, context tier, fit seed 0.

  | Quantity | My replication (fit seed 0) | V20 report (two fits) |
  |---|---|---|
  | Shifted log-loss gain over the joint neural reader | 0.0535 [0.0375, 0.0697] | 0.05462 [0.03891, 0.07136] |
  | Brier, candidate versus V20 MLP | 0.92659 versus 0.92790 | 0.92683 versus 0.92789 |
  | Exact accuracy, candidate versus V20 MLP | 8.15% versus 9.35% | 8.08% versus 9.52% |
- One fit seed per condition unless stated. Intervals are 4,096-resample whole-world bootstraps and descriptive.
- Timings come from a cloud container, not the workstation.
- Scripts, verbatim module copies and outputs are in `GHOST_V20_DIAGNOSTICS_2026-10-10.zip`.

**Priority:**
- **A:** change before dispatch.
- **B:** change during the first implementation day.
- **C:** correction to the record.

---

## GS-1 (A). V20's confirmed shifted win comes from duplicated evidence

**Program text.**
- Section 1 preserves: "The frozen changed-execution/context candidate passed on 32 untouched worlds: capped log-loss gain over the joint neural rival was 0.05462 nats [0.03891, 0.07136]."
- Section 2 lists "duplicate evidence" among rivals that "can each manufacture an apparent reconstruction benefit."

**Finding.** Duplicate evidence manufactured this one.
- **Two features record nearly the same fact.** In the native law, repair copies the claim (`execute()` in `world.py`).
  - So the current episode's `claim` and `evidence` features agree in 90.7% of confirmation-world cases.
  - Under changed execution they agree in 9.4%.
- **The structured candidate counts that fact twice.** It is naive Bayes over features given the 128-way label (`readers.py` lines 21–23 and 60–64).
  - Removing the evidence feature from the current and past episodes improves its native log loss by 0.061 [0.057, 0.065] nats.
- **The shift cancels the double count.** Under changed execution, the native-trained claim and evidence factors disagree and partly cancel.
  - The forecasts soften: mean entropy rises from 2.784 to 2.880 nats.
  - The candidate's loss improves by 0.050 [0.038, 0.062] relative to its own native loss.
  - The shifted worlds are not easier. The ceilings barely move: supplied-law oracle 2.710 to 2.703; law-marginal 2.839 to 2.834 (GS-3).
- **The whole shift gain lives in the duplicated feature.**
  - With the evidence feature removed, the candidate's loss changes under the shift by −0.004 [−0.014, +0.006].
  - With the claim feature removed instead, the shift costs it 0.100 [0.088, 0.113], because the flipped evidence alone misleads it.
  - V20's second training draw (fit seed 1) repeats the pattern. Its loss changes under the shift by −0.045 [−0.057, −0.033] with all features, and by −0.003 [−0.013, +0.007] without the evidence feature.
- **The confirmed gain is almost entirely the cancellation gain.**
  - Natively, the candidate and the V20 MLP tie: −0.002 [−0.019, +0.015].
  - Under the shift, the candidate gains 0.050 and the V20 MLP loses 0.005.
  - The native −0.002 plus those two changes gives the 0.054 shifted gain.
- **A properly fitted direct reader shows the opposite pattern (GS-2).**
  - It is the best native reader, at 2.899.
  - It collapses under the shift: 5.339 nats and 5.1% exact accuracy, against a 3.152 prior.
  - A reader that learned the native dependency well is confidently wrong when that dependency flips.

**Change.**
1. **Do not edit V20 reports.** In the v21 packet, and wherever the result is cited, carry it as: "a native-trained naive-Bayes reader loses less than a fixed-schedule MLP under changed execution, because the shift cancels its native double counting of claim and evidence." It is not evidence that structured inference transfers.
2. **Make this A01's first case and B02's positive control.** Duplication within an episode is the known instance; B02 currently varies only duplicated episodes.
3. **Report shift effects reader by reader.** In every shift comparison, report each reader's paired native-to-shift change. For any product-of-factors reader, also report a variant with the duplicate removed.

---

## GS-2 (A). V20's direct rival is underfit, so the program's "strong matched direct reader" does not exist yet

**Program text.**
- Section 1 asks what helps "beyond a strong matched direct reader."
- Section 4 reuses `v20/readers.py` for "Frequency, template, MLP and naive-Bayes rivals."
- Section 2 lists "better optimization" as a rival.

**Finding.**
- **The V20 direct MLP runs a fixed schedule.** It is 64 tanh units trained by 120 full-batch gradient steps at learning rate 0.7 (`readers.py` lines 24–32).
- **Quadrupling the labels barely helps it.** On 16 held-out validation lineages (264–279), with training lineages 200–263, its unmasked log loss is:
  - 3.6605 at 32,768 labels;
  - 3.6556 at 131,072 labels.
- **The same network trained to convergence does much better.**
  - Same architecture and features, minibatch Adam, early stopping on those lineages.
  - It reaches 3.0368 and 2.9531, stopping at epochs 22 to 32.
- **On the confirmation worlds** (native law, 131,072 labels from lineages 0–7):
  - converged direct reader: 2.899;
  - V20 MLP: 3.011;
  - structured candidate: 3.013.
  - The converged reader beats the structured candidate by 0.113 [0.100, 0.128].
- **Every V20 direct-rival comparison therefore measured an optimizer setting as well as a representation.** The factorized MLP uses the same fixed schedule (lines 34–42), which C01 already suspects.

**Change.**
1. Add a converged direct rival to every learned comparison in the B, C, D and F cards. Use early stopping on validation lineages disjoint from training, development and confirmation, and report the validation curve.
2. Declare a rival `INSTRUMENT FAILED` if its validation loss does not improve when labels quadruple while a converged rival's does.
3. Keep the V20 MLP as a compatibility arm, labeled "fixed-schedule MLP."

---

## GS-3 (A). Use the ceiling a reader can reach: the law-marginal posterior, not the supplied-law oracle

**Program text.**
- A01: "Compute exact conditional entropy, Bayes modal accuracy, Brier uncertainty and regret for each reader."
- Section 5: "For supplied-law p and forecast q, exact expected Brier regret can be evaluated as `sum((q-p)^2)`."

**Finding.**
- **Readers never see the test world's coefficients.** The best any reader can do is the Bayes posterior under the generator: average over coefficient laws, then update on the current evidence and the past episodes.
- **I computed this law-marginal posterior by Monte Carlo** over laws from lineages 5000 and up.
  - It is stable: 40, 160 and 400 laws give 2.862, 2.864 and 2.865 on a six-world test set.
- **On the confirmation worlds:**
  - native: supplied-law oracle 2.710; law-marginal 2.839; gap 0.129 [0.106, 0.152];
  - changed execution: gap 0.131 [0.106, 0.155].
- **About 0.13 nats of what A01 would call regret is law uncertainty.** No learner trained on other worlds can remove it.
- **Exact enumeration is feasible.**
  - `law()` draws route, select, revise, goal and inspect coefficients from 2 to 6, and skill from 5 to 8.
  - It also draws `presentation`, which `enumerate_world()` never uses.
  - That leaves 5^5 × 4 = 12,500 distinct laws. The law tables are shared across all test worlds.

**Change.**
1. Split A01's error budget into three terms, in nats and in Brier:
   - irreducible: supplied-law entropy;
   - law uncertainty: law-marginal minus supplied law;
   - reader excess: reader minus law-marginal.
2. Use the law-marginal as the ceiling in A01, A02 and C03, and for the `INFORMATION BOUNDARY` disposition.
3. Keep the supplied law as the privileged diagnostic.
4. Compute in log space at history 64 or more.

---

## GS-4 (A). Compute a headroom map before admitting B and C screens

**Finding.**
- **Headroom bounds every structured gain.** A representation can gain over the converged direct reader at most that reader's distance to the law-marginal ceiling.
- **Measured headroom** (native law, context tier, four past episodes):

  | Labels | Training laws | Test worlds | Headroom (nats) |
  |---|---|---|---|
  | 131,072 | 8 | confirmation worlds | 0.060 [0.042, 0.080] |
  | 131,072 | 64 | confirmation worlds | 0.029 [0.013, 0.046] |
  | 32,768 | 64 | 16 other worlds | 0.107 [0.080, 0.133] |
- **V20's practical margin was 0.02 nats.** At V20's largest label budget, B01, B02 and C02 have almost nothing left to win natively.
- **Headroom grows where labels are scarce, and under shift.** Under changed execution the converged reader sits 2.4 to 2.5 nats above the shift-aware ceiling.

**Change.**
1. A01 first produces a headroom table over tier × history × labels × shift, using the converged direct reader and the law-marginal ceiling.
2. Admit a B or C screen only in cells where the lower bound of the headroom interval exceeds the practical margin.
3. Elsewhere the expected disposition is `PARITY`, so skip the run and record why.

---

## GS-5 (A). V20's world hides maker process dispositions from every symmetric history; B01 needs asymmetric evidence

**Program text.**
- B01's grid is "history 0/1/4/16/64" over the V20 tiers.
- Local prediction 1: "Shared-maker modeling should help most when several dependent episodes jointly identify persistent dispositions."

**Finding** (supplied-law oracle, 8 worlds, 1,024 cases each):
- **Two choices leave no trace in the artifact.** `execute()` ignores route and inspection.
- **Past episodes reveal nothing the current one hides.** `inputs()` projects past episodes at the current episode's tier.
  - So a past episode reveals the maker's route preference only in tiers where the current route is already visible.
  - It never reveals the review disposition unless inspection is also observed for the current episode.
- **Result:**
  - With context-tier histories, going from 4 to 16 past episodes leaves route accuracy at 0.75 and inspection at 0.49. Only the aims improve (aim A from 0.84 to 0.96).
  - Inspection is a coin flip in every symmetric tier.
  - The supplied-law oracle's exact seven-bit accuracy is 12.3% at context tier with four episodes, and only 43.9% even with complete records.
- **Asymmetric evidence changes this.** Keep the current episode at context tier, but give past episodes complete records with inspection observed:
  - at 16 episodes, route accuracy rises to 0.80 and inspection to 0.57;
  - at 4 and 16 episodes, oracle log loss falls by 0.13 and 0.10 nats;
  - for an artifact-only current episode it falls by 0.25 and 0.21.
- **This asymmetric case is the product case:** complete Git history for earlier posts, an endpoint for the new one.
- **The inspection ceiling stays low by construction.** `law()` draws the inspect and route coefficients from 2 to 6, so even a known disposition moves the probability to at most 0.8.

**Change.**
1. Give v21 `inputs()` a separate `history_tier` and history inspection visibility. Make complete past records with an endpoint-only current episode B01's primary regime.
2. Score per bit, and report the Bayes per-bit accuracy beside each reader.
3. If B01 should be able to detect process sharing, widen the disposition coefficient ranges in `v21/world.py`, or report the ceiling the current ranges imply.

---

## GS-6 (B). Training-law diversity is an unmeasured axis

**Finding.**
- **V20 trains every reader on 8 coefficient laws** (lineages 0–7). Each label budget is spread over those 8.
- **More laws helped every reader.** Same confirmation worlds and the same 131,072 labels, but 64 training laws (200–263) instead of 8. Change in native loss:
  - structured candidate: −0.048 [−0.096, −0.001];
  - converged direct reader: −0.031 [−0.065, +0.003];
  - V20 MLP: −0.022 [−0.055, +0.010].
  - That is one training draw each, so it is suggestive only.
- **Meta-learning has a direct precedent.**
  - Below a task-diversity threshold, transformers trained on regression tasks behave like the Bayes estimator whose prior is the finite pretraining task set, and fail on unseen tasks ([Raventós et al. 2023](https://arxiv.org/abs/2306.15063)).
  - Prior-fitted networks repeatedly draw a task from the prior and learn to predict its held-out labels, approximating Bayesian inference over that prior ([Müller et al.](https://arxiv.org/abs/2112.10510)).

**Change.**
1. Cross B01's label budgets with training-law count (8 and 64) at fixed labels.
2. Add a "fresh-law" direct rival trained on a new generator draw per minibatch. It should approach the law-marginal ceiling, and it asks whether structure reaches the ceiling with fewer laws or labels.
3. This is an OWNER DECISION below, because it changes what "data scarcity" means in Ghost.

---

## GS-7 (B). The theory's transfer claim needs adaptation data: add a sparse-mechanism-shift card

**Finding.**
- **V20's shift test gives structure no way to help.**
  - Readers get zero shifted labels.
  - The structured reader has no separate execution factor.
  - The measured "transfer" is an artifact (GS-1).
- **The theory's claim is modular.** If the maker model is separate from the execution law, a change in execution should require re-learning only the execution factor.
- **Causal machine learning states the testable version.** The correct factorization adapts faster to a modified distribution when the change is concentrated in one or a few mechanisms ([Bengio et al.](https://arxiv.org/abs/1901.10912)).

**Change.** Add card **G21-B07, adaptation under a sparse execution change.**
- *Readers:*
  - a modular reader, with a maker/policy factor learned natively and an execution factor re-estimated from shifted episodes;
  - the converged direct reader, fine-tuned on the same shifted episodes;
  - the fresh-law rival, fine-tuned the same way.
- *Adaptation data:* k ∈ {0, 16, 64, 256} labeled changed-execution episodes, drawn from shifted training lineages.
- *Score:* adaptation curves against the shift-aware law-marginal.
- *Promising* if the modular reader reaches a fixed loss with a quarter or fewer of the shifted labels the direct reader needs.
- *Stop* the transfer claim if direct fine-tuning matches it at every k.

---

## GS-8 (B). Measured cost for sizing (cloud container, not the workstation)

- **One complete block took 1,485 to 1,501 s on one core.** The block:
  - 131,072 labels from 8 or 64 training lineages, four past episodes, context tier;
  - the V20 readers plus the converged MLP;
  - 32 worlds × 2 laws × 2,048 cases;
  - a 160-law law-marginal for each law.
- **Breakdown:**
  - data and fits: 489 to 507 s;
  - 160 law references per law: 43 to 48 s;
  - each world evaluation: 13 to 15 s;
  - resident memory: about 1.1 GiB when sampled.
- **CPU is not the constraint.** The opening core sits far below the 48 to 72 CPU-hour envelope; review time binds. Re-measure on the workstation, as section 8 requires.

---

## GS-9 (C). Smaller corrections

- **Brier normalization.** Section 5's Brier identities are for full multiclass Brier, which V20 reports (about 0.927). Sounding Line reports half-Brier (SL-3), so name the normalization in every F03 fixture.
- **A01's "do not train for this card"** is fine for V20's retained readers. Label their excess as including the optimizer shortfall (GS-2); the converged rival belongs to B01 and C02.
- **Opening core and calendar.**
  - A01 should start with GS-1's case and GS-4's headroom map.
  - B01 should start in the asymmetric regime (GS-5).
  - B07 joins the Monday lane with B01 to B03.

---

## Decisions for the curator

1. **Direct rival definition (GS-2, GS-6).** Example answers:
   - "A converged rival on the same finite training laws only; scarcity is part of the question."
   - "Also add the fresh-law rival; the question is structure versus a direct reader at the generator limit."
2. **World revision (GS-5).** Example answers:
   - "v21 adds asymmetric history tiers and widened disposition coefficients."
   - "Asymmetric tiers only; keep V20's coefficient ranges for comparability."
   - "Keep V20's world frozen."
