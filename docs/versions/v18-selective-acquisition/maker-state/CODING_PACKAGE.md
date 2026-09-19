# Ghost Scale V18.2: persistent maker state, perspective and selective learning

Coding-agent handoff, 19 September 2026. Status: proposed bounded implementation and execution package; no jobs were launched by the analyst. This extends V18.1 without rewriting its completed results or restarting V17.

## What this continuation is for

The curator wants a genuine mechanism search for simulating another maker. Ghost should establish which representations can recover and use a maker's persistent skill/preferences and changing goals/beliefs in a world whose generator is known. Sounding Line will test reader implementations and neural interventions separately. Neither project should wait for the other to finish setup.

Run useful CPU comparisons immediately while adding later branches. Retain the established Sunday checkpoint: **20 September 2026, 15:00 UTC / 08:00 PDT**, or the earlier operator-recorded discussion time. Pacific time is an inherited planning assumption. Late receipt shortens the available window; no rolling 48-hour reset. Begin the report two hours before the checkpoint. If received after the appointment, prepare the bounded runnable packet and ask for the next schedule rather than inventing a new one.

Use the actual current machine allocation, predominantly Gear 1: one low-priority scientific CPU worker and existing numerical-thread/cooling controls. No GPU, cloud, paid API, large downloads or background data crawl. Sounding Line retains the shared GPU. Respect existing delegation rules; an execution queue is not a commission for an agent fleet.

Reviewed main: `7602f7e69975c96095ce187b9d8731af66b0ab1f`. Reconcile current native worktrees and ongoing workers first; do not reset to this commit or duplicate an already completed branch. New code belongs in a sibling namespace such as `ghostscale/validation/soundingline/v18_2/`, with an ordinary runner and separate results root. Preserve V18.1 definitions and frozen primary outcomes.

## Current evidence and the actual delta

V18.1 already contains the following; these are not new tasks:

- In the complete candidate menu, two observations isolate the true law on the tested 192 contexts. With physical preparation charged, dependency-based and equal-query known-law methods both reach 57.81% task success, primitive search 33.33%, and the direct compiler 100%.
- With the true law omitted and setup supplied, one observation can leave one wrong candidate. Two observations empty the candidate family and let candidate-aware methods abstain on all 192 tested contexts.
- In the latest physical misspecification packet, chains reach that diagnosis; forks and groups do not under the tested support and resource conditions. Forks partly exhaust the budget and still fail to diagnose after a completed pair; groups exhaust after one probe. Fresh-object provisioning/reset costs remain outside this experiment.

See [physical query result](https://github.com/EmotiveAutomaton/ghost-scale-sim/blob/7602f7e69975c96095ce187b9d8731af66b0ab1f/docs/exchange/v18-1-cyclic-physical-response.md), [missing-law result](https://github.com/EmotiveAutomaton/ghost-scale-sim/blob/7602f7e69975c96095ce187b9d8731af66b0ab1f/docs/exchange/v18-1-cyclic-misspecified-response.md), [physical missing-law result](https://github.com/EmotiveAutomaton/ghost-scale-sim/blob/7602f7e69975c96095ce187b9d8731af66b0ab1f/docs/exchange/v18-1-cyclic-misspecified-physical-response.md).

The next question is not simply whether an observer can identify one of a few world laws. It is whether an observer can maintain the right **maker state** when the world, immediate goal, private information and acquired expertise can vary independently.

Existing V16 modules already test mechanism families and representation recodings. Inspect `mechanism_models.py`, `mechanism_reader.py`, `attack_recoding.py`, the acquisition/attention helpers, and the V18/V18.1 runtime/evaluator. Reuse their execution and observation boundaries. The new contribution is temporally persistent, role-separated state and its interventions, not another renaming-invariance check.

## Minimal world extension

Use one existing executable world first. Add a thin maker-policy wrapper rather than a new simulation framework. Retain the strongest direct compiler/known-law comparison wherever its information assumptions apply.

Give each maker four distinct generative components:

| Component | Concrete initial implementation | Why it is separate |
|---|---|---|
| Persistent tradeoff | A finite distribution over two or three outcome features, with a declared choice-temperature/noise model | A preference is not the same as the currently assigned task. |
| Expertise | An actually acquired library/repertoire that changes available or affordable operations | Skill must affect execution, not merely a descriptive label. |
| Current goal | A task-conditioned target or utility term which can change across episodes | Temporary purpose should not automatically rewrite the persistent state. |
| Belief/information state | A posterior or finite belief state determined by observations the maker received | The maker may act on incomplete or false beliefs even when the evaluator knows the world. |

Use a declared decision rule combining task reward, stable tradeoff and cost. Avoid claiming its numerical coefficients are human values. Vary these components independently enough to expose confounds, and also include deliberately confounded cases where identification is impossible from the supplied evidence.

Initially, a small finite parameter grid should permit exact enumeration. Do not choose a massive latent space merely to make direct inference look bad. If multiple states induce the same observable distribution, preserve the equivalence class. An informative new context may separate them; without it, calibrated ambiguity is correct.

The public packet contains only permitted behavior, artifacts, contexts and observations. It does not contain true preference labels, private evaluator state, future outcomes or generator internals unavailable to the comparator. For perspective tests, explicitly define whether the reader knows which observations the maker received. Do not silently give that information to one method alone.

Generate real programs/actions from the maker policy and replay them through the independent evaluator. Reference probabilities must sum to one and match direct enumeration. A sample from a stochastic policy is one outcome, not the full true distribution.

## Methods to compare

Keep methods simple and their information tiers explicit:

1. **Direct/no-history:** strongest available task-specific predictor under the declared public information.
2. **Raw-history predictor:** nearest-neighbor or empirical conditional predictor, with a small flat sequence model if the existing stack supports it.
3. **Persistent executable posterior:** finite candidate state with sequential updates and actual predictive likelihoods.
4. **Split learned state:** compact encoder of longer history plus a current-state encoder, with a shared prediction head.
5. **Flat learned state:** same evidence, comparable parameter count/training budget, without the imposed slow/fast split.
6. **Full-information oracle:** evaluator-side bound; never treated as a deployable reader.

The learned-state comparison is inspired by existing machine theory-of-mind work, not a claimed invention or full reproduction. [ToMnet](https://arxiv.org/abs/1802.07740). Separating private and shared representations also has explicit precedent in [LatentToM](https://arxiv.org/abs/2505.09144); reuse the distinction without importing its full robotics framework.

Use an already available CPU numerical stack for a small encoder, preferably at most roughly 100,000 trainable parameters and a bounded training set. No GPU dependency. If a working training path cannot be admitted within two focused engineering hours, run methods 1–3 and 6 and mark the learned comparison instrument-limited. Do not replace it with an untested hand-coded score and call that a neural representation result.

## Branches

### G0 — enumeration, ambiguity and an immediate persistent-state comparison

Start this while implementing the other branches. Construct 64 development maker/world histories and 128 held-out histories, with up to eight earlier episodes and four future probes per history. Group all probes by maker/world lineage. Split makers and world structures, not just random rows. Supply an exact reference posterior on the finite grid.

Compare inference from one artifact, a sequence of artifacts, and permissible process observations. Within an information tier, all methods receive the same evidence. Report the gain from added evidence separately from the gain from a new algorithm.

Include cases with identical visible artifacts but different histories, and cases whose histories remain observationally equivalent. Check that predictions sharpen only when the evidence warrants it. Measure predictive log loss/Brier score, task success, state-equivalence coverage, and cost. Exact posterior agreement is an implementation test; superiority over direct prediction is the substantive comparison.

### G1 — persistent expertise with changing immediate goals

Cross stable skill with stable/changed local goal and stable/changed preference. Give each reader the same available evidence. Compare a model that retains and updates state with one rebuilt independently for each task and with equal-budget raw-history prediction.

Use interventions that alter one generator component while holding the others fixed. Ask whether an inferred skill predicts future construction across a goal change and whether the model incorrectly interprets that goal change as a new lifelong preference. Conversely, include real changes to persistent policy so the observer cannot win by never updating.

For learned states, swap the candidate skill or goal subrepresentation between matched makers and test the corresponding counterfactual. Unstructured latent embeddings may lack uniquely identifiable coordinates; compare their predictions rather than imposing a convenient psychological label after seeing the result.

Keep evaluator-selected counterfactual interventions separate from the reader's ordinary predictions. Knowing which component the experimenter changed is useful for testing a causal representation; it must not leak into a claimed deployable reconstruction advantage.

**Next branch whether positive or null:** longer delays, novel goals, and history compression at matched inference budgets. Do not count the same maker replayed many times as independent persistence evidence.

### G2 — false belief and self–other separation

Hold the true world fixed while varying what the maker observed. Separately vary what the reader knows. Include shared true belief, target false belief, reader-only correction, and target correction followed by a real behavioral change.

The decisive test is selective response: changing the target's evidence should change predictions where the policy depends on it; adding a reader-only fact should not magically update the target's mind. Compare perspective-separated state with a predictor that uses a shared world representation and with a flat predictor trained on the same data.

Do not require a narrative explanation to score this. Use executable predicted action distributions. Then export a small text rendering for Sounding Line, without forcing that project to wait for this branch.

**Next branch:** partially unknown observation access and uncertainty over whether the maker noticed an event. This makes the problem harder in a specific way; it is not a blanket increase in random noise.

### G3 — missing hypotheses and context that changes the model family

V18.1 has already tested empty-family detection. Extend it to a **bounded alternative-family update** for the maker model, using a predeclared vocabulary of changes such as a missing skill, a changed goal, or a mistaken belief.

Compare: fixed-family posterior, fixed-family safe abstention, bounded family expansion, and a strong empirical predictor. Charge candidate proposal, evaluation and any additional observation acquisition. A revision can occur only after the disconfirming evidence, and it is judged on a later held-out probe. Neither generator truth nor the later probe chooses the new hypothesis.

Include an attractive but wrong context card and two reports copied from one underlying source. Duplicate evidence must not count as two independent observations unless its generative dependence justifies that treatment. Report how often revision repairs predictions, overfits the latest event, or remains unresolved.

**Next branch:** charge physical access/preparation when the existing adapter is already operational. Keep a separate label for supplied setup versus paid setup. Do not reopen the whole fresh-object provisioning problem as an overnight prerequisite.

### G4 — representation geometry under a real computational constraint

Using the same latent generator, produce a transparent representation, an invertibly rotated representation, and a compressed or mixed representation. Keep the underlying episodes identical. Compare exact transformed inference, learned flat state and split state.

First verify coordinate invariance for a full invertible transform with the inverse correctly applied. No gain is expected from changing coordinates alone. Then impose a declared rank, memory or decoder constraint and test whether learned alignment preserves useful role information more efficiently than random axes or a generic variance-maximizing projection.

Separate linear rotations from nonlinear mixing. A nonlinear transformation is a different test, not an orthogonal rotation. Define train/dev/test transforms and whether the learner knows the transform. Never grant an inverse map to one reader and withhold it from the rival without labeling that information difference.

**Next branch:** crossed belief/goal states absent from training, then a new world family. This extends V16's interface-isomorphism checks into learned compression and causal role transfer.

### G5 — attention, learning weight and selective uptake

Reuse acquisition helpers to test the curator's tentative attention/trust hypothesis without defining it true in the code. Create demonstrations containing a useful procedure and a goal/tradeoff the learner does not share. Use neutral synthetic goals; no extremist material is needed.

Independently manipulate which observations are acquired, how strongly they update the learner, and which learned procedure is applied. Start with separable task features, then introduce an explicit dependency that makes selective transfer harder. Equalize available information and charge attention/query costs. Do not insert a goal-transfer penalty directly into only the attention arm.

Measure useful task transfer, uptake of the unwanted goal/tradeoff, retained uncertainty, and construction cost separately. This can establish a dissociation or dependency within the constructed learner. It cannot establish that human disgust, trust and conscious attention are one mechanism.

**Next branch:** compare attention alone, update weighting alone, and their combination under the same budget; then a held-out dependency. If these repeat a prior V16/V18 result exactly, record the equivalence and skip to the new dependency rather than republishing it.

### G6 — a small check on the anti-capture premise

This is a maximum 30 CPU-minute mathematical counterexample exercise, not an alignment architecture. Define one synthetic population mean and one individual's policy parameter. Vary within-person observations, number of people, heterogeneity, selection bias and duplicated records. Report coverage and error for each **different estimand**.

Demonstrate where a well-observed individual is more precisely estimated than a poorly sampled population and where the reverse holds. An individual interval and a population interval are not interchangeable claims. A concentrated but biased posterior is a failure, not evidence of safety.

No secret confidence floors, deliberately wrong personal predictions or invented proof that all humanity is necessary. The curator's humanity-wide objective belongs in explicit governance and aggregation choices. This branch supplies concrete counterexamples for that later discussion; it does not choose the objective.

## Expansion queue and stopping rules

Give each branch a complete small block before a large expansion. When a block finishes, dispatch the next admitted branch or its explicit follow-on. No positive result is needed to continue the program. Prioritize a new explanatory contrast over more seeds in an unchanged setup.

Allowed expansion axes, within the same resource ceiling:

1. Additional independent makers and held-out combinations of the existing factors.
2. Meaningfully different existing world topologies with the same target roles.
3. Longer histories with a local-goal change and with a persistent-policy change.
4. Observation-access uncertainty and context contradictions.
5. Matched compression/search/attention budgets.
6. Independent training seeds for the learned-state comparison, once the preceding contrasts are covered.

Start with 128 test histories per complete comparison; extend to 512 or 2,048 histories only when measured cost and the question justify it. Training data is generated independently. For a learned model, start around 10,000 training episodes and cap the initial fit by time; enlarge only if the validation curve and available time make the comparison interpretable. Always report actual counts and independent clusters.

Resource ceiling: **18 cumulative worker-plus-child CPU hours**, further limited by actual available wall time and existing machine policy. Single-thread libraries remain single-threaded. Stop on the first of the checkpoint, authorized resource exhaustion, a machine safety condition, or exhaustion of all distinct valid branches. Do not run meaningless repetitions to fill the clock.

Estimate runtime from the first 16 completed units in each materially different branch. Maintain both observed and twice-as-fast queue forecasts, reflecting prior overestimates. Fit/evaluator/search costs are included. Keep admitted follow-on work sufficient for the faster scenario; do not leave a three-minute queue and assume it occupies a day. If a branch is too cheap to justify more samples, proceed to its next conceptual contrast.

Setup is layered: G0 runs first; add G1/G2 while it runs; prepare learned-state fitting and G3/G4 next; G5 remains an independent branch. Give each failed adapter one two-hour active repair allowance, counted within the same schedule. Failures do not block unrelated branches. Native transition supervision follows the repo's current rules and retains the absolute deadline on resume.

## Shared packet with Sounding Line

As soon as the first fixture is verified, export a small versioned public packet containing:

- Stable case/maker/world identifiers and artifact/input hashes.
- Public history, observation-access description, current context and allowed action choices.
- A task type and evidence tier; no evaluator truth or latent labels in the public fields.
- Separate evaluator truth, equivalence classes and counterfactual siblings kept behind the existing truth boundary.
- Renderer version, chronology and source lineage; an explicit synthetic-data label.

A dozen verified counterfactual families is a useful first export. The independent performance sample stays distinct from chosen illustrations. Sounding Line may start with its own fixtures and consume this later; no synchronization gate. The analyst is not authorized to message another operator on the user's behalf, so normal repo exchange is the handoff route.

## Validity checks that matter

Verify finite-reference likelihoods, fresh generator replay, observation access, temporal separation, identity interventions, transformed-coordinate equivalence, and the strongest direct rival. Inspect at least one selected disagreement end to end. Do not write a large test suite that merely mirrors implementation.

Keep exact-posterior correctness, learned predictive performance, and interpretability of a learned state separate. Ground truth is defined by this generator; it does not establish a human ontology. A model can have a useful representation that is identifiable only up to a transformation, and different histories may remain genuinely equivalent.

Use proper scoring rules and coverage in addition to task success. Account for compute spent learning and compiling policies, not just final action execution. Let a strong direct compiler win where it should. If a task is saturated, report that and move to an explicitly harder, independently defined condition; do not weaken the rival after seeing the result.

Group uncertainty by independent maker/world lineage. Report exposed versus fresh support, all failed/partial cells, and whether a comparison was chosen after observing earlier results. These are discovery comparisons; an attractive result opens a later frozen confirmation.

## Sunday packet

Deliver the ordinary result, verification, progress and exchange records, plus a short mechanism table stating what actually ran. Include six reconstructed examples: persistent skill across a changed goal, a false-belief correction, an unidentifiable maker pair, an unsuccessful context explanation, a selective-transfer case and a representation failure or success. If a category does not occur, say so rather than manufacture it.

The report should tell the curator which mechanism deserves a week, which cheap rival remains sufficient, where more evidence rather than more computation is needed, and what was too costly or broken to judge. Connect results to the theory's separate targets: reader-enactable technique, historical correspondence, local purpose, expertise and persistent preference. No automatic promotion to human values, consciousness, or a universal anti-capture guarantee.
