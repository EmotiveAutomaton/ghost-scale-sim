# Changed-tool transfer: pooled survival hides changed-endpoint reversal

Does a learned composition advantage survive a changed primitive operation? No overall advantage survives: learned propagation differs from direct prediction by -0.00081 nats with original training and -0.00931 with composition held out. Within the held-out composition whose original primitives were visited, it helps by 0.14026 nats overall but loses by 0.15154 on endpoints actually changed by the tool. Independent reconstruction and paired replay-verified estimates establish this exploratory constructed-method result, not historical process correspondence or human intent.

The question is whether a composition benefit learned under the original operation
law persists when the tool changes. Both count-table training draws and both
training modes remain frozen. Eight previously evaluated coefficient lineages are
paired, with each rule retaining its own native path weights. All 640 queries
supply skill, belief, starting artifact and three requested operations. This is
forward endpoint prediction with supplied operations, not inverse local-goal inference.

The table reports exact learned propagation minus direct endpoint counts under
the changed rule; negative values favor propagation. Original training uses the
complete saved episode selection; composition-holdout excludes the declared action
sequence. All/changed/stay groups separate every query from queries whose true
endpoint changes or stays fixed across laws. All support means every query;
heldout-seen-primitives means the withheld action composition whose original-law
primitive transitions were visited. It does not mean the changed primitive was
observed. Each interval averages two draws within lineage, then resamples the
eight paired lineages 10,000 times. Intervals are exploratory, conditional on these
fits and lineages, and are not simultaneous family-wide confidence statements.

| Training table | Endpoint group | Support group | Loss difference [95% interval], nats |
|---|---|---|---:|
| original | all | all | -0.00081 [-0.01256, +0.01107] |
| original | all | heldout-seen-primitives | -0.04046 [-0.04127, -0.03967] |
| original | changed | all | +0.14878 [+0.13449, +0.16246] |
| original | changed | heldout-seen-primitives | -0.01867 [-0.02016, -0.01696] |
| original | stay | all | -0.05544 [-0.07548, -0.03475] |
| original | stay | heldout-seen-primitives | -0.11634 [-0.11634, -0.11634] |
| composition-holdout | all | all | -0.00931 [-0.02100, +0.00254] |
| composition-holdout | all | heldout-seen-primitives | -0.14026 [-0.15658, -0.12636] |
| composition-holdout | changed | all | +0.12700 [+0.11396, +0.13947] |
| composition-holdout | changed | heldout-seen-primitives | +0.15154 [+0.15075, +0.15241] |
| composition-holdout | stay | all | -0.05908 [-0.07869, -0.03885] |
| composition-holdout | stay | heldout-seen-primitives | -0.62211 [-0.62211, -0.62211] |

For the whole native population, original training gives practical equivalence
within the fixed +/-0.02-nat margin: [-0.01256, +0.01107]. Composition-holdout
training is inconclusive, with [-0.02100, +0.00254]; a near-zero mean is not an
equivalence verdict. In contrast, propagation's original-law advantage exceeds
the margin in both modes: -0.05726 [-0.07565, -0.03811] and -0.05562
[-0.07342, -0.03702]. The original-law numbers reproduce the completed parent
experiment and are identity controls, not additional independent evidence.

The held-out composition with visited original primitives preserves an aggregate
advantage under composition-holdout training, -0.14026 [-0.15658, -0.12636].
Separating endpoint changes reveals a reversal, +0.15154 [+0.15075, +0.15241],
and a large stay-query advantage, -0.62211. The identical stay-query value across
lineages reflects this narrow conditional estimand, not universally precise
generalization. The full record includes both fit modes, every support group,
all empty strata and their actual denominators; no favorable subset was selected.

Logarithmic loss is the negative log probability of the true endpoint, in natural-log
units. All methods and both law oracles use final 1/32 uniform smoothing. Under the
changed rule, the sixteen-path sampler loses to direct by 0.20356/0.20010 nats for
original/holdout training; one- and four-path samplers also lose. The original-law
oracle has mean loss 1.51057, while the privileged true-law oracle has 0.02772.
The wrong-law floor is 1/256 on an incorrect deterministic endpoint. These are
frozen, matched-smoothing comparisons, not a claim that any oracle is a reader.

Cross-rule loss rises 0.45226/0.41154 nats for learned propagation and
0.39581/0.36523 for direct counts. Those differences include changed native
population weights; they do not by themselves isolate a fixed-query mechanics
effect. Original support labels describe the old law, never observed new semantics.
Squared probability error and true-endpoint probability remain separate measures
in all 480 means and 660 contrasts; logarithmic loss controls the stated margin.

Independent reconstruction verifies all 221,184 native paths, four saved count
tables reconstructed from ordered training episodes, uniforms, forecasts, support
labels, 327,680 raw scores and 7,680 strata, with maximum discrepancy 3.11e-15.
A separate review recombines all 3,420 metric estimates, draw means, contributing
lineages and bootstrap intervals. All 47 deterministic outputs match original,
adjacent and extracted-source executions; all 442 checker sources and 53 inputs
verify. Eight isolated checker controls include a complete two-law fixture,
empty denominators, paired regrouping and deliberate forecast corruption.
Numerical reconstruction and replay address different implementation failures;
neither measures general architecture severity. Miniature — architecture untested.

READER.zip contains supplied execution queries only: no actual rule, target,
support label, lineage, weight or score. SCIENTIFIC_EVALUATOR.zip retains complete
original paths, models, forecasts, uniforms, parent inputs and source. Separate
REPLAY_RECEIPTS.zip and VERIFICATION.zip bind all executions and the independent
checker. These scientific/evaluator exports are never reader inputs.

**Warrant:** whole-population practical equivalence for original training,
inconclusive holdout-trained aggregate, support-specific advantage and
changed-endpoint reversal. Exploratory constructed-method evidence; no historical
process correspondence, human intent or confirmation. **Pursuit:** implement the
prepared fixed-budget practice-support mixtures. Off-midpoint actual changes and
bounded primitive-feedback repair remain independent prepared alternatives.
The week and confirmation reserve remain open.

[Frozen comparison](SUPPORT_TRANSFER_PROTOCOL.md),
[independent review](SUPPORT_TRANSFER_REVIEW_PROTOCOL.md),
[practice alternative](SUPPORT_MIX_PROTOCOL.md),
[evidence roles](../../../results/v19/G19-F-support-transfer-1/README.md).
