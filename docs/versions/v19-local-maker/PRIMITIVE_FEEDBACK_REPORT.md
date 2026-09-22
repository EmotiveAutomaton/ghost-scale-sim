# Fixed primitive feedback: independently verified

We tested whether observed tool transitions can repair a frozen forward model without harming unchanged queries. With 64 truthful inputs, row replacement improves changed-tool forecasts by 0.19116–0.19524 nats under original training, but worsens unchanged queries by 0.02170–0.02482 nats. Independent reconstruction and both complete replays verify this constructed-method tradeoff; historical process correspondence and human intent remain unestablished.

Logarithmic loss is forecast error measured in nats; lower is better. The table
shows update-minus-unchanged loss with 95% paired coefficient-lineage intervals
for the changed tool at 64 truthful input reports. Rows cross the two fixed
training-support conditions, both input orders and both updating methods.
Columns retain all queries, queries whose true endpoints change under the tool,
and queries whose true endpoints stay unchanged. Native query probabilities
weight each lineage; the two saved training draws are averaged within lineage.

| Training support | Feedback order | Update | All queries | Changed endpoints | Unchanged endpoints |
|---|---|---|---:|---:|---:|
| Original | forward | Add one observation | -0.01438 [-0.01460, -0.01415] | -0.05525 [-0.05606, -0.05425] | +0.00063 [+0.00056, +0.00072] |
| Original | forward | Replace observed row | -0.19116 [-0.19472, -0.18683] | -0.77909 [-0.79263, -0.76377] | +0.02482 [+0.02202, +0.02794] |
| Original | reverse | Add one observation | -0.01405 [-0.01428, -0.01383] | -0.05399 [-0.05472, -0.05308] | +0.00062 [+0.00054, +0.00071] |
| Original | reverse | Replace observed row | -0.19524 [-0.19855, -0.19137] | -0.78573 [-0.79764, -0.77170] | +0.02170 [+0.01915, +0.02457] |
| Composition withheld | forward | Add one observation | -0.01463 [-0.01486, -0.01439] | -0.05610 [-0.05701, -0.05500] | +0.00060 [+0.00053, +0.00068] |
| Composition withheld | forward | Replace observed row | -0.19012 [-0.19323, -0.18645] | -0.76055 [-0.77355, -0.74599] | +0.01943 [+0.01723, +0.02188] |
| Composition withheld | reverse | Add one observation | -0.01438 [-0.01461, -0.01414] | -0.05506 [-0.05587, -0.05406] | +0.00057 [+0.00050, +0.00065] |
| Composition withheld | reverse | Replace observed row | -0.19417 [-0.19702, -0.19095] | -0.76665 [-0.77797, -0.75339] | +0.01615 [+0.01421, +0.01832] |

Row replacement has a clear overall advantage at this budget in both training
conditions and input orders. It does not achieve repair without collateral loss.
With original training and forward input order, unchanged-query damage exceeds
the declared 0.02-nat margin throughout its conditional interval. The reversed
order's interval crosses that margin. With composition withholding, the forward
interval also crosses the margin, while the reversed interval stays below it.
All four stay estimates are positive. One-count updates have much smaller
overall improvements, below the practical margin in every corresponding interval.

The gain is not monotonic across nested feedback budgets. Under original
training, row replacement with 4 inputs changes overall changed-tool loss by
-0.00281/-0.00339 nats for forward/reverse order; with 16 inputs it worsens loss
by +0.00271/+0.00179; with 64 it improves loss by -0.19116/-0.19524. These are
fixed lexicographic input sets, not randomized evidence-quantity trials.
Wrong feedback at 64 inputs raises row-replacement loss over truthful feedback
by 0.25966–0.26231 nats for the changed tool across both support modes/orders.
That comparison checks the role of observed outcomes; it does not establish
that an observer could identify which reports are wrong.

Equal-query estimates retain a different population: all 640 queries have the
same weight, including those with zero native probability. This second table
shows 64-input truthful row-replacement differences for each tool. Rows cross
support and order; columns identify the original and changed tool. Forecasts
repeat across all eight coefficient lineages, so their degenerate lineage
intervals do not quantify training or feedback-population uncertainty.

| Training support | Feedback order | Original tool | Changed tool |
|---|---|---:|---:|
| Original | forward | -0.03344 | -0.17439 |
| Original | reverse | -0.03370 | -0.17978 |
| Composition withheld | forward | -0.03507 | -0.17121 |
| Composition withheld | reverse | -0.03494 | -0.17623 |

Every budget, update, order, feedback correctness, tool, support mode, changed/
stay subset and original-support stratum remains in the complete 15,360-estimate
regroup, including empty strata. The 92,160 original aggregate cells retain both
native and equal-query weights and complete denominators. No favorable class
replaces either declared population. Eight paired coefficient lineages are the
uncertainty units; 10,000 bootstrap resamples use seed 190971, while both saved
fit draws remain within lineage. These intervals condition on the retained fits
and deterministic feedback sets; they are exploratory, not confirmation.

The unchanged model, one-count addition and row replacement share final 1/32
forecast smoothing. The additive arm preserves the saved unit pseudocount once.
Row replacement changes observed rows only; it does not generalize to unobserved
undo-buffer aliases. Skill, belief and operation metadata are supplied teacher
privileges. Unique input reports count once, and conflicting duplicates are
rejected. Different input rows are not independent sampled trials.

**Validation:** original, adjacent and extracted-source replay output maps match
across all 831 deterministic files. The original binds 508 sources and 33 inputs;
the independent checker binds 512 sources and 834 inputs. Source archives,
environment/plan identities, timing receipts and all 831 exported members verify.
The checker independently reconstructs feedback, updated tables, all 245,760
forecasts, 1,966,080 paired scores, native masses from 221,184 paths, support
strata and 26 anonymous reader packets. Its largest absolute discrepancy is
3.10863e-15, below the frozen 1e-12 tolerance. It uses the separately validated
independent mechanics and explicit path-sum implementation; saved parent fitting
and path-probability validation are inherited from their bound reviews.

Ten original controls and nine isolated checker controls pass. Checker fixtures
include deliberate table, forecast, score, population and denominator corruption.
The first fixture attempt remains retained: its intentional corruption tried an
immutable write; the repair permits mutation only in temporary corruption fixtures.
No scientific criterion or result was changed. Replay, independent reconstruction
and the broader regression refresh remain separate validation claims.

**Warrant:** exploratory constructed-method advantage with collateral reversal;
the no-stay-damage hypothesis fails in this fixed comparison. Miniature —
architecture untested. This establishes neither interpolation to unseen inputs,
historically correct processes, selective causal access nor human intent.

**Pursuit:** the prepared primitive input-pooling comparison tests whether known
undo-buffer equivalence removes an avoidable support bottleneck, with a matched
wrong grouping. The independent retrospective-evidence comparison tests whether
future-schedule compression preserves updates about the past. Pooling now passes ten isolated controls and has a frozen executable plan;
its original and both full replays use the existing serial owner. Retrospective
evidence updating remains prepared, alongside the independent hidden-metadata
boundary. The later regression completion event awaits its separate review.

READER_INPUTS.zip remains unchanged and contains only queries and exact observed
budget slices under anonymous content hashes. Rule identities, pairing maps,
weights, targets and scores remain in separate SCIENTIFIC archives. REVIEW.zip
adds the complete independent regroup and numerical review; it is evaluator
evidence, never reader input. No new fit, architecture setting, sampled episode
or protected confirmation lineage was consumed.

[Experiment](PRIMITIVE_FEEDBACK_PROTOCOL.md),
[independent review](PRIMITIVE_FEEDBACK_REVIEW_PROTOCOL.md),
[evidence](../../../results/v19/G19-F-primitive-feedback-1/README.md).
