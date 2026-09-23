# Broader goal reports: independently verified specificity tradeoff

We tested whether broader local-goal reports improve accuracy while retaining less specificity. At 2,048 labels, the predictive bank's three binary goal vocabularies lower always-reported step error from 13.45% to 4.60–9.94%, while retaining 1.64–1.79 fine alternatives per claim. Independent reconstruction, all 9,600 paired estimates and both complete replays verify this constructed-method reporting tradeoff. Fine-goal forecasts are unchanged; historical correspondence and human intent remain unestablished.

All three binary partitions are retained: meaning versus dependency/presentation,
dependency versus meaning/presentation, and presentation versus meaning/dependency.
Fine goals and the one-class endpoint anchor specificity. Each frozen forecast
either maps its fine coordinate mode or selects a coarse coordinate mode, then
always reports or abstains separately at costs 0.1, 0.25 and 0.5. Smaller codes
break exact modal ties; equality abstains. These costs are stipulated, not measured
user preferences. Evaluator truth scores reports without choosing them.

The table uses the predictive bank at 2,048 labels, original restricted forecast,
coarse modes and abstention cost 0.1. Coverage is reported steps per opportunity;
conditional error is incorrect reports divided by reports made. Fine alternatives
per claim counts how many precise goals remain consistent with the coarse claim.
Each row averages sixteen paired development laws, two training draws and five
fit seeds, with native evidence weighting. Ratios divide pooled masses.

| Goal vocabulary | Policy | Coverage, percent | Conditional error, percent | Fine alternatives per claim | Native cost per step |
| --- | --- | ---: | ---: | ---: | ---: |
| fine | always | 100.00000 | 13.45341 | 1.00000 | 0.134534 |
| fine | abstain | 76.42785 | 0.11945 | 1.00000 | 0.024485 |
| meaning | always | 100.00000 | 9.81680 | 1.67126 | 0.098168 |
| meaning | abstain | 76.66593 | 0.18559 | 1.64243 | 0.024757 |
| dependency | always | 100.00000 | 4.60255 | 1.78848 | 0.046025 |
| dependency | abstain | 83.93795 | 0.34899 | 1.75275 | 0.018991 |
| presentation | always | 100.00000 | 9.94425 | 1.64307 | 0.099443 |
| presentation | abstain | 76.51216 | 0.12782 | 1.63064 | 0.024466 |
| one-class | always | 100.00000 | 0.00000 | 3.00000 | 0.000000 |
| one-class | abstain | 100.00000 | 0.00000 | 3.00000 | 0.000000 |

Broadening the vocabulary changes what counts as correct. It does not recover
more fine goals. The one-class endpoint makes every goal possible and is certain
by construction. All three binary always-report vocabularies reduce error; their
abstention behavior is not uniformly better. Meaning-versus-rest reporting has
0.18559% conditional error and 76.66593% coverage, compared with 0.11945% and
76.42785% for fine-goal abstention. Dependency-versus-rest gives 83.93795% coverage
and 0.34899% error. Presentation-versus-rest gives 76.51216% and 0.12782%.
These tradeoffs do not select a universal preferred vocabulary.

Fine-goal logarithmic loss remains 0.913367 nats in every illustrated policy.
Coarse losses answer different target questions and cannot establish an improved
fine forecast. Coarse-mode and mapped-fine choices remain separately reported;
combining classes and choosing a mode need not commute. All four readouts, both
frozen forecasts and every budget remain in the full regroup. No architecture
winner or practical bank advantage is inferred from this illustrative cell.

Original and both complete replays match all 668 deterministic outputs. All plan,
source, archive, environment, input, output and timing bindings verify. The
independent checker reconstructs 307,200 learned score strata, 960 native rows,
225,280 frame forecasts, 704 reader packets and 30,720 fine-parent identities.
Maximum score error is 3.56e-15; parent error is 1.78e-15. Integer goal labels,
scalar projected sums and direct fine-label support scoring independently check
coarse probabilities, partial reports, correctness, costs and specificity.
Saved binary64 marginals are checked before exact modal and threshold decisions;
this establishes arithmetic agreement, not independent numerical-library validity.

Original and reconstructed rows separately regroup with scalar means and
bootstrap multiplicities. A third direct-index bootstrap of original rows
reproduces all 7,680 budget contrasts, 1,920 normalized log-budget areas and 1,920
means within 1.56e-15. Seed 191013 fixes 10,000 paired lineage resamples. Draws and
fit seeds remain inside each lineage; their variation is separate. Intervals
condition on the retained fits. Ratios do not inherit additive-metric intervals.
Zero-coverage ratios remain undefined. No report has zero native support on this
roster; compatibility still does not identify the historically realized goal.

Twenty-six isolated checker controls passed, including every partial-report form,
all partitions, coarse-correct/fine-ambiguous fixtures, modal noncommutation,
one-class certainty, exact and adjacent thresholds, empty reports, pooled ratios,
complete synthetic execution and eighteen corruption cases. Original fitting and
native-law construction remain inherited accepted inputs. This is exploratory
miniature evidence, architecture untested beyond the admitted roster, not a
confirmation of the primary bank claim.

Reader exports contain artifacts, context and witnessed operations only. Forecasts,
reports, target distributions and scores remain in separate scientific/evaluator
archives. No new fitting, samples, protected lineages or tiny settings were used.
Both shared tiny settings are consumed. Sixty-four batches have independent
acceptance; the 96-hour total, protected 16-hour reserve and immutable report times
remain unchanged. Fixed-bin goal-confidence calibration and retrospective updating
are independent alternatives.

[Review protocol](GOAL_COARSENING_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-D-goal-coarsening-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-D-goal-coarsening-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-coarsening-1/EVIDENCE_ROLES.json).

Goal-confidence calibration passed 55 isolated controls and all three admitted executions completed through the existing serial queue. Reproduction and numerical acceptance await their independent completion-event review. Retrospective updating and complete class-wise goal reliability remain prepared, unimplemented alternatives. No new fit, sample or protected lineage was used.
