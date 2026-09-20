# V18.4 purpose alignment with matched target content

Does useful compression depend on matching supplemental purposes to the right histories, even when their target content is unchanged? In this finite test, the aligned four-symbol portfolio has lower future logarithmic loss than all seven shifted portfolios: 0.92620 under the original rules and 0.23258 under the alternative rule. Several shifts nearly tie it, while others are worse than training only for the original purpose. Target content alone therefore does not determine transfer; its pairing with history and other tasks matters. This is a descriptive constructed-method result with uniform history allocation and supplied-law decoding, miniature — architecture untested.

Twenty fresh coefficient draws cross 16 fixed architecture cells and the
alternative lexicographic rule, giving 640 assigned finite worlds. Every world
enumerates all 4,140 partitions of eight histories. Uniform history allocation
makes all seven cyclic shifts preserve supplemental target content and its
information measures exactly. The original passive purpose stays fixed; each
portfolio shifts both supplemental target vectors together. Future answers never
select a code. The marginal-only control removes history information.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows name the training selector. Columns cross the rule family with future-purpose
or original-purpose loss. Each value averages the architecture cells and twenty
paired coefficient draws; shifts and cardinalities are not independent replications.

| Training selector, four symbols | Original rules: future | Original rules: old | Alternative rule: future | Alternative rule: old |
|---|---:|---:|---:|---:|
| Original purpose only | 0.95617 | 1.19961 | 0.28445 | 0.51167 |
| Aligned portfolio | 0.92620 | 1.20360 | 0.23258 | 0.51191 |
| Marginal-only supplement | 0.95695 | 1.19961 | 0.28894 | 0.51167 |
| Cyclic shift 1 | 0.96401 | 1.20122 | 0.35403 | 0.51209 |
| Cyclic shift 2 | 0.92836 | 1.20817 | 0.23318 | 0.51940 |
| Cyclic shift 3 | 0.96360 | 1.20195 | 0.33931 | 0.58808 |
| Cyclic shift 4 | 0.92685 | 1.20396 | 0.23261 | 0.51202 |
| Cyclic shift 5 | 0.96351 | 1.20364 | 0.35404 | 0.51212 |
| Cyclic shift 6 | 0.92844 | 1.20999 | 0.23341 | 0.52354 |
| Cyclic shift 7 | 0.96402 | 1.20968 | 0.34019 | 0.60220 |

Original rules: the seven shifted future losses average 0.94840, ranging from 0.92685 to 0.96402.
Alternative rule: the seven shifted future losses average 0.29811, ranging from 0.23261 to 0.35404.

The near ties under several shifts are retained without a significance claim.
These rotations can preserve useful structure; they are not all possible
permutations or guaranteed complete misalignment. Their compatibility with the
unchanged passive purpose and future questions can differ. Thus the comparison
does not isolate a task-independent relevance scalar or match partition difficulty.
Compared with original-purpose selection, the aligned portfolio's future gain
also comes with a small original-purpose cost in both rule groups.

The next table exposes other storage budgets. Rows are the maximum number of code
symbols; columns compare aligned selection, original-purpose selection and the
mean of seven shifts, first under original rules and then the alternative rule.
All figures are future logarithmic loss in nats. Full capacity removes the code
selection difference because every history has its own symbol.

| Symbols | Original: aligned | Original: old only | Original: mean shift | Alternative: aligned | Alternative: old only | Alternative: mean shift |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.99122 | 0.99122 | 0.99122 | 0.40333 | 0.40333 | 0.40333 |
| 2 | 0.95329 | 0.96995 | 0.97107 | 0.33636 | 0.35502 | 0.35587 |
| 4 | 0.92620 | 0.95617 | 0.94840 | 0.23258 | 0.28445 | 0.29811 |
| 8 | 0.92100 | 0.92100 | 0.92100 | 0.23249 | 0.23249 | 0.23249 |

Independent verification reconstructs all 8,320 means and exactly replays eight
fixed complete worlds. Aggregation independently scores every selected code in
all 640 worlds; source, plans, raw blocks and portable reassembly verify. The
protocol's known-answer, zero-information, matched-content, noninterference and
corruption controls pass. This is bounded whole-world replay, not independent
new-world confirmation. It is exact finite compression, not neural learning or
native construction. V15 C11/M01 failures remain unchanged.

Science cost 686.890625 charged CPU seconds and 696.372329 wall seconds; replay
added 6.703125 CPU seconds and 7.294509 wall seconds, 11.73 minutes combined.
A useful follow-on would separate content-preserving transformations that retain
the same optimal partitions from those that change them. More random shifts alone
would not resolve that structural explanation. The new matched-exposure training
round addresses the related learning question separately.

[Portable evidence](../../results/v18/exploratory-loop/P2-purpose-alignment-1/SCIENTIFIC_MANIFEST.json).
