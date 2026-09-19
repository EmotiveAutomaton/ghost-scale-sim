# V18.3 research results in progress

## A: purpose and paid investigation

Does the purpose of investigation change which evidence is useful? In the uniform-access comparison, choosing demonstrations for learning to act produced 45.31% own-task success, versus 25.94% for uncertainty reduction when both made exactly three attempts. The action-focused reader predicted later behavior less accurately: expected logarithmic loss, where lower is better, was 0.997 versus 0.638. When access depended on hidden maker state, the corresponding success rates were 32.19% and 9.06%. This is descriptive evidence about constructed mechanisms: useful evidence depends on the reader's purpose, and learning to act is not interchangeable with identifying a maker. It does not establish human learning or a universal selector advantage.

The table compares native three-cell task success and future predictive loss. Every row makes exactly three query attempts; missing responses remain charged. Values average 16 architecture cells within each of 20 coefficient draws. Logarithmic loss is measured in nats and lower is better.

| Access mechanism | Selector objective | Own-task success | Future logarithmic loss |
|---|---|---:|---:|
| Uniform | Learn to act | 45.31% | 0.99745 |
| Uniform | Reduce state uncertainty | 25.94% | 0.63825 |
| Query-dependent | Learn to act | 39.38% | 0.95358 |
| Query-dependent | Reduce state uncertainty | 22.50% | 0.64050 |
| Hidden-state-dependent | Learn to act | 32.19% | 1.04767 |
| Hidden-state-dependent | Reduce state uncertainty | 9.06% | 0.79390 |

The stopping prediction selector makes 1.22 attempts under uniform access and reaches loss 0.64096; the three-attempt fixed diagnostic reaches 0.64010. Fewer requests alone do not establish a total-cost advantage. Selector cost counts likelihood-table entries evaluated under an explicit price, not hardware instructions. Primitive observation, practice, planning and final execution costs remain separate. Whole-run CPU is recorded independently. The robust selector tests three access-prior perturbations, not an unrestricted distributionally robust optimum.

### Repair, validity and scope

The first implementation recorded four robust likelihood evaluations but charged only one when deciding whether to stop. The repair charges all four in the stopping utility. A planted regression control passes only if the ordinary selector continues and the robust selector stops at a utility between those costs. Sixteen isolated tests passed. Matched three-attempt task and information arms were also added. The four original packets and their complete proofs remain **superseded**, never an independent replication; the four `-r1` packets replace them. Ordinary existing task outputs are unchanged.

The current 1,088 evaluations comprise three access conditions of 320 and 128 known-answer controls. Means and uncertainty intervals in [A_ROLLUP.json](../../../../results/v18/research-extension/A_ROLLUP.json) average architecture cells within coefficient draw; all intervals are descriptive. Complete raw checks and mean reaggregation passed, with eight fixed whole-unit source-extracted replays per packet, 32 current replays total. This is bounded replay, not a full rerun. No-information, resolved-state, failed-access and identical-evidence controls remain explicit. Native craft uptake reuses actual repeated-fragment learning. Historical state discrimination concerns the declared finite state family, not an arbitrary recovered biography.

The miniature varies decision rule, curriculum interference, available opportunities and repeated sources. Those 16 constructions are tested architectures, not a representative architecture population. Findings remain discovery; fresh third-rule severity and the remaining families are pending.
