# V18 worked cases

Outcome-selected illustrations, chosen by lowest case/target hash per observed type.
Absent types are not fabricated. These contain evaluator truth; they are not blind reader requests.
Actions 0–3 place cells 0–3; 4–7 remove cells 0–3. Endpoints below list occupied cells.

## focused acquisition benefit

Case `061e86636736467af97a0d864dcd01e9ef553739cb6e85d89cc3d84f290b8666`; budget 32; compatible target [0, 2, 3].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [0, 3] (A), [1, 2] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: A, 1: B, 2: B, 3: A, 4: B, 5: B, 6: A, 7: A, 8: B, 9: A, 10: B, 11: B, 12: A, 13: A, 14: B, 15: A, 16: B, 17: A, 18: B, 19: B, 20: A, 21: B, 22: B, 23: A, 24: B, 25: A, 26: B, 27: A, 28: A, 29: A, 30: B, 31: A.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]; learned [[1, 2]]; top ties [[1, 2]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 1 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 2 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 3 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 4 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 5 | B | [1, 3] | [1, 3] | [1, 3] | False |
| 6 | A | [6, 3] | [6, 1] | [1] | False |
| 7 | A | [5, 3] | [5, 3] | [3] | False |
| 8 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 9 | A | [0, 7] | [0, 7] | [0] | False |
| 10 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 11 | B | [1, 3] | [1, 3] | [1, 3] | False |
| 12 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 13 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 14 | B | [1, 5] | [1, 5] | [] | False |
| 15 | A | [1, 3] | [1, 6] | [1] | False |

Focused selected [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 15, 17, 20, 23, 25]; learned [[0, 3]]; top ties [[0, 3]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 1 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 2 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 3 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 4 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 5 | B | [1, 3] | [1, 3] | [1, 3] | False |
| 6 | A | [6, 3] | [6, 1] | [1] | False |
| 7 | A | [5, 3] | [5, 3] | [3] | False |
| 9 | A | [0, 7] | [0, 7] | [0] | False |
| 12 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 13 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 15 | A | [1, 3] | [1, 6] | [1] | False |
| 17 | A | [0, 0] | [0, 0] | [0] | False |
| 20 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 23 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 25 | A | [0, 3] | [0, 7] | [0] | False |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [] | [] | False | 0 / 32 / 0 |
| broad-unchecked | [1, 2] | [] | [] | False | 0 / 31 / 0 |
| focused-unchecked | [0, 3] | [0, 3, 2] | [0, 2, 3] | True | 0 / 22 / 3 |
| broad-checked | [0] | [] | [] | False | 2 / 30 / 0 |
| focused-checked | [0, 3] | [0, 3, 2] | [0, 2, 3] | True | 2 / 22 / 3 |
| broad-checked-extra | [0] | [] | [] | False | 2 / 32 / 0 |
| focused-checked-extra | [0, 3] | [0, 3, 2] | [0, 2, 3] | True | 2 / 22 / 3 |

Broad checking: action 1: [] -> [1], target distance 3 -> 4; action 2: [1] -> [1, 2], target distance 4 -> 3. Retained [].

Focused checking: action 0: [] -> [0], target distance 3 -> 2; action 3: [0] -> [0, 3], target distance 2 -> 1. Retained [[0, 3]].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.

## checking benefit

Case `cfe8dc48b35b5213aff5196ce5af07704ef7b04ea48804bca0826fd970b4bc9a`; budget 32; changed target [0, 3].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [1, 3] (A), [2, 0] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: B, 1: B, 2: A, 3: B, 4: A, 5: B, 6: A, 7: A, 8: B, 9: B, 10: A, 11: A, 12: B, 13: A, 14: B, 15: B, 16: A, 17: B, 18: B, 19: B, 20: B, 21: B, 22: A, 23: A, 24: A, 25: B, 26: A, 27: B, 28: A, 29: A, 30: A, 31: A.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]; learned [[1, 3]]; top ties [[1, 3], [2, 0]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 1 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 2 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 3 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 4 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 5 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 6 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 7 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 8 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 9 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 10 | A | [5, 3] | [5, 3] | [3] | False |
| 11 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 12 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 13 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 14 | B | [2, 0] | [2, 5] | [2] | False |
| 16 | A | [1, 3] | [1, 3] | [1, 3] | True |

Focused selected [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 16, 22, 23, 24, 26]; learned [[1, 3]]; top ties [[1, 3]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 1 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 2 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 3 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 4 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 5 | B | [2, 0] | [2, 0] | [0, 2] | True |
| 6 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 7 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 10 | A | [5, 3] | [5, 3] | [3] | False |
| 11 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 13 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 16 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 22 | A | [0, 3] | [0, 3] | [0, 3] | False |
| 23 | A | [1, 3] | [1, 3] | [1, 3] | True |
| 24 | A | [1, 3] | [1, 7] | [1] | False |
| 26 | A | [1, 3] | [1, 3] | [1, 3] | True |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [0, 3] | [0, 3] | True | 0 / 16 / 2 |
| broad-unchecked | [1, 3] | [] | [] | False | 0 / 31 / 0 |
| focused-unchecked | [1, 3] | [] | [] | False | 0 / 31 / 0 |
| broad-checked | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| focused-checked | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| broad-checked-extra | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| focused-checked-extra | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |

Broad checking: action 1: [] -> [1], target distance 2 -> 3; action 3: [1] -> [1, 3], target distance 3 -> 2. Retained [].

Focused checking: action 1: [] -> [1], target distance 2 -> 3; action 3: [1] -> [1, 3], target distance 3 -> 2. Retained [].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.

## checking budget cost reversal

Case `b7adf4135f1057c7715602a91c6f7fc51ab0901b6c8eac6c951e8cf89bfb5b99`; budget 32; changed target [1, 3].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [0, 3] (A), [1, 2] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: B, 1: A, 2: B, 3: B, 4: B, 5: B, 6: B, 7: A, 8: A, 9: B, 10: B, 11: A, 12: B, 13: A, 14: B, 15: A, 16: A, 17: A, 18: B, 19: B, 20: A, 21: A, 22: A, 23: B, 24: A, 25: B, 26: B, 27: A, 28: B, 29: A, 30: A, 31: A.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17]; learned [[0, 3]]; top ties [[0, 3], [1, 2]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [3, 2] | [3, 1] | [1, 3] | False |
| 1 | A | [0, 7] | [0, 7] | [0] | False |
| 2 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 3 | B | [7, 2] | [7, 2] | [2] | False |
| 4 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 5 | B | [1, 2] | [1, 3] | [1, 3] | False |
| 6 | B | [1, 4] | [1, 2] | [1, 2] | True |
| 7 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 8 | A | [4, 3] | [4, 5] | [] | False |
| 9 | B | [2, 2] | [2, 2] | [2] | False |
| 10 | B | [5, 2] | [0, 2] | [0, 2] | False |
| 11 | A | [0, 3] | [0, 6] | [0] | False |
| 13 | A | [0, 1] | [0, 1] | [0, 1] | False |
| 15 | A | [0, 5] | [0, 5] | [0] | False |
| 16 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 17 | A | [0, 3] | [0, 3] | [0, 3] | True |

Focused selected [0, 1, 2, 3, 4, 7, 8, 11, 13, 15, 16, 17, 20, 21, 22, 24]; learned [[0, 3]]; top ties [[0, 3]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [3, 2] | [3, 1] | [1, 3] | False |
| 1 | A | [0, 7] | [0, 7] | [0] | False |
| 2 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 3 | B | [7, 2] | [7, 2] | [2] | False |
| 4 | B | [1, 2] | [1, 2] | [1, 2] | True |
| 7 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 8 | A | [4, 3] | [4, 5] | [] | False |
| 11 | A | [0, 3] | [0, 6] | [0] | False |
| 13 | A | [0, 1] | [0, 1] | [0, 1] | False |
| 15 | A | [0, 5] | [0, 5] | [0] | False |
| 16 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 17 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 20 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 21 | A | [0, 3] | [0, 0] | [0] | False |
| 22 | A | [0, 3] | [0, 3] | [0, 3] | True |
| 24 | A | [0, 3] | [0, 3] | [0, 3] | True |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [1, 3] | [1, 3] | True | 0 / 32 / 2 |
| broad-unchecked | [0, 3] | [] | [] | False | 0 / 31 / 0 |
| focused-unchecked | [0, 3] | [] | [] | False | 0 / 31 / 0 |
| broad-checked | [0] | [] | [] | False | 2 / 30 / 0 |
| focused-checked | [0] | [] | [] | False | 2 / 30 / 0 |
| broad-checked-extra | [0] | [1, 3] | [1, 3] | True | 2 / 32 / 2 |
| focused-checked-extra | [0] | [1, 3] | [1, 3] | True | 2 / 32 / 2 |

Broad checking: action 0: [] -> [0], target distance 2 -> 3; action 3: [0] -> [0, 3], target distance 3 -> 2. Retained [].

Focused checking: action 0: [] -> [0], target distance 2 -> 3; action 3: [0] -> [0, 3], target distance 3 -> 2. Retained [].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.

## no success gain with an acquired library

Case `95d36d5b47f599f73e3903342ff20a5daadae9a76cc253e2b2916f863402a430`; budget 128; changed target [0, 3].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [0, 1] (A), [2, 3] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: A, 1: A, 2: A, 3: B, 4: A, 5: A, 6: A, 7: A, 8: B, 9: B, 10: A, 11: A, 12: B, 13: A, 14: A, 15: B, 16: B, 17: A, 18: A, 19: A, 20: A, 21: B, 22: A, 23: B, 24: B, 25: B, 26: B, 27: B, 28: B, 29: B, 30: B, 31: B.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 21, 23]; learned [[0, 1]]; top ties [[0, 1]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 1 | A | [0, 7] | [0, 7] | [0] | False |
| 2 | A | [2, 1] | [2, 1] | [1, 2] | False |
| 3 | B | [7, 3] | [7, 3] | [3] | False |
| 4 | A | [0, 1] | [0, 0] | [0] | False |
| 5 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 6 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 7 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 8 | B | [5, 3] | [1, 3] | [1, 3] | False |
| 9 | B | [2, 6] | [2, 6] | [] | False |
| 10 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 12 | B | [2, 3] | [2, 3] | [2, 3] | True |
| 15 | B | [2, 4] | [2, 4] | [2] | False |
| 16 | B | [2, 3] | [2, 3] | [2, 3] | True |
| 21 | B | [0, 3] | [0, 7] | [0] | False |
| 23 | B | [7, 3] | [7, 3] | [3] | False |

Focused selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]; learned [[0, 1]]; top ties [[0, 1]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 1 | A | [0, 7] | [0, 7] | [0] | False |
| 2 | A | [2, 1] | [2, 1] | [1, 2] | False |
| 3 | B | [7, 3] | [7, 3] | [3] | False |
| 4 | A | [0, 1] | [0, 0] | [0] | False |
| 5 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 6 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 7 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 8 | B | [5, 3] | [1, 3] | [1, 3] | False |
| 9 | B | [2, 6] | [2, 6] | [] | False |
| 10 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 11 | A | [0, 1] | [0, 5] | [0] | False |
| 12 | B | [2, 3] | [2, 3] | [2, 3] | True |
| 13 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 14 | A | [0, 1] | [0, 1] | [0, 1] | True |
| 17 | A | [0, 1] | [0, 1] | [0, 1] | True |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [0, 3] | [0, 3] | True | 0 / 16 / 2 |
| broad-unchecked | [0, 1] | [0, 3] | [0, 3] | True | 0 / 48 / 2 |
| focused-unchecked | [0, 1] | [0, 3] | [0, 3] | True | 0 / 48 / 2 |
| broad-checked | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| focused-checked | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| broad-checked-extra | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |
| focused-checked-extra | [0] | [0, 3] | [0, 3] | True | 2 / 16 / 2 |

Broad checking: action 0: [] -> [0], target distance 2 -> 1; action 1: [0] -> [0, 1], target distance 1 -> 2. Retained [].

Focused checking: action 0: [] -> [0], target distance 2 -> 1; action 1: [0] -> [0, 1], target distance 1 -> 2. Retained [].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.

## residual interference after checking

Case `eaa14b6d07ab7184184e6d00b200aee7513386f34d9965e2ad3c780723b76f55`; budget 32; changed target [1, 3].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [3, 2] (A), [0, 1] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: B, 1: A, 2: B, 3: B, 4: A, 5: A, 6: B, 7: A, 8: B, 9: A, 10: A, 11: A, 12: B, 13: B, 14: A, 15: B, 16: B, 17: A, 18: B, 19: A, 20: A, 21: B, 22: A, 23: B, 24: B, 25: A, 26: B, 27: A, 28: B, 29: A, 30: A, 31: B.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]; learned [[3, 2]]; top ties [[3, 2]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [0, 1] | [3, 1] | [1, 3] | False |
| 1 | A | [3, 0] | [6, 0] | [0] | False |
| 2 | B | [0, 1] | [0, 1] | [0, 1] | True |
| 3 | B | [0, 1] | [0, 1] | [0, 1] | True |
| 4 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 5 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 6 | B | [2, 1] | [2, 1] | [1, 2] | False |
| 7 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 8 | B | [0, 1] | [0, 1] | [0, 1] | True |
| 9 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 10 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 11 | A | [3, 2] | [3, 1] | [1, 3] | False |
| 12 | B | [5, 1] | [5, 1] | [1] | False |
| 13 | B | [0, 7] | [0, 7] | [0] | False |
| 14 | A | [1, 2] | [1, 2] | [1, 2] | False |
| 15 | B | [3, 1] | [3, 1] | [1, 3] | False |

Focused selected [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 17, 19, 20, 22]; learned [[3, 2]]; top ties [[3, 2]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [0, 1] | [3, 1] | [1, 3] | False |
| 1 | A | [3, 0] | [6, 0] | [0] | False |
| 2 | B | [0, 1] | [0, 1] | [0, 1] | True |
| 3 | B | [0, 1] | [0, 1] | [0, 1] | True |
| 4 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 5 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 6 | B | [2, 1] | [2, 1] | [1, 2] | False |
| 7 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 9 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 10 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 11 | A | [3, 2] | [3, 1] | [1, 3] | False |
| 14 | A | [1, 2] | [1, 2] | [1, 2] | False |
| 17 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 19 | A | [3, 5] | [3, 5] | [3] | False |
| 20 | A | [3, 2] | [3, 2] | [2, 3] | True |
| 22 | A | [3, 2] | [1, 2] | [1, 2] | False |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [1, 3] | [1, 3] | True | 0 / 32 / 2 |
| broad-unchecked | [3, 2] | [] | [] | False | 0 / 31 / 0 |
| focused-unchecked | [3, 2] | [] | [] | False | 0 / 31 / 0 |
| broad-checked | [0] | [] | [] | False | 2 / 30 / 0 |
| focused-checked | [0] | [] | [] | False | 2 / 30 / 0 |
| broad-checked-extra | [0] | [1, 3] | [1, 3] | True | 2 / 32 / 2 |
| focused-checked-extra | [0] | [1, 3] | [1, 3] | True | 2 / 32 / 2 |

Broad checking: action 3: [] -> [3], target distance 2 -> 1; action 2: [3] -> [2, 3], target distance 1 -> 2. Retained [].

Focused checking: action 3: [] -> [3], target distance 2 -> 1; action 2: [3] -> [2, 3], target distance 1 -> 2. Retained [].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.

## unchecked interference

Case `b17856c16519d2e9872899c28f264ae740de4384793aa3f6ad0411b3b7549176`; budget 32; changed target [0, 1].
Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.
Private source motifs: [1, 2] (A), [3, 0] (B).
These are evaluator annotations, not learner evidence.

Available offers (index: topic): 0: B, 1: A, 2: B, 3: A, 4: B, 5: A, 6: A, 7: B, 8: B, 9: A, 10: A, 11: B, 12: A, 13: B, 14: B, 15: A, 16: B, 17: A, 18: B, 19: B, 20: B, 21: A, 22: A, 23: A, 24: B, 25: A, 26: A, 27: A, 28: B, 29: B, 30: A, 31: B.

Broad selected [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]; learned [[1, 2]]; top ties [[1, 2], [3, 0]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 1 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 2 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 3 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 4 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 5 | A | [0, 2] | [0, 2] | [0, 2] | False |
| 6 | A | [4, 2] | [4, 2] | [2] | False |
| 7 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 8 | B | [3, 0] | [3, 2] | [2, 3] | False |
| 9 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 10 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 11 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 12 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 13 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 14 | B | [3, 5] | [3, 5] | [3] | False |
| 15 | A | [1, 2] | [1, 2] | [1, 2] | True |

Focused selected [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 15, 17, 21, 22, 23]; learned [[1, 2]]; top ties [[1, 2]].

Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.

| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |
|---:|---|---|---|---|---|
| 0 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 1 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 2 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 3 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 4 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 5 | A | [0, 2] | [0, 2] | [0, 2] | False |
| 6 | A | [4, 2] | [4, 2] | [2] | False |
| 7 | B | [3, 0] | [3, 0] | [0, 3] | True |
| 9 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 10 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 12 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 15 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 17 | A | [1, 7] | [1, 0] | [0, 1] | False |
| 21 | A | [2, 2] | [2, 2] | [2] | False |
| 22 | A | [1, 2] | [1, 2] | [1, 2] | True |
| 23 | A | [1, 7] | [1, 7] | [1] | False |

Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.

| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |
|---|---|---|---|---|---|
| primitive | [0] | [0, 1] | [0, 1] | True | 0 / 12 / 2 |
| broad-unchecked | [1, 2] | [] | [] | False | 0 / 31 / 0 |
| focused-unchecked | [1, 2] | [] | [] | False | 0 / 31 / 0 |
| broad-checked | [0] | [0, 1] | [0, 1] | True | 2 / 12 / 2 |
| focused-checked | [0] | [0, 1] | [0, 1] | True | 2 / 12 / 2 |
| broad-checked-extra | [0] | [0, 1] | [0, 1] | True | 2 / 12 / 2 |
| focused-checked-extra | [0] | [0, 1] | [0, 1] | True | 2 / 12 / 2 |

Broad checking: action 1: [] -> [1], target distance 2 -> 1; action 2: [1] -> [1, 2], target distance 1 -> 2. Retained [].

Focused checking: action 1: [] -> [1], target distance 2 -> 1; action 2: [1] -> [1, 2], target distance 1 -> 2. Retained [].

The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.
