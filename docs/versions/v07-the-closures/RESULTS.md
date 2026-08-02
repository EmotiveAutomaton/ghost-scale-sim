# Version 7: closing what was held back, and going back at the withdrawn claim

Generated from [results/v7/](../../../results/v7/). Regenerate with `python scripts/write_results_v7.py`. Criteria hash-locked at `796084debdf32cb8` before any cell ran.

---

## At a glance

| item | question | outcome |
|---|---|---|
| **C-1** (E31) | Does what a reader takes on track its own estimate of how much thinking went into the work, whichever channel produced that estimate? | ONE_MECHANISM_CONFIRMED_ON_THE_METHOD |
| **C-2** (N21) | Is depth just effort wearing a hat? | EFFORT_CAN_MANUFACTURE_DEPTH |
| **C-3** (E20) | Do the collapse and the invention peak occupy the same narrow band? | CO_LOCATION_RETIRED |
| **C-4** (E35) | Does a reader worn down by content that gives it nothing disengage from work it has never seen? | DEPLETION_CARRIES_TO_UNSEEN_WORK |
| **E45** | What does modelling a maker actually buy? | SIMULATION_IS_A_SAMPLE_EFFICIENCY_DEVICE / SIMULATION_READS_AN_INTENT_IT_HAS_NEVER_SEEN |
| **E46** | Can you reject something and be unchanged? | REJECTION_IS_NOT_PROTECTION |
| **E47** | Does the coverage figure survive the mechanism question? | THE_COVERAGE_FIGURE_IS_ROBUST_TO_THE_MECHANISM |

---

## E45: what modelling a maker actually buys

The experiment that made this project withdraw a claim asked whether you NEED to imagine a maker to end up confidently wrong about one. You don't. But that was never the interesting question. If imagining a maker is worth its cost, the payoff is that you already own the machinery -- so you need far less evidence, and you can read an intention you have never encountered. Neither had been measured.

**Evidence needed to reach 0.80 accuracy:**

| reader | examples needed |
|---|---|
| simulates a maker | **4** |
| counts co-occurrences | **512** |

**On an intention it has never seen:** the simulator scores **0.767**; the counter's best across every training size is 0.573, against chance of 0.50. the counter's training set contains zero examples of the held-out intention and more training makes no difference, which is the point: it is not short of data, it is short of a way to represent something it has not seen

> *E21 is not overturned and its withdrawal stands. What changes is the scope: 'not necessary to produce the signature' is what E21 established, and 'not necessary' is what it has been read as. These are different claims and only the first was tested.*

---

## E46: the gate cannot fully close

In this model a reader that disagrees with something absorbs exactly none of it. People do not work that way, and the theory never said they did: to decide you disagree with a claim you have to work out what it says, and working out what it says means running it. The act of refusing is itself a small act of taking on.

*the preprint already contains this term -- 'the calculation of the value disagreement itself implies simulation, which inherently drives Hebbian learning due to gating imperfections... likely the mechanism for indoctrination and propaganda'. The code never had it. This is the same class of finding version 6 made three times.*

| leak | drift after repeated rejection | accumulates? |
|---|---|---|
| 0.00 | 2.356e-05 | rank correlation 0.968 |
| 0.02 | 0.0017 | rank correlation 0.995 |
| 0.05 | 0.0089 | rank correlation 0.995 |
| 0.10 | 0.0276 | rank correlation 0.992 |
| 0.20 | 0.0724 | rank correlation 0.962 |

**The reader who studies it carefully drifts 7.09× more than the one who skims.** the reader who studies it carefully in order to refute it is more affected than the one who skims, because the leak passes a fraction of what was RECOVERED and careful reading recovers more

**And a reader absorbs its own invention.** On content with no recoverable intent at all, drift is 0.0148. fully foreign content is the condition this project has spent four versions showing makes readers confidently WRONG. Such a reader has recovered something -- a fabrication -- and the leak passes it. So a reader that invents an intent then absorbs its own invention, and the drift is comparable to what real intent produces.

> *version 6 replaced the binary engagement decision with a sigmoid, and a sigmoid never reaches zero. So the graded gate already leaked a little and nobody noticed -- E42 reported integration as 0.00 because that is what 3e-06 looks like at two decimal places. The only versions of this model that could protect a reader perfectly were the ones with the binary gate.*

---

## The four closures

### C-1 — E31

**Does what a reader takes on track its own estimate of how much thinking went into the work, whichever channel produced that estimate?**

Scored on the **method**: 0.932, interval [0.771, 0.977], against a bar of 0.70. Scored on the **purpose**, which the construction holds constant: 0.622.

History: None under the approximate solver, None under exact arithmetic, None in the retrofit reconstruction. **This is E31's own design.**

### C-2 — N21

**Is depth just effort wearing a hat?**

The pre-registered contrast gives 1.26 against a bar of 3.00 and **fails**. On what actually transfers, depth dominates effort by 96.7×.

So the reader's *estimate* of depth is contaminated by effort, and what it *takes away* is not. The original criterion decides and is reported as failing.

### C-3 — E20

**Do the collapse and the invention peak occupy the same narrow band?**

Peak at 0.05. The crash signature fires at: **nowhere**.

**the claim comes OUT of the README and the prediction card. The peak is unchanged and is not in question; what does not survive is the coincidence, which held under the approximate solver because the reader was left more uncertain at partial overlap than exact arithmetic leaves it.**

### C-4 — E35

**Does a reader worn down by content that gives it nothing disengage from work it has never seen?**

Relative drop 1.000, monotone at -0.977, over 30 encounters. The pre-registered absolute clause reads 0.142.

> *an absolute bar on a quantity whose baseline varies two-fold between seed blocks cannot be stable, and it was not: the mechanism reproduced on three blocks and the clause passed on one. Retained and reported, as every superseded criterion in this project is.*

---

## E47: does the policy number survive?

The project's one policy number says about a third of machine content has to be labelled. That number was produced by a model in which telling the truth is fully protective. If the published theory is right that trust lowers the guard, then an honest label no longer keeps the material out, and the number should move.

**THE_COVERAGE_FIGURE_IS_ROBUST_TO_THE_MECHANISM.** The threshold sits at 0.80 under the code's mechanism and 0.80 under the paper's.

*nothing here establishes that the coupled mechanism is the true one. E41 established only that the two are distinguishable. This says what the policy number would be IF the paper's mechanism is right, which is the question that has to be asked before relying on a threshold.*

*E16's figure is not overturned. It is the answer under one of two mechanisms, and until the mechanism is settled the honest statement is a range rather than a number.*

---

## What was deliberately not built

- no follow-up to the tool-hypothesis negative. Looking and absorbing are not separable in the way it would need, which is H7.3's whole point, and an affordance that reads as 'do not look at this' would never be adopted by the people who have to apply it. It has to read as 'interact with this differently'.
- no creator agent, so the Zahavian security argument stays untested
- no recursion

*Generated from results/v7/ by `scripts/write_results_v7.py`. Every number above is read out of a verdict file; none is typed in.*
