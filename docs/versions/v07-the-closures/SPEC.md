# Version 7 — closing what was held back, and attacking E21 properly

**Written before any V7 code. Not edited afterwards. Deviations are logged in RESULTS_V7.md.**

---

## §0. Two jobs

**Job one: close four results that were deliberately not drawn.** Version 6 produced a walkthrough
and four findings were held out of it because a picture makes a claim hard to qualify and each of
those four carried an open question. Holding them was right. Leaving them held is not.

**Job two: attack E21 on the right axis.** E21 is the experiment that made this project withdraw a
claim, and it has never sat well. The objection is specific and it is not "I dislike the answer":

> *The reason modelling another mind is necessary is that you use your own architecture to simulate
> theirs. You are cheating the solution space. That cannot be what small-sample overfitting is
> doing.*

**That objection is correct about what E21 did not test, and E21's conclusion is correspondingly
narrower than it has been stated.** E21 asked whether a reader without a maker-model can produce the
confident-and-contradictory *signature*. It can. It did not ask what the maker-model *buys*, and
"necessary to produce a signature" and "necessary to be viable" are different claims.

If simulation is an efficiency device — which is what an active-inference account would predict,
because nature does not pay for machinery that buys nothing — then its advantage is not in the
signature at all. It is in **how much evidence you need** and **whether you can handle something you
have never seen**. Neither has been measured.

---

## §1. Hypotheses

**H7.1 — theory of mind is a sample-efficiency device.**
A reader that infers a maker by running its own goal machinery reaches competence on far less
evidence than one that learns the mapping by counting, because it is not learning the mapping — it
already has it, and is only inferring which goal is active.
Predicted: at small evidence the gap is large; it closes as evidence grows.
**Fails if** the two learning curves are the same shape, in which case simulation buys no
efficiency and E21's conclusion stands as stated.

**H7.2 — theory of mind is what lets you read an intent you have never encountered.**
A simulator can represent a goal it has never seen because it can generate it. A counter cannot: no
examples, no entry.
Predicted: on a held-out goal, the simulator retains substantial recovery and the counter is at
chance.
**Fails if** the simulator is also at chance, which would mean its advantage is entirely
within-distribution and the "cheating the solution space" account is wrong.

*Together H7.1 and H7.2 are the real necessity claim. Neither is refuted by E21 and neither has been
run.*

**H7.3 — the gate cannot fully close, and that is why propaganda works.**
The acceptance gate in this model can shut completely: a reader that rejects material integrates
exactly none of it. **The theory says otherwise and always has.** The preprint is explicit that
computing the value disagreement *itself* requires simulating the thing, and that simulation drives
learning through gating imperfections — "likely the mechanism for indoctrination and propaganda".
So the gate should have a floor it cannot go below.
Predicted: with any leak, repeated exposure to rejected material produces cumulative drift in the
reader's own priors, and **the drift is larger for readers who engage more closely** — the reader
who studies something carefully in order to refute it is more affected than one who skims.
**Fails if** drift is flat in exposure, which would make the leak a scaling constant rather than a
mechanism.

**H7.4 — trust coupling raises the labelling coverage a disclosure regime needs.**
If trust lowers the guard rather than merely misinforming, then honest labels stop being fully
protective for readers who trust the source, and the coverage threshold should move.
Predicted: the threshold rises with the coupling.
**Fails if** it does not move, in which case the coverage figure is robust to which mechanism is
right, which would be good news and worth knowing.

---

## §2. The four closures

Each names the open question and what would settle it. None is a new question about the world.

**C-1 — the two-gates result, scored on the quantity the theory names.**
Its criterion has been unstable across solvers for two passes (0.89 approximate, 0.60 exact, twice).
The retrofit showed it resolves at 0.83 when scored on how much of the maker's *method* the reader
recovered rather than the *purpose*, which the construction holds constant. **But that number came
from a reconstruction of the experiment, not from the experiment.** Re-run E31's own design with
process uptake as a scored primary.

**C-2 — the depth-versus-effort null, same treatment.** N21 reverses under exact inference and has
never been scored on the method measure. Same fix, and the last unresolved solver disagreement.

**C-3 — the crash and the invention peak.** The README and the prediction card both assert these
occupy the same band. Under the repaired model the peak is unchanged and the crash signature fires
nowhere, because the reader at partial overlap ends up *less* uncertain and drops below a threshold
in a conjunctive criterion. **Establish it under exact inference or retire it from both documents.**

**C-4 — depletion, at a length the mechanism needs and on a criterion that can work.** The
pre-registered clause is an absolute drop, and baseline engagement varies two-fold between seed
blocks, so it passes on one block of three while the mechanism reproduces on all three. Re-run
longer, score on the relative drop, and report the absolute clause as the failing original.

---

## §3. What is NOT being built, and why

- **A follow-up to the tool-hypothesis negative.** Rejected on two grounds, both the author's and
  both good. **Modelling:** looking and appreciating are not separable in the way the follow-up
  would need — you cannot look without absorbing a little, which is H7.3's whole point, so an
  intervention that says "look but do not take anything in" is not a thing a reader can do.
  **Practical:** an affordance that reads as *do not look at this* will never be adopted by the
  people who have to apply it. It has to read as *interact with this differently*, or the Ghost
  Scale is a non-starter before its effectiveness is even a question. **That is a design constraint
  on the proposal and it is recorded here rather than lost.**
- **No creator agent, still.** The Zahavian security argument remains untested. Unchanged from V6.
- **No recursion.** Unchanged from V6.

---

## §4. Nulls

N31 — **the leak reduces to the current gate.** At leak = 0, V7 reproduces V6 exactly.

N32 — **the leak does not manufacture learning.** A reader exposed to material with no structure in
it must not drift, whatever the leak. The leak passes a fraction of *what was recovered*; where
nothing was recovered there is nothing to pass.

N33 — **the sample-efficiency comparison is fair.** Both arms see the same artifacts, from the same
tape, with the same priors. Any difference is the reader.

N34 — **the held-out goal is genuinely held out.** The counter's training set contains zero examples
of it, and the simulator receives no privileged information about it beyond the shared family it
already had.

---

## §5. Design decisions that need a call

Recorded here at the point they were made, and flagged to the author because they are choices rather
than derivations.

1. **The leak's size is swept, not chosen.** There is no principled value, so picking one and
   reporting the consequence would be reporting the choice. It is swept from zero, and the
   interesting output is the *shape* of drift against leak rather than a number at one setting.
2. **The leak stays OFF by default.** It would otherwise change every result in the repository, and
   an addition that silently rewrites the record is the accretion problem the repair pass was
   written against. What it changes is reported; what it changes is not adopted here.
3. **H7.1 is scored on evidence needed to reach a fixed competence**, not on competence at a fixed
   evidence level. The second confounds the ceiling with the rate, and the claim is about the rate.
