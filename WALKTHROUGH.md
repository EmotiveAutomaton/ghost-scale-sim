# The walkthrough

**Seventeen pictures, in the order that makes the argument.** Each one is meant to be readable in
about two seconds. If you get to the end you will know what this project claims, what it withdrew,
and what it still cannot answer.

Every number on every plate is read out of a committed results file, named in the plate's own
footer. Regenerate the whole set with `python scripts/make_walkthrough_plates.py`.

*The older per-experiment charts have moved to [figures/archive/](figures/archive/). They are still
correct as records of what their version measured; they are research figures rather than
explanations, and several of them now mean something different from what they meant when they were
drawn.*

---

## Part one — what happens when you lie about who made something

### 1. The central result

![A false label moves you away from the truth](figures/walkthrough/01_false_label_moves_you_wrong.png)

The measure that made this visible is newer than the result. For five versions the model scored
*how far a reader's beliefs travelled*, which cannot tell being convinced from being fooled — a
reader who ends up confidently wrong has moved just as far as one who ends up right. Given a sign,
the three cells stop looking similar and start having opposite signs.

### 2. The two witnesses

![Two witnesses arrive with every glance](figures/walkthrough/16_two_witnesses.png)

Why the lie works at all. The reader gets two pieces of evidence about origin at every glance: what
the label says, and what the work itself says. On a lie they point in opposite directions, and which
one wins is decided by an inequality you can solve on paper. **This was computed with no simulation
at all, and it predicted the shape of several results that had already been run.**

### 3. Honesty is not always enough

![If trust lowers your guard, honesty stops being enough](figures/walkthrough/06_honesty_is_not_enough.png)

The most consequential thing version 6 found, and it is a disagreement between the published theory
and its own implementation. The code models a **con**: you were lied to, you were fooled, the fix is
disclosure. The paper models something worse: trust *itself* lowers the guard, so a trusted source
gets absorbed **even when it tells you exactly what it is**. Disclosure does not fix that one.

Both accounts produce the famous result. Only one of them survives being told the truth.

### 4. Reputation blindness

![The readers most inclined to believe a label can never learn the labeller lies](figures/walkthrough/07_reputation_blindness.png)

Follow the previous plate through and you get a prediction the earlier model structurally could not
make. To learn that a source lies you have to notice the label and the work disagreeing — and above
the crossover, the label has already won that argument before the disagreement can register.

Not slow learning. Learning that cannot start.

---

## Part two — what unreadable content does to a reader

### 5. Where it breaks

![Invention peaks in the middle, not at the empty end](figures/walkthrough/02_invention_peaks_in_the_middle.png)

The most robust result in the project: same location under exact arithmetic, on a different seed
block, at double scale, in every cell of the robustness sweep, and in every resampled run. **The
worst place to be is nearly understandable.** Total nonsense is safe; total clarity is safe; the
danger is having just enough handholds to build a story on.

### 6. Legible and empty

![Legible and empty is a different failure](figures/walkthrough/03_legible_and_empty.png)

The model used to describe unreadable machine content as *written in a vocabulary you don't have*.
That is not what people report. What they report is reading every word and finding nobody there.

So version 6 built a third condition: content made of entirely familiar material whose maker cannot
be reconstructed from it, because several different maker-states produce the same surface. Full
vocabulary, no inversion. It produces a signature neither existing condition does.

### 7. Two kinds of damage

![One kind of damage scales; the other is already there at zero](figures/walkthrough/12_two_kinds_of_damage.png)

Absorbing bad material gets worse the more of it there is — the obvious kind. The other kind is
driven by *walking away*, so it is fully present in a corpus with no machine content in it at all.
The second has the strongest independent support of anything in this project.

### 8. Labelling needs a convention

![Labelling only protects readers who know the labelling exists](figures/walkthrough/15_labels_need_a_convention.png)

The policy number. And a lower bound, because the aware reader here is handed the true coverage,
which is the most generous assumption available.

---

## Part three — reading the maker

### 9. Depth moves the method, not the purpose

![Depth changes how much of the method you pick up](figures/walkthrough/04_depth_moves_the_method.png)

**Five versions looked for an effect in the one place the design guarantees there isn't one.**

Depth here is the Zen master's circle against the child's scribble: what differs is compressed
practice, not effort spent. The construction deliberately makes a deep work and a shallow one state
their purpose *equally clearly* — that is what stops "depth" being "clarity" with a new name.

And then the experiment measured whether depth changes how much of the **purpose** you get. It
couldn't. Not "it didn't" — it *couldn't*. Measured on how much of the **method** transfers, it
moves.

### 10. Intent unlocks the method

![Work out what someone was trying to do and their choices start making sense](figures/walkthrough/05_intent_unlocks_the_method.png)

The claim: you work out what someone was *for* first, and that is what makes their choices
readable — every move can then be read as being in service of it.

It is in neither the preprint nor the essay. It came out of a conversation, and it holds — inside a
single reading, on the same object, before and after the reader settles on the purpose.

### 11. The master cannot explain themselves

![The more practised the work, the less its maker can say why](figures/walkthrough/11_the_master_cannot_explain.png)

A novice can tell you exactly which rule they were following, because they are still following it on
purpose. Practice compresses decisions into automatic routines, and compression is precisely what
puts them out of reach of report.

Note what is *not* happening here: nobody is setting the maker's self-blindness. Depth sets it.

### 12. Looking is not the same as being changed

![Paying attention and being willing to be changed are not the same thing](figures/walkthrough/10_looking_is_not_being_changed.png)

Attention is what you spend. Whether what you find is allowed to change you is a separate decision,
and every combination of the two is reachable. The model always kept them apart; no version had ever
reported them separately.

---

## Part four — the unwelcome results

### 13. You don't need a mind to invent one

![A counting classifier does it too](figures/walkthrough/13_no_mind_needed.png)

**The framework used to claim that confident, mutually contradictory readings of empty content
require a reader that models the maker as a mind. That is false, and the experiment that killed it
is in this repository.**

A classifier that counts features and never represents a maker reproduces the pattern, through
nothing more than small-sample overfitting. What it *cannot* do is respond to a label, or keep
paying attention to something it cannot resolve — and those are the two the framework now rests on.

### 14. A knee, not a cliff

![Competence bends rather than falling off a cliff](figures/walkthrough/14_a_knee_not_a_cliff.png)

The author's own claim, tested knowing it could only survive or weaken. A genuine threshold gets
sharper as you gather evidence; this one did not move across sixteen times the data, so its shape
comes from the model rather than from a boundary in the world.

### 15. Expertise substitutes

![Learning to read machine work swaps a skill out](figures/walkthrough/08_expertise_substitutes.png)

The prediction was that people who understand these systems would be spared the crash. They are —
by trading away the human channel. The adaptation that protects you is the same adaptation that
costs you.

### 16. Pays more, gets less

![Optimise the signal of depth and readers pay more for less](figures/walkthrough/09_pays_more_gets_less.png)

A third failure mode, distinct from the other two. It is not the crash: the reader is fully engaged.
It is not the lie: nobody lied, and the reader is right about where the work came from. It is a
reader correctly reading something built to trip its own heuristic for deciding what is worth
reading.

This is the alignment argument, in miniature, inside the model.

### 17. The experiment that stayed withheld

![One experiment has been withheld three times](figures/walkthrough/17_the_withheld_experiment.png)

It asks whether damage compounds as each generation learns from the last — the most consequential
question the project has, and the one it cannot answer. Its honesty check has failed three times, so
real decay cannot be told apart from the instrument's own noise.

Narrowly missing is exactly the case a no-exceptions rule exists for.

---

## What is not drawn yet, and why

A plate makes a claim legible in two seconds, which also makes it hard to qualify. Four results
carry an open question that a plate would paper over, so they are named here instead.

| result | why it is not drawn |
|---|---|
| **The collapse and the invention peak occupying the same band** | The peak is settled. The *co-location* is not: under exact arithmetic the reader at that point ends up less uncertain, so the conjunction stops holding. Needs re-establishing or retiring before it is drawn. |
| **The two-gates criterion at 0.83** | That number comes from the retrofit, which re-ran the relevant cells with the new measure. The experiment itself has not been re-run with process uptake as its scored primary. The reading is almost certainly right and the number should not be posted until it comes from the experiment rather than from a reconstruction of it. |
| **Depletion carrying to unseen work** | The mechanism reproduces on every seed block tested. The pre-registered magnitude threshold is met on one of three, because it is an absolute threshold on a quantity whose baseline moves. The direction is drawable; the size is not, and a plate would imply the size. |
| **Depth versus effort** | Reverses under exact arithmetic and has not been re-scored on the method measure. Same fix as the two-gates item, and probably the same outcome, but it has not been run. |

**Four experiments' worth of work would close all of it**, and that work is listed in
[FINDINGS.md](FINDINGS.md) under "What is still open".

---

## What the whole thing does not do

- **It is a simulation of a mechanism, not a study of people.** There is no human data anywhere in
  it and nothing here is evidence about what real readers do.
- **The shapes are the claims; the numbers are properties of this model's dimensions.** Rebuilt from
  the prose alone by someone with no access to the code, the central effect points the same way and
  comes out fifteen times smaller. Quote directions. Do not quote multiples.
- **There is no forward test yet.** One is written, sealed with a hash, and unbuilt. Until it
  exists, this is a theory checked carefully against itself.
