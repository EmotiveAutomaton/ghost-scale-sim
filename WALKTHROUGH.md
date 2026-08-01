# The walkthrough

**Twenty-one pictures, in the order that makes the argument.** Each one is meant to be readable in
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

## Part five — what the machinery is for, and what it cannot keep out

### 18. What imagining a maker actually buys

![Imagining a maker is about being cheap](figures/walkthrough/18_what_imagining_a_maker_buys.png)

Plate 13 is the result that made this project withdraw a claim: you do not need to imagine a maker
to end up confidently wrong about one. True, and it stands.

But it answered a narrower question than it has been read as answering. It asked whether a
maker-model is needed to produce a *signature*. It never asked what the maker-model *buys* — and if
imagining another mind is worth what it costs to run, the payoff was never going to be a signature.

### 19. Reading a purpose nobody has shown you

![You can recognise a purpose nobody has shown you](figures/walkthrough/19_reading_an_unseen_intent.png)

This is the sharper half, and it is what *cheating the solution space* means. A reader that can
**run** the generator has the whole space available. A reader that has to **observe** the space only
ever has the part it happened to see — and more examples do not fix that, because its problem was
never a shortage of examples.

### 20. Rejection is not protection

![You cannot reject something and be unchanged](figures/walkthrough/20_rejection_is_not_protection.png)

To decide you disagree with something, you first have to work out what it says. Working out what it
says means partly running it. So refusing is itself a small act of taking on, and it compounds.

The theory always contained this term; the code never did. And the second half is worse than the
first: **the reader who studies it carefully in order to refute it drifts seven times more than the
one who skims.**

### 21. A disagreement that turned out to be a measurement error

![Pointed at the wrong thing](figures/walkthrough/21_pointed_at_the_wrong_thing.png)

The project's longest-running open question, settled by changing *what* was measured rather than
*how*. The criterion was scored on the work's purpose — which the model deliberately holds equally
readable at every depth, so it could never have moved.

---

## What is not drawn yet, and why

Version 6 held four results out of this walkthrough because each carried an open question a plate
would paper over. **Version 7 closed all four** — and one of them closed by being retired.

| result | what happened |
|---|---|
| **The two-gates criterion** | **Settled, and drawn** (plate 21). Re-run on E31's own design and scored on the method: 0.93, with the interval excluding the bar. |
| **Depletion carrying to unseen work** | **Settled.** At thirty encounters the probe falls essentially to nothing, monotone at −0.98. The direction was never in doubt; the pre-registered magnitude clause was the wrong *shape* of criterion and is reported as such. |
| **Depth versus effort** | **Still open, and reported as failing.** The pre-registered contrast still returns *effort can manufacture depth*. On what actually transfers, depth dominates effort ninety-fold — so the reader's *estimate* of depth is contaminated by effort while the *transfer* is not. Not drawn, because a plate cannot carry that split. |
| **The collapse and the invention peak sharing one band** | **Retired.** The peak is unchanged and is not in question. The co-location held only under a superseded solver, and it has been removed from the README and from the prediction card. |

One result stays undrawable, and it is the honest kind: **N21's split**. Two measures disagree about
the same manipulation, one of them is the pre-registered one, and it fails. That is a paragraph, not
a picture.

---

## What the whole thing does not do

- **It is a simulation of a mechanism, not a study of people.** There is no human data anywhere in
  it and nothing here is evidence about what real readers do.
- **The shapes are the claims; the numbers are properties of this model's dimensions.** Rebuilt from
  the prose alone by someone with no access to the code, the central effect points the same way and
  comes out fifteen times smaller. Quote directions. Do not quote multiples.
- **There is no forward test yet.** One is written, sealed with a hash, and unbuilt. Until it
  exists, this is a theory checked carefully against itself.
