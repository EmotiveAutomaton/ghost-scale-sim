# The walkthrough

**Twenty-seven pictures, arranged as one story.** Read in order, they say what a reader does when
it looks at made work, what machine-made content does to that reading, why lying about origin is
the sharpest damage, what a steady diet costs, what actually protects, and exactly how much of
all that you should believe. Each plate is meant to be readable in about two seconds. If you get
to the end you will know what this project claims, what it withdrew, and what it still cannot
answer.

> **Mixed, and scored plate by plate. The mark sits at the end of each heading.**
> **◐ Curator, 60%: ten plates. ○ Ghost, 5%: seventeen plates.**
> **This page's own prose is ◐ Curator, 60%.**

Every number on every plate is read out of a committed results file, named in the plate's own
footer. Regenerate the whole set with `python scripts/make_walkthrough_plates.py`, which also runs
an automatic audit that fails any plate with text running off the canvas or printed over other text,
because both of those happened here and neither is visible to anything but an eye on the image.

**The numbering here is reading order, not drawing order.** The image filenames keep the order the
plates were drawn in. Neither ordering changes any number. The story runs in six acts, and the
closing section, *what the whole thing does not do*, is the one to read if you read nothing
else.

---

## Read the title colour first

**The Ghost Scale is running on these slides, and the plates are scored as artifacts rather than as
claims.**

| what you see | tier | what it means |
|---|---|---|
| **A black title** | **◐ Curator, 60%** | A person chose the claim, wrote the headline, and checked it against the results file. A machine drew everything else. The headline itself is Polished, the words are the author's, the grammar is not. |
| *A grey title* | **○ Ghost, 5%** | A machine wrote the headline from the author's numbers, end to end. **Grey means not yet been through the author. It does not mean less important, and it does not mean less true.** |

Grey plates are still being worked through. Several of them carry the project's strongest results and
are waiting on a headline in the author's own hand, not on better evidence.

---

## Act one · *a reader really does recover the maker*

*The premise, before anything breaks: when you look at made work, part of what you do is
reconstruct the person, what they were for, and how they went about it. These four plates are the
evidence that the reconstruction is real, ordered, and sometimes better than the maker's own
memory.*

### 1. The math of empathy ◐

![Working out the purpose is what makes the method readable](figures/walkthrough/05_intent_unlocks_the_method.png)

The claim, in plain words: you work out what someone was *for* first, and that is what makes their
choices readable. It is in neither the preprint nor the essay. It came out of a conversation. It
has now been tested four times and the fourth is the one to read.

| form | what it did |
|---|---|
| **pre-registered, between readers** | readers who ended up *right* about the goal against readers who ended up *wrong*, on whole-rollout process recovery. Gap 0.047 against a required 0.15. **Fails, reported as failing.** |
| **temporal, added and declared** | method uptake before the goal settles, 0.050, against after, 0.130. Decisive-looking, with an obvious hole: *after* is simply **later**. |
| **V-11a, placebo split** | cut each reading at a sham point drawn from the settling-time distribution. Survives at **+0.020**, [+0.001, +0.038]. Looked thin. |
| **V-11c, this plate** | the control was eating the signal, and the axis the hypothesis actually names had never been used. |

**Why V-11a understated it.** Settling times bunch hard at the front of a reading: 104 of 320
readings settle at step 2. A sham drawn from that distribution lands within two steps of the truth
about a third of the time, so a third of the "control" readings were themselves post-settling.
Forced four steps clear, the same test returns **+0.031** rather than +0.020.

**Where the information actually arrives.** Align every reading on its own settling step and average
the reader's information about the maker's true execution mode, step by step. It sits at **+0.036**
in the five steps before, and **+0.097** from the arrival step onward. The jump is at lag −1, one
step before the detector fires, which is correct: the observation that drops the goal entropy below
threshold is the observation that carries the information, and the threshold is noticed afterwards.
Fit the pre-event slope and carry it forward and a pure accumulation predicts 0.067. The actual
level is **+0.030 above that line.** It is a step, not a ramp.

**And the axis nobody had used.** Beta is how legible the maker's goal is, and the hypothesis names
it directly: if working out the purpose is what unlocks the method, then the unlock must die when
the purpose cannot be read.

| how readable the purpose is | extra method picked up |
|---|---|
| fully | **+0.123** [+0.079, +0.168] |
| half | **+0.046** [+0.014, +0.080] |
| not at all | **−0.002** [−0.040, +0.034] |

**Monotone, and zero at the bottom.** Elapsed time does not know what beta is. No amount of *it just
accrued* produces that shape.

**And it is understanding rather than commitment.** Readers who settle on the *wrong* goal gain
**+0.007**. Readers who settle on the right one gain **+0.070**, a difference of **+0.064**,
interval [+0.012, +0.116]. It is not that the reader stopped being uncertain. It is what it became
certain *of*.

**On the size.** The whole process signal a reader extracts across an entire reading is about
**0.091 nats**, so the far-sham difference is about **a third of everything the reader ever gets
about how the work was made**, and at full goal legibility the unlock is larger than the average
whole-reading total. The raw numbers look small because nats are small here. Against their own
denominator they are not.

*Declared: V-11a first ran at 40 readings per cell with an interval crossing zero and was re-run at
120. The point estimate barely moved. Raising n after a null is a forking path and it is on the
ledger.*

### 2. Depth moves the method, not the purpose ○

![Depth changes how much of the method you pick up](figures/walkthrough/04_depth_moves_the_method.png)

**Five versions looked for an effect in the one place the design guarantees there isn't one.**

Depth here is the Zen master's circle against the child's scribble: what differs is compressed
practice, not effort spent. The construction deliberately makes a deep work and a shallow one state
their purpose *equally clearly*, that is what stops "depth" being "clarity" with a new name.

And then the experiment measured whether depth changes how much of the **purpose** you get. It
couldn't. Not "it didn't", it *couldn't*. Measured on how much of the **method** transfers, it
moves.

### 3. Expertise bakes in, and skills transfer anyway ◐

![The more practised the work, the less its maker can say why](figures/walkthrough/11_the_master_cannot_explain.png)

A novice can tell you exactly which rule they were following, because they are still following it
on purpose. Practice compresses decisions into automatic routines, and compression is precisely
what puts them out of reach of report. The maker names its own purpose correctly **0.98** of the
time on a scribble and **0.68** on a master's work, while how much of their **method** a reader
lifts off the work rises from **0.27 to 0.47**. Two lines crossing.

Note what is *not* happening: nobody is setting the maker's self-blindness. **Depth sets it**,
which is a stronger and different claim from E33, where self-blindness was a knob.

**The second sentence of the title was an extension, and V-11b went and tested it.** Nothing in E43
puts the maker in the reader's seat, it compares a maker's declared accuracy against a *separate*
reader's. So: take the artifact a maker produced, hand it to an observer with that maker's own body
plan and a flat prior over goals, the same person, with no memory of which intention was running,
and see what it recovers.

The statistic that decides it is an **interaction**, not a level, because on a scribble the maker
*should* win: a novice can simply tell you what they were doing. And that is what happens. Reading
your own work loses by **0.16** on a scribble and wins by **0.16** on a master's work, a tipping of
**+0.32, interval [+0.18, +0.45]** at 120 readings per depth.

**Past a certain depth, your own work is a better record of what you meant than your memory of it.**
That is *happy little accidents*, and it is now measured rather than asserted.

*What it still cannot show: self-report is modelled as a probability that falls with depth rather
than derived from the maker's own machinery. This establishes that the artifact carries recoverable
intent at depths where the declared channel is degraded. It does not derive the degradation. E43 is
the same in this respect and says so.*

*A plate that used to stand ahead of the next one has been deleted. E45's
efficiency result turned out to be an oracle against a learner, the simulating
reader was constructed with the world's own emission map, so it needed no training
examples by definition and the test could not fail. The measurement is invalid, so
the plate is gone rather than caveated. What it claimed is recorded in
`results/v7/e45_tom_efficiency.json` under `what_this_cannot_show`, and the check is
committed as `scripts/audit_e45_oracle.py`.*

### 4. Reading a goal nobody has shown you ◐

![You can recognise a purpose nobody has shown you](figures/walkthrough/19_reading_an_unseen_intent.png)

The obvious objection to the withdrawn plate is *that is just a prior; give the counter more data.*
So: a goal held out of training entirely.

A reader that can **run** the generator has the whole space available. A reader that has to
**observe** the space only ever has the part it happened to see, and more examples do not fix that,
because its problem was never a shortage of examples. The simulator gets it right **0.82** of the
time, pooled over the eight measurements. The counter sits at guessing across a 128-fold increase
in training data and never leaves.

**The simulator's line is not flat, and it should not have looked flat.** The verdict used to
publish a single number for it, the first of eight, which happened to be the lowest, and the
plate drew that one number straight across the axis, asserting a stability nothing had measured.
The eight real values run 0.77 to 0.85. That spread is noise: chi-square 5.94 on 7 degrees of
freedom against a critical value of 14.07, at n=150 per point.

What *is* true is that the simulator has no learning curve at all, because training size is
consumed by the counter's classifier and by nothing else. Its eight values move only because
building that classifier draws a size-dependent number of values from the generator the artifacts
come from next. Give the classifier its own generator and the simulator returns **0.8467 at every
one of the eight sizes**, identically. Both facts are now on the plate.

**Why this hypothesis survives the audit that withdrew the one above it.** The oracle objection
applies here too, the simulator holds the world's emission map, including the row for the goal it
has never been shown. So the test that matters is what happens when that map is taken away.
Perturbing the simulator's own signature away from the world's, using the codebase's own
inexpertise parameter, the held-out-goal advantage is still standing at a **half-random likelihood**
(0.71 against the counter's 0.53, chance 0.50), where the withdrawn efficiency advantage has
already gone to nothing. **The two hypotheses lean on the oracle to very different degrees, and only
one of them falls over when it is removed.** That sweep is a scratch audit and is not yet a
committed experiment; it should become one.

### 5. Values come into focus across a body of work ○

![Show a reader more works by one maker, and the maker's values come into focus](figures/walkthrough/27_values_into_focus.png)

One work tells you what its maker was doing. A body of work tells you what its maker cares
about. Version 11 gave the world a persistent maker, a standing set of priorities behind every
piece, and asked whether a reader can recover it: 53% from a single work, 98% from fifty, and
works shuffled across makers never leave chance. The signal is the person, not the pile.

The honest riders travel with it. Recovery this clean depends on reader and maker drawing from a
shared, bounded space of possible priorities; take that away and the leftover error grows nearly
thirtyfold, which is the first time that assumption has a price. And the matching test of the
reader's expertise came back too cheap to see at this scale, failed its own pre-specified bar,
and is reported as failing.

### 6. An absent drive shows only under commission ○

![A drive a maker doesn't have is nearly invisible, until you commission the work](figures/walkthrough/28_the_aperture.png)

Can you tell a maker who LACKS a drive from one who simply never uses it? In their ordinary
work, barely: the two read almost alike. Commission work aimed at the missing drive and they
separate completely, because instruction can only amplify a drive that exists. The maker without
it routes through substitutes, and the routing shows.

The control is the point: strip out how the goal is pursued and keep only the commissioned
surface, and the discrimination collapses to an exact coin flip. The reader is reading the
pursuit, nothing else. This is the first working mechanism for work that reads as made under
duress, and it says where to look: commissioned work whose brief demands what its maker does not
have.

---

## Act two · *machine-made work breaks the reading while feeling fine*

*Everything in act one assumed there was a person to find. These four plates are what happens when
there is not. The failure is walking away satisfied,
holding an answer you built yourself.*

### 7. Not understanding is the safe failure ◐

![Not understanding a painting is safe; AI work lets you believe you understood](figures/walkthrough/03_legible_and_empty.png)

Three paintings, and one measure: how finished you felt walking away.

**The middle one is the control that makes the point.** It is a real maker with a real purpose,
working in a tradition you have no training for. Versions 1 to 5 of this model believed that was
what machine content was. You take almost nothing from it, and the point is that **you know that.**
Being stopped at the door is the protection.

The one on the right is what version 6 built to replace that description, because *legible and
empty* is the complaint people actually make about generated work. Four maker states, only two
distinct surfaces, on material you recognise completely. Nothing is hard to look at, and there is no
route from any of it back to a state you could occupy.

**And you do not come away empty. That is the part worth being alarmed about.** Your belief moves
**1.03** on the AI image against **1.40** on a real painting, so nearly as far, and it moves the
wrong way. You finish at 76% settled, holding a confident answer you constructed yourself about a
maker who was never there. Then you keep it.

The tell is that viewers who all feel finished **disagree with each other** about what they found,
nearly as much as on genuine human work. On the painting nobody could read, nobody disagrees at all,
because nobody committed.

**Failing to understand something protects you. Being satisfied by it is how the false thing gets
in.**

### 8. Where it actually breaks ◐

![Invention peaks in the middle, not at the empty end](figures/walkthrough/02_invention_peaks_in_the_middle.png)

The most robust result in the project: same location under exact arithmetic, on a different seed
block, at double scale, in every cell of the robustness sweep, and in every resampled run. **The
worst place to be is nearly understandable.** Total nonsense is safe; total clarity is safe; the
danger is having just enough handholds to build a story on.

### 9. Expertise substitutes ○

![Learning to read machine work swaps a skill out](figures/walkthrough/08_expertise_substitutes.png)

The prediction was that people who understand these systems would be spared the crash. They are,
by trading away the human channel. The adaptation that protects you is the same adaptation that
costs you.

### 10. Pays more, gets less ○

![Optimise the signal of depth and readers pay more for less](figures/walkthrough/09_pays_more_gets_less.png)

A third failure mode, distinct from the other two. It is not the crash: the reader is fully engaged.
It is not the lie: nobody lied, and the reader is right about where the work came from. It is a
reader correctly reading something built to trip its own heuristic for deciding what is worth
reading.

This is the alignment argument, in miniature, inside the model.

*Attention goes from almost none to more than a third, and the reader learns exactly the same amount
either way, a negative amount. The ratio is off a denominator of 0.02, so quote the direction.*

---

## Act three · *the label is the weapon*

*Act two's failures need no deception. These six are what deception adds, and the model's central
result lives here: exposure to hollow content wastes your time, but a believed lie about who made
it rewrites you.*

### 11. What a false label does ◐

![A false label moves you away from the truth](figures/walkthrough/01_false_label_moves_you_wrong.png)

The measure that made this visible is newer than the result. For five versions the model scored
*how far a reader's beliefs travelled*, which cannot tell being convinced from being fooled: a
reader who ends up confidently wrong has moved just as far as one who ends up right. Given a sign,
the three cells stop looking similar and start having opposite signs.

Told honestly that a machine made it, a reader barely moves: **−0.09**. Told a person made it,
**−5.96**. Sixty-six times further, and negative, which means movement *away from* the truth rather
than a failure to learn.

### 12. The two witnesses ○

![Two witnesses arrive with every glance](figures/walkthrough/16_two_witnesses.png)

Why the lie works at all. The reader gets two pieces of evidence about origin at every glance: what
the label says, and what the work itself says. On a lie they point in opposite directions, and which
one wins is decided by an inequality you can solve on paper. **This was computed with no simulation
at all, and it predicted the shape of several results that had already been run.**

### 13. Reputation blindness ○

![The readers most inclined to believe a label can never learn the labeller lies](figures/walkthrough/07_reputation_blindness.png)

To learn that a source lies you have to notice the label and the work disagreeing, and above the
crossover, the label has already won that argument before the disagreement can register.

Not slow learning. Learning that cannot start.

### 14. Honesty is not always enough ○

![If trust lowers your guard, honesty stops being enough](figures/walkthrough/06_honesty_is_not_enough.png)

The most consequential thing version 6 found, and it is a disagreement between the published theory
and its own implementation. The code models a **con**: you were lied to, you were fooled, the fix is
disclosure. The paper models something worse: trust *itself* lowers the guard, so a trusted source
gets absorbed **even when it tells you exactly what it is**. Disclosure does not fix that one.

Both accounts produce the famous result. Only one of them survives being told the truth.

### 15. Rejection is not protection ◐

![You cannot reject something and be unchanged](figures/walkthrough/20_rejection_is_not_protection.png)

To decide you disagree with something, you first have to work out what it says. Working out what it
says means partly running it. So refusing is itself a small act of taking on, and it compounds.

The theory always contained this term; the code never did. And the second half is worse than the
first, which is why it is the picture rather than a caption on it: **the reader who studies it
carefully in order to refute it drifts seven times more than the one who skims**: same content,
same guard, eight times the looking. The effort you spend disagreeing is the channel.

Seal the guard completely and the drift really does fall to 0.00002. No version of this reader
after version 5 has a guard that seals, because version 6 replaced the binary gate with a sigmoid,
and a sigmoid never reaches zero. Shown content with no creator to recover, the reader invents one
and then absorbs its own invention, and that lands about as hard (0.015) as a real intent does.

### 16. Looking is not the same as being changed ○

![Paying attention and being willing to be changed are not the same thing](figures/walkthrough/10_looking_is_not_being_changed.png)

Attention is what you spend. Whether what you find is allowed to change you is a separate decision,
and every combination of the two is reachable. The model always kept them apart; no version had ever
reported them separately. The release valve at the end of an uncomfortable set.

---

## Act four · *what a steady diet does*

*One artifact at a time, the previous acts are recoverable. These three are about volume: what
changes when the contaminated stream is what you live on, and the one question about it this
project refuses to answer rather than answer badly.*

### 17. Two kinds of damage ○

![One kind of damage scales; the other is already there at zero](figures/walkthrough/12_two_kinds_of_damage.png)

Absorbing bad material gets worse the more of it there is, the obvious kind. The other kind is
driven by *walking away*, so it is fully present in a corpus with no machine content in it at all.
The second has the strongest independent support of anything in this project.

### 18. A knee, not a cliff ○

![Competence bends rather than falling off a cliff](figures/walkthrough/14_a_knee_not_a_cliff.png)

The author's own claim, tested knowing it could only survive or weaken. A genuine threshold gets
sharper as you gather evidence; this one did not move across sixteen times the data, so its shape
comes from the model rather than from a boundary in the world.

There was no version of that test where the claim got stronger.

### 19. The experiment that stayed withheld ○

![One experiment has been withheld three times](figures/walkthrough/17_the_withheld_experiment.png)

It asks whether damage compounds as each generation learns from the last, the most consequential
question the project has, and the one it cannot answer. Its honesty check has failed three times, so
real decay cannot be told apart from the instrument's own noise.

Narrowly missing is exactly the case a no-exceptions rule exists for.

---

## Act five · *what actually protects*

*Given all of the above, what would help? Three plates on the obvious answers: detect it, label
it, punish dishonesty, each of which works less than hoped, and then the one result in the whole
project that is genuinely a defence.*

### 20. Why detection is a losing race ○

![Better detectors mean fewer false accusations, until the content fights back](figures/walkthrough/26_the_arms_race.png)

An earlier version of this project found that detectors misfire *less* as they sharpen, and
published that as its own prediction failing. It had measured a world with no adversary in it.

Put evasion in, so that detection and evasion improve together, and the misfiring stops falling and
turns back up: at its worst, **two thirds of careful human writing wrongly flagged.** Switch evasion
off and the clean decline comes straight back, which is what makes this a finding about the world
rather than about a new piece of code.

There is a second half that does not fit on the same picture. A reader whose detector stops updating
does not gradually get fuzzier. Its accusation rate does not move at all while its accuracy drains
away underneath, until it is firing on people and machines at exactly the same rate with exactly the
same confidence. **A coin flip that still believes it is a detector.**

### 21. Labelling needs a convention ○

![Labelling only protects readers who know the labelling exists](figures/walkthrough/15_labels_need_a_convention.png)

The policy number. And a lower bound, because the aware reader here is handed the true coverage,
which is the most generous assumption available.

### 22. Honest marking is self-policing, at a price ○

![Honesty pays above a detection rate](figures/walkthrough/23_honesty_pays_at_a_price.png)

The proposal's answer to bad actors, simulated for the first time in eight versions. It works, and
it comes with a condition rather than a reassurance: **you have to catch a quarter of the liars.**

*The framework reaches this through Zahavian signalling, on which honesty is stable because the
honest signal is wasteful. Signalling theory has since moved to a trade-off account, and a
detection-rate threshold is a trade-off result. The simulation landed on the current position
without being aimed at it.*

### 23. The one thing here that is actually a defence ◐

![Filtering on how good it looks does nothing](figures/walkthrough/25_a_defence_that_works.png)

Every result in this project except this one describes something going wrong. This is the only one
that does something about it.

Networks now publish at industrial scale specifically so that **AI systems** will absorb them, not
so that people will read them: 150+ domains, roughly three million articles a year, and almost no
human visitors, which is the tell. Filtering that out by how good the writing looks does **nothing**,
because looking good is the whole objective. It is the same shape as the RLHF result: optimise the
signal and the signal stops carrying information.

Asking *who made this, and why* cuts the damage by about a quarter, restores the learner's reading
of genuine human work, and **costs nothing on a clean stream**, while never once looking at a
label. That last part matters most. The Ghost Scale failed twice as something makers apply. **This is
the same idea as something readers do, and nobody has to agree to it.**

*Read the direction, not the size: 83% of randomly parameterised models of this shape do it too, so
most of this is architecture rather than evidence for the specific theory.*

---

## Act six · *how much of this to believe*

*The project's habit of auditing itself is the reason to trust anything above. These four plates
are the audits: a withdrawn claim, a randomisation test on the model's own settings, an ablation of
its own commitments, and a measurement error found and named.*

### 24. You don't need a mind to invent one ○

![A counting classifier does it too](figures/walkthrough/13_no_mind_needed.png)

**The framework used to claim that confident, mutually contradictory readings of empty content
require a reader that models the maker as a mind. That is false, and the experiment that killed it
is in this repository.**

A classifier that counts features and never represents a maker reproduces the pattern, through
nothing more than small-sample overfitting. On empty content carrying a creator's label the two
readers end up **0.92 certain against 0.92, disagreeing 0.99 against 0.99**, the same object, to
two decimal places, on both halves of the signature.

**A version of this plate drew the label response and has been reverted.** It showed my reader
collapsing from 0.92 certain to 0.04 when the same empty artifact is honestly labelled, against a
counter that does not move, which reads as a decisive win and is not a measurement.
`run_no_tom_classifier` computes its posterior from feature counts and the class prior;
`declared_signal` never enters it, and the classifier has no provenance state to put it in. **"It
cannot hear the label" is true by construction**, in exactly the way E45's efficiency result was.

What is left is the withdrawal, and the withdrawal is empirical: it did not have to come out this
way, and it did. The two numbers being dull to look at is a fact about the finding rather than a
reason to go looking for a livelier one.

### 25. I threw my own settings away ◐

![How much of this is the theory](figures/walkthrough/22_how_much_is_the_theory.png)

The most important number in this project is not a result. Keep the shape of the model, randomise
everything the theory specifies, and count how often the finding still appears.

**Two of three headlines appear in almost every draw, one in all 60, one in 59 of them.** They
are properties of building a reader this shape at all, which *is* the theory, but is the part
shared with any account of the same shape. It does not distinguish this framework from a competitor
built the same way.

One result survives at zero. That is where the specific commitments earn their keep.

### 26. And what is each finding actually made of? ◐

![What each finding is made of](figures/walkthrough/24_what_its_made_of.png)

The severity rate on plate 25 says how much of a result is architectural. It does not say **which
part**. So the complement: keep the settings and strip the shape instead: remove one structural
commitment at a time, six of them, each a decision about what a reader *is*.

**Every finding dies the moment the reader stops simulating the creator and starts pattern-matching
a surface.** That is the one load-bearing commitment in the project. The model can lose its
hierarchy, its costly attention, and its separate belief about origin, and keep every result.

And *legible and empty* is the odd one out for the second time. It is the only finding that needs
the reader to hold a **distribution** rather than a best guess, which is exactly right, because
"I read every word and there was nobody there" *is* a statement about the shape of an uncertainty.
It is also the only finding with a 0% false-positive rate. Two unrelated audits, same answer.

*One row is missing from this plate: sustained futile attention did not reproduce in the ablation
harness's own baseline, so it has no answer here rather than a bad one.*

### 27. A disagreement that turned out to be a measurement error ○

![Pointed at the wrong thing](figures/walkthrough/21_pointed_at_the_wrong_thing.png)

The project's longest-running open question, settled by changing *what* was measured rather than
*how*. The criterion was scored on the work's purpose, which the model deliberately holds equally
readable at every depth, so it could never have moved.

---

## What is not drawn yet, and why

*The older per-experiment research charts have been removed. They were working figures from versions
1 to 5, drawn before this repository had a house style and never brought up to it, and several of
them meant something different by the end than they did when they were drawn. Every claim they
carried is in the plates above, redrawn from the same committed data.*

A plate makes a claim legible in two seconds, which means it also makes a claim hard to qualify.
Version 6 held four results out of this walkthrough because each carried an open question a plate
would paper over. **Version 7 closed all four**, and one of them closed by being retired.

| result | what happened |
|---|---|
| **The two-gates criterion** | **Settled, and drawn** (plate 27). Re-run on E31's own design and scored on the method: 0.93, with the interval excluding the bar. |
| **Depletion carrying to unseen work** | **Settled.** At thirty encounters the probe falls essentially to nothing, monotone at −0.98. The direction was never in doubt; the pre-registered magnitude clause was the wrong *shape* of criterion and is reported as such. |
| **Depth versus effort** | **Still open, and reported as failing.** The pre-registered contrast still returns *effort can manufacture depth*. On what actually transfers, depth dominates effort ninety-seven-fold, so the reader's *estimate* of depth is contaminated by effort while the *transfer* is not. Not drawn, because a plate cannot carry that split. |
| **The collapse and the invention peak sharing one band** | **Retired.** The peak is unchanged and is not in question. The co-location held only under a superseded solver; it is gone from the README and stands in the committed prediction card only under a SUPERSEDED marker. |

Three plates from the ledger remain undrawn: **plate 25 reprised** as an anchor rather than a
caveat, **the withheld version-10 rider** (which damaged a learner reading nothing but honest human
writing and was recorded in advance as the check most expected to fail), and **the forking-paths
ledger**: eighteen places across four versions where a design or a criterion changed after a
result was seen.

One result stays undrawable, and it is the honest kind: **N21's split**. Two measures disagree about
the same manipulation, one of them is the pre-registered one, and it fails. That is a paragraph, not
a picture.

**One version-11 result, the smear, has no plate yet**; it is tabled in
[FINDINGS.md](FINDINGS.md) and [docs/theory/READING_INTENT.md](docs/theory/READING_INTENT.md) §9.

---

## What the whole thing does not do

- **It is a simulation of a mechanism, not a study of people.** There is no human data anywhere in
  it and nothing here is evidence about what real readers do.
- **The shapes are the claims; the numbers are properties of this model's dimensions.** Rebuilt from
  the prose alone by someone with no access to the code, the central effect points the same way and
  comes out fifteen times smaller. Quote directions. Do not quote multiples.
- **There is no forward test.** There was one sealed prediction and its status was withdrawn in
  version 8, because the author does not recognise authoring it. This is a theory checked carefully
  against itself.
- **The proposal it is named after has no demonstrated mechanism.** Twice this project has gone
  looking for how a Ghost Scale label would actually protect a reader, and twice it has come back
  without one: E39 and E54. The proposal is not refuted. It is unsupported, which is a different
  thing and should be said in its own sentence.

---

*[← the fastest true picture](README.md) · [every question and its current answer](FINDINGS.md) ·
[what the world published, next to what this predicted](EVIDENCE.md)*
