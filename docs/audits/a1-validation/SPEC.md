# Ghost Scale Simulation — Validation Pass and Release Preparation

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending `ghost-scale-sim`
**Depends on:** V1–V5 specs, all `RESULTS_*.md`, `EXPERIMENTS_TABLE_CONSOLIDATED.md`

This is not a new version. **Nothing here asks a new question about the world.** Every item asks
whether the answers already recorded can be trusted, and then prepares what survives for
publication.

---

## 0. What this is defending against

**This is exploratory modelling, and all of it is confirmatory by construction.** Every
prediction in this repository was derived from one prior theory. The simulations formalise that
theory and test whether its parts fit together, which means agreement between model and theory is
the expected outcome rather than evidence for it.

That is a legitimate mode of work and it has a specific failure that validation exists to catch:
**a model can reproduce its own assumptions and be indistinguishable, from the outside, from a
model that discovered something.** Separating those two is the entire job here.

The project's own record shows the risk is live rather than theoretical. **Seven times, an
instrument was answering a different question than the one being asked, and every time the wrong
answer looked completely reasonable:**

- a stale threshold file silently overriding a requested sample size
- a null evaluated at smaller scale than the experiment it gated
- a statistic computed at the wrong timestep, understating an effect roughly fiftyfold
- a fallback hypothesis that was uniform to machine precision and sat closer to synthetic content
  than to any real goal
- a lucky random seed producing an apparent confirmation that vanished across four draws
- a baseline comparison that flattered the framework until fairness clauses were added
- an inference approximation that was **confidently blind** to the quantity being measured

Six of those were caught by targeted checks written for other reasons. **This pass makes the
checking systematic instead of lucky.**

---

## 1. Release-blocking checks

**No figure ships and no claim is published until these four pass or their failures are
documented in the claim itself.** Each writes a verdict file and a section of `VALIDATION.md`.

### V-1 — Does the inference approximation distort the headline results?

**The highest-priority item, because the failure is confirmed rather than hypothetical.**

pymdp's mean-field solver could not see hierarchical structure at all and reported near-certainty
while being wrong. The mechanism is Jensen's inequality penalising any hypothesis that carries
latent structure. Everything before V5 ran on that solver.

Nothing before V5 had hierarchical depth, so there was nothing latent to penalise — **but
"content whose structure the observer cannot express" is uncomfortably adjacent to "hypothesis
carrying latent structure," and that is exactly what the foreign-content experiments are made
of.**

**Do:** implement exact filtering for the observer's inference and re-run, on identical
observation tapes, at reduced but adequate scale:

- the collapse-and-invention sweep across the readability axis
- the sustained-attention result on unreadable content
- the two-gates result
- the silent-versus-loud failure comparison
- the original label-induced disagreement result

**Report:** side-by-side, exact against approximate, for every primary outcome.

**The specific thing to watch:** the readability axis has an interior peak where invention is
highest and the collapse signature fires, and a great deal is anchored to that location. **If the
peak moves under exact inference, every claim that references it moves with it,** including the
prediction card built for a human study.

**Pass:** peak location within one grid step; primary outcomes within their existing run-to-run
spread. **Fail:** re-anchor every dependent claim and say so in the README, above the table.

### V-2 — Would a model of this shape produce these results anyway?

Two arms. The second has never been run and is the stronger one.

**(a) Scrambled provenance, at scale.** Permute the mapping from provenance to content properties
and re-run every headline experiment. All effects must vanish. This exists as a null in one
experiment; it should exist for all of them.

**(b) Random-parameter false-positive rate.** Draw the likelihood structure at random, subject
only to the format constraints — column-stochastic, separation floor, entropy ceiling — and run
the headline experiments unchanged. Repeat a few hundred times. **Count how often a randomly
parameterised model of this architecture returns something that would have been reported as a
finding.**

That number is the apparatus's false-positive rate and nobody has it. If a meaningful fraction of
random models produce a twenty-two-fold label effect, the label effect is a property of the
architecture rather than of the theory.

**Pass:** headline effects fall outside the random distribution at a threshold committed before
the run. **Fail:** the affected claim is reported as architecture-dependent.

### V-3 — Are the headline results knife-edge or robust?

For each surviving headline, sweep every free parameter over a defensible range and record
whether the verdict flips.

At minimum: effort cost, feature count, goal count, observer count, planning horizon, prior
strength, the separation floor between goal signatures, the opacity-to-readability values, and
the number of encounters in the sequential designs.

**Deliverable:** one matrix, result by parameter, each cell recording whether the verdict holds,
weakens, or flips.

**This is a scoping exercise, not a pass/fail.** A result that holds across the whole range and a
result that holds in a narrow window are both real; they are different claims and the README must
say which is which. **Any result that survives only in a window narrower than the range that was
actually explored during development is reported as tuned.**

### V-4 — What is forced by construction?

The systematic version of the checks that caught the uniform fallback hypothesis and the
noise-versus-unidentifiability distinction.

**For every headline claim, write down in advance:** which property of the construction, if
altered, would eliminate this result? Then alter it and confirm the result disappears.

**Two architectural choices deserve their own audit** because a great deal rests on them and
neither was chosen on theoretical grounds:

1. **The disjoint human/foreign feature partition.** Adopted because no such partition existed at
   the original feature count, so the space was doubled. Every claim about the readability axis
   is downstream of it. Enumerate those claims explicitly and state what a different partitioning
   scheme would do to them.
2. **The rebuilt effort parameter.** It was changed specifically so that "offhand but deep" is
   representable — which means the dissociation the model was built to test was made possible
   before it was measured. Report the dissociation as a construction commitment rather than an
   emergent finding, and check whether it survives under the original parameterisation in any
   form.

---

## 2. Secondary checks

Run after the four above. Each is cheap.

**V-5 — Recompute every superseded criterion.** For every logged deviation across all versions,
compute the original criterion as written and report which verdicts would change under it. One
deviation has this already; the rest do not. Publish the table, including cases where the verdict
is unchanged, because that is the informative case.

**V-6 — Cross-version consistency.** Wherever two versions measure the same quantity, check they
agree, and check every boundary regression still holds — the reductions where a later model must
reproduce an earlier one exactly.

**V-7 — Seed and scale independence.** Re-run every headline at a different seed block and at
double scale. Report effect sizes, not just verdicts. Anything that moves materially with scale
is under-powered and gets said so.

**V-8 — Independent reimplementation of the single strongest result.** Take the two-gates
finding — same artifact, one word in the label, twenty-two-fold difference in uptake — and
reimplement it from the prose description alone, in separate code, without reading the original
implementation. **If it replicates, that is the strongest evidence available anywhere in this
project. If it does not, finding out why is worth more than any new experiment.**

**V-9 — One genuinely out-of-sample prediction.** The literature check is retrospective and every
simulation was designed by people who knew the theory. There is currently no forward test.

Pre-register a full prediction — outcome, direction, magnitude, and named failure branches — for
an experiment that has not been built. **The obvious candidate is the observer equipped with a
hypothesis for what a maker was avoiding**, which is the natural next experiment anyway. Write
the prediction, hash-lock it, then build it. That is the only true out-of-sample test the project
can generate for itself.

---

## 3. The README rebuild

The landing page is at V3 and does not mention anything after it. Rebuild as follows.

### 3.1 Structure

1. **One-sentence description in plain language.** No jargon in the first line.
2. **What this is**, three or four sentences. A model of how people work out what someone was
   trying to do when they made something, what happens when nothing was trying, and what a
   provenance label does about it. Simulation, not a study of people.
3. **What was found** — the consolidated table from `EXPERIMENTS_TABLE_CONSOLIDATED.md`, with one
   column added: **validation status**. Not a separate technical table. One table, plain-English
   questions and answers, numbers inline, and a column for published work located *after* the
   simulations ran, marked as such. Two tables become two truths and drift apart within three
   revisions, which is the same failure this repository has caught seven times.

   **Include one row inside the table**, in position, noting that two experiments were removed
   because they tested a construct later found to be wrong, and pointing to the section below.
   A reader scanning the table should not have to discover the gap on their own.
4. **What was not reported** — the withheld experiment, its own section, high on the page. "We
   could not measure" is not "we found nothing," and the distinction is the point.
5. **What was removed** — the two experiments run against a construct later found wrong.
   Uninterpretable, not embarrassing, and the distinction stated.
6. **What died** — the seven dead ideas, named, including the author's own and the framework's own
   necessity claim.
7. **How this was kept honest** — pre-registration as executable code, nulls for every headline,
   deviations logged with originals retained, and the validation pass summarised with a link.
8. **Scope and limits**, generously stated. A simulation of a proposed mechanism, not evidence
   about people. Specific numbers are properties of this model's dimensions and do not transfer;
   the shapes are the claims. The labelling-coverage figure is a lower bound because the
   convention-aware reader is handed the true coverage, which is the most generous assumption
   available.
9. Install and run, unchanged.
10. Relationship to existing work, unchanged. It is honest about what is and is not novel.
11. Links, licence, citation.

### 3.2 Required edits regardless of structure

- **Replace the goal-empty framing throughout.** The page still describes synthetic content as
  having no goal. The model now treats it as having a goal the observer cannot represent, and the
  two make different predictions. **Define both plainly, because it is the most misunderstood
  distinction in the project:** goal-empty is wood grain, goal-foreign is a page in a script you
  cannot read, and neither one is "the goal is repugnant."
- **State the withdrawn claim.** A counting classifier with no model of a maker reproduces the
  confident-disagreement pattern. Say it plainly, near the top, along with what survives.
- **Em-dash pass**, per the author's standing convention. Several are load-bearing and need the
  sentence rebuilt rather than the character swapped.
- **Add every results file** to the layout section.
- **Rewrite the repository description field.** It is the preview caption everywhere the link is
  pasted and it currently opens with three pieces of jargon. Add topics.
- **Record the attention-gradient limitation**, with the observation that this repository's own
  charting code already declined to use the published opacity ramp and substituted a monotone one.
  That is the argument having been made in code before it was made in prose.

---

## 4. Figures for distribution

**Nothing here is built until §1 passes.** A figure is a claim with a picture attached; a claim
that has not survived validation does not get one.

### 4.1 Visual identity — unchanged from the prior specification

Ghost Scale opacities as the palette: 100%, 95%, 60%, 5% black on white, with the 5% tier
carrying the isolating bounding box the published framework requires. **The charts obey the
framework they report on.**

**The accessibility constraint is load-bearing and is not a limitation.** The 60% tier sits
exactly at the minimum contrast ratio for body text and the 5% tier is far below it. Therefore
all text renders at 100% or 95% only; only non-text data marks use the lower tiers. Assert at
render time that no text element is assigned a lighter value. A carousel about a
contrast-grounded transparency framework that fails contrast requirements would be caught by
precisely the audience being addressed.

Non-tier series: 100% black and 60% grey, distinguished additionally by line style so the
contrast survives greyscale and feed compression.

### 4.2 Output

Five slides at 1080 × 1350, plus the same pages assembled as a PDF, plus one social preview image
at 1280 × 640. Write to `figures/social/`. Do not overwrite research figures — they have
different jobs and different constraints.

Minimum 24pt type. One idea per slide. Generous margins. Assume one-handed viewing on a phone.

### 4.3 The five slides

**Slide 1 — title.** No chart.

> I built my theory as a working model.
> It disagreed with me seven times.

Subhead: a simulation of how people read intent, and where it breaks. Optionally the four tier
swatches as a strip, unlabelled — a design element to a general viewer and a signature to anyone
who knows the framework.

**Slide 2 — the two-gates result. The new headline.**

> The same object. One word changed.

The strongest and most legible thing in the project: identical machine-made content, labelled
honestly against passed off as human, and a twenty-two-fold difference in how much the viewer
takes on. Two bars, or a before-and-after pair.

**Required annotation:** the ceiling. The observer in this build cannot doubt the label it
conditioned on, so the multiple is an upper bound. **Put that on the slide, not in a footnote.**
It is the difference between a finding and an overclaim, and it is the thing a hostile reader
looks for.

**Slide 3 — the competence result.**

> Same documents. Only the reader changed.

The strongest standalone claim, because it uses no synthetic content at all. Include the
second finding on the same slide if it fits legibly: belief accuracy degrades well before choice
accuracy does, and choices are what preference data collects.

**Slide 4 — silent versus loud failure.**

> A novice looking at a person beats an expert looking at a machine.

The most counterintuitive measured result. At an identical information deficit, the unskilled
reader of human work is substantially more accurate than the expert reader of machine work,
because a badly aimed template still points near the truth while unreadable content points
nowhere.

**Slide 5 — the close.**

> Seven ideas died. One was mine.
> One experiment failed its own test three times and I didn't report it.

Repo URL, larger than the other footers. No call to action, no "let's connect."

**This is the slide that does the work with the audience that matters.** Every other slide could
be anyone's. This one cannot.

### 4.4 Held for later posts

The mark that exists and cannot be read is the most interesting result in the project and the
hardest to compress to a single image. It wants its own post with room to explain, not a slot in
a carousel.

---

## 5. Order, and what gates what

| stage | work | gate |
|---|---|---|
| 1 | **V-1**, the solver check | if the interior peak moves, stop and re-anchor before anything else |
| 2 | **V-2**, null and random-parameter | any headline inside the random distribution is pulled from the figures |
| 3 | **V-3**, **V-4** | scoping; changes wording rather than blocking |
| 4 | README rebuild with validation status column | needs 1–3 complete |
| 5 | **V-5** through **V-7** | cheap, run alongside |
| 6 | Figures | **only from results that survived stages 1 and 2** |
| 7 | **V-8**, independent reimplementation | slower, does not block release, upgrades the strongest claim |
| 8 | **V-9**, the out-of-sample pre-registration | write the prediction now, build later |

---

## 6. Constraints

- **A result that fails validation is not deleted.** It is reported with its failure attached, in
  the same cell as the claim. The project's discipline has been to keep unwelcome findings and
  remove only uninterpretable ones; validation failures are unwelcome, not uninterpretable.
- **`VALIDATION.md` is written from the verdict files after the runs**, never from the
  expectations in this document.
- **The withheld experiment stays withheld**, its failing test stays in the suite, and the
  open residual stays open.
- **No figure ships for a claim that has not passed stages 1 and 2.**
- **Every check in §1 must be able to return the unwelcome answer**, and the reporting sections
  must be written before the results are known.
- **The whole body is presented as exploratory and theory-derived**, in the README and in
  `VALIDATION.md`. Every prediction came from one prior theory; the simulations formalise it and
  check that its parts fit. That framing is stated once, plainly, near the top, and then not
  relitigated. **It is the accurate description of the work and it pre-empts the objection rather
  than inviting it.**
