# Ghost Scale — public-facing assets specification

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent with access to the `ghost-scale-sim` repository
**Two deliverables:** (1) a five-slide figure carousel for social distribution, (2) a rewritten
`README.md` for a repository that is about to become the primary public landing page for this work.

---

## 0. Context you need before starting

This repository is currently written for someone who already knows the project. That is about to
stop being true. Within days it will be linked from a public post and from email to academic
contacts, and most people arriving will have **no prior context, no interest in pymdp, and about
forty seconds of patience**. Both deliverables exist to serve that reader without alienating the
much smaller number of readers who will actually check the code.

The two audiences want opposite things and you must serve both:

- **The general reader** wants to know what was found and whether it matters. They will not read
  a methods section. They need plain language and a chart they can parse in five seconds.
- **The technical reader** (active inference researchers, alignment people, HCI academics) wants
  to know whether the result was rigged. They will look for null tests, pre-registration, and
  withheld results before they look at anything else.

The single most credible fact about this project is that **one experiment failed its own
acceptance test three separate times and was never reported.** Surface that early and
prominently in both deliverables. It is worth more than any positive result.

**Verify every number you use against the CSVs in `results/`.** Numbers quoted in this spec are
transcribed from `RESULTS_V2.md` and `RESULTS_V3.md` and are provided so you know which columns
matter, not as a substitute for reading the data. If a number here disagrees with the CSV, the
CSV wins and you flag the discrepancy. This project has been bitten four times by an instrument
answering a different question than the one being asked; do not become the fifth.

---

## PART ONE — the carousel

### 1.1 Output

- Five slides, exported as **PNG at 1080 × 1350** (4:5 portrait, which occupies the most vertical
  space in a mobile feed).
- Also assemble the same five pages into a single **PDF at the same aspect ratio**, since document
  carousels are uploaded as PDF.
- Write to `figures/social/` — a new directory. Do not overwrite anything in `figures/`.
- Filenames `slide_1.png` … `slide_5.png`, and `carousel.pdf`.

### 1.2 Visual identity: the Ghost Scale is the palette

The charts are rendered in the framework they report on. This is the whole design idea and it
should be executed precisely, not gestured at.

**The four tiers, as published, are opacities of black on white:**

| tier | opacity | in charts |
|---|---|---|
| CREATOR | 100% | `#000000` |
| POLISHED | 95% | `#0D0D0D` |
| CURATOR | 60% | `#666666` |
| GHOST | 5% | `#F2F2F2`, **and it requires an isolating bounding box** |

**The bounding box is not decoration and is not optional.** The published framework states that
Ghost-tier material at 5% opacity cannot maintain legibility under WCAG without an isolating
bounding box. So any GHOST data mark in these charts gets a thin 100%-black outline. This is the
framework's own rule applied to itself, and an informed viewer will notice. Implement it as a
`markeredgecolor='#000000'` / `edgecolor='#000000'` with `linewidth≈1.2` on the GHOST series only.

**Hard accessibility constraint, and it is load-bearing.** The 60% tier sits *exactly* at the WCAG
minimum contrast ratio of 4.5:1, which is the threshold for body text. The 5% tier is far below it.
Therefore:

- **All text** — titles, annotations, axis labels, tick labels, captions — renders at **100% or 95%
  only**. Never 60%, never 5%.
- **Only non-text data marks** — lines, bars, scatter points, fills — may use 60% or 5%.

A carousel about a WCAG-grounded transparency framework that itself fails WCAG would be an
embarrassment that the exact audience you want is equipped to spot. Add an assertion in the
plotting module that no text artist is assigned a color lighter than `#0D0D0D`.

**Non-tier series** (anything that is not one of the four provenance tiers, e.g. "with averaging"
versus "without") use 100% black and 60% grey, distinguished additionally by line style (solid
versus dashed) so the distinction survives greyscale and low-quality feed compression.

**Everything else:**

- Background pure white `#FFFFFF`. No grid, or a very light grid at `#EEEEEE` if a chart is
  genuinely unreadable without one.
- No chartjunk. No 3D, no shadows, no gradients, no colored accents. The whole point is that this
  palette is monochrome by design.
- Despine top and right axes.
- Sans-serif. Check font availability at runtime and fall back gracefully rather than assuming a
  font is installed; a missing font silently substituting is a real failure mode. Set a stack and
  verify.

### 1.3 Typography and layout, per slide

Assume one-handed viewing on a phone on a train. That is the binding constraint.

- **Headline:** 44–52 px equivalent, bold, 100% black, top of slide, left-aligned, maximum two
  lines. This is the claim. It is not a chart title in the academic sense; it is a sentence.
- **Subhead / annotation:** 26–30 px, 95%, immediately under the headline, one line.
- **Chart:** occupies the middle 55–60% of the slide.
- **Axis and tick labels:** minimum 22 px. If that makes the chart crowded, remove ticks rather
  than shrinking type.
- **Footer:** 20 px, 95%, bottom-left. Slides 2–4 carry `ghost-scale-sim` plus the experiment ID
  so a screenshot is traceable back to the source. Slide 5 carries the repo URL.
- Generous margins. Minimum 80 px on all sides. Crowding reads as amateur far faster than
  whitespace reads as empty.

**One idea per slide.** If a slide needs two sentences to explain, it is two slides or it is cut.

### 1.4 The five slides

Numbers below are from the results files. Confirm each against the CSV before rendering.

---

**Slide 1 — title card. No chart.**

Headline, large, centered or upper-left, dominating the slide:

> I built my theory as a working model.
> It disagreed with me five times.

Subhead:

> An active inference simulation of how people read intent, and where it breaks.

Optional: render the four tier swatches as a small horizontal strip at the bottom — four
rectangles at 100 / 95 / 60 / 5%, the last one boxed — with no labels. It reads as a design
element to a general viewer and as a signature to anyone who knows the framework.

---

**Slide 2 — the competence result. Source: `results/e15_competence_cliff.csv`.**

Headline:

> Same documents. Only the reader changed.

Subhead:

> Intent recovery collapses as reader competence falls, with the material held constant.

Chart: goal accuracy against reader inexpertise `d`. Line at 100% black, markers at each swept
point.

| d | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| goal accuracy | 0.9993 | 0.9973 | 0.9851 | 0.9186 | 0.6980 | 0.5110 | 0.2390 |

Annotate the midpoint at **d50 = 0.878**. Do not use the words "cliff" or "phase transition" —
the width arm refuted that, and the defensible word is **knee**.

Footer: `E15 · ghost-scale-sim`

---

**Slide 3 — the dissociation. Source: same CSV.**

This is the sharpest technical claim in the set. Give it its own slide.

Headline:

> Belief breaks before choice does.

Subhead:

> RLHF collects choices. Choices are the last thing to fail.

Chart: two curves against `d` on shared axes — belief-quality (psi / posterior entropy, d50 ≈
0.70) at 100% black solid, and choice accuracy (d50 = 0.878) at 60% grey dashed. Shade the region
between the two midpoints very lightly and label it:

> the gap where the signal still looks clean

Footer: `E15 · ghost-scale-sim`

---

**Slide 4 — confident fabrication. Source: `results/e17_tier_stats.csv`.**

Headline:

> Confident. And nobody agreed.

Subhead:

> Told an artifact is human, readers invent a purpose. The less intent it carries, the more
> confidently they invent.

This slide is where the Ghost Scale palette earns itself, because the x-axis **is** the tiers.

Chart: grouped bars or a paired dot plot per tier, showing within-observer confidence against
between-observer disagreement. Each tier's marks rendered at that tier's own opacity. GHOST gets
its bounding box.

| tier | α | within (confidence) | between (disagreement) |
|---|---|---|---|
| CREATOR | 1.00 | 0.0000 | 0.0000 |
| POLISHED | 0.95 | 0.0000 | 0.0000 |
| CURATOR | 0.60 | 0.0111 | 0.1083 |
| GHOST | 0.05 | 0.0894 | 1.3794 |

Two required annotations:

- A dashed reference line at **1.386** labeled `maximum possible disagreement`. GHOST at 1.3794 is
  essentially at ceiling and the viewer needs the ceiling to see that.
- A single callout: `told the truth instead, uncertainty rises to 1.297`. This is the framework's
  entire proposition in one number and it must appear on this slide.

**Plot against α, not against tier index.** The tiers are unevenly spaced by construction and
plotting against index would manufacture a straight line out of that spacing. If α spacing makes
the chart hard to read, use categorical positions but state the α value under each tier label.

Footer: `E17 · ghost-scale-sim`

---

**Slide 5 — the close. No chart, or a very sparse one.**

Headline:

> Ten results held.
> One experiment failed its own test three times.
> I didn't report it.

Body, smaller:

> Five hypotheses died getting here, including one of mine. Every null test, every deviation, and
> every withheld result is in the repo.

Footer, larger than the others and clearly a link:

> github.com/EmotiveAutomaton/ghost-scale-sim

This slide is doing the heaviest lifting with the audience that matters. Do not soften it, do not
add a call to action, do not add "let's connect."

### 1.5 Implementation notes

- New module `ghostscale/social_figures.py`. Do not modify existing figure code; the research
  figures and the social figures have different jobs and different constraints.
- Read from `results/*.csv`. Never hardcode a number that exists in a CSV.
- Every slide function takes no arguments and writes one file, so slides can be regenerated
  individually.
- One CLI entry point regenerates all five plus the PDF.
- Add a check that asserts the rendered numbers match the CSVs, so a stale CSV cannot silently
  produce a wrong slide.

---

## PART TWO — the README

### 2.1 What changes and why

**Read the current README before touching it. It is in better shape than a rewrite instruction
implies.** The honesty architecture is already present and already well done: the withheld
experiment appears in the top blockquote, the methodological finding about measuring the
instrument has its own section, the crosswalk is honest about what is and is not novel, and the
V2 learner-prior section explains a design choice as a theoretical commitment rather than
excusing it. **Do not restructure any of that. Do not delete any of it.**

What is actually wrong is narrower:

1. **The first screen is written for someone who already has the vocabulary.** The opening
   sentence uses "Theory-of-Mind inference," "goal-empty synthetic content," and "provenance
   signal" before establishing any of them. A reader arriving from a social link bounces.
2. **The experiments table lists only e1 through e6.** There are eighteen. The full set appears
   in prose under repository layout, but the table a reader actually scans is two versions stale.
3. **There is no findings section.** A visitor cannot learn what was found without reading three
   results files. The numbers are the reason to stay and they are not on the page.
4. **Em-dashes throughout**, against the author's standing convention. This is a substantial
   pass, not a find-and-replace, because several are doing real syntactic work and need
   restructuring rather than substitution.
5. **No `LICENSE`. No `CITATION.cff`.**

So: **additive and surgical, not a rewrite.** Add a plain-language opening and a findings
section above the existing content, update the experiments table, do the em-dash pass, add the
two missing files. Everything else stays where it is.

### 2.1a A live problem: the results are invisible to visitors

The repository layout section states that CSVs and PNGs are gitignored, with only JSON verdicts
committed. **For a private working repo that is correct. For a public landing page it is a
serious problem.** Someone arriving from a link cannot see a single chart or check a single
number, and the README references figure paths that do not resolve for them.

Two consequences, both of which need resolving before the repo is promoted:

- **Commit the figures.** At minimum the handful referenced in the README and used in the
  carousel. PNGs at this size are trivial in a git history and the alternative is a public
  research repo with no visible results.
- **Commit the CSVs backing any number quoted in the README or on a slide.** A reader who cannot
  verify a claim will discount it, and "verifiable" is the entire proposition of this repository.
  If total size is a concern, commit the summary CSVs and leave the raw per-observer files out,
  and say which is which.

**This also affects Part One.** The carousel spec instructs you to read numbers from
`results/*.csv`. Those files exist in the author's working tree but not in a fresh clone. Build
the figures from the local files, and treat committing them as a prerequisite for anything that
quotes a number publicly.

### 2.2 Required structure, in order

1. **Title and one-sentence description.** Plain language. No jargon. A reader should know whether
   they care by the end of the first line.

2. **What this is, in three or four sentences.** Something close to: this is a computational model
   of how people work out what someone was trying to do when they made something, what happens to
   that process when the thing was generated rather than made, and what a provenance label does
   about it. Companion to a preprint. Simulation, not a study of human subjects.

3. **What was found.** Four or five bullets, each one sentence, each with a number and an
   experiment ID. Lead with:
   - reader competence caps intent recovery with material held constant (E15)
   - belief accuracy degrades before choice accuracy, which is what RLHF measures (E15)
   - a false provenance signal produces confident fabrication that scales with missing intent;
     an honest one produces honest uncertainty (E2, E17)
   - unlabeled synthetic content gets folded into the learner's model of human work (E7)
   - ~31% labeling coverage suffices, but only for readers who know the convention exists (E16)

4. **What did not work.** Its own section with its own heading, above the fold if possible. E8
   failed its acceptance null three times and is not reported. Five hypotheses died. Name them.
   **This section is the reason a serious reader will trust the rest of the page.** Do not bury it
   and do not apologize for it.

5. **How this was kept honest.** Short. Zero observer preference over provenance. Every headline
   result has a matching null. Criteria pre-registered in executable code so the written and
   applied criteria cannot drift. Deviations logged, including the two criteria that were changed
   after seeing data.

6. **Scope and limits.** Explicit, in plain language, and generous. This is a simulation of a
   proposed mechanism, not evidence about human subjects. Numbers like d50 = 0.878 are specific to
   this model's feature and goal cardinality and do not transfer; the shape is the claim, not the
   number. The 31% figure is a **lower bound** — the regime-aware reader in the model is handed
   the true coverage, which is the most generous assumption available. **Stating limits plainly is
   a credibility asset, not a hedge.**

7. **Install and run.** Existing content, unchanged.

8. **Experiment index.** A table: ID, one-line question, status, output files. Every experiment
   E1 through E18, including the withheld one, marked withheld.

9. **Relationship to existing work.** The existing crosswalk section, unchanged. It is good and it
   is honest about what is novel and what is not.

10. **Links.** Preprint DOI, essay, Figma kit, in that order.

11. **License and citation.**

### 2.3 Voice

Match the author's existing writing, which the preprint and the essay both demonstrate:

- Contractions mandatory.
- No em-dashes. Use commas, colons, or a new sentence.
- Direct. Short sentences carrying real content. No throat-clearing.
- No AI-isms. Specifically avoid: "delve," "leverage" as a verb, "it's worth noting," "in today's
  rapidly evolving landscape," "unlock," "harness," "robust" as a filler adjective, and any
  sentence built on "not just X, but Y."
- No marketing register. No exclamation marks. Do not describe the work as exciting, groundbreaking,
  or novel; state what it does and let the reader decide.
- Lead with what exists rather than what did not materialize.

### 2.4 License

The repository needs one. A public research repo without a license is legally ambiguous and
technical readers notice immediately.

Recommendation, which is a suggestion and not legal advice, and which the author should confirm:
**MIT for the code, CC BY 4.0 for the prose and figures**, stated in a short `LICENSE` file plus a
line in the README. This is the conventional pairing for academic software with an accompanying
write-up and it imposes essentially no friction on reuse.

Also add a `CITATION.cff` so anyone citing the repo produces a consistent reference, pointing at
the Zenodo DOI `10.5281/zenodo.19407789`.

---

## PART TWO-B — the link preview

Two repository settings control what a link to this repo looks like when it is pasted into
LinkedIn, Slack, or email. Neither lives in the README and both are currently defaults.

**The About field is the preview caption.** It currently reads:

> Active-inference model of the generative crash and trust exploit: how Theory-of-Mind inference
> disengages from goal-empty synthetic content, and how a trusted dishonest provenance signal
> turns that into confident fabrication.

That is an accurate abstract and a poor preview. It is the single line of text a scrolling reader
sees under the link card, and it opens with three pieces of jargon. Replace it with something a
general reader parses in one pass while still being true, on the order of:

> A simulation of how people read intent into things other people make, what breaks when nothing
> made it, and what a provenance label does about it.

Add repository topics as well, since they drive GitHub's own discovery: `active-inference`,
`ai-alignment`, `pymdp`, `theory-of-mind`, `ai-transparency`, `provenance`,
`computational-modeling`.

**The social preview image is currently GitHub's auto-generated card.** GitHub allows a custom
image under repository Settings, and the recommended size is 1280 × 640. Produce one as a sixth
output of the carousel module, at that size rather than 4:5, using slide 4's chart (the tier
dose-response) with a short headline. That chart is the most visually distinctive asset in the
project and it renders the framework in its own palette, which makes it recognizable on sight.

Write it to `figures/social/social_preview.png`.

- [ ] Every number on every slide traced to a CSV, not to this document
- [ ] No text artist in any slide lighter than `#0D0D0D`
- [ ] GHOST series has its isolating bounding box everywhere it appears
- [ ] Slide 4 plots against α, not tier index
- [ ] The words "cliff" and "phase transition" appear nowhere
- [ ] Slides legible at 400 px wide (open one at that size and actually look)
- [ ] Slides legible in greyscale and at low compression quality
- [ ] README first screen makes sense to someone who has never heard of this project
- [ ] The withheld experiment is still prominent after the edit, not pushed down by new content
- [ ] Experiments table lists all eighteen, not six
- [ ] A findings section with numbers exists above the technical content
- [ ] No em-dashes anywhere in the README
- [ ] `LICENSE` and `CITATION.cff` exist
- [ ] Figures referenced in the README are committed and resolve for a visitor
- [ ] CSVs backing any publicly quoted number are committed
- [ ] Every link in the README resolves
- [ ] About field rewritten; repository topics added
- [ ] `social_preview.png` produced at 1280 × 640 and set in repository settings
- [ ] Slide 5 shows `github.com/EmotiveAutomaton/ghost-scale-sim`
