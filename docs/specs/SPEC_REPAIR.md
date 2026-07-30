# Repair pass: from demonstration to measurement

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending `ghost-scale-sim`
**Depends on:** `VALIDATION.md`, `DIAGNOSTICS.md`, `docs/specs/`

---

## 0. The governing principle

The diagnostics pass named the real problem: **this apparatus was built to demonstrate and is now
being audited as though it measures.** Those have different standards and most of what came back
is the gap between them.

The evidence is direct. κ is unidentifiable because **nothing in the model was ever built to
estimate κ** — it is a knob you turn to ask what a reader with that disposition would do. The two
parameters that recovered cleanly recovered because someone wrote an estimator for them *during
the diagnostics pass*. They had none before either.

This pass closes that gap where it can be closed and states plainly where it cannot.

### The rule for this pass, and it is a hard rule

**Every change must either make something measurable that was not, or remove something.**

The model has grown every version since V1 and nothing has ever been taken out. That accretion is
what produced a construction where two of the interior peak's dependencies **could not be broken
because the model refuses to run without them.** A repair pass that adds machinery in the ordinary
way makes that worse.

An addition qualifies only if it converts a free choice into a measured quantity. Anything else
is deferred to the minimal-model programme in §6, which is subtraction.

### On compute

Compute is not the binding constraint and cloud time is not needed. The validation pass ran in
35 minutes; these are small discrete models. **What is scarce is implementation time**, so the
tiers below are ordered so that partial completion still leaves the project better off, and each
tier is useful on its own.

---

## 1. Tier one — recomputation, no new simulation

Everything here runs on committed data. Highest value per unit of work in the whole document.

### R-1 — Recompute the three under-powered criteria

Five of nine headline criteria are computed over fewer than twelve independent units. Three had
far more data sitting in the same run and did not use it:

| criterion | computed over | available |
|---|---|---|
| update magnitude tracks recovered depth (the public headline) | 6 cells | 2,400 per-reader pairs |
| depth recovery is not effort recovery | 4 cell means | 1,920 |
| depth changes how much the reader takes on | 3 levels | 3,600 |

**Recompute each over the per-reader units, with a bootstrap interval.**

The reason this is first: the diagnostics pass established that **both 0.886 and 0.600 sit inside
what six points produce by chance**, so the validation pass's headline finding was not a flip. The
criterion was never able to tell in either direction. Recomputing over 2,400 pairs either produces
a real answer or establishes there is none, and both are better than what is on record.

**Report the original value beside the recomputed one in every case.** The originals are not
deleted.

### R-2 — Pairwise divergence beside every disagreement figure

Add mean pairwise Jensen-Shannon divergence between readers' full posteriors. **Beside** the
existing modal-goal entropy, never instead of it, so committed numbers stay readable.

The existing statistic cannot distinguish readers who are each certain of a different answer from
readers who are all equally lost and guessing. Both produce the same count vector. 41% of its
variance across the checked cells is explained by a straight line on within-reader uncertainty, so
it is close to a restatement of how unsure the readers are.

The replacement separates three pairs of cells the current one reports as identical. Recompute it
everywhere disagreement appears, including in every results file and the README.

### R-3 — Bootstrap intervals on every headline point estimate

A measurement instrument reports intervals. Almost nothing in this project does.

Resample readers with replacement, at least 2,000 draws, for every headline quantity. **The one
that matters most: the location of the fabrication peak.** Resample, recompute the index across
the overlap grid, take the argmax, and report the *distribution* of argmax locations.

That location is what every downstream claim is anchored to, including the prediction card, and it
currently has no error bar at all. If the bootstrap distribution spans several grid points, the
prediction card has to say so.

### R-4 — Correct the reader-count bias in entropy estimates

The bias is analytic at roughly (K−1)/2N nats, confirmed empirically, and correctable in one line.
It is negligible at 4,000 readers and material at 15, which means **the validation pass's
random-draw figures are not on the same scale as the headline figures they were compared against.**

Add the correction as a separate function and report both corrected and uncorrected. Do not
substitute, which would change what every committed number means without changing any file.

---

## 2. Tier two — repair the measures themselves

### R-5 — Decompose the uptake measure

This is the most consequential item in the pass.

The current measure is the distance between what a reader believed before and after. The
diagnostics pass showed it is **U-shaped in how well the reader understood**: a reader who becomes
confidently wrong ends far from its prior too, and scores 87% of what a correct reader scores.

**So "how much the reader took on" has never measured understanding. It measures belief movement,
and being fooled moves you almost as much as being right.**

Replace one number with three, all reported together:

| quantity | definition | behaviour |
|---|---|---|
| **movement** | KL(posterior ‖ prior) | what is currently reported; U-shaped |
| **error reduction** | KL(prior ‖ truth) − KL(posterior ‖ truth) | signed, monotone in accuracy, negative when the reader moves away from the truth |
| **the trust factor** | −ln(1−κ) | reported separately, never multiplied in |

**Error reduction is the measure that is not U-shaped**, and it is the one most experiments
should have been using. It goes negative for a confidently wrong reader, which is the correct
behaviour and which the current measure cannot express.

The trust factor is separated because it changes by **43× across the range one experiment sweeps**,
which is larger than most effects in this project. A sweep that varies trust and reports uptake is
reporting the product of two things.

**Recompute all three on every committed run where the posterior was saved.** Where it was not,
record which experiments would need re-running to get them.

### R-6 — Write an estimator for every parameter that lacks one

Currently: depth has a posterior mean (compressed at slope 0.32), readability and the value gate
got fitted estimators during the diagnostics pass, trust has one that does not work, and **effort
has none at all** — no agent carries a belief about it and it enters no likelihood an analyst
could profile, which is why one of the two joint-recovery checks could not be run.

Every parameter that appears in a claim needs an estimator and a recovery classification. Where an
estimator is an *analyst* fitting a model rather than the *reader* inferring a state, say so in
the same line as the number. For readability that distinction is the entire V4 reframe.

### R-7 — Replace the seed function, in parallel

The per-reader seed function is documented as collision-resistant and is not: 48% duplicates on
the overlap sweep, with a closed-form structure. Every collision is across cells rather than
within them, so the direction is conservative and no reported statistic is affected — but the
structure moves inside a cell under a different choice of cardinalities, so it is a live hazard
rather than a historical one.

A verified collision-free replacement already exists, unwired.

**Wire it, and run one headline experiment both ways.** Report the two side by side. Adopting it
silently would change every committed number; adopting it with the comparison published is a
measurement of how much the collisions were worth, which is the useful version.

Fix the docstring regardless. It asserts something false today.

---

## 3. Tier three — identifiability

### R-8 — Make trust measurable

This is the move that converts the apparatus, and it is the one place where adding machinery is
justified under §0's rule.

**Why trust is currently unmeasurable, precisely.** The likelihood is monotone in κ rather than
peaked. A sharper label channel makes whatever signal arrived more probable under *some*
provenance, and the reader is free to move its provenance belief to accommodate. So the estimate
runs to whichever end of the grid the data pushes it, and it runs to *opposite* ends on the two
datasets. Within a single artifact, trust and the provenance posterior trade off exactly.

**What breaks the trade-off: repeated encounters with a named source.**

Give each source an identity and an honesty rate the experimenter knows and the reader does not.
The reader encounters each source many times. Because provenance is also inferable from content,
every encounter produces an observable event: **does the label agree with what the content says?**

The mismatch rate over encounters identifies the source's honesty. The reader's trust is then
identified by **how much weight it puts on the label relative to content given that history** —
which is a different quantity from the provenance belief and no longer trades off against it.

**This is V4's C3, specified and never built.** It also makes the Zahavian signalling appendix
testable for the first time, since a source can now build a reputation and defect.

**Committed before the run:** trust must reach RECOVERED or PARTIALLY RECOVERED under this design,
or it is reported as structurally unidentifiable in this framework and every claim resting on it
stays conditional permanently. **That is a real possible outcome and it must be reportable.**

### R-9 — Re-run parameter recovery on the repaired model

Every parameter, the marginal pre-check, the single-parameter sweep, and every runnable pair.
The pre-check is necessary and not sufficient — trust passes it comfortably and is still
unidentifiable — so report both.

---

## 4. Tier four — coverage

### R-10 — Build the exact learning path

Nine experiments cannot run under exact inference. Six of those are blocked by Dirichlet learning,
and **two of the six carry public claims**: the unlabelled learner losing about a third of its
ability to read human work, and the 31% versus 74% coverage figures.

Those are the two results most likely to appear in outreach and neither can currently be checked
at all.

The exact agent should carry Dirichlet counts over the joint rather than the factorised
likelihood. Where that is intractable, say which experiments remain unreachable and why, rather
than approximating quietly.

### R-11 — Re-run the selectivity measure under exact inference

**The cheapest outstanding check in the project.** The shortcut's error peaks at timestep 2 —
where the joint belief is furthest from the product of its marginals — and decays afterward.

The selectivity measure is taken over **the first three free steps**, deliberately, to catch the
decision that matters. It sits exactly in the error peak, and it is one of the experiments that
could not be swapped to exact inference until the diagnostics pass fixed the accessors.

It is now possible and cheap. Run it.

### R-12 — Re-run everything reachable under exact inference

Fourteen experiments are reachable and have never been checked. The validation pass checked five
and three of those moved.

**A one-in-three movement rate on the checked sample is not a reason to leave fourteen unchecked.**
Run all of them, report every quantity both ways, and flag every verdict that moves.

---

## 5. Tier five — the reruns

### R-13 — Depth and the generous fallback, on a fair footing

Both came back inconclusive and both can now be rerun properly.

**Two constraints that pull against each other, and both must hold at once:**

1. Goal recovery must be genuinely uncertain, which the difficulty probe located at reader
   inexpertise around 0.85 with six observations. Note the knob that gets there is inexpertise;
   signature separation and goal count are dead, because they change how *fast* the reader reaches
   certainty rather than *whether* it does.
2. Both arms must sit on the **same side of the uptake trough**, which is at inexpertise 0.90 —
   uncomfortably close to the middle of the difficulty band.

**Use error reduction from R-5 as the primary outcome rather than movement.** It is monotone in
accuracy, so the trough problem does not arise for it, which may make constraint 2 moot. Report
both measures either way, because if they disagree that is itself the finding.

The generous-fallback experiment additionally needs its positive control rebuilt: it fails under
exact inference *and* under its own original criterion, which is two independent failures on the
same control.

---

## 6. Then, and only then: the minimal-model programme

Not for today. Recorded because it is where this is heading and because it is the subtraction that
balances §0's rule.

**For each surviving result, find the smallest model that still produces it.** If the interior peak
requires the split feature space, what is the smallest object with a split feature space? Strip
everything else and check whether the peak is still there.

Three things at once: it identifies what is genuinely load-bearing, it makes results comparable
across experiments, and it produces models simple enough to explain in a paragraph.

Then the minimal models become a **family**, and comparison replaces single-model validation as
the frame. With one model, a failure is uninterpretable — theory or implementation, no way to
tell. With a family, a failure tells you *which commitment* was wrong. "Depth does not move
uptake" is unreadable. "The depth-gated variant loses to the flat variant" is a result.

The channel-accounting result is a down payment on this: knowing that provenance inference is a
two-channel race with an analytic crossover tells you which commitment to try removing first.

---

## 7. Order, and what to do when time runs out

| tier | items | why here |
|---|---|---|
| 1 | R-1 … R-4 | no new simulation; three undetermined verdicts become determined |
| 2 | R-5, R-6 | R-5 changes what several experiments were measuring; do it before any rerun |
| 4 | R-11 | cheapest single check in the project, run it as soon as tier 2 lands |
| 3 | R-8, R-9 | the conversion; largest build |
| 4 | R-10, R-12 | unblocks two public claims and fourteen unchecked experiments |
| 2 | R-7 | parallel comparison, no urgency |
| 5 | R-13 | needs R-5 in place first |

**If only tier one completes, the day was worth it.** Three verdicts move from undetermined to
determined, every headline gains an interval, and the disagreement figure stops being quotable
alone.

---

## 8. Constraints

- **Every original number is retained and reported beside its replacement.** No committed value is
  silently superseded. This applies to R-1, R-2, R-4, R-5 and R-7 without exception.
- **The withheld experiment stays withheld**, its failing test stays in the suite, and the open
  residual stays open.
- **R-8 must be able to return "trust is structurally unidentifiable in this framework."** The
  reporting section is written before the result is known.
- **No claim is upgraded on the strength of a repair alone.** A criterion recomputed at higher
  power reports what it reports; if it comes back null at 2,400 pairs, that is the answer and the
  six-point version is not preferred because it was friendlier.
- **Additions must convert a free choice into a measured quantity.** Anything else waits for the
  minimal-model programme. The model has grown for five versions and the audit is now finding
  constructions that cannot be broken because the model will not run without them.
- **`REPAIR.md` is written from the verdict files afterward**, never from the expectations here.
