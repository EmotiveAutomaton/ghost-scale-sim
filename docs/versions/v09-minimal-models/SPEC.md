# Version 9 — the minimal-model programme, and the two experiments the literature asked for

**Written before any V9 code. Not edited afterwards. Deviations are logged in RESULTS_V9.md.**

---

## §0. Why this is the last version

Version 8's severity pass asked: *keep the model's shape, throw the settings away — does the finding
survive?* For two of three headlines it did, every time. Which means those findings come from the
**shape**, and no amount of further parameter work will change that.

**The complementary question has never been asked, and it is the one that actually discriminates:
keep the settings and strip the shape.** Remove one structural commitment at a time and see which
removal kills the finding. What survives every ablation was never the theory's. What dies to a
specific removal tells you exactly which commitment is load-bearing.

That is the minimal-model programme, deferred since the repair pass, and it is the last piece of
modelling work this project needs. After it, the remaining questions are human-subject questions and
this apparatus cannot answer them.

Two small experiments ride along, both of which came out of the author's reading of places the
published literature disagreed with the simulation. Both would reconcile a disagreement rather than
choose a side.

---

## §1. The minimal-model programme

### The six structural commitments

Every one of these is a decision the model makes about what a reader *is*. None is a parameter.

| commitment | what it means | what removing it gives you |
|---|---|---|
| **generative** | the reader models a *maker producing* the work, not a mapping from surface to label | a discriminative classifier |
| **costly attention** | looking is expensive and the reader chooses | a reader that always looks |
| **provenance as state** | where it came from is something the reader *infers*, held separately from what it was for | a reader for whom the label is just another feature |
| **hierarchy** | the maker has levels, and the reader represents them | a flat reader |
| **distributional belief** | the reader holds a distribution, not a best guess | a point-estimate reader |
| **shared likelihood** | reader and maker draw on the same family — the body plan | a reader whose templates are its own |

### The design

Four surviving findings × six ablations. For each cell: **does the finding still appear?**

The four findings are chosen to span the severity range, so the programme can be checked against a
result whose answer is already known:

- **the label effect** (false authorship misleads) — severity 100%, so it should survive nearly
  everything, and if it does not the programme is miscalibrated;
- **legible and empty** — severity 0%, so it should be fragile, and *which* removal kills it is the
  most informative single cell in this version;
- **depth transmits method** — severity 98%;
- **sustained futile attention** — never severity-tested.

**A finding's minimal model is the set of commitments whose removal does not kill it.** Reported per
finding, as a set, not as a score.

### What would make this uninformative

**If every ablation kills every finding**, the ablations are too destructive and the programme
measures whether the model runs rather than what it needs. Guarded by null N41: the label effect
must survive at least one ablation, because a finding with a 100% false-positive rate cannot
plausibly require all six commitments.

**If no ablation kills anything**, the ablations are not reaching the mechanism. Guarded by N42: at
least one finding must die to at least one removal, or the programme has not been run.

---

## §2. E53 — the surface detector

**The author's reading of a published disagreement.** Eye-tracking finds *less* attention on
AI-generated content, not the sustained futile attention E19 predicts. The proposed explanation is
not that E19 is wrong but that a layer is missing: **readers have begun learning the surface
signature of generated content and disengaging on it, before any attempt to read intent happens.**

That is a learned detector sitting in front of the mechanism this model implements, and it predicts
something specific:

**H9.1 — the detector works, and it misfires.** A reader that has learned a surface correlate of
machine origin disengages faster from machine content — reconciling E19 with the eye-tracking — and
**fires on human work that happens to share those features.**

**H9.2 — and the misfiring gets worse as the detector gets better.** A sharper detector is a more
confident one, and confidence on a surface correlate is exactly what produces false accusation.

*Fails if* detector accuracy and false-alarm rate move together in the same direction, which would
mean the detector is reading something real rather than a correlate.

---

## §3. E54 — the adversarial mode

**The author's reading of the other published disagreement**, and the more consequential of the two.

Counterarguing research finds that people encouraged to counterargue show **less** attitude change,
which cuts against E46's finding that a reader who studies something carefully to refute it drifts
*more*. The proposed resolution: there is a **pre-emptive adversarial mode** — a stance in which the
gate is shut *before* engaging, rather than closing reactively once the material has been
understood. Given how much of human history is adversarial social play, it would be strange if that
channel did not exist.

**H9.3 — a pre-shut gate protects where a reactively-shut one does not.** At matched engagement,
adversarial reading produces less drift than sympathetic reading of the same material.

**H9.4 — and this is what the Ghost Scale should actually be doing.** E39 found that a "there is no
maker here" hypothesis buys nothing, because it is redundant with what the reader already knows
about origin, and concluded the affordance would have to act on the *gate*. Adversarial mode is a
gate intervention. So a label that says **read this differently** should protect better than a label
that says **do not read this**.

Predicted: the mode-switch label outperforms the dismissal label on uptake of misleading content,
*without* costing accuracy on genuine work.

*Fails if* the two labels are equivalent, in which case the affordance's value really is dismissal
and the practical objection to it — that nobody applies a *do not look* label to their own work —
stands unanswered.

**This is also the answer to a practical objection raised several rounds ago and set aside.** A
label reading *don't look at this* is a non-starter: no maker applies it. A label reading *engage
with this differently* is something a maker would plausibly adopt. If H9.4 holds, the adoptable
version is also the effective one.

---

## §4. Nulls

N41 — the label effect survives at least one ablation.
N42 — at least one finding dies to at least one ablation.
N43 — the surface detector carries **no** goal information. It is a correlate of origin, not a
second legibility channel.
N44 — adversarial mode does not simply reduce engagement. If it protects by making the reader look
less, it is dismissal wearing a new name, and the comparison must be at matched engagement.

---

## §5. What is deliberately NOT built, and this is the end of the list

- **The acquisition test.** Exposing a reader to work above its level and measuring what it can then
  *produce* is the sharpest test of H8.3, and in humans it is the strongest empirical move available
  to this project. It needs human subjects and money. **Named as the top external priority and not
  simulated**, because simulating it would only re-demonstrate a rule already written.
- **Distributed authorship, tool-delegation, recursion.** Unchanged.

---

## §6. Pre-mortem

1. **The ablations are too blunt** and everything dies. N41.
2. **The ablations are too gentle** and nothing dies. N42.
3. **The surface detector is just legibility again.** N43.
4. **Adversarial mode protects by disengaging.** N44 — matched engagement is required, not optional.
