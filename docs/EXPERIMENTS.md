# What was tested, and what came back

Every question below was derived from the theory and then tested in a simulation. The
**"found afterward"** column lists published work located *after* the simulations ran, in a
literature search done specifically to check whether any of this was already known. Nothing in
that column informed the design. It is a coherence check, not evidence.

---

## The results

| The question | What came back | Found afterward |
|---|---|---|
| **Does a viewer give up on something made without a purpose behind it?** | Yes, and it falls out of ordinary cost-benefit reasoning. No special mechanism for disliking machines was needed, and none was built in. | — |
| **If you lie about who made something, what happens to the viewer?** | Every viewer becomes confident. No two agree on anything. Told the truth about the same object, they become appropriately unsure instead. | Well replicated. Several studies hold the artwork constant and change only the stated author, and find the same collapse in appreciation. Mind-perception mediates it. |
| **Does the labelling scheme pay for itself?** | Roughly halves wasted effort. Costs one to three points of accuracy on genuinely human work, because a label-aware viewer occasionally walks away from something real. | — |
| **Where exactly does invented meaning happen?** | Trust has a sharp switch, not a slope: below roughly one-fifth trust in the label, invention stops. Even a fully sceptical viewer with no label at all invents about one time in five. | — |
| **Is trust in provenance just general decisiveness renamed?** | No. Trust changes the *gap* between how a viewer treats human and machine work. Decisiveness only moves the overall level and never produces the gap. | — |
| **Can you learn to spot hollow content without being told?** | No. The learner folds machine structure into its picture of what humans are like, and loses about a third of its ability to read genuine work. With honest labels it builds a clean picture quickly. | Model collapse is established in machine learning. No human equivalent has been measured. |
| **Are there two different kinds of damage?** | Yes, separable. Absorbing bad material scales with how much of it there is. Not absorbing good material does not — it is fully present even at zero contamination, because it is driven by walking away, not by what is in the pile. | **Strong support for the second kind.** Cognitive offloading reduces engagement including self-monitoring; skipping effort impairs skill acquisition; and users perform worse than never-users once the tools are removed. Almost nothing on the first kind. |
| **Does a viewer's own skill cap what can be extracted?** | Yes, on a corpus with zero machine content anywhere. Hold the material perfectly constant, vary only the reader, and extraction collapses. | Expertise is known to moderate aesthetic processing broadly, and artists' eye movements are measurably less driven by surface features than novices'. The threshold shape is untested. |
| **Is that collapse a cliff or a knee?** | A knee. A real cliff sharpens as you add evidence; this one did not budge across sixteen times the data. **And belief accuracy breaks down well before choice accuracy does** — a rater's internal picture rots while their picks stay right. | **One direct hit.** Experts rating AI safety responses agreed so little that roughly nine-tenths of the variance in a label reflected the rater rather than the response. Reward models trained on that learn rater habits. |
| **How much labelling is enough?** | About a third of machine content — **but only for viewers who know the labelling convention exists.** Viewers who do not need three-quarters, and never build a reliable picture at any coverage. | **Directly relevant, and it complicates us.** The implied truth effect: warning-labelling *some* false headlines makes the unlabelled ones look truer. Replicated for AI content as an implied authenticity effect. Same inference, opposite valence. The literature calls coverage the key variable and has never produced a threshold. |
| **Does invention scale with how hollow something is?** | Yes, smoothly. Telling the truth about hollow content converts near-certainty into honest uncertainty. | — |
| **Is mislabelling symmetric?** | No. Same confidence either way, but the disagreement differs enormously. Human work called machine-made is still read correctly. Machine work called human produces maximum disagreement. | **Direction holds, consequences reverse.** Expert artists detect AI images well but produce more false accusations than automated tools, and false accusation is socially costly. We measure damage to understanding; the world measures damage to people. |
| **How miscalibrated does a false label make you?** | Every one of four thousand viewers landed in the highest confidence band while performing at chance. Not a bad tail. Unanimous near-certainty about nothing. | — |
| **Does the collapse survive a more generous set of explanations?** | Yes. Adding "they were just exploring" as an available explanation absorbed exploratory *human* work convincingly and did nothing at all for machine work — it was chosen *less often than random guessing.* | — |
| **Do viewers actually disengage from machine content, or keep paying?** | **They keep paying.** Content with real structure the viewer cannot parse holds attention indefinitely, because every look keeps promising an answer that never arrives. This inverts the earlier prediction. | **One suggestive hit.** Eye-tracking found AI-labelled artworks produce more dispersed gaze. Dispersed is not disengaged — it is searching without settling. |
| **Where along the readability axis does it break?** | In the middle, not at the empty end. Invention peaks where the content is about a tenth readable — enough familiar structure to make an explanation seem available, not enough to make it right. **The collapse and the invention peak occupy the same narrow band**, which the framework had always treated as two separate phenomena. | — |
| **Is a model of the maker's mind actually necessary?** | **Partly, and the unwelcome half comes first.** A simple counting classifier that never represents a maker at all reproduces the confident-and-inconsistent pattern, through nothing more than small-sample overfitting. What it *cannot* do: respond to a label, or keep paying attention to something it cannot resolve. Both of those need the full machinery. | Nobody has asked this question. Our negative is the only data point that exists. |
| **Are the collapse and the trust exploit the same mechanism?** | **Yes.** The same machine-made object, labelled honestly, reads as shallow and moves the viewer barely at all. Passed off as human, it reads as deeper and moves them twenty-two times as far. What a viewer absorbs tracks how much thought it believes went in, no matter which channel produced that belief. | — |
| **Is unreadable content the same as an unskilled reader?** | No, and they are opposites. At an identical information deficit, the unskilled reader quits almost immediately and feels reasonably settled; the expert facing unreadable content keeps working and stays lost. **The second dimension is whether you can tell you are failing** — a badly aimed template fails silently, out-of-range content fails loudly. **And the unskilled reader of human work is substantially more accurate than the expert reader of machine work.** | — |
| **Can a viewer know a maker better than the maker knows themselves?** | Yes, and the margin grows the more wrong the maker's self-account is. The viewer's accuracy stays flat as the maker's self-report degrades to nothing. *Scope: the viewer is told how unreliable the report is, so this is a calibrated reader discounting a known-bad source.* | — |
| **Does self-blindness leave a mark on the work?** | **Yes on the object, no to the viewer.** Work made by a maker driven by something they cannot see is measurably marked, and work by a liar or a system with no self-model is not. But no viewer in this model can tell the three apart — the readings differ in the fourth decimal place. **The mark exists and is unreadable**, which is a different result from there being no mark, and only measuring the object directly could distinguish them. | — |
| **Does how much thought went in change how much you take away?** | **Inconclusive, and the construction is at fault.** Depth was visible and absorption was flat — but the measure written down in advance could not have moved, because depth was deliberately built so the goal is equally readable at every level. Two of the three depth levels also turned out indistinguishable. **What did show up unpredicted: depth drove attention roughly six-fold and absorption not at all.** | — |
| **Is "depth" just "effort" wearing a hat?** | No. Depth reading tracks depth about six times more than it tracks effort, and effort cannot make a viewer see depth that is not there. *Two caveats: the pass condition was rewritten after the first version failed, with the original retained and reported as failing; and the effort parameter was rebuilt so that "offhand but deep" is representable at all, which builds the dissociation in before measuring it.* | — |

---

## Not reported

**One experiment was withheld three times and is not in the table.** It asked whether damage
compounds across generations, as people who learned from contaminated material become the source
for the next round. Its own honesty check — *with zero contamination, show zero damage* — failed
every time. The relay leaks, so real generational decay cannot be told apart from the
instrument's own noise. The failing test is kept in the suite as a visible marker rather than
deleted, so any future fix has to switch it off deliberately.

**This is not "we found no effect."** It is "we could not measure." The four diagnostic runs that
chased the leak are part of that investigation and are not standalone results.

---

## Removed from the record

Two experiments were run against a version of the model that was later found to be wrong, and are
**uninterpretable rather than embarrassing.** Reading them requires holding an assumption we now
know to be false, which makes the numbers meaningless rather than inconvenient.

- **A test of whether "how hard the maker was trying" gates uptake.** The construct was wrong.
  What gates uptake is not effort but the depth of thinking behind the work, which is a different
  quantity — a fully committed shallow effort and a master's offhand sketch sit on opposite
  corners. Superseded by the depth test above.
- **A test of whether three separate gates behave differently.** One of the three gates was the
  mis-specified effort construct, so the design was testing something that does not exist.
  Re-run correctly, with two gates, as the same-mechanism test above.

**Kept deliberately:** every result that came back against the framework. The counting-classifier
result that withdrew a claim, the knee that weakened a stronger framing, the prediction that
missed once all four variants were run rather than the flattering one. Those are unflattering and
interpretable, which is the distinction being drawn.

---

## Seven ideas that died

| Whose | The claim | What killed it |
|---|---|---|
| The framework's | The leak across generations is ordinary sampling noise | It did not shrink across a hundred times more data |
| The framework's | Passing one reader's estimate forward was the whole problem | Fixing that channel left the damage where it was |
| Mine | Viewers were quitting before they worked it out | Forcing them to keep looking made it worse |
| **The author's** | The competence threshold is a cliff | Its width did not change across sixteen times the evidence |
| Mine | Zero effort and "just exploring" are the same hypothesis | They give different answers, for a structural reason |
| The framework's | Modelling another mind is required for confident disagreement | A counting classifier does it |
| The pre-registration's | A spike in value-divergence separates the gates | Low effort spikes too, for unrelated reasons |

The fourth one died to a test the author approved knowing it had two possible outcomes: his claim
survives, or his claim weakens. There was no version where it got stronger.

---

*Every number quoted above is in the results files, with the criteria written down before each
run and the outcome branches named in advance. Where a criterion changed after a measurement, it
is logged as a deviation with the original retained and reported.*
