# What the world has found, next to what this model predicted

**A retrospective literature check, one row per experiment.** It was run *after* the simulations and
informed no design. It is a coherence check, not evidence for the model.

**How to read it.** Regular text means the published work **agrees** with the simulation's finding.
*Italic text means it disagrees, complicates, or cuts against it* — those are the rows worth reading
first, because a coherence check that only finds agreement has not been run properly.

Where a row says **no direct evidence located**, that is what it says: nobody appears to have asked
the question. Those are the project's forward predictions by default, and they are the most
interesting rows in a different way.

---

## The label, and what it does to a reader

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E2** | A false claim of authorship makes every reader certain and no two agree; told the truth about the same object they are appropriately unsure. | Messingschlager & Appel, *New Media & Society* (2025), [Mind ascribed to AI and the appreciation of AI-generated art](https://journals.sagepub.com/doi/10.1177/14614448231200248) — labelling an artwork AI-generated reduces appreciation, **and the effect is mediated by perceived mind of the artist**. That mediation is the model's proposed mechanism, measured independently. | **Yes, including the mechanism** |
| **E2** | The effect is about the *label*, not the object. | Malecki, Messingschlager & Appel (2025), [The impact of exposure to generative AI art](https://doi.org/10.1177/14614448251344590) — identical artworks, only the attribution varies, appreciation drops. | **Yes** |
| **E17** | Invention is graded by how little intent survives, not binary. | [Beyond ethics and aesthetics: perceived lay theory violations and agency disruption in AI-generated art](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1716814/full) (2026) — graded agency disruption rather than a single categorical rejection. | **Yes** |
| **A1** | Mislabelling is asymmetric: human work called machine-made is still read *accurately*; what is lost is willingness to look. | [Understanding Reader Perception Shifts upon Disclosure of AI Authorship](https://arxiv.org/pdf/2510.24011) (2025) — disclosure "drained the text of its warmth" while comprehension was unaffected. Attention and evaluation move; understanding does not. | **Yes** |
| **E4** | Even a fully sceptical reader with no label invents about one time in five. | *No direct evidence located.* The literature almost always supplies a label; the no-label baseline is rarely run. | *Untested* |

## The crash, and what kind of failure it is

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E37** | A distinct failure mode: content that is **legible and empty** — every word familiar, nobody behind it. Not "I cannot parse this". | [Understanding Reader Perception Shifts](https://arxiv.org/pdf/2510.24011) (2025) — participants: *"The text seemed well-written, but learning it was from an AI made it feel like it lacked a soul."* And [Exploring the difference and quality of AI-generated versus human-written texts](https://link.springer.com/article/10.1007/s44217-025-00529-z) (2025) — "structurally sound and coherent... but lack depth". Writers in [Precarity and Solidarity](https://arxiv.org/pdf/2412.04575) describe AI writing as "soulless... devoid of intentionality". | **Yes, and in participants' own words.** This is the model's strongest external match, and it is the one result with a **0%** false-positive rate. |
| **E20** | Confident invention peaks where content is *almost* readable, not where it is empty. | The uncanny valley is the same shape in a different domain — [systematic review, *Frontiers in Psychology* (2025)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1625984/full): discomfort peaks at near-human, not at obviously artificial. Independent arrival at an interior maximum. | **Yes, by analogy** |
| **E19** | Readers **keep paying** on content with real structure they cannot parse. | ***Disagrees.*** [Attention, Emotion, and Authenticity: Eye-Tracking Evidence from AI vs. Human Visual Design](https://www.biorxiv.org/content/10.1101/2025.10.22.684027v1) (2025) — human-made visuals drew **longer** viewing (M = 7035 ms), more fixations, and broader exploration. Real AI content produces *less* attention, not sustained futile attention. | ***No — and this is the single most useful disagreement in the table*** |
| **E1** | Readers disengage from work with no intent behind it. | The same eye-tracking result **supports E1** while contradicting E19. | **Yes** |
| **E34** | Not answerable in simulation: where does real generated content sit on the readability axis? | ***The eye-tracking result is a partial answer, and it points at the disengagement end.*** Which means the prediction card can be scored — and the branch that fires is E1's, not E19's. | ***Answered against the newer prediction*** |

## What a contaminated corpus does over time

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E6 · E9** | Two separable damages. Absorbing bad material scales with how much there is; *failing to absorb good material* is fully present at zero contamination, because it is driven by walking away. | Gerlich (2025), [AI Tools in Society: Impacts on Cognitive Offloading and the Future of Critical Thinking](https://www.mdpi.com/2075-4698/15/1/6) — 666 participants; heavier AI use predicts lower critical thinking, **mediated by cognitive offloading**. The mediator is the walking-away channel. MIT's essay-writing study found weaker neural connectivity in LLM users. | **Yes, strongly, for the second damage** |
| **E7** | An unlabelled learner folds machine structure into its picture of people and loses about a third of its ability to read genuine work. | Shumailov et al., *Nature* (2024), [AI models collapse when trained on recursively generated data](https://pubmed.ncbi.nlm.nih.gov/39048682/); and [Strong Model Collapse, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/284afdc2309f9667d2d4fb9290235b0c-Paper-Conference.pdf) — degradation appears even when synthetic data is a small fraction. | **Yes, in machines. No human equivalent has been measured.** |
| **E16** | About a third of machine content must be labelled — but only for readers who know the convention exists. | ***Complicates severely.*** [Implied Authenticity Effect, ICWSM (2025)](https://ojs.aaai.org/index.php/ICWSM/article/view/42721) — labelling *some* content raises the perceived authenticity of the unlabelled, at about a fifth the size of the direct effect. **Partial labelling has an asymmetry the model does not contain.** [Labeling AI-generated media online, *PNAS Nexus* (2025)](https://academic.oup.com/pnasnexus/article/4/6/pgaf170/8151894). | ***Partly — same inference, opposite valence*** |
| **E16** | The convention-aware reader needs less coverage. | [Does labelling AI content make users more sceptical?](https://www.hertie-school.org/en/news/detail/content/does-labelling-ai-content-make-users-more-sceptical) — the more comprehensive the labelling, the less the implied-authenticity dynamic can be exploited. Which is the model's coverage threshold, arrived at from the other direction. | **Yes** |

## The reader's own limits

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E10** | A reader's own skill caps what can be recovered, measured on a corpus with *zero* machine content. | [Art Expertise Reduces Influence of Visual Salience on Fixation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4319974/) — artists' fixations are less driven by low-level features; they extract on high-level structure. Experts and novices reading the same object differently is the whole claim. | **Yes** |
| **E15** | Competence collapse is a **knee, not a cliff** — its width does not narrow with more evidence. And belief accuracy breaks down before choice accuracy. | [Correcting Human Labels for Rater Effects in AI Evaluation](https://arxiv.org/pdf/2602.22585) (2026) and RLHF work reporting expert–annotator agreement around 70% — rater identity carries a large share of the variance, and it does not resolve with scale. | **Yes** |
| **E48** | A reader can only see as far up a hierarchy as it has built itself; expertise is possessing a structure, not being well-calibrated. | [The Influence of Art Expertise and Training](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0134241) — high-expertise viewers find work "significantly more interesting and less confusing"; expertise raises *cognitive* facets while affective ones stay flat. The split the model now makes between depth and calibration. | **Yes** |
| **E48** | Reading and making are the same machinery, so appreciation installs capability. | [From abstract painting to action painting: rethinking embodied simulation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405211/) (2025) — viewers motor-simulate brushstrokes; [Dissociating embodiment and emotional reactivity in motor responses to artworks](https://pubmed.ncbi.nlm.nih.gov/33761410/) found *muscle-selective* facilitation for brushstroke over pointillist canvases. Perception recruiting the motor system is measured. ***What is not measured is the second half: that this leaves the viewer able to produce.*** | **Half yes.** The mechanism is evidenced; the acquisition claim is not. |
| **E43** | The more practised the work, the less reliably its maker can name its own purpose. | [The paradox of human expertise](https://discovery.ucl.ac.uk/id/eprint/48372/1/Dror_PB_paradoxical_human_expertise.pdf) — "given the nature of automaticity, experts cannot fully account and explain, or even recall, their actions." Polanyi's tacit knowledge; the expert blind spot. | **Yes, and it is well established** |

## Depth, density, and attention

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E49** | Artfulness is **density** — hierarchy per unit of observable extent — which is what lets a readymade be dense rather than empty. | [Compression ensembles quantify aesthetic complexity and the evolution of visual art, *EPJ Data Science* (2023)](https://link.springer.com/article/10.1140/epjds/s13688-023-00397-3) — compression-based complexity tracks human judgements of visual complexity. [Kolmogorov compression complexity differentiates schools of iconography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9232591/). | **Yes, the measure is established.** *The bimodality prediction is not tested anywhere.* |
| **E50** | Grabbing attention and keeping it are two decisions; shock art and slop are different objects. | [Art Expertise Reduces Influence of Visual Salience on Fixation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4319974/) — capture by surface salience is separable from sustained expert attention, and expertise moves the second without moving the first. | **Yes, indirectly** |
| **E30 · E31** | Depth moves what a reader takes of the **method**, and provably cannot move what it takes of the **purpose**. | *No direct evidence located.* Nobody appears to have separated method-uptake from purpose-uptake in an art-perception paradigm. | *Untested — and this is a clean forward prediction* |
| **E36** | Resolving what something was *for* unlocks reading *how* it was done, within a single encounter. | *No direct evidence located.* The nearest relative is schema-driven comprehension, but the temporal ordering claim inside one exposure is not something the literature has tried to measure. | *Untested — the sharpest forward prediction here* |

## Trust, disclosure, and bad actors

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E41 · R-8b** | Trust may work by *lowering the guard* rather than by misinforming — so a trusted source is absorbed even when it tells the truth. And a sufficiently trusting reader can never learn a source lies. | The **sleeper effect** is the closest published relative: [Pratkanis et al.](https://faculty.washington.edu/agg/pdf/Pratkanis&GLB.JPSP.1988.pdf) — a discounting cue decays faster than the message, so initially-rejected content gains force. ***But the literature is contested and some researchers doubt the effect exists.*** | *Partly, and on contested ground* |
| **E46** | You cannot read something, reject it, and be unchanged — and the reader who studies it carefully to refute it drifts **more** than the one who skims. | ***Cuts both ways.*** Counterarguing research finds those encouraged to counterargue show **less** attitude change, which is the opposite of the model's prediction. The sleeper-effect literature supports delayed absorption of rejected material. **The model sides with the minority.** | ***Contested, and the model takes the less-supported side*** |
| **E51** | Honest marking is self-policing — above a detection rate of about 0.50. | ***Refines the framework's own framing.*** Modern signalling theory has moved off Zahavi: [Honesty in signalling games is maintained by trade-offs rather than costs, *BMC Biology* (2022)](https://link.springer.com/article/10.1186/s12915-022-01496-9) and [general signalling theory, *J. Evol. Biol.* (2026)](https://academic.oup.com/jeb/article/39/2/171/8362708) — honesty is stable when deception is *costly*, not when signals are *wasteful*. **The simulation independently landed on the trade-off account rather than the handicap account it set out to test.** | **Yes to the result, *no to the framework's stated mechanism*** |

## The one that was withheld

| # | What the model found | What has been published | Agrees? |
|---|---|---|---|
| **E8** | Withheld three times. Its own honesty check failed every time; generational decay cannot be told apart from the instrument's noise. | The machine-learning literature has the effect the simulation could not measure: [recursive collapse within five generations](https://arxiv.org/pdf/2412.17646), and [mixing ≥5% real data prevents long-term collapse](https://arxiv.org/html/2509.16499v1). | **The effect exists in machines. The simulation still cannot measure it, and that remains true.** |

---

## What this check actually found

**Six agreements that matter**, because they arrived independently and one of them arrived in participants' own words: the label effect *and its mind-perception mediation*, legible-and-empty, cognitive offloading as the second damage channel, model collapse, the expert's inability to explain themselves, and compression as an aesthetic measure.

**Four places the world disagrees or complicates:**

1. **E19's sustained attention.** Eye-tracking finds *less* attention on AI content, not more. The model's own E1 prediction fits and its E19 prediction does not — which effectively **scores the E34 prediction card**, and it scores it against the newer of the two accounts.
2. **E16's coverage threshold.** The implied-authenticity effect gives partial labelling an asymmetry the model does not contain: labelled content loses trust *and unlabelled content gains it*.
3. **E46's careful-reader prediction.** Counterarguing research says engaging carefully produces *more* resistance, not less. The model sides with the sleeper-effect minority.
4. **E51's mechanism.** Signalling theory abandoned the handicap principle two decades ago. The simulation's result is right and the framework's stated justification for it is out of date.

**And four rows where nobody has asked the question** — E4's no-label baseline, E30/E31's method-versus-purpose split, E36's within-encounter ordering, and E49's bimodality. Those are the project's real forward predictions, arrived at honestly: not sealed in advance, but genuinely unasked.

---

## The author's reading of the four disagreements

Recorded because a disagreement with no interpretation attached is just a loose end, and because
two of these are testable.

**E19 — less attention on AI content.** *Not a failure of the finding so much as a missing layer.*
The proposed reading: readers have begun learning the **surface signature** of generated content and
disengage on that, before any attempt to read intent. That is a learned detector sitting in front of
the mechanism this model implements — and it predicts something specific and unpleasant: **it will
misfire.** A detector trained on surface features will fire on human work that happens to share
them. **Buildable, and it would reconcile E19 with the eye-tracking result rather than choosing
between them.**

**E16 — the implied-authenticity effect.** Accepted without argument. The reading: this is a
**transition cost**. A convention only works once enough people hold it, and until then partial
adoption has a corrosive effect on exactly the content it fails to cover. That is a real price of
implementation rather than a flaw in the proposal, and it should be stated as one — a disclosure
scheme is worse than nothing at low coverage and better than nothing above it, which is what E16's
threshold already says.

**E46 — counterarguing produces more resistance, not less.** The proposed reading, and it is the
most interesting of the four: there is probably a **pre-emptive adversarial mode** — a stance in
which you take something apart at a distance with the gate already shut, rather than closing it
reactively once you have understood. Given how much of human history is social adversarial play, it
would be strange if that channel did not exist.

*The implication is the useful part.* If such a mode exists, it is architecture that could be
**pointed at generated content deliberately** — which would make the Ghost Scale's protective effect
larger than anything measured here, because a label that triggers adversarial reading does more than
a label that triggers dismissal. **Buildable: a reader whose gate is pre-closed rather than
reactively closed, and the prediction is lower drift at the same engagement.**

**E51 — signalling theory.** Accepted. The framework arrived at its security argument through
Zahavian signalling and modern theory has moved to a trade-off account. The simulation independently
produced a trade-off result. The conclusion stands; the stated mechanism is updated wherever it
appears.

---

*Searched August 2026. Nothing here informed any design.*
