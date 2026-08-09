# Reading intent: every claim this simulation has tested, under its umbrella hypotheses

> **"To appreciate something is to identify the actions taken to create an artifact, then to use
> those actions to infer the goals of the creator, and to subsequently connect with and learn from
> the creator if those goals are deemed worthy."** *(the essay, 2026-04)*

**The theory in three lines.** A reader looks at something made and runs an inference about the
maker, what they were for, how they worked, whether to let any of it in, under a metabolic
budget. Generated content breaks that inference; a believed false label suppresses the breakdown
and the reader fabricates. The Ghost Scale is the proposed affordance that lets the machinery
stand down cheaply.

**The one empirical commitment, stated where it can be argued with:** the model reads the
published tiers' opacities directly as the fraction of the maker's intent that survives into the
work. The checks establish that only the *ordering* matters, the results survive compressing and
stretching the ramp, but the mapping itself is a modelling choice, not a measurement.

**How to read this file.** Sections run in decreasing load-bearing order, each: the claim in the
curator's words, what it says, then every result that bears on it, with status. Format, status
legend and source legend are in [README.md](README.md). Detail for any row:
[FINDINGS.md](../../FINDINGS.md) (the method archive), then the version document the row names.
**This file holds living hypotheses; what is actually implemented and executable, and where the
implementation's claims stop, is stated in the
[repository gateway](../../README.md#what-is-implemented).** An OPEN row here is a claim nothing
has run against; a (run) row is a mechanism the simulator has executed.
Statuses here are **(run)**, a committed verdict in this repository, unless marked **(lit)**,
published work from [EVIDENCE.md](../../EVIDENCE.md). Every (run) row is a **method** result
about a constructed world; whether the mechanism transfers to people is exactly what a simulation
cannot say, and rows that depend on a real-world fact say so.

---

## §1. The core commitment: reading is inverse inference of a maker

**2026-04, the essay:**

> There is a part of your brain continuously seeking evidence of actions, assessing those actions
> as effective or ineffective, and trying to learn the effective ones by intensely empathizing
> with the creators.

**What it says.** The reader holds a model of a maker rather than a classifier of surfaces: a
purpose, a process, a provenance, and inverts the artifact against it. This is the commitment
everything else stands on, it was attacked directly, half of it fell, and the half that survived
is now the best-established thing in the record.

| # | hypothesis | status | notables |
|---|---|---|---|
| E21 | Confident, contradictory readings of empty content require modelling a mind | **WITHDRAWN (run)** | A counting classifier (200 examples, no maker anywhere) reproduces the signature via small-sample overfitting. The claim of *necessity* is dead and marked at every point it was made |
| E45 | What the maker-model buys: evidence efficiency | **WITHDRAWN (run)** | The efficiency comparison handed the simulating reader the world's own emission map, it needed no examples by construction, so the test could not fail. At the counter's full budget the gap is five points |
| E45 | What the maker-model buys: reading a goal never shown | **SUPPORTED (run)** | The counter sits at chance at any training size; the maker-model reads an unseen intent, and holds when its map is perturbed halfway to random. The two things no baseline reproduces: label-response and sustained attention on the unresolvable |
| MIN | Every surviving finding needs the maker-model | **SUPPORTED (run, V9)** | Remove one structural commitment at a time: **every finding dies when the reader stops modelling a maker and starts classifying a surface.** Hierarchy and costly attention are free, no finding needs them |
| E36 | Resolving the purpose unlocks the method | **SUPPORTED (run) in its temporal form; the between-reader form REJECTED** | Within one reading, process recovery roughly doubles *after* the goal settles. Between readers, getting the goal right does not predict getting the method. The pre-registered between-reader test is reported as failing |
| E10 | The reader's own expertise caps what can be recovered | **SUPPORTED (run)** | On a corpus with zero machine content anywhere: hold the material constant, vary the reader, extraction collapses |
| E48 | A reader sees only as far up a hierarchy as it has built itself | **SUPPORTED (run)** | Expertise is possessing a structure, not being well-calibrated. (lit READ: expertise raises cognitive facets of viewing while affective ones stay flat) |
| E15 | The competence collapse is a cliff | **REJECTED (run)** | The author's own claim, tested knowing it could only survive or weaken. A knee: width unchanged across sixteen times the data. Sharper secondary: **belief accuracy rots before choice accuracy**, the internal picture fails while the picks stay right |
| E33 | A reader can know a maker better than the maker knows itself | **SUPPORTED (run), scoped** | The margin grows as the self-account degrades; the reader is told how unreliable the report is. The self-blindness half splits: the mark on the work **exists and no reader in this model can read it**: readings differ in the fourth decimal |
| E43 | Practice removes the maker's own reasons from report | **SUPPORTED (run)** | Compression is what makes a decision unavailable to its own maker, while the reader is unaffected. (lit READ: the expertise literature states this directly: automaticity costs experts the ability to account for their actions) |

**What these add up to.** The necessity claim died and the sufficiency claim survived, and the
project is better off for the trade: *"a maker-model is required for confident invention"* was
false, but **no finding in ten versions survives removing it, and two behaviours, responding to a
label, and paying attention to something unresolvable, have no baseline reproduction.** The core
commitment now rests on ablation rather than assertion. Within one encounter the inference has an
order (purpose first, then method: E36), a ceiling set by the reader rather than the artifact
(E10, E48), and a failure mode where the reader's picture degrades invisibly while its choices
still look right (E15), which is the shape of every quiet failure in the rest of this file.

## §2. The wall (familiar words, nobody home) is a distinct failure, and the reader keeps paying

**2026-04, the essay, on realising a thing is generated:**

> It's not that there isn't a creator — but rather that the math becomes so complicated that you
> simply stop trying... The black box is so unbreakable you may as well not even try.

**What it says.** The theory predicted a *crash*: disengagement to protect the budget. The
simulation's most distinctive finding is that the crash is only half the story, and the other
half inverts it: content carrying real-but-foreign structure holds a reader indefinitely, and
content whose every word is familiar but whose maker cannot be reconstructed produces a signature
of its own: **legible and empty**, which is the complaint people actually make.

Four content models, kept distinct because they behave differently: **intent-empty** (wood grain,
structure with nothing deciding), **intent-foreign** (a script you cannot read: real purpose in
a vocabulary with no entry), **intent-unrecoverable** (every word familiar, the route back to a
maker-state does not exist), and **value-divergent** (a person who wants what you do not). None of
them is "the intent is repugnant"; the model does not contain that.

| # | hypothesis | status | notables |
|---|---|---|---|
| E1 | Readers abandon intent-*empty* work, from plain cost–benefit | **SUPPORTED (run)** | No built-in dislike of machines exists to explain it: provenance preference is asserted zero at every construction (N7). Rider: the tier meant to say *don't spend here* is the one readers spend most on; **the model and the published design disagree and the design is probably right** |
| E19 | Readers keep paying on intent-*foreign* work | **SUPPORTED (run), CONTESTED (lit)** | The inversion of the framework's own prediction. Eye-tracking finds **less** attention on AI content, not more, the single most useful disagreement in EVIDENCE.md; which branch fires depends on how much human-shaped structure real content carries, which is E34's card |
| E37 | The wall is a missing inversion, not a vocabulary deficit | **SUPPORTED (run), and the one finding that is genuinely the theory's** | *Legible and empty* on familiar features. **0% false-positive rate under random settings (V8)** and the only finding needing the reader to hold a **distribution** rather than a best guess (V9). (lit READ: participants verbatim: *"well-written... but it lacked a soul"*) |
| E20 | Confident invention peaks at an interior point of the readability axis | **SUPPORTED (run)** | About a tenth readable: enough handholds to build a story, not enough to make it right. Same location under exact inference, all seventeen robustness cells, disjoint seeds, double scale, 100% of bootstrap draws, the most robust single result here |
| E20 | The collapse and the invention peak share one band | **RETIRED (run, V7)** | Did not survive exact arithmetic. Removed from the README; the committed prediction card carries it only under a SUPERSEDED marker |
| E32 | Unreadable content ≡ unskilled reader | **REJECTED (run), they are opposites** | At matched deficit the novice quits and feels settled; the expert stays and stays lost. The second dimension is **whether you can tell you are failing**. Survives exact inference on all five measures |
| E14 | Readers were quitting before they worked it out | **REJECTED (run)** | Forcing them to keep looking made it worse |
| E39 | A no-maker *hypothesis* lets the reader stand down | **REJECTED (run)** | It redirects rather than relaxes: invention −38%, no resolution because a no-maker hypothesis about the goal is redundant with what the reader knows about provenance. If the Ghost Scale is to relax a brain, it must act on the gate, not the hypothesis space |
| E34 | Where real generated content sits on the axis | **NOT ANSWERABLE IN SIMULATION**, a prediction card | (lit) The eye-tracking result is a partial score **against the newer branch**: it points at the disengagement end, E1's branch rather than E19's |
| E53 | A learned surface heuristic fires before intent-reading, and sharpening it worsens false accusation | **HALF-REJECTED (run)** | The heuristic is real (0.63 vs 0.23) and never stops misfiring, but sharpening makes misfiring **rarer**. The conflict it was built to reconcile was a comparison error between two axes |
| E57 | ...until the content fights back | **SUPPORTED (run)** | With evasion tracking detection, false accusation stops falling and peaks at **65% of careful human work**; evasion off reproduces E53's decline exactly. E57b: a detector that stops updating stops discriminating without stopping accusing: hit rate 0.575 → 0.175 with false alarms flat |

**What these add up to.** The wall is the project's one distinguishing finding, the only one with
a 0% architectural false-positive rate, the only one requiring a posterior rather than a point
estimate, and the one the outside literature confirms in participants' own words. Around it sits
an honest, unresolved fork: whether real generated content is *empty* (readers disengage, E1,
and the eye-tracking says this) or *foreign* (readers burn on it, E19). **The fork is a fact
about the world, not the model**, E34 is the card that scores it, and the current external
evidence favours the branch the project moved *away* from on theoretical grounds. The detection
rows add the adversarial floor: even a correct heuristic, sharpened, peaks at accusing careful
human work two times in three once content adapts, so *detect-and-drop* has a structural ceiling
that *reading intent* (§6) does not share.

## §3. The trust exploit: a believed false label moves the reader away from the truth

**2026-04, the preprint:**

> A vulnerability appears when an observer encounters a generative artifact lacking extractable
> intent while believing it to be human... the brain hallucinates a non-trivial local minimum to
> satisfy the computational requirement of the open gate.

**What it says.** The label and the work are two witnesses. On a lie they disagree, and above a
computable trust threshold the label wins before the disagreement can register, the reader
builds a confident theory about someone who was never there, and ends further from the truth than
it started.

| # | hypothesis | status | notables |
|---|---|---|---|
| E2 | A false human label produces confident, mutually contradictory readings; the truth produces calibrated unsureness | **SUPPORTED (run), with four standing qualifications** | The *confident* half is architectural (100% of random models); the claim is scoped to labels trusted above the crossover; the disagreement figure cannot be quoted alone; and on the signed measure the lie moves readers **away** from the answer: opposite signs, not different sizes. The central result. Independent rebuild from prose alone: same direction, **fifteen times smaller; quote the direction, never the size** |
| A1 | Mislabelling is symmetric | **REJECTED (run): sharply asymmetric** | Human work called machine is still read *accurately*; what dies is the willingness to look. Machine work called human produces disagreement at 99.5% of the theoretical ceiling. One costs attention, the other costs the model. (lit READ: direction holds; the world adds a cost axis we do not model: false accusation harms people) |
| A2 | Miscalibration under a false label is a bad tail | **REJECTED (run), it is unanimous** | All four thousand readers in the top confidence band while performing at chance |
| E17 | Invention grades with how little intent survives | **SUPPORTED (run)** | Four tiers give three steps; state it as an ordering, not a slope |
| E4 | Invented meaning is a slope in trust | **REJECTED (run), a switch** | Below ~one-fifth trust, invention largely stops. The half less quoted: a fully sceptical reader with no label still invents about one time in five |
| D-1 | The exploit is one mechanism | **REJECTED (run, diagnostics), it is two witnesses** | Label-evidence and work-evidence arrive at every glance and disagree on a lie; which wins is arithmetic with a crossover at trust 0.54. **Every claim "a label does X" is really "a label trusted above 0.54 does X"** |
| R-8b | A trusting reader eventually notices a lying source | **REJECTED (run, repair)** | Not slow learning; learning that cannot start: above the threshold the label has already won the disagreement that noticing requires. A prediction the fixed-trust model could not make |
| E41 | The paper's mechanism and the code's mechanism are the same | **REJECTED (run, V6)** | The paper says trust switches the alarm off (predicting absorption even under a believed *truth*); the code says the label out-argues the work. Run on the label cells: a truth-told reader integrates almost nothing under the code's mechanism and about half under the paper's. **An unresolved fork in the theory itself, in code** |
| E31 | The depth collapse and the trust exploit are one mechanism | **SUPPORTED (run, V7-closed), with a logged audit qualification** | A dishonest label inflates the estimate of the thinking behind the work; re-scored on process uptake, 0.83 against the original bar. V5-4 (2026-08-08): the tracking correlation was scored on open-gate cells only: as pre-registered "across all cells" it is −0.02 by construction, and a fabrication-gap bar the run fails (0.025 vs 0.05) went unreported |
| E46 | You can read a thing, reject it, and be unchanged | **REJECTED (run), CONTESTED (lit)** | Refuting requires partly running the argument; the careful refuter takes on more than the skimmer. Counterarguing research finds the opposite; the sleeper-effect minority agrees with us. **The model takes the less-supported side** |
| E42 | Attention implies openness | **REJECTED (run)** | A stable regime exists: high engagement, accurate reading, closed gate, nothing integrated |
| E5 | Trust in provenance is decisiveness renamed | **REJECTED (run)** | Only trust produces the human–machine gap; decisiveness moves the level and never the gap |

**What these add up to.** The exploit survived three audits by becoming more precise: it is now a
claim about **labels trusted above a stated crossover**, on a **signed** measure, quoted as a
**direction**. Its two sharpest consequences are structural, not parametric, a sufficiently
trusting reader *cannot* learn it is being lied to (R-8b), and rejection is not immunity (E46).
The unresolved item is E41: the paper and the code disagree about the mechanism, the paper's
version predicts a reader that is told the truth, believes it, and absorbs anyway, and the code
cannot produce that reader. **That fork decides what H6 means** (§10) and is the single most
important theoretical question the record leaves open. (lit: the mind-perception mediation the
paper proposed has since been measured directly and holds, the one place the mechanism, not just
the effect, has outside support.)

## §4. Depth is compressed practice, not effort; method, not purpose, is what transmits

**2026-04, the essay:**

> Automaticity is the caching of human struggle.

**What it says.** What gates uptake is not how hard the maker tried but how many levels of
compressed decision-making reach the surface, the Zen master's circle against the child's
scribble. Built so that depth lives **only in the order** of the work (identical feature
histograms at every level, a counting reader cannot see depth at all), which is the commitment
that stops "depth" being "legibility" renamed. And what a reader takes from deep work is the
**method**; the purpose provably cannot carry it.

| # | hypothesis | status | notables |
|---|---|---|---|
| E30 | Depth changes what a reader takes away | **SUPPORTED (run) on method; structurally unmeasurable on purpose** | The construction holds purpose equally readable at every level, so the pre-registered purpose measure could not have moved, bounded near zero and explained. On process recovery, depth moves it with the interval excluding zero. *Was: an unexplained null* |
| N21 | Depth is not effort renamed | **SPLIT (run), reported as failing** | The pre-registered contrast returns *effort can manufacture depth* (the reader's **estimate** is contaminated); on what actually **transfers**, depth dominates effort ninety-seven-fold. Both reported; the original decides, and it fails. The effort axis was also rebuilt to make "offhand but deep" representable *before* measurement; logged |
| E28 / E29 | The rationality-of-the-maker construct | **SUPERSEDED (run)** | Kept as records of a mis-specified construct: a fully committed trivial effort and a master's offhand sketch sat in the same cell. Replaced by depth; the two removed experiments are named in the README |
| E56 | You can take someone's technique without their aims | **SUPPORTED (run), and it is where indoctrination lives** | A guard raised early blocks purpose and values by about half and method by **four percent**, because method arrives from the first line and purpose resolves late. The guard's direct protection is ~5% on the rescored harm measure (4.7, interval [3.8, 5.4]); the rest rides in on practised method, where a maker's unspoken commitments are stored. **100% architectural (V10 severity): a property of building a reader this shape, not distinguishing evidence** |
| E38 | Machine literacy stacks with human literacy | **REJECTED (run), it substitutes** | A machine-tuned reader reads machine work perfectly and gives up nearly three-quarters of its accuracy on human work (1.00 → 0.28). A crossover, not an upgrade |
| E40 | Surface appeal and endorsement combine additively; optimising the surface cue is harmless | **Half SUPPORTED, half REJECTED (run)** | Additive as predicted, and direct optimisation of the depth-signal produces a third failure mode: **readers pay more and get less.** The RLHF argument in one line, and the prediction E55's surface-filter null later confirmed from the other side |
| E49 | Artfulness is density: hierarchy per unit of observable extent | **SUPPORTED (run)** | What lets a readymade be dense rather than empty. (lit READ: compression-based complexity tracks human judgement; the bimodality prediction is untested anywhere) |
| E50 | Grabbing attention and keeping it are one decision | **REJECTED (run), two decisions** | Shock art and slop are different objects. (lit: capture-by-salience and sustained expert attention separate, and expertise moves only the second) |
| E43 / E33 | Practice hides the maker's reasons from the maker, not from the work | **SUPPORTED (run)** | See §1, the mark exists on the object and is unreadable to any reader in this model; only measuring the object directly could tell the three sources apart |

**What these add up to.** Depth earned its place the hard way: its first measure could not move
(E30's construction), its second is contaminated (N21's estimate leg), and what survives is the
**transfer asymmetry**: deep work transmits method overwhelmingly and purpose not at all, and a
guard can only block what arrives late, which is purpose. Put E56, E38 and E40 side by side and
the section's claim sharpens into the project's darkest result: **the channel that carries skill
is the channel that carries values, it is open before any gate can close, and both optimising the
signal of depth and tuning the reader to the machine make things worse in ways that look locally
like improvement.** N21's failing original is retained on purpose: the reader's *estimate* of
depth is effort-contaminated even where the *transfer* is not, and no single number states both.

## §5. A diet of unlabelled machine content damages the reader twice, separably

**2026-04, the essay:**

> Your brain wasted so much energy in the past seeking the ghost of an author in the world around
> it. Nowadays, it keeps being disappointed after its search for meaning... You will literally
> become too tired to care.

**What it says.** Two damages, different mechanisms. Absorbing bad material scales with dose.
Failing to absorb good material does not, it is fully present at zero contamination, because it
is driven by the walking-away habit, not by what is in the pile.

| # | hypothesis | status | notables |
|---|---|---|---|
| E6 / E6b / E9 | The two damages are one thing | **REJECTED (run): separable, and the second is dose-independent** | The starvation signature is fully present at zero contamination. The strongest independent support of anything here (lit READ: cognitive-offloading studies land on the walking-away channel almost exactly; the absorption channel has no human measurement). Regenerated at full scale under the fixed harness, 2026-08-08: directions held, E6's curve steeper post-fix |
| E7 | A learner can spot hollow sources unaided | **REJECTED (run)** | It folds machine structure into its picture of *people* and loses about a third of its reading of genuine work. Honest labels: a clean picture ≥6× faster. Validated under exact inference after the repair pass built the path. (lit: model collapse is established in machines; no human equivalent measured) |
| E35 | The damage accumulates and carries to unseen human work | **SUPPORTED (run) in direction; magnitude unstable** | Mechanism reproduces on all three seed blocks; the absolute threshold passes one in three because the baseline varies two-fold. **The direction is the claim** |
| E12 | The generational leak is sampling noise | **REJECTED (run)**, the framework's own repair hypothesis, killed by its own gate | The leak does not shrink across a hundredfold more material. *Committed numbers predate the 2026-08-08 harness fix; direction-level until regenerated* |
| E13 | The freeze and the leak are two defects | **REJECTED (run), one shared axis** | The criterion produced a usable-looking number and was thrown away for lacking its precondition. *Same regeneration caveat as E12* |
| E18 | Passing one reader's estimate forward was the whole leak | **REJECTED (run)** | Fixing that channel left the damage in place. **A second contributor exists and has not been found, the oldest open mechanism question in the record** |
| E8 | Damage compounds across generations | **WITHHELD (run), three times** | Its honesty check (zero contamination must show zero damage) failed every time. Not a null: *could not measure*. The failing test stays in the suite as a visible marker |

**What these add up to.** The diet section holds the project's most externally-supported claim
(the dose-independent starvation channel) and its most honest refusal (E8, withheld three times
rather than reported around). The causal chain it cannot yet close: the leak is real (E12), it is
one axis rather than two (E13), the obvious channel is not it (E18), and the generational
compounding that everything points toward cannot be measured with the current instrument. **That
is a mechanism gap, not an evidence gap**, and any future fix must switch off a deliberately
failing test to claim otherwise.

## §6. Reading intent is a defence; inspecting the surface is not

**2026-08, the curator's row on the front page (◐):**

> Yes, and filtering on writing quality does nothing at all... Asking who made it and why cuts the
> damage, restores its grip on human writing, and costs nothing on clean data.

**What it says.** The last version turned the machinery around: against content published to be
*absorbed* rather than read, carrying real structure under a false claim of origin: a
surface-quality filter is worthless by construction, because surface quality is the attacker's
objective. Reconstructing the maker is a defence that reads no label.

| # | hypothesis | status | notables |
|---|---|---|---|
| E55 | An intent gate defends a learner where a quality filter does not | **SUPPORTED (run)** | Quality filter vs no filter: 0.826 vs 0.827: indistinguishable, as E40 predicted. The intent gate cuts damage 23%, restores reading of genuine work 0.84 → **1.00**, costs nothing on a clean corpus, never reads the label. 83% architectural (severity draw): mostly the shape, partly the theory |
| E55b | Convincing mimicry is the danger | **REJECTED (run): disguise is** | Every reader does *better* on perfectly-mimicked content than on a clean corpus: content reproducing genuine intent-structure teaches true things about how purposes make surfaces. The predicted-against-interest hypothesis failed and the failure is the more useful finding |
| E57 / E57b | Detection substitutes for reading | **REJECTED (run)** | See §2, under adaptation, false accusation peaks at 65% of careful human work, and a stale detector keeps accusing after it stops discriminating. The defence that survives an arms race is the one that reads structure the attacker must *keep* to succeed |
| H10.4 | Values ride in on process even through a shut gate | **WITHHELD (run)** | The version's most attractive idea. Its test arm damaged a learner on a corpus with no contamination in it, a broken update, not a mechanism. In the human-reader arm the effect is real and under its own pre-registered bar (0.193 vs 0.2). **Not refuted, not established, and the thing most worth building properly** |
| N45 / N50 | The gate's clean-corpus cost, and drift with nothing to detect | **Reported as failing (run)** | Two of the three failing nulls in the suite belong to this section and stay visible |

**What these add up to.** The constructive result, and the record's largest self-correction lands
here rather than in §7 where it started: **the Ghost Scale failed twice as a label makers apply
and succeeded as a capability readers run**: reject what you cannot attribute a maker and a
purpose to. It needs no adoption, no honesty from anyone, and no detector, which is exactly what
the arms-race rows say a durable defence requires. The gap between E56 (method passes any guard,
100% architectural) and H10.4 (values-on-method, withheld) is the section's live question: the
defence blocks the *purpose* channel, and whether the part it cannot block carries the part that
matters is precisely the withheld experiment.

## §7. The Ghost Scale as a label: where the proposal itself now stands

**2026-04, the essay:**

> We need to actually sign our work with intentionality.

**What it says.** The published proposal is a maker-side signing convention. The simulation's own
verdict on it is the least flattering section of this file, and the project's credibility rests on
saying so plainly.

| # | hypothesis | status | notables |
|---|---|---|---|
| E3 | The labelling scheme pays for itself | **SUPPORTED (run)** | Roughly halves wasted effort, at one to three points of accuracy on genuinely human work, the cost of occasionally walking away from something real |
| E16 / E47 | A minority of labelled content protects the stream | **SUPPORTED (run), scoped, and (lit) complicated** | Committed thresholds: **31%** coverage for a reader who knows the convention exists, **74%** for one who does not, a lower bound by construction (the aware reader is handed the true coverage). Holds under the V7 coupling. (lit READ: the implied-authenticity effect, labelling some content makes the *unlabelled* look more authentic, is an asymmetry the model does not contain; same inference, opposite valence, and the two results together say the scheme is worse than nothing at low coverage and better above a threshold no human study has located) |
| E5 | κ is a distinct quantity | **SUPPORTED (run)** | Trust in the label is not decisiveness; see §3 |
| E51 | Honest marking is self-policing | **SUPPORTED (run), mechanism updated** | Only above a detection rate of **0.25**, a trade-off result. (lit READ: signalling theory has moved from handicap to trade-off accounts; the simulation landed on the current position without being aimed at it. The conclusion survives; the preprint's stated mechanism should be revised) |
| E39 | A maker-hypothesis is the affordance's mechanism | **REJECTED (run)** | See §2: redundant with provenance; the relaxation must act on the gate |
| E54 | A *read-this-differently* label carries the affordance | **REJECTED (run), twice-measured** | The pre-shut stance is real and protects (+0.015 [0.011, 0.019], ~6%); applied to half a stream, **no label separates from no label.** Second failure to find the Scale's mechanism as a label, confirmed by the E54-R rescore |
| E1-rider / H3 | The published attention gradient matches the model's | **REJECTED (run), and the design is probably right** | The tier meant to say *do not spend here* is the most expensive for the model's readers; the repository's own charting code quietly substituted a monotone ramp before the prose admitted the conflict. Needs human subjects (H3) |
| E52 | The sealed forward prediction | **Primary held (run); status withdrawn** | The author does not recognise authoring the commitment, so it is not a forward test and the forward-test count is zero |

**What these add up to.** As economics, the scale works in-model (E3, E16, E51). As a *label
mechanism* it has failed every attempt to find it (E39, E54), its published attention gradient is
probably backwards for real readers (E1's rider), and its security argument needs its stated
mechanism replaced with the trade-off account the simulation itself produced. What survived is
§6's inversion: **the value of the Ghost Scale in this record is as a description of what a
reader should do, not of what a maker should sign.** The label experiments' standing contribution
is the pair of coverage thresholds and the self-policing floor, quantities a human study could
target, plus one warning the literature sharpened: partial adoption has a cost side (implied
authenticity) the model cannot see.

## §8. What the apparatus is entitled to claim about itself

**2026-08, the curator's front page (◐), introducing the against-the-framework section:**

> The section a stranger should use to decide whether to trust the rest.

**What it says.** The record's honesty machinery is itself a set of tested claims: how much of
each finding is architecture, which commitment carries it, what the instruments can measure at
all, and where the criteria failed at their own jobs.

| # | hypothesis | status | notables |
|---|---|---|---|
| V8 | The headline findings are the theory's, not the shape's | **REJECTED for two of three (run)** | Random-settings reproduction: false-label effect **100%**, depth-moves-method **98%**, the wall **0%**. V10 added: gate-blocks-purpose 100%, intent-gate-beats-surface 83%, stale-detector asymmetry 75%. A held prediction with a 100% false-positive rate is true and non-distinguishing |
| V9 | Some finding survives without the maker-model | **REJECTED (run)** | None does. The complementary pass to V8, and the two agree from opposite directions on which result is genuinely the theory's (the wall) |
| A2 | The instruments can answer what they are asked | **Four limits (run)** | Trust was unmeasurable as fitted (repair later recovered its lower half); label and work are two witnesses with a 0.54 crossover; uptake is U-shaped in accuracy (a confidently wrong reader moves as far as a correct one, the repair split it into a signed measure and the headline cell reversed sign); the disagreement figure cannot be read alone |
| A1/A3 | The recorded answers survive recomputation | **Five of nine checks against the work (run)** | Two verdicts were artifacts of the inference shortcut; the independent rebuild reproduced the mechanism at one-fifteenth the size; the exact-solver switch (V-1) re-runs every experiment through its own unmodified code path |
| — | Criteria did their own jobs | **Four failures, each caught by a later pass (run)** | A six-cell rank correlation deciding a headline; a permutation check that could never pass; a monotonicity criterion punishing guaranteed ties; an absolute threshold on a two-fold-varying baseline. A fifth caught before it ran. **This is the failure mode the project has most of** |
| — | The record is forward-tested | **REJECTED: count zero** | Eighteen logged places where a design or criterion changed after seeing a result (7 found by V8, 4 added by V9, 7 by V10); the one sealed prediction was disowned and E52's held primary does not restore it. **The largest single thing the project owes** |
| — | The harness the corpus family ran on was sound | **REJECTED, then repaired (2026-08-08)** | A reused observer's prior was silently wiped from the second artifact onward (E6–E9, E12, E13, calibration). E6/E6b/E7/E9 regenerated at full scale: every direction held. E8 withheld regardless; E12/E13 direction-level until regenerated |

**What these add up to.** The apparatus's honest self-description in one sentence: **directions
and orderings are trustworthy; sizes are properties of this model's dimensions; two of three
headlines are shared with any model of this shape; the wall is the theory's own; and the whole
record is a theory checked against itself, with zero forward tests.** Every criterion failure was
found by a *later* pass, which is the argument for keeping the pass cadence rather than for
trusting any single layer of it.

## §9. Service results for the sibling: validated rulers, and instruments killed before use

**2026-08-07, the curator, commissioning the batch:**

> It sure does feel like there's fertile ground for high-fidelity simulation on some of the things
> we are doing over here... I do worry that it runs us astray sometimes.

**What it says.** Sounding Line reads real text and cannot construct ground truth; this
repository can. The T/S modules test the sibling's statistics against planted answers before real
corpora are spent on them. They are harvested as `(sim)` rows in the sibling's theory folder;
this table is the home record. **All of these are method results, the strong kind, the kind that
transfers.**

| # | question | status | notables |
|---|---|---|---|
| T-1 | Is empathy three coupled inferences or a chain? | **Chain, on the substitute triangle (run)** | Goal a sink (at exact ceiling, 1.000, in the flagship cell; 0.83/0.495 off-ceiling), process the source (+0.840 to depth), edges additive with small positive excesses (+0.008–0.010) in off-ceiling cells. **The values vertex does not exist in this model and was not invented**, the verdict's own SUBSTITUTION field says it must be built before T-1 can be asked as posed |
| T-6 | The exact information budget of one observation | **Values vertex VOID (run)** | H(values \| goal) = 0, a deterministic coarsening; four of six edges are properties of the matrix, not measurements. The correction to T-1's reading |
| T-2 / T-9 | Motivational breadth is measurable as recovered-purpose breadth | **INSTRUMENT DEAD (run), twice** | `purpose_breadth` tracks task difficulty, not variety: at matched difficulty the diversity excess is −0.013 to −0.025. Killed the sibling's construct before a corpus was spent on it |
| T-3 | Decisions are countable from the posterior | **REJECTED (run)** | The sub-goal posterior floors at ~2.33 effective modes in the most favourable regime; "a decision was recovered" never becomes a well-defined event |
| T-10 | The decision's *location* is recoverable where its identity is not | **Weakly SUPPORTED (run)** | Posterior travel beats a circular-shift null by +0.081 [0.072, 0.091]: real, small. The transportable half: canonical trajectory features read the maker's *switch rate* at \|r\| ≈ 0.44 |
| T-4 | The leaked/emblematic split survives an uncertain reader | **SUPPORTED (run) for heavy concealment** | The louder-shield direction holds under a reader wrong about almost everything, including a 50% channel swap, and fails at 25% concealment: it reads *effort spent hiding*, not hiding. Post-fix re-run strengthened it (0.546 → 0.723 on the fair baseline) |
| T-5 | The process posterior beats the goal posterior as a maker-detector | **Neither dominates (run)** | 6 of 14 contested cells to process, median difference −0.002; both near ceiling at high channel counts. Posterior travel is the best single feature |
| T-8 | A multivariate trajectory bank beats the best single feature | **REJECTED (run)** | Median gain on fresh seeds +0.019, worst −0.084, helps in 4 of 8 cells: one number somebody thought of is as good |
| S-1 | The unlock ratio inherits E36's support | **REJECTED (run)** | At the only threshold where it is defined often enough to read (0.95), agreement with the true process measure is ~0.08 across cells; at 0.90 it fails N28 outright (reads 0.11 where the truth demands 1.0). The sibling's primary has to earn its own support |
| S-2 | Purpose-breadth measures flattened intent | **WITHDRAWN (run)** | Same difficulty confound as T-2/T-9 |
| S-3 | The leak channel is readable; concealment is divergence | **QUALIFIED (run)** | Leak readable at 0.90; concealment shows as an *amplified* display, not a suppressed one |
| S-45 | Inference order changes the answer | **REJECTED (run), by exactly zero** | Anomaly-first settles ~5% sooner: a cost saving, not a finding. Order-independence is a theorem only for the static factors; checked empirically for the temporal chain |
| S-6 | Practised polish decays faster than depth; synthetic polish is flat | **SUPPORTED (run), miniature** | 6.5× faster decay; the flat-surface machine signature is the positive claim worth having. A purpose-built construction, not the shared world, no severity coverage yet |
| S-11 | The component-count pipeline recovers a planted number | **QUALIFIED (run)** | Parallel analysis returns **~250 components from structureless data** (250 at n = 1200 in the committed shuffled-null arm, 254 at 2400, growing with sample size, which is the retracted failure reproduced on demand); the bias-corrected participation ratio and bi-cross-validation recover the planted count. The counting method the sibling retracted was broken, not biased |
| S-15 | Value-profile recovery converges with artifacts per maker, and the residual is priced | **SUPPORTED (run, V11), with the expertise half of C2 REJECTED as pre-registered** | 0.53 → 0.98 over 1..50 artifacts, residual L1 0.009. The bounded family (convergent midbrains) is worth **0.24 L1**; breaking the reader's expertise cost 0.0003 against a 0.05 bar: goal reading saturates at this length, so the price is *invisible, not zero* (queued with T-11). **Corpus price: 20 works per maker** at curator-tier noise. Construction discrimination: conjunctive satisfaction reads a profile from one artifact (0.97), amplification cannot (0.53); **the sibling's single-artifact failure record is the amplification signature** |
| S-14 | An absent drive is recoverable | **SUPPORTED (run, V11)** | Near-invisible in spontaneous work (0.61), perfect under commission toward the missing channel (1.00), and pure compliance collapses to exactly 0.5, the discriminator is *how the goal is pursued*, nothing else. Instruction amplifies multiplicatively; a zero cannot be amplified. The made-under-duress mechanism, first working form |
| S-12 | A three-locus structure with a noisy middle reads as one mid peak | **SUPPORTED (run, V11), both halves** | The smear: 100% of runs, and 100% of random re-parameterisations: architectural, which is the point: published mid-peak profiles are uninformative against a three-locus truth. The residual instrument (G22's route) separates the worlds at AUC 0.87, in **25%** of random parameterisations, so it has an operating regime, not universality. Its own gate caught the identity arm's first construction: amplitude alone can smear fixed loci (D-V11-1) |

**What these add up to.** The service work killed three instruments (breadth, the unlock ratio,
decision-counting) and one criterion family (parallel analysis) *before* real corpora were spent
on them, a dead ruler found here costs an afternoon; found there it costs a corpus, and V11 has
now built the object whose absence T-1 and T-6 recorded: **the values vertex exists, and its
first three measurements came back.** Values converge across artifacts and the ambiguity floor is
small under the theory's assumptions; the assumption that carries the weight is the shared
bounded family (0.24 L1), not reader expertise (unpriced at current difficulty, the honest open
edge); an absence is readable exactly where the theory says attention lives (commission,
multiplicative amplification); and the field's depth-profile instrument cannot distinguish the
trimodal architecture from its own consensus, while the residual route can, within a measured
operating regime. What the sibling should take: size the follower corpus at ~20 works per maker,
read its own single-artifact failure record as evidence for amplification over conjunctive
satisfaction, and build the residual analysis before conceding anything to the mid-peak
literature. Queued here: T-12's supply arms (G47 at last askable), T-11's off-ceiling
restatement, and the habit-residue build (T-13).

## §10. The preprint's six hypotheses: the standing crosswalk

**2026-04, the preprint:** *"Six empirical hypotheses are proposed to test the framework."*

The full curator-voiced statement is on the front page; this is the ledger form. None of the six
is refuted; none is confirmed in people.

| # | claim | standing |
|---|---|---|
| H1 | The metabolic drop on learning a thing is AI-made is an autonomic step | **Sharpened (run + lit).** The drop is real *when the reader identifies the content*, and protective. The failure is the detector, not the drop: unlabelled, readers do the opposite (E19). Needs the pupillometry study |
| H2 | Raters disagree far more about AI artifacts | **Supported and relocated (run).** Driven by the **label**, not the origin (E2, A1/A2); whether theory-of-mind is *required* for it is not decidable by this apparatus in either direction (E21, E45) |
| H3 | Ghost Scale tiers produce a dose-dependent metabolic trade-off | **Not simulated; the model disputes the design's gradient and the design is probably right** (E1's rider). Needs human subjects |
| H4 | Copying AI geometry degrades the artist's own fluency | **Not simulated; nearest analogue holds** (E35, direction only). The acquisition test remains the top external priority, deliberately unsimulated |
| H5 | Intent-dense training data yields less sycophancy at equal benchmarks | **The one V10 ran, and the closest thing to a confirmation** (E55: quality filtering does nothing, intent-gating defends at zero clean-corpus cost). The real test is a reward model on real corpora, the sibling is that build |
| H6 | Bypassing the firewall writes AI content straight to predictive models | **Answered worse than posed (run).** No hypnosis needed: the firewall is porous by default (E46), the guard blocks ~5% (E54/E56), and the *mechanism* of the bypass is exactly the E41 fork, which the code cannot currently express in the paper's version |

---

## The maintenance rule

**A result lands here in the same pass that lands it in [FINDINGS.md](../../FINDINGS.md).**
FINDINGS is organised by what was run; this file by what is believed. A result recorded only in
FINDINGS gets lost; a claim recorded only here loses its provenance. When any table changes, the
paragraph under it is revisited in the same edit, a stale conclusion under a fresh table reads as
current, which is worse than none.
