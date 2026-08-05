# Second batch for the Ghost Scale Simulation

**2026-08-05.** Four questions. All four are about **mechanism**, which is why they go there rather
than being run on real text without ground truth.

The first is the important one and it reorganises the whole framework.

---

## T-1 · Is empathy three coupled inference problems? ★

**The claim, from the curator, arrived at independently of anything in this repository:**

> Empathy is effectively a variational inference problem — **three separate variational inference
> problems being solved in parallel, and each one bootstraps the others.** The more information you
> have in one, the easier it is to solve the others.
>
> 1. the **proximal goal**
> 2. the **process**
> 3. the **values / drives**
>
> This is why an expert can instantly understand what a novice was thinking. This is why being
> close friends with someone, you can read their book and get more of a sense of why they made
> certain choices. **This is why information passes more easily between people who are close.**

**Why this matters to the simulation specifically.** E36 established **goal → process** within a
single encounter, and called it the sharpest forward prediction in the project. If the triangle is
right, **E36 measured one of six directed edges** and the other five were never asked about.

### The test

The V5/V6 reader already carries a goal posterior and a sub-goal (mode) posterior. E36 already
splits a rollout at the point the goal settles. **So the machinery is there and what is missing is
the other conditioning directions.**

For each edge, supply ground truth at one vertex as a prior and measure recovery at another,
against a control given no prior:

| supply | measure | status |
|---|---|---|
| goal | process | **E36 — done, ~doubles** |
| process | goal | |
| values | goal | |
| values | process | |
| goal | values | |
| process | values | |

**Three things worth reporting beyond pass/fail:**

1. **Are the edges symmetric?** Does goal→process equal process→goal? The curator says the vertices
   have *"relative strengths, relative difficulties"*, which predicts asymmetry.
2. **Does supplying two vertices beat supplying one by more than the sum?** That is what
   *bootstrapping* means as opposed to *independent evidence*, and it is the whole claim.
3. **Is there an edge that does nothing?** A dead edge would be the most informative single result
   here, because it would say the structure is not a triangle.

**Values may need constructing.** V5 has goal and mode; if there is no values factor, the honest
options are to use the rationality knob (`beta`) as a stand-in and say so, or to report T-1 on the
goal–process pair only and mark the rest as needing a model that does not exist yet. **Please do not
invent a values factor and then measure it** — that is the trap this project keeps naming.

---

## T-2 · Does goal diversity rise with automaticity, at fixed decision count?

**The curator's second thought, which supplies a mechanism the framework lacked:**

> When we talk about something having **soul**, what that means is **a variety of motivations**. And
> it tends to travel with expertise — because as processes are baked in with automaticity, you lose
> conscious access to them and **they start to be tied more to your drives**.

So: practice moves a decision out of deliberate control, and what makes it instead is a drive.
An expert's artifact carries more motivational variety **without the expert choosing that**.

### The test

Vary automaticity — how much of the creator's emission comes from cached structure rather than
deliberate selection — **holding the number of decisions constant.** That constancy is the whole
design, and it is the thing no real corpus can do.

**Predictions:**
- the **diversity** of the goal mixture behind the emission **rises** with automaticity
- **goal recovery accuracy stays flat** — it is one goal; the sources feeding it multiply
- the reader's **posterior entropy over goals rises**, which is `purpose_breadth` in the other
  project and is already validated there at matched density (S-2)

**The trap:** if goal recovery *falls* with automaticity, "diversity" has quietly become "noise",
and the construct is confounded. That check is the point of running it here.

---

## T-3 · Is a count of recovered decisions ever recoverable?

S-1 established that a count ratio fails N28 at 17×, is uncorrelated with the graded measure, and
is undefined in 81–100% of cases — because the reader's sub-goal posterior stays diffuse throughout,
never below 75% of maximum entropy in 288 steps.

**The question that leaves open:** is that a property of *this* reader, or of the problem?

Sweep whatever makes the posterior sharpen — more evidence, higher legibility, a stronger prior,
longer rollouts. **Does the sub-goal posterior ever concentrate enough that "a decision was
recovered" is a well-defined event?**

- **If yes**, the count is recoverable in a regime, and that regime is worth naming — Sounding Line
  could aim at it.
- **If no**, discrete decision-counting is dead in principle rather than in practice, and every
  instrument built on "how many decisions" should be abandoned rather than repaired. **That is the
  more useful answer** and it would retire a whole family of designs.

---

## T-4 · Does leaked/emblematic divergence survive an uncertain reader?

S-3 found the two-channel emitter works: state recovers from the leak alone at 0.899, divergence
identifies concealment, and amplifying the shield makes concealment **more** detectable — the
curator's counter-intuitive claim, which held.

**That was measured with a reader that knows the channel structure.** A real reader does not.

Degrade the reader — wrong priors over the emission model, uncertainty about which channel is
which, a reader that does not know concealment is possible — and find **where divergence stops
working.**

**Why it matters here:** Sounding Line's version of this reads a language model, which has no
correct prior over a human's emission channels at all. If divergence collapses under mild reader
uncertainty, the whole leaked-layer programme is a simulation result that cannot be transported,
and that should be known before more is built on it.

---

## What to send back

Verdict, numbers, and **what would have falsified it** — the format that made the first batch
useful. **T-1 first if only one runs**, because it decides whether the framework is a chain or a
triangle, and Sounding Line's entire failed primary was built on assuming a chain.

---

## Addendum — the curator's directional predictions for T-1, recorded before the run

Dictated 2026-08-05, after reading the above and before any of it executed.

> **There is absolutely not going to be symmetry.**
>
> The easiest one to figure out is **the goal** — the proximal goal, why this artifact was made —
> and that informs the other two.
>
> That said, I suspect **the strongest predictive help will be the process.** If you have the
> process, then you can reproduce the artifact.

So the prediction has two parts and they are separable:

| | |
|---|---|
| **easiest to recover** | **goal.** Highest accuracy from the artifact alone, no prior supplied. |
| **most useful when supplied** | **process.** Largest improvement at the other two vertices when given. |

**Those can come apart, and that is what makes it a real prediction.** A vertex that is easy to
recover is not necessarily the one that helps most once you have it — and if the same vertex wins
both, the triangle has a natural entry point and the loop should be rebuilt around it.

### And the consequence he drew from it

> If you know the person better, then you can extract their proximal goal and their process more
> easily. Which makes **showing something you've written a kind of intimacy** — and more so to
> someone who knows you well.

That is the values vertex acting as a **prior held before the artifact is seen**, rather than as
something recovered from it. Which means T-1's values→goal and values→process edges are not just
two more conditions: **they are the formal statement of why sharing work with someone who knows you
is a different act from publishing it.**

Worth measuring for that reason alone, whatever the numbers do.

### On T-3

> I suspect T-3 is going to come back negative. But you know what, we're going to keep on trying.
> **Diversity of motivations is probably more easily extractable than decision density.**

Recorded as a prior. If T-3 returns negative — discrete decision-counting dead in principle — the
follow-on he names is already specified: **diversity as the recoverable quantity, density as the
one that is not.** T-2 is the test of that and it is in this batch.
