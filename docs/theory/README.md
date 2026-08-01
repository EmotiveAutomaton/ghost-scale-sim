# The theory this simulation implements

This folder holds the documents that say what the model is *for*. The simulation is downstream of
them: it writes the theory down as running code and checks whether the parts fit together.

**The theory itself lives outside this repository**, in the preprint and the artist-facing essay:

- *Art as an Algorithmic Virus: Unifying the Generative Crash and AI Value Convergence via
  Cognitive Affordances* — Zenodo, DOI
  [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)
- The plain-language version: <https://abrahamhaskins.org/art>

## The one-paragraph version, in the theory's own words

**Art is compressed intent.** Anything into which a thinking being concentrated a high density of
decisions in the service of a goal — a painting, a bridge, a speech, a route, a smile. To
**appreciate** something is to identify the actions that produced it, use them to infer the maker's
goals, and let the values those goals imply decide whether you take the method on board. That is
inverse reinforcement learning, executed in wetware, under a hard metabolic budget.

Generative output breaks the inference. Not because it is bad, but because there is **no coherent
maker-state to recover** — so the calculation does not converge, and the brain shuts the process
down to protect its budget. That shutdown is the **generative crash**. And when the reader has been
told a person made it, the shutdown is suppressed, the calculation runs against nothing, and it
**fabricates** an answer. That is the **trust exploit**.

The proposed remedy is a **cognitive affordance** — the Ghost Scale — that signals intent density
directly, so the reader's machinery can stand down cheaply instead of burning itself out on
something with nobody in it.

## What is in here

| file | what it is |
|---|---|
| [WHY_V6_EXISTS.md](WHY_V6_EXISTS.md) | The origin document for version 6. Written as a working register after reading the preprint and the essay against the shipped code, it is where the three missing equation terms were found and where the six extensions and two corrections came from. Effectively the reasoning behind [SPEC_V6](../specs/SPEC_V6.md). |

## The vocabulary, and how it maps onto the code

The simulation's variable names are not the theory's words. This is the crosswalk.

| the theory says | the model calls it | what it governs |
|---|---|---|
| compressed intent, artfulness | **depth** (μ) | how many levels of the maker's decision hierarchy reach the surface |
| appreciation | **engagement** | whether the reader spends the effort to look closely |
| epistemic trust | **trust in the label** (κ) | how much the reader weights what it is told about origin against what it can see |
| the disgust firewall | **the acceptance gate** (θ, λ) | whether what was recovered is allowed to change the reader |
| precision weighting (ω) | **the attention decision** | how much the reader is currently crediting what it sees |
| metabolic reserve (E) | **the reserve** | how worn down the reader is, carried across encounters |
| the generative crash | **the crash signature** | unresolved, and no longer looking |
| fabrication | **the fabrication index** | confident *and* mutually contradictory: invention, as against honest confusion |
| the Ghost Scale tiers | **provenance**, and its opacity α | how much of the maker's intent survives into the object |

**One term is used differently in the two places and it is worth flagging.** In the preprint, ω is
the reader's own precision weighting — an *output*, which collapses when the inference fails. In the
code, ω is feature overlap: a fixed property of the content, an *input*. The code's analogue of the
preprint's ω is the attention decision. They share a letter and are different objects.

## The one empirical commitment

The published Ghost Scale draws its four tiers at 100%, 95%, 60% and 5% opacity. **The model reads
those percentages directly as the fraction of the maker's intent that survives into the work.**

That is the load-bearing assumption of the whole framework, and it is stated in the open rather than
buried in an implementation file. If you think transparency-as-a-design-value and
recoverable-intent are not the same quantity, that is the line to argue with. What the checks
establish is that the *ordering* of the tiers is what matters: the results survive compressing and
stretching the ramp.
