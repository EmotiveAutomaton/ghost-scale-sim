# H3: frozen V26 protocol

GLOBAL_REVISION. Does revision history reveal the goal a writer settled on, and the one she started with? (ceiling 4)

**Why.** The curator: global revision would occur "when you have allowed your expertise to slot in words automatically, and then you need to go back afterwards... and make changes to align those automatic outputs with the higher order functional hierarchy positions."

**Worlds.**
- A maker has one habit (H2's grid) and one provisional goal g0 (one of 6, with H2's weighting), both fixed across her documents. Each document is a 24-unit draft made by H2's process under g0 (deliberation probability 0.6, no difficulty pauses), so the draft carries g0.
- At a discovery point (after unit 12 or after unit 24, equally likely) the document's settled goal g* is drawn: g* = g0 with probability 0.5, otherwise uniformly among the other five. After an early discovery, units 13 to 24 are drafted under g*.
- At the end of the draft she revises once: each unit drafted before the discovery point whose weight under g* is below 1 is redrawn from the habit times g*'s weights with probability r (0.3, 0.6, 0.9), whether or not g* equals g0.

**Observations:** (a) the final text; (b) plus which positions were revised; (c) plus old and new units; (d) plus the discovery point.

**Readers:** exact posterior over (g0, g*, discovery point, r, habit). A record is one document; the records-needed figure on g0 counts one maker's documents, with g* drawn afresh for each.

**Case and check.** The case: a final text that still mixes g0- and g*-consistent units.
- Check: at r = 0.6, reader (a)'s posterior on the true g* is below 0.8 in at least 50% of worlds where g* differs from g0.

**Primary.** Evidence about g* from (c) beyond (a), per revision event, at r = 0.6. Under 0.05 nats per event counts as none. Also: records (documents) needed to reach 1 nat on g0, the goal she started with, from (c).

**Falsifiable arm.** None: (c) carries weakly more information than (a) by construction. The primary's size is the result, and the g0 records-needed figure is the open question.

**Known-answer checks.**
- All goal weights equal to 1: no unit is deliberated or revised, (c) equals (a), and every posterior over (g0, g*) equals the prior.
- r = 0: (b) and (c) carry no revision events, and their posteriors over (g0, g*) equal (a)'s.

**Texture grid:** draft lengths (12, 48), discovery points, partial revisions that move a unit only part way (redraw from a mixture of g0 and g*).

**Shed order:** r = 0.3; (d).

**Route.**
- To Sounding Line's S2 (what gets revised later) and S3 (whether revision moves text toward the finished piece): how much revision history should add over the final text, per event.
- To ToMpathy: whether revised spans deserve a "realigned here" mark in the expertise-and-habit panel, as constructed support.


## Declared implementation choices

Six makers, one per initial goal, cyclic habits. Documents share initial goal and habit. Settled goal is .5 same or .1 each other; discovery equally half/end. r prior uniform .3/.6/.9, shared across all documents of one maker. Prefixes1/2/4/8/16. Revise flag includes redraws with unchanged content. Primary pooled gain/event within a world. Local settled goals are marginalized before accumulating maker evidence.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[
  {
    "r": 0.6,
    "units": 12
  },
  {
    "r": 0.6,
    "units": 48
  },
  {
    "r": 0.6,
    "discovery_fraction": 0.25
  },
  {
    "r": 0.6,
    "discovery_fraction": 0.75
  },
  {
    "r": 0.6,
    "partial": 0.25
  },
  {
    "r": 0.6,
    "partial": 0.5
  },
  {
    "r": 0.6,
    "partial": 0.75
  }
]
```
