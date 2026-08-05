# Tests to run in the Ghost Scale Simulation

**Written 2026-08-03 from Sounding Line, for the sim repo.** Carry this file over.

Six tests. **Every one of them is a question about a MECHANISM, not about real text** — which is
why they belong in an environment with ground truth, a maker who actually acts, and four audit
passes of scaffolding, rather than in an environment where the only ground truth is a corpus label
somebody guessed.

**They are CPU/numpy work.** Sounding Line is GPU-bound on a Gate 3 run until ~03:30. **These will
not contend with it.** Run them whenever.

Ordered by what they would settle.

---

## S-1 · Is method unlock a broken statistic? ★ run this first

**The problem it settles.** Sounding Line's primary measure is

```
unlock = decisions_recovered_after_purpose_settles / decisions_recovered_before
```

a **count ratio with no ground truth**. E36's measure is `process_error_reduction` — mean
log-probability of the true mode against a uniform baseline. Those are different quantities, and
E36's own file rejects count-style statistics explicitly:

> the first implementation used ACCURACY. That was wrong... it came out BELOW nominal chance,
> which no amount of information could produce. **The fourth instance in this project of an
> instrument answering a nearby question.**

And Sounding Line's version already failed its control: **machine-generated artifacts unlocked at
1.111 against competent commercial work at 0.917.** With no execution chain there is nothing for a
settled purpose to unlock and the ratio should sit at 1.0.

**The test.** In the sim, where the true mode chain is known, compute BOTH quantities on the same
rollouts:

1. `process_error_reduction` before/after the goal settles — E36's existing measure, already
   implemented in `v6/e36_process.py` around the `RESOLVED_ENTROPY` split.
2. A **count ratio**: number of sub-goal steps whose posterior argmax is non-uniform (or however
   "a decision was recovered" best maps onto the V5 reader), after ÷ before.

Then:

- **Do they correlate across cells?** If not, Sounding Line's primary is not measuring E36's
  quantity and inherits none of its support.
- **Run the count ratio at mu = 1**, where N28 says there is no process. **Does it sit at 1.0?**
  If it moves, the count ratio fails N28 *in an environment with ground truth*, and Gate 3's
  primary is reading something else — which is the single most consequential thing this list can
  establish.

**Why it matters most.** Gate 3 finishes tonight. If the count ratio fails N28 in the sim, the
result is uninterpretable whatever its p-value, and we would know that from mechanism rather than
from argument.

---

## S-2 · Does flattened intent show up as posterior concentration? (C-22)

**The claim**, from the curator, corrected by him from an earlier stronger version:

> It is unfair to say corporate goals are singular... the **share** of the goal is
> disproportionately large. You'd expect that to be a larger piece, **not the whole pie**.

**The test.** Build two creator classes:

- **layered** — several terminal values, comparable weight
- **flattened** — the same *number* of instrumental decisions, but one terminal value carrying most
  of the weight

Hold **decision density constant** across both. That is the whole point and the thing real corpora
cannot do: the curator's position is that corporate work is *not* thinner, only more concentrated.

**Measure:** normalised entropy of the reader's goal posterior (`purpose_breadth` in Sounding Line).

- Does it separate the two classes?
- Does **goal-recovery accuracy** stay equal? It should — a flattened maker is *easier* to read,
  not harder.
- Does **process recovery** stay equal? If flattened makers show lower process recovery at matched
  density, "flattened" is quietly "shallower" and the construct is confounded.

**Why here.** In real text, "same density, different concentration" cannot be constructed. In the
sim it is two lines of config.

---

## S-3 · Do two affect layers produce a measurable divergence?

**The claim.** `family_v3.yaml`: emission has an **involuntary** component (Panksepp primary, leaks)
and a **chosen** component (Barrett tertiary, displayed). The curator's correction to my first
version, which is the part to build:

> Leaked greater than emblematic doesn't even count as concealment. If anything the emblematic
> would get **larger**. You perform louder to cover up. **The shield matches the leak.**

**The test.** Give the creator two emission channels over one alphabet:

- `leak` — a function of the creator's true state, **not** under its policy's control
- `display` — chosen by the policy, and under a *concealment* condition, chosen **to oppose** the
  leak and amplified

Then ask:

- Can a reader recover the true state from `leak` alone? (If not, the layer is unreadable in
  principle and Sounding Line's leaked layer is a dead end.)
- Does **divergence** between the two channels identify the concealment condition?
- **Does amplified opposition make concealment MORE detectable rather than less?** That is the
  curator's shield claim, and it is counter-intuitive enough to be worth a mechanism.

---

## S-4 · Does the loop run backwards?

**The claim**, discovered by the curator while reading aloud:

> If you figure out part of either piece — the values or the method — you can immediately use that
> to jump into the goal, and then use that to figure out the rest of the process.

E36 established purpose → method within a reading. It never established that method → purpose
cannot happen. Sounding Line's loop is unidirectional and purpose-first; the curator ran it in
reverse three times out of three.

**The test.** Two readers on identical rollouts:

- **forward** — settle the goal, then read modes under it (current behaviour)
- **reverse** — condition on the observed mode sequence first, derive the goal from it, then
  re-read modes under the settled goal

Compare final goal accuracy, process recovery, and steps-to-settle. **Does the reverse path ever
win?** Under what conditions — low goal legibility (`beta`), high depth (`mu`), both?

If it never wins, E36 was the whole story and Sounding Line's loop is correct as built.

---

## S-5 · Is an anomaly a better entry point than the whole artifact?

**The claim.** The curator entered every successful reading through a specific oddity — an absence
of jargon, an admitted fib, a self-serving ordering — never through the artifact as a whole.

**The test.** Two conditioning orders on the same observation sequence:

- **uniform** — condition on observations in order (current)
- **anomaly-first** — condition on the **least likely observation under the reader's prior** first,
  then the rest in order

Bayes says final posteriors are identical if all observations are used. **So the test is about
COST**, which is exactly what this simulation is built to measure: does anomaly-first reach the
same posterior in **fewer DEEP steps**, under a reader that pays per look and can disengage?

If yes, the anomaly stage is not a better inference — it is a **cheaper** one, and under a
metabolic budget cheaper *is* better. That is a sharper claim than the one Sounding Line is
currently making, and it is testable only where attention has a price.

---

## S-6 · Does surface thickness decay while depth does not?

**The claim**, `docs/theory/SURFACE_AND_DEPTH.md`, derived from automaticity: content decisions are
practised and cached, so they cost little and do not decay across one artifact; surface decisions
are a performance held consciously, so under a metabolic budget they degrade.

**The test.** Give the creator two decision streams and a budget that depletes:

- **content** — drawn from automatised structure, cheap, non-depleting
- **surface** — costs the creator, and the cost is not cached

**Predictions:**
- surface density declines across the artifact; content density does not (**S-1** in that doc)
- a creator with **no** automatised structure — a novice — shows surface decay *and* content decay
- a synthetic creator with no budget at all shows **flat surface**, which is the sharper prediction
  (**S-2** there): the machine signature is not thin depth but *a surface that does not move*

---

## What to send back

For each: the verdict, the numbers, and — in this project's usual style — **what would have
falsified it**. S-1's N28 check is the one to prioritise if only one gets run.

Sounding Line will treat any of these as evidence about **mechanism only**. None of them says
anything about real artifacts, and none should be quoted as if it did.
