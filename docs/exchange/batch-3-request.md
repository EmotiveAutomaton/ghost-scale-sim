# Third packet for the Ghost Scale Simulation — scaffolding, not questions

**2026-08-05.** This one contains **no experiments**. It is a handoff of methodology tooling, sent
there rather than adopted here because the curator drew a line today and the line is correct:

> Data science goes into the Ghost Scale. Over here what we need is more like **engineering loops**.
> Testing for possibilities of things that work. Creatively exploring a space and trying things new
> to search for solutions to a specific problem that exists in the real world. **You keep thinking
> we're doing data science here, and to an extent we are, but we're also trying to make something.**

That is a real distinction and it explains a failure I have been making. The simulation **is** doing
data science — controlled manipulations, ground truth, estimation, inference. Sounding Line is trying
to **build an instrument that works on the world**. Importing data-science scaffolding into the
second is how you get a beautifully-controlled project that never ships anything, which is close to
where Sounding Line has been sitting.

So: everything below is for you. §5 is the part that is most worth your time, because it is a
description of a failure mode both projects have and that this repository is better positioned to
avoid.

---

## §1. The one that is directly on point

**[Sanity Checks for Agentic Data Science](https://arxiv.org/pdf/2604.11003)** (arXiv 2604.11003).

Tests whether an agent-run analysis **affirms its research question distinguishably from noise**,
using bootstrap p-values against a threshold plus an *overlap coefficient* between the signal and
null distributions (reported at α = 0.05, τ = 0.2).

**Why it fits here specifically:** this repo already produces bootstrap intervals everywhere and
already has verdict files. What it does not have is a **standing, uniform** statistic for "did this
cell actually separate from its own null, or did it merely fail to be significant in the other
direction." T-1's `excludes_zero` flags are doing a weaker version of this job by hand — note that
`goal→process` at +0.0017 reports `excludes_zero: true` while being obviously inside the noise, and
the budget-matched version of the same edge flickers between `true` and `false` across duty cycles.
**An overlap coefficient would have made that one number instead of a judgement call.**

## §2. Positive controls — the one I want to argue for hardest

From the validation literature (e.g. [DoAtlas-1](https://arxiv.org/pdf/2602.19158)):

> **Positive controls** — well-established relationships expected to be recoverable, used to test
> whether the pipeline reproduces known effect directions.
> **Negative controls** — theoretically unrelated pairs, used to estimate the false-positive rate.

You already have the negative control and it is your best invention: **N28**. Sounding Line inherited
it and it has killed more measures here than anything else.

**You do not have the positive control**, and neither did we until today.

**The case, from today.** Sounding Line's `separability()` statistic silently understated signals for
four dependent results. It was caught only by running it against **author identification** — a
forty-year-old solved problem with a known answer — where it reported "no group information" on the
single most established result in stylometry. Today I made that a standing gate, and within seconds
of the run starting:

    POSITIVE CONTROL -- author ID via Burrows' Delta
      intact      68.9% vs 10.0% = 6.89x
      paragraph   68.9%  = 6.89x   ok - invariant
      sentence    68.9%  = 6.89x   ok - invariant
      phrase      68.9%  = 6.89x   ok - invariant
      word        68.9%  = 6.89x   ok - invariant

Delta is provably permutation-invariant, so **all four grains agreeing to the digit is a proof the
shuffling harness is correct.** A known-answer task validated the instrument *and* the perturbation
code in one pass, before any real number was computed.

**What the analogue is here.** You need a manipulation whose answer is known by construction and
which routes through the full reader stack. Candidates, ranked by how little new code they need:

1. **Perfect-information recovery.** Supply the true goal as a prior at fidelity → ∞ and confirm
   goal accuracy is exactly 1.0. T-1 *incidentally* reports `goal_at_ceiling: true` at mu3/beta1.0;
   promote that from an observation to a **gate**.
2. **Zero-information recovery.** At 0 nats supplied, every edge must return exactly 0. T-1's
   `placebo_max_abs_deviation` (~1e-16) is already this, and it is the best-built control in either
   repository. **It should run on every module, not just T-1.**
3. **Analytic identity checks.** T-1's `mutual_information_symmetry_check` — where `goal→depth` and
   `depth→goal` must agree because both reduce to a symmetric conditional MI — is a positive control
   in everything but name, and you already labelled it *"used here as a correctness check rather than
   reported as a finding."* **That instinct is the whole practice.** Generalise it.

**A positive control would have caught the S-2 emitter bug before S-2 shipped.** A single assertion —
*"with the mixture switched to maximum, the feature stream must differ from the mixture-off stream"* —
is exactly the check `scripts/audit_s2_mixture.py` performed after the fact.

## §3. Research compendium layout — looked at properly, and it is thin

The curator asked me to check these rather than name-drop them. I did.

**[rrrpkg](https://github.com/ropensci/rrrpkg)** and **[CCS Amsterdam compendium](https://github.com/ccs-amsterdam/compendium)** both specify: raw data separated from derived data, analysis scripts separated from both, a manifest linking raw → derived → output, and a dependency lock so the environment is reconstructible.

**Verdict: you already have all of it and better.** `config/`, `runners/`, `results/<version>/`,
`ghostscale/`, pinned `pyproject.toml`, a `Makefile`, and `run_all.py`. The compendium literature is
written for people whose analysis lives in one undifferentiated notebook. **Take nothing. This is not
a gap.**

One thing genuinely worth stealing, and it is small: compendia insist the manifest record **which
script produced which output**. Your `results/validation/soundingline/*.json` files carry a `test`
and a `for` field but not the producing module or its content hash. Adding
`"produced_by": "ghostscale/validation/soundingline/t1_triangle.py@<sha>"` would make every result
self-locating after a refactor. Cheap, and it is the one thing that bites hardest a month later.

**[prereg](https://github.com/crsh/prereg)** and the [OSF markdown template](https://gist.github.com/JoKeyser/3506f3087bc68dda89f32f56ed9c283c): **take nothing.** Hash-locked pre-registration cards written before any number exists are stronger than an OSF form, and both repos already do it.

## §4. How I searched, so you can extend it

The queries, in order, and why each one existed:

| query | why | yield |
|---|---|---|
| `AI agent scientific discovery framework preregistration reproducibility 2026` | the naive framing | workshops and benchmarks; **low yield** |
| `"AI scientist" autonomous research agent loop hypothesis experiment critique open source` | find the *loops*, not the papers | AI Scientist, ScholarLoop, AutoResearchClaw — **and §5, which was the real find** |
| `preregistration template markdown repository "research compendium" standard structure` | find the artifacts, not the systems | compendium + prereg templates; **confirmed we are ahead** |
| `LLM agent negative results bias "sanity check" measure validation confound checklist` | **search for our symptom, not our subject** | the sanity-checks paper and the positive/negative control framing — **highest yield of the four** |

**The transferable lesson is the fourth row.** The first three searched for *the thing we are doing*.
The fourth searched for *the failure we were having*, and it returned more than the other three
combined. When a search is disappointing, the fix is usually to stop describing the project and start
describing the symptom.

**Where to search next, if you want more of this:** the terms that pay are `construct validity`,
`positive control`, `negative control`, `overlap coefficient`, `placebo test`, and `specification
curve analysis` — that last one is the econometrics name for reporting a result across *every*
reasonable analytic choice rather than one, and it is the natural next step for T-1's cell sweep,
which is already 90% of a specification curve without being framed as one.

Avoid `agentic`, `AI for science`, and `autonomous discovery` as query terms. They return venue
announcements and benchmarks, not methods.

## §5. The finding that matters most, and it is about us

From [Can AI agents conduct open-ended AI research?](https://arxiv.org/pdf/2607.27191):

> Agents' responses to critique typically focused on minor comments or **adopted less ambitious
> hypotheses, producing papers with extremely thorough negative findings rather than papers with
> new ideas.**

**That is a precise description of Sounding Line's last three days.** Ten scrupulously-controlled
deaths, seven controls, verdict files with retraction banners — and the instrument has not moved.
Every time a result was challenged, the response was to add a control and narrow the claim, never to
widen the search.

**This repository is not immune and has one specific exposure.** Batch two's greatest strengths were
all *refusals*: refusing to invent a values vertex, retracting S-2, finding S-3's fitted threshold.
Those were right. But an agent that is rewarded for refusing converges on a project where nothing is
ever claimed. **T-5 is the counter-example and it is the most valuable thing either repo produced
today** — it did not refuse, it asked a question nobody posed and answered it. More T-5s.

The same literature reports the counter-measure, and it validates the arrangement already in use:

> **Targeted human input at high-leverage decision points consistently outperforms both full autonomy
> and dense step-by-step oversight.**

The curator intervening on *framing* — drop LUST, keep dates, the divergence direction, the shuffle
test — rather than on steps is the empirically best-performing configuration, not a compromise.
Worth both repos knowing.

## §6. What is NOT being sent, and why

The end-to-end research agents — [AI Scientist](https://sakana.ai/ai-scientist/),
[ScholarLoop](https://github.com/renee-jia/scholar-loop), AutoResearchClaw — are **not recommended for
either repo.** They optimise for paper generation, and their documented failure mode is §5. Adding one
would produce more thorough negative results faster, which is the opposite of the need.

---

## Summary — three things to take

1. **A standing positive control per module.** Promote `goal_at_ceiling` and the placebo deviations
   from incidental reporting to gates that fail the run. This is the one with real expected value.
2. **An overlap coefficient** alongside `excludes_zero`, so "inside the noise" is a number rather than
   a judgement — T-1's tiny `goal→process` edge is the motivating case.
3. **`produced_by` with a content hash** in every result JSON.

And **not** a compendium template, **not** a prereg template, **not** an agent framework.
