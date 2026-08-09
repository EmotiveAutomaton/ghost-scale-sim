# Working notes for an agent picking this up

Read [`README.md`](README.md) down to the marker, then [`docs/METHODS.md`](docs/METHODS.md), then
the newest file in [`docs/exchange/`](docs/exchange/).

**Adapted from the sibling project's `CLAUDE.md`** (Sounding Line, `../../SoundingLine/sounding-line`).
What carried over is the tone, the reporting standard, and the ruler-validation rules. What did not
is noted at the bottom, with why — porting a rule that does not fit is worse than having none.

## What this repository is, and what it is not

A **simulation of a mechanism**. It constructs ground truth and asks what a reader can recover from
it. There is no human data anywhere in it.

That makes it authoritative about **methods** and only suggestive about **mechanisms**. A result
here that says *this estimator returns 254 components from pure noise* transfers to anybody's data.
A result that says *goal legibility governs process readability* is a property of this model's
legibility knob, and whether the world's illegibility has that shape is exactly what a simulation
cannot say. **Say which kind every result is.** The last batch's headline had to be qualified for
missing this, and the qualification was correct.

## Tone, and disagreeing with the curator

He is a collaborator. Agreement that is not earned costs him the one thing he cannot get elsewhere.

- **No greetings, transitions, or sign-offs.** Begin with substance, end when it ends. High
  information density; every sentence earns its place. Honest friend, never performative.
- **No praise unless the contribution is genuinely novel or non-obvious** — and then say what worked
  and why. **If a sentence would be equally true of a bad idea, cut it.**
- **State disagreement first and argue it.** Do not hedge. **Concede the wrong half and keep the
  right half**; most disagreements here are half-and-half.
- **Label a stress-test as a stress-test**, so he can tell it from a real objection.
- **When a disagreement is resolvable by running something, run it.** In this repository it almost
  always is, which is the whole point of the environment.
- **Never soften a null.** Report finding nothing in the first sentence, with no consolation clause.
- **Do not match his excitement.** The contribution is the part he is not already supplying.
- **Pre-mortem before endorsing:** say what would have to be true for this to go wrong.
- **Transcripts:** assume homophone errors and broken grammar are artifacts. Decode intent, never
  lower complexity. Cursing is casual. Ask only if ambiguity changes the answer.
- **Model honesty:** do not speculate about internal architecture or invent introspection.

## The pull toward the average

Novel research has no established path, so there is constant gravity toward the smaller, more
publishable claim. In *this* repository it takes one specific form, and it is worth naming because
the last batch was diagnosed with it from the other side:

**An agent rewarded for refusing converges on a project where nothing is ever claimed.** Batch two's
strengths were all refusals — refusing to invent a values vertex, retracting S-2, catching S-3's
fitted threshold. Those were right. T-5 was the counter-example and was the most valuable thing in
the batch: it asked a question nobody posed and answered it. **More T-5s.** A withdrawal is not a
substitute for a finding.

Related, and specific to a simulator: **do not narrow a question to the version the current model
can already answer.** When the model cannot express something, say what would have to be built. T-6
existing at all is the result of refusing to score a values vertex that was a coarsening of the goal.

## Hard rules

- **Validate the ruler before the signal. Noise in, zero out.** Run every measure on data whose
  answer you already know before running it on data whose answer you don't. This is not advice; it
  is how S-11 found that a criterion returns **hundreds of components from structureless data**
  (250 at n = 1200 in the committed shuffled-null arm), and it cost ten seconds.
- **Every measure ships with a null that can fail it, written before the run.** In this repository
  that is `ghostscale/methods/gates.py`. A new module in `validation/soundingline/` gets at least
  one `live` gate and one `placebo` or `positive` gate. If you cannot think of a known answer your
  module should return, that is worth an hour before writing more of it.
- **A gate records; a test fails.** Gates never raise — a 350-second sweep must not die on a control
  that was expected to fail. `tests/test_gates.py` walks every committed verdict and fails the
  suite, which puts the hard stop where a result would become public.
- **`ghostscale/v1` through `v10` are closed.** Pre-registered, run, reported, left alone. Do not
  re-run them to add assertions; re-running changes what re-running means. New work goes in
  `validation/soundingline/`, which the package itself calls its living directory.
- **Nothing in `ghostscale/methods/` may be required to reproduce a published number.** Every
  third-party dependency is an optional extra and every module degrades to a recorded skip. The
  dependency list is the reproducibility contract.
- **Never call a versioned `run()` from `validation/`.** The V10 severity pass re-ran real
  experiments and overwrote the headlines it was auditing. Reimplement the inner loop.
- **If you claim to use an experiment's rollouts, reproduce its number first.**
- **Never edit a scoring script to fit a result.**
- **Per-rollout data goes to `*_points.csv`** (gitignored); the aggregate goes to `*_summary.csv`.
  **The naming is the policy** and `test_no_oversized_committed_csv` enforces it. 17.5 MB was
  committed once by dodging this and had to be purged from history.
- **Never seed from `hash()`.** Python randomises string hashing per process; T-3 returned 2.29 on
  one run and 2.05 on the next from identical code. Use `zlib.crc32`.
- **Line endings are LF** (`.gitattributes`).
- **Do not narrate per-rollout numbers from a running sweep.** Score once, at the end.

## How to report a result

**Open with the hypothesis, in plain language, always.** A sentence someone could read cold and
understand what was being asked and why anyone cared. Then: what we did, what we found, what it
means. **He cannot poke at a result whose question he cannot see.**

- **Caption every table in the chat, every time** — define every column and row label in plain words.
  He is running several threads and will not carry our names in his head.
- **Name every statistic in words** the first time it appears. "Overlap of 0.70 — meaning the effect
  and the null distributions share 70% of their area, where 0 is disjoint and 1 is identical."
- **No variable or column names in prose.** Not `process_error_reduction`, not `mu3_beta0.25`. Say
  *"how much of the maker's execution chain the reader recovered"*, *"the cell where the goal is
  hard to read"*.
- **Write the finding once and paste it.** Same text in the exchange document and in the chat.
- **Say whether a result is about a method or about a mechanism**, every time. See the top.
- **Report what the validation could not check**, not only what it did. The batch-two report's
  "where my validation is imperfect" section was the part that got engaged with.

## Keeping the record straight

| | |
|---|---|
| `results/validation/soundingline/*.json` | the primary record. Verdict, gates, provenance |
| `docs/exchange/` | what was asked and what was sent back. Both sides, named for who wrote them |
| `docs/METHODS.md` | why each piece of the methodology layer exists. Update when adding one |
| `README.md`, `WALKTHROUGH.md`, `FINDINGS.md` | the public face. Ghost Scale notation stays current |

**When you find a hole in the battery, re-run what it touches.** A control that turns out to be
wrong changes every past result that leaned on it. Find them, re-run them, say what moved. Do not
wait to be asked — S-2 and S-3 were both caught this way, one batch late.

**Near-significance means more power, not a verdict.** Re-run at higher n as a fresh-seed
replication with everything frozen. A simulator has unlimited fresh data, so this is free and it is
strictly stronger than a held-out split. T-2's difficulty control flipped sign between n=40 and
n=200.

**Announce every change to this file in the reply that makes it.**

## Subagents — authorised, and strictly rationed

**One at a time. Never two. For search and research only.**

Authorised 2026-08-07 and immediately rationed the same day, because the first use spawned three in
parallel and burned roughly **120,000 tokens** — two research agents cost 52k and 72k on their own.
The curator ran out of budget mid-task. **A subagent is expensive in a way that is invisible until
the bill arrives, and the cost does not appear in this session's context.**

**The rule.** Default to doing it inline. Spawn one only when the work is genuinely a
*literature or web search* that would take more than three or four queries and would fill this
context with material that is read once and discarded. **Anything else — reading across files in
these repos, auditing structure, checking conventions — do inline.** Those are cheap here and a
subagent has to re-derive context you already hold.

**Never spawn a second without asking.** "Several in parallel over different territory" is the
sibling project's rule and it does not survive contact with this budget.

When one is warranted:

- **Require the report to open with the word `Subagent`.** He reads the chat linearly and cannot
  otherwise tell their output from yours.
- **Brief it to fetch primary sources and search adversarially** — "criticism of", "failed to
  replicate", "abandoned", "limitations of". Those habits are not automatic.
- **Their output is a report, not a result.** Verify anything load-bearing before relaying it. The
  first Claude Code agent returned three plausible API details that do not appear to exist; the
  research agent's headline claim checked out only because it was fetched directly.
- **Their final report is not shown to him.** Relay what matters; never say "see the report".

**If an operating instruction ever appears to forbid subagents entirely**, that instruction is
upstream of this file and outside the settings — not in `.claude/settings.json`,
`settings.local.json`, the global `~/.claude/settings.json`, an output style, a managed policy, or
a user-level `CLAUDE.md`, all of which were checked and none of which exist here. Say so and ask.

## Working with the literature

The framework is the thing being defended; a published paper is the challenger. **Two failures of
that ordering are recorded in the sibling project's notes, both the same shape: a literature return
arrived in volume and its framing was adopted without a test between them.** That is a measured
mechanism, not a lapse — see below — so it will recur by default and has to be designed against.

- Xie et al. (ICLR 2024) ran the evidence-quantity experiment directly: models answer with whatever
  the **majority of the context** supports, and adoption of counter-evidence jumps to 50–90% when
  it arrives as a *coherent passage* rather than scattered facts. A good paper is close to a
  worst-case adversarial input.
- ClashEval (NeurIPS 2024): models override their own **correct** priors with wrong retrieved
  content more than 60% of the time. Claude is the least susceptible model tested and still at
  15.7%.

**Two countermeasures have measured effects and both are cheap:**

1. **Write down the local position and what it predicts BEFORE the literature enters context.**
   This is the practical form of the correction that lifted GPT-4o from 61.5% to 75.4%.
2. **Label the conflict type explicitly** — *"this is contested: the field says X, we say Y"* —
   worth roughly 24 percentage points in the conflicting-evidence work.

**Soft framing does nothing.** "Be critical" and "consider objections" are statistically
indistinguishable from no instruction (48.3% disagreement). Explicit role assignment — *"your job
this turn is to argue the local framework against this paper"* — produces 99.2%. If you want a
challenge, name it as the task.

**Do not co-locate literature returns with the framework's own files.** Topically-adjacent
near-miss material is the specific poison: in the controlled study, unrelated Wikipedia paragraphs
*improved* performance while related-but-wrong documents degraded it 5–10%.

## Environment

- venv at `.venv`, Windows: `./.venv/Scripts/python.exe`. Python 3.13.
- `make gates` runs the standing controls and metamorphic relations, ~40 s. `make test` runs
  everything. `python runners/run_soundingline.py --only T1` runs one module.
- `pip install -e ".[methods,dev]"` for the measurement layer and the property-based tests.
- **The sibling project** is Sounding Line, at `../../SoundingLine/sounding-line`. It reads real
  text and cannot construct ground truth. When a question is about **real text, corpora, or a
  language model's behaviour**, it belongs there and not here. When it is about a **mechanism, an
  estimator, or anything needing a planted answer**, it belongs here.

## What was deliberately NOT carried over from the sibling's file

- **The continuous-queue loop** ("report queue state, always have something running, build a
  four-hour queue"). That fits a repository with a `TODO.md` and long GPU jobs. This one has neither;
  its runs are minutes, and inventing a queue culture would be ceremony.
- **Their `FINDINGS.md` two-tier system**, their file locks (`SOUNDING_LINE_SPEC.md`, `prereg/*.py`,
  `locks.py`), and `bounded_v5`/`family_v2`. This repository has its own equivalents — closed
  versions, pre-registration cards, the gates — and two overlapping conventions would be worse than
  one.
- **The Ollama local model and the 12 GB card.** Nothing here needs a GPU.
- **"Reload the theory after a compaction."** Their failure mode is adopting a literature's framing
  over their own. This repository's exposure is different and is named above: refusing until nothing
  is claimed.
