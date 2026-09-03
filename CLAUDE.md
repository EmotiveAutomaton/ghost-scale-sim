# Working notes for an agent picking this up

Read [`README.md`](README.md) down to "The complete record", then
[`docs/theory/READING_INTENT.md`](docs/theory/READING_INTENT.md) — the hypothesis store, where
every claim stands — then [`docs/METHODS.md`](docs/METHODS.md), then the newest file in
[`docs/exchange/`](docs/exchange/).

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
- **`ghostscale/v1` through `v10` are closed.** Pre-specified and hash-locked, run, reported, left alone. Do not
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
- **A purpose-built miniature ships its own random-draw severity check, or its verdict carries
  "miniature — architecture untested".** The severity passes cover the shared model only; a
  miniature (S-3, S-6, S-12) is a new architecture whose false-positive rate nobody has measured
  unless it measures its own. S-12 is the template: twenty redraws of its generative constants,
  reproduction rates in the verdict.
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
| `docs/theory/READING_INTENT.md` | **the hypothesis store — the dense channel.** Every claim under its umbrella hypothesis, with status. **A result lands here in the same pass that lands it in `FINDINGS.md`**, and the paragraph under a changed table is revisited in the same edit. Format: `docs/theory/README.md` |
| `docs/exchange/` | what was asked and what was sent back. Both sides, named for who wrote them |
| `docs/METHODS.md` | why each piece of the methodology layer exists. Update when adding one |
| `README.md`, `WALKTHROUGH.md`, `FINDINGS.md` | the public face and the method archive — the wide channel. Ghost Scale notation stays current |
| `docs/archive/` | superseded documents. Nothing deleted, only moved, each with a supersession note |

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

- venv at `.venv`, Windows: `./.venv/Scripts/python.exe`. Python 3.13. **System `python` is not
  this interpreter and lacks the dependencies.**
- `make gates` runs the standing controls and metamorphic relations, ~40 s. `make test` runs
  everything. `python runners/run_soundingline.py --only T1` runs one module.
- `pip install -e ".[methods,dev]"` for the measurement layer and the property-based tests.

### V15 is the live program. It runs for 168 hours. Check before you touch anything.

V13 and V14 are **closed**. V15 — The Boundary Map — is a 112-card, 24-attack program on one
continuous seven-day window, driven by `python -m runners.run_v15 --stage all` under
`runners/watchdog_v15.py`.

**Launch is always module form.** `python -m runners.run_v15`, never `python runners/run_v15.py`.
The sibling project's `tools/orphan_sweep.ps1` runs at every gear-script startup and taskkills any
python whose command line matches `runners[\/]run_`. That killed V14's runner seven times, cost
that program its first 24-hour window, and retro-explains four of V13's "unexplained" silent
deaths. Attack X24 reads `runners/run_v15_wrapped.ps1` and checks the module form is still there.

**Read state from the records, never from prose — this file included.**

| | |
|---|---|
| `results/v15/RUNNER_STATUS.json` | stage, card, pid, heartbeat. **Check the pid is alive and the heartbeat is fresh** |
| `results/v15/DEADLINE.json` | the one immutable UTC deadline. Restarts inherit it; nothing shortens it |
| `results/v15/WORKER_OCCUPANCY.json` | the §9.4 receipt, including `RUNTIME_FAILED` and its reasons |
| `results/v15/COVERAGE.json` | cards resolved by state and trunk |
| `results/v15/COMPLETION.json` | verdict path, hash and receipt per card |
| `results/v15/QUEUE_MANIFEST.json` | per-card status, criterion status, lanes |
| `results/v15/coverage/blocks.jsonl` | the executed prefix of the balanced coverage stream (gitignored) |

- **A relaunch resumes; it never re-prepares.** `--stage all` with an open window takes
  `stage_resume`: straight to science, skipping prepare/smoke/pilot/open. Two reasons, both paid
  for on day one (44 refuse-and-exit relaunches, 3.7 window-hours lost): `stage_prepare`
  regenerates the lock-input files, every generated file embeds a `written` timestamp, so
  regeneration always breaks the scientific lock; and the smoke pass is not hermetic once real
  discovery verdicts exist — P07 walks the committed record even under `GS_V15_SMOKE` and blocks
  on a partial record. If the lock is ever broken by a stray prepare: the lock inputs' HEAD blobs
  match the scientific lock's recorded hashes, so `git checkout --` of the six hashed files
  restores it byte-exact.
- **The runner heartbeats through long cards.** One T3 card can run for an hour with no new
  checkpoint, verdict, or coverage line, and the watchdog killed a healthy runner 48 minutes into
  C02 for exactly that. The runner now appends `kind=heartbeat` checkpoint lines every 5 minutes
  and the watchdog (on disk; a restarted one) also counts process-tree CPU and the RUNNER_STATUS
  heartbeat as progress. Do not "clean up" the heartbeat lines; they are what keeps a live
  watchdog's stall counter moving.
- **Instrument repairs re-run through `runners/run_v15_amendment.py`**, never by hand. It uses
  the science stage's own `run_card`, tier, lane and seeds, preserves the original verdict under
  `<lane dir>/amended/`, and records the swap in `results/v15/AMENDMENTS.json`. It can express a
  gate or receipt repair; it cannot express a criterion, estimator or factor change (lock
  amendment, curator). First used 2026-09-02 for X23, X24 (receipt miscounted units) and H03
  (placebo gate carried the criterion statistic).
- **Never run `python -m runners.run_v15 --stage smoke` (or any stage) beside a live runner.**
  `main()` writes `RUNNER_STATUS.json` with its own pid; the watchdog reads that pid, sees it
  exit, and launches a *second* runner over the live one. Scratch smokes go through `run_card`
  from a file-based script (Windows `spawn` cannot re-import a stdin script as `__main__`; the
  pool respawns workers forever).
- **No packet before the deadline.** Spec §9.1 forbids result prose, HTML, Markdown summaries,
  bridge packets and curator-facing charts until hour 168. `runners/report_v15.py` refuses;
  `--draft` writes to a scratch directory only. A checkpoint, a dashboard or a bridge file is
  exactly how an early packet gets created without anyone deciding to create one.
- **`LANDED` is not a held criterion.** Every verdict carries `state` and `criterion_status` as
  separate columns, and the reporting code will not print one without the other.
- **A gate bar is never a criterion bar.** `cards.battery` has no parameter through which a
  magnitude could reach a gate, and `tests/test_v15_gates.py` fails the suite if one does. V14 lost
  three small *real* effects to that conflation and had to repair it mid-window.
- **Hash-locked generators.** The nineteen files in `prereg_v15.GENERATOR_FILES` are hashed; any
  byte change breaks the lock and `--stage science` refuses to run. `manifest.py`, `runtime.py`,
  `atomicio.py`, `runners/` and `tests/` are outside the lock.
- **A run may be live right now.** Workers are Windows `spawn` processes and re-import from disk,
  so an edit lands in workers created *after* it while the parent keeps what it imported. Work in a
  worktree and deploy at a boundary.
- **`RUNTIME_FAILED` has no softened form.** If the queue empties or workers wait for the deadline,
  the flag is set and stays set. Spec §9.4 permits reporting the results and does not permit
  claiming the seven-day contract; there is deliberately no "short run but complete".

### V13 is a long-running queue. Check before you touch anything.

*(V13 is closed. This section is kept because its rules about spawn workers, hash locks and confirmation amendments still describe how the machinery works.)*

The per-module runs above are minutes. **V13 is not**: 152 cards on a tier-calibrated,
checkpointed, multi-day queue, driven by `runners/run_v13.py --stage all` under
`runners/watchdog_v13.py`, which relaunches a dead run up to six times, gated on ledger growth.

**Read state from the records, never from prose — this file included.** Nothing below is a
count; the counts live in files that are rewritten as the program runs:

| | |
|---|---|
| `results/v13/RUNNER_STATUS.json` | current stage, card, pid, heartbeat. **Check the pid is alive and the heartbeat is fresh**; its embedded `coverage` block is a snapshot and goes stale |
| `results/v13/COVERAGE.json` | cards resolved by state and trunk, with `written` |
| `results/v13/COMPLETION.json` | the completion ledger: verdict path, hash, receipt per card |
| `results/v13/QUEUE_MANIFEST.json` | per-card status, tier, lineages |
| `docs/versions/v13-common-ground/HEALING_PLAN.md` | known repairs and pending curator decisions |

- **A run may be live right now.** Before editing any imported module, check for a live runner.
  Workers are Windows `spawn` processes: they re-import from disk, so an edit lands in workers
  created *after* it while the parent keeps the code it already imported. That mixes two versions
  inside one card's results. Work in a worktree and deploy at a boundary.
- **Do not** reset the ledger, delete checkpoints, or clear a lock to make an edit easier.
- **Some files are hash-locked and editing them halts the program.** `ghostscale/prereg_v13.py`
  hashes *itself*; `common.py`, `schemas.py`, `world.py`, `priors.py`, `exact.py`, `attention.py`,
  `costs.py`, `goals_trust.py`, `hierarchy.py`, `projection.py` and `pymdp_reader.py` under
  `validation/soundingline/v13/` are hashed as generators. Any byte change flips
  `lock_status()["locked"]` false and `--stage discovery` exits refusing to run. Changing one is a
  lock amendment — a curator decision — not a routine fix. `manifest.py`, `runtime.py`,
  `atomicio.py`, `runners/` and `tests/` are outside the lock.
- **Inspect any entry point named smoke or validation before invoking it**; several execute real
  experiments. `GS_V13_SMOKE=1` is refused by the scientific stages on purpose.
- **Reviewed is not confirmed.** A discovery verdict whose criteria passed is a *candidate*.
  Confirmation re-runs a frozen packet on an untouched lineage.
- **Widening the confirmation packet is an amendment, and it is recorded, never silent.**
  `runners/run_v13_confirmation.py` freezes the promoted set before the first confirmation world
  and verifies it — discovery hashes, card and criterion identity, both lock hashes — on every
  later entry. Adding a card (which is what `HEALING_PLAN.md`'s confirmation step does, and the
  curator kept that plan on 2026-08-28) writes an amendment preserving the original packet beside
  the replacement, in `results/v13/CONFIRMATION.json` and `results/v13/AMENDMENTS.json`. Added
  cards get an untouched lineage for free: `rng_for` seeds on the card id. `--no-amend` refuses to
  widen instead. **Never widen the packet by editing the ledger by hand.**
- **The sibling project** is Sounding Line, at `../../SoundingLine/sounding-line`. It reads real
  text and cannot construct ground truth. When a question is about **real text, corpora, or a
  language model's behaviour**, it belongs there and not here. When it is about a **mechanism, an
  estimator, or anything needing a planted answer**, it belongs here.

## What was deliberately NOT carried over from the sibling's file

- **The continuous-queue loop** ("report queue state, always have something running, build a
  four-hour queue"). Their version is a *culture* — keep the GPU fed, always have something
  running. That is still not carried over. **What is no longer true is the reason originally given
  here** ("this one has no queue; its runs are minutes"): V13 is a manifest-driven, multi-day card
  program with its own ledger and watchdog, described under Environment. The distinction that
  survives is that V13's queue is a *pre-registered program that ends*, not a loop to keep full —
  the correct response to an idle machine here is to report state, not to invent work.
- **Their `FINDINGS.md` two-tier system**, their file locks (`SOUNDING_LINE_SPEC.md`, `prereg/*.py`,
  `locks.py`), and `bounded_v5`/`family_v2`. This repository has its own equivalents — closed
  versions, pre-registration cards, the gates — and two overlapping conventions would be worse than
  one.
- **The Ollama local model and the 12 GB card.** Nothing here needs a GPU.
- **"Reload the theory after a compaction."** Their failure mode is adopting a literature's framing
  over their own. This repository's exposure is different and is named above: refusing until nothing
  is claimed.
