# Ghost Scale V23: commissioning spec for the round after V21

**For the curator, the Ghost planning thread and the coding agent.**
- **Commissioned** by the curator on 14 October 2026 at 20:00 PDT ("we should give V23 for sure").
- **No edits to V21.**
- **Start:** V23 starts when V21's final report is posted.

**As of 14 October 2026, 20:15 PDT.**
- **Pin read:** Ghost `main` at `32ef58e4`, the V22 closure commit (13 October, 20:18 PDT), sparse checkout. V21's documents are outside it, so V21's dates come from `AGENTS.md`.
- **V22: closed** on 13 October. Seven finite cards, all independently checked and fully replayed; 1.33 charged CPU-hours.
- **V21: running.** Scheduled dates:
  - discovery freeze, 15 October 00:00 PDT;
  - confirmation opening, 06:00;
  - science cutoff, 16 October 10:00;
  - final report, 16 October 18:00.
- **V21's final report** had not posted as of 20:00 PDT. The curator has asked V21 to finish early. Day 0 of V23 is whenever that report posts.
- **Sounding Line** keeps host priority; its joint final packet is due 16 October 18:00 PDT.

**Relation to V21 and V22.** This round, **V23**, starts after V21's final report and shares nothing with V21 before then. It repairs four V22 controls (Part R), adds eight mechanism families (Part N), builds four synthetic twins of real testbeds (Part T), and makes every Ghost world emit the engine's decision ledger (Part B). V22's closed code is untouched; repairs reimplement its inner loops in a new `v23` package.

**Companion specs (same date).**
- `TOMPATHY_ENGINE_SKELETON_SPEC_2026-10-14.md`: the ToMpathy 0.3 engine, in `tompathy/engine/`. It owns the canonical ledger schema at `engine/tompathy_engine/ledger/schema.py`, and Ghost pins it by ToMpathy commit.
- `SOUNDING_LINE_NEXT_ROUND_SPEC_2026-10-14.md` holds the real-data families whose stopping rule these twins calibrate (its Part 0.2) and whose spike-in controls they supply (its Part 0.3).

**Decisions.** V23 is commissioned, and the ledger is owned by the ToMpathy engine. All other choices are defaults in force (Section 8).

---

## 1. Program rules

### 1.1 Families, variants and batches

The curator asked for a research division: sweep large spaces fast and return **textured results**, many variants shown together, instead of one thorough result.
- A **family** is one mechanism question: one theory row, one world generator, its readers and controls, and one declared **primary contrast**, the paired score difference between the mechanism reader and its matched rival (in nats unless stated).
- A **variant** is one point of the family's sweep grid.
- A **batch** runs every active variant on fresh world identities, scores every reader, and completes the independent check and full replay.
- The **useful margin** m is 0.02 nats, and 0.02 for probabilities (V22's margins).

**Base rung (proposal):** 16 worlds × 64 cases per variant, with laws drawn from a menu of at least 16 distinct values. V22's eight worlds per card drew on only two to four.

### 1.2 The stopping rule

**Why not count null batches.** If one batch of a real family shows a clear win with probability p (its **per-batch power**), stopping after k consecutive null batches kills that family with probability (1 - p)^k:

| Per-batch power p | k = 2 | k = 5 | k = 7 |
|---:|---:|---:|---:|
| 0.1 | 0.81 | 0.59 | 0.48 |
| 0.2 | 0.64 | 0.33 | 0.21 |
| 0.3 | 0.49 | 0.17 | 0.082 |
| 0.5 | 0.25 | 0.031 | 0.0078 |

*Rows: per-batch power. Columns: consecutive null batches required to stop. Cells: the chance of killing a family whose effect is real.*

Cheap batches have per-batch power of 0.1 to 0.3, where counting five to seven nulls kills 8% to 59% of real families. The analyst's earlier rough estimate is that the pooled rule below loses about a third as many; calibration replaces it.

**The rule, per family.**
1. **Pool.** After each batch b ≥ 2, fit a hierarchical model to all variant estimates so far: each variant's true gain θ comes from a family-wide normal distribution with mean μ (the family's typical gain) and spread τ (how much variants differ), and each batch estimate is θ plus sampling error sized by that batch's whole-world bootstrap standard error. Proposed priors: μ normal around zero, scale 0.05 nats; τ half-normal, scale 0.03. A grid makes the fit exact in seconds.
2. **Decide.** **q** is the posterior probability that the best shrunk variant (highest posterior mean) truly gains more than m.
   - **Burn-in:** batches 1 and 2 always run, every variant at the base rung.
   - **Kill** if q < 0.10, recorded as "stopped below margin" with its q trajectory; revivable by the curator.
   - **Nominate** the best variant for confirmation if q ≥ 0.90.
   - **Cap** at 7 batches; ending there without crossing a boundary is **inconclusive**, never "null".
3. **Halve, factor 3.** From batch 3 the top third of variants by posterior mean continue with three times the worlds (16, 48, 144). Parked variants keep their data in the model and on the curve; allocating by data already in the model does not bias the posterior. With one to three workers, the asynchronous version reduces to plain halving.

**Calibrate before launch** (CPU minutes). Simulate families of 27 or 81 variants (μ in {-0.02, 0, 0.01, 0.02, 0.03}; τ in {0.005, 0.01, 0.02}; base-rung standard errors {0.01, 0.02, 0.03} nats, shrinking by √3 per rung), including slow starters whose gain at rung r is θ × (1 - 0.5^(r+1)). Targets: kill at most 10% of real families (best true gain above m + 0.01) and nominate at most 20% of non-real ones (best gain at most m). Choose the kill threshold from {0.05, 0.10, 0.15} and the prior scales, report counting rules at k = 2, 5 and 7 beside it, and freeze the choice before any family runs.

### 1.3 Confirmation and false discovery control

A nominated variant gets one **confirmation** on never-used worlds: 128 worlds × 256 cases (256 worlds if its shrunk estimate is within 0.02 of the margin), against a direct reader fitted to that variant alone.

**The e-value** is a nonnegative score whose average is at most 1 when the true gain is at or below the margin; large values are evidence of a real gain. Proposal: E = the product over worlds w of (1 + λ_w × (x_w - m)), where x_w is world w's paired gain, truncated below at a declared floor (-0.1 nats, or lower if screening saw lower; truncations reported), and λ_w is a betting fraction set only from earlier worlds, at most half the value that keeps every factor positive. In an analyst scratch simulation (not a Ghost result) with V22-like per-world spread (0.04 nats), a true gain of 0.04 gave E ≥ 100 in 98% of runs at 128 worlds, and a gain at the margin averaged E near 1.

**e-LOND.** Promotions are tested in a fixed order, strongest first. As the analyst understands the method, the j-th is confirmed if e_j ≥ 1 / (α × γ_j × (D + 1)): α is the target false discovery rate (0.10), γ_j a spending sequence summing to 1 (1/(j(j + 1)): 0.5, 0.167, 0.083 and so on), and D the number of confirmations so far. The coding agent verifies the formula against the primary source before implementing it. Nothing is called confirmed without this step.

### 1.4 Readers, controls and gates

Carried forward from V22 unless marked new.
- **Exact reader:** the law-marginal posterior over finite supports, which is the ceiling; the supplied-law oracle is reported separately.
- **Ablations:** the exact reader minus one ingredient (pooling, weights, a cue, a class model).
- **Direct readers:** V22's validation-stopped network and protocol, at 2,048 and 8,192 labels with two fit seeds. **New:** fitted once per family across its grid (variant settings as inputs) and frozen across batches, plus a **fresh-law rival** trained on a new generator draw per minibatch (10 October errata, GS-6).
- **Engine slot (new):** any module offering fit(ledger, population_prior) returning a posterior, predict(posterior, decision) returning a distribution over options, and explain(posterior, maker_id) returning hypotheses is scored on the same worlds through the ledger. Ghost's exact readers ship wrapped as reference modules.
- **Controls:** every family has a **positive control** (a planted effect the exact reader must find) and a **negative control** (nothing to find; the mechanism reader must not beat its rival by more than the margin). Known-answer gates run first. A failed control halts the family as **instrument failed**, which is not a kill.
- **Headroom gates:** learned superiority needs both fits stopped on validation and a matched-stratum interval for direct minus exact loss wholly above 0.02 nats; wholly at or below 0.02 is parity; otherwise unresolved.
- **Verification:** an independent checker rebuilds every exact posterior and score; a full replay must match byte for byte, timing excluded, before a batch counts toward any decision.
- **Seeds:** SHA-256-derived from (family, variant, batch, world, case) in a `v23` namespace, with bootstrap seeds per comparison and fit seeds per family (pre-mortem item 6).
- **Repairs and result kind:** one bounded repair per family under a new identity, original retained. N1, N4 and N6 also make constructed-mechanism claims; every other card is a method result. None supports a human conclusion.

### 1.5 Textured reporting

Each family report leads with a **specification curve**: every variant's primary contrast, raw with interval and shrunk, sorted low to high, over a panel marking each variant's setting on every axis, so the reader sees which settings carry the effect. Two-axis families add a heat map. Beneath come the q trajectory, the verdict, the controls, and every reader's proper scores by stratum.

### 1.6 CPU and how it scales

V22 spent 965.6 CPU-seconds on science, checks and full replays for 14,336 cases, about 0.07 seconds per case, so a base variant-batch (1,024 cases) projects to 0.2 to 1.5 CPU-minutes, depending on latent size and reader count. Halving keeps each later batch near one full base batch, so a family costs about V × c × B: V variants, c CPU-minutes per base variant-batch, B batches (about 3.5 expected, 7 at cap). N1 also scales with makers × records. Pilots replace these projections before admission.

---

## 2. Part R: repairs of V22's flawed controls

### R1. A2 repaired: the vanishing-gap control with matched hypothesis spaces

**Original prediction.** The content-uptake gap between readers with and without the steering process "vanishes when steering and informing artifacts are indistinguishable in process."

**Why it failed.** The control removed the signature but kept different hypothesis spaces: the missing-process reader cannot represent steering and believes every claim, while the full reader still holds steering possible (posterior 0.51) and discounts. The gap stayed at 0.130 (0.144 with the signature): the library carries both a prior possibility of steering and signature recognition, and the control separated neither.

**Corrected design.**
- **Generator:** V22's A2 world, law menu widened to 16 reliabilities from 0.60 to 0.90.
- **Readers on the same artifacts:** steering-aware with the signature visible; **the same reader with selection bits masked**; missing-process (labelled misspecified); steering-aware with prior steering probability 0.01, separating prior from library; law oracle; direct readers with and without selection bits.
- **Contrasts:** **signature value** (masked minus visible uptake); **skepticism value** (missing-process minus masked); **false-skepticism cost** (extra content loss on informing sources). **Primary:** content-property log loss, masked minus visible.
- **Grid (36):** steering prevalence {0.05, 0.2, 0.5} × signature strength {0.5, 0.6, 0.7, 0.9} (the chance a selection bit points to the true goal; 0.5 is none) × bits {2, 8, 32}.

**Predictions.**
1. At strength 0.5, signature value is zero within tolerance for every matched pair: the vanishing-gap control done right, run as a known-answer gate.
2. Signature value exceeds the margin at strength ≥ 0.7, ≥ 8 bits and prevalence ≥ 0.2.
3. Skepticism value stays positive at prevalence ≥ 0.2 even without a signature: the real part of V22's persistent gap.
4. At prevalence 0.05, steering-aware readers pay a false-skepticism cost on informing sources, which a visible signature shrinks.

**What outcomes mean.** If 1 fails, the mask leaks. If 2 and 3 hold, persuasion knowledge protects mostly through skepticism, and through signature recognition only above a measured strength. A near-zero skepticism value would mean V22's gap came from misspecification alone.

### R2. A3 repaired: withdrawing every purpose-diagnostic channel

**Original prediction.** Withdrawal "pays in lower purpose-recognition accuracy."

**Why it was untested.** V22 withdrew the content and value channels, but the flag likelihood, itself purpose evidence, stayed at full precision after withdrawal and under the early guard. "No recognition cost" cannot measure the cost of not looking.

**Corrected design.**
- **Generator:** V22's sequential A3 world, with streams extended from 16 to 32 steps and flag reliability on a 16-value menu from 0.65 to 0.95.
- **Grid (16):** procedure coupling {separable, goal-bearing} × prefix {4, 8, 16, 32} × benign share {0.5, 0.8}.
- **Policies, all on every variant:** no gate; procedure-only; early guard, now covering flags; **withdraw-content** (V22's); **withdraw-all**, dropping flag precision after firing to a floor φ in {0.5, 0.2, 0.05, 0}. Thresholds {0.5, 0.7, 0.9}; release rule none, or release when the flag posterior falls below 0.5 (only if φ > 0). Matched-technique comparisons use V22's fixed no-gate mixtures.
- **Metrics:** purpose and value memory change; technique accuracy; purpose recognition (log loss and modal accuracy), split flagged versus benign and fired versus not; benign false alarms and releases. **Primary:** recognition log loss, withdraw-all (φ = 0.05, threshold 0.7) minus no gate.

**Predictions.**
1. A recognition cost above the margin appears at prefixes 16 and 32.
2. The accuracy cost falls on falsely flagged benign makers, whose purpose posterior freezes on the wrong side. On truly flagged makers it is near zero at thresholds above 0.5, because the gate fires only after the reader already believes the flagged purpose; they lose confidence (log loss), not the answer. This is a derivation, checked as a known answer.
3. Release with φ ≥ 0.2 recovers most of the benign cost, with protection within the margin of withdraw-all.
4. Withdraw-all protects at least as much as withdraw-content at matched technique.

**What outcomes mean.** If 1 and 2 hold, the flag channel hid the cost, and false alarms pay for not looking: the case for N6's design. If no cost appears, recognition is front-loaded and withdrawal is nearly free at these thresholds.

### R3. B1 repaired: the fully choice-independent costly-label null

**Original specification.** "Choice-independent costly labels": recorded costs that never enter the maker's utility. If weighting by them still helps, weighting is a heuristic, not effort evidence.

**Why it was not run.** V22's "independent effort" costs were random but still entered utility. Effort was already in the exact likelihood, so weighting counted it twice; its gain was 0.009 nats, under the margin.

**Corrected design.**
- **Generator:** V22's homeostatic makers (two to four internal variables, set points on {-1, 0, 1}), with 16 laws on finer grids of drive exponent and choice precision.
- **Cost arms:** relevant (V22 native); **irrelevant labels** (recorded costs uniform on 0 to 1, utility uses zero cost); half-relevant; V22's arm as a reference.
- **Readers:** exact with the true cost role; **relevance-aware exact** (infers the role from {enters, label, half}); exact assuming costs enter; effort-weighted (V22's power likelihood); **matched tempering** (effort weighting's average weight on every record); cost-ignorant; frequency; direct.
- **Metrics:** set-point interval coverage and width; future-choice log loss. **Primary:** future-choice loss, unit weighting minus effort weighting, under irrelevant labels; a stop below margin is the predicted outcome.
- **Grid (72):** cost arm (3) × records {2, 8, 32, 128} × dimensions {2, 3, 4} × perturbation {native, larger}.

**Predictions.**
1. Under irrelevant labels, effort weighting gains nothing beyond the margin and lowers coverage.
2. Under relevant costs, matched tempering reproduces at least half of effort weighting's narrowing.
3. The relevance-aware reader puts at least 0.9 on the true role by 32 records and matches the true-role reader by 128.
4. Assuming costs enter loses more than the margin under irrelevant labels from 32 records.

**What outcomes mean.** If 1 and 2 hold, effort informs only through the exact likelihood, so the engine should model costs, not reweight by them. If weighting gains under irrelevant labels, it is tempering, and V22's narrowing cannot be credited to effort.

### R4. C1 repaired: a root-level cue for top-down entry

**Original prediction.** Joint inference beats either direction alone at coverage of at least 50%.

**Why it failed at the root.** Only leaves were observed, so top-down inference had nothing to enter with and bottom-up equalled joint at the root. The curator's route in by function ("this would cut, so it's a cutting tool") was never tested.

**Corrected design.**
- **Generator:** V22's trees (depth four, branching two or three, leaf accuracy 0.9); the chance r that a child copies its parent comes from 16 values from 0.60 to 0.98.
- **Cues:** a **root cue** (a stated purpose or title) with reliability ρ; a **function cue** on one depth-one node; a misleading arm (ρ = 0.3 while a fixed-trust reader assumes 0.9).
- **Readers:** joint with cues (exact sum-product); bottom-up; top-down (cues and a forward pass only); V22's joint without cues; joint marginalizing ρ; direct with cue features. **Primary:** root log loss, bottom-up minus joint with cue.
- **Grid (90):** ρ {0.5, 0.7, 0.9, 0.99} × leaf coverage {0, 0.1, 0.25, 0.5, 1.0} × branching {2, 3} × cue position {root, function}, plus 10 misleading-cue variants.

**Predictions.**
1. At ρ ≥ 0.9 and coverage ≤ 0.25, the cue beats bottom-up at the root by more than the margin; at full coverage with branching three it adds less.
2. At coverage 0 with a root cue, top-down equals joint, and the cue's value at depth d shrinks with the root-to-node agreement (2r - 1)^d.
3. A function cue helps its own subtree more than an equally reliable root cue, and helps the root through its parent.
4. Under misleading cues the fixed-trust reader loses more than the margin; the ρ-marginalizing reader keeps the loss under half that.

**What outcomes mean.** The ρ × coverage surface maps where top-down entry matters and sets how edit summaries enter the engine (T3). If no cue helps even at coverage 0, the engine should enter at mid-level functions, not the root.

---

## 3. Part N: new mechanism families

**Shared maker core.** N1, N2, N4, N7 and the twins share one generator. **Drives** are Panksepp's seven primary systems (SEEKING, RAGE, FEAR, LUST, CARE, PANIC/GRIEF, PLAY) plus two bodily needs (Solms); base worlds use four drives, SEEKING and one need, to keep exact inference cheap. Options carry a 0 or 1 affordance per drive. **Values** are **recipes**: sparse sets of drives from a finite dictionary of singletons and pairs, held with weights in {1, 2}. Choice is a softmax over recipe-weighted utility. Every latent goes to the truth sidecar.

### N1. "The only way out is together"

**Question.** How much can a reader learn about one maker's values from that maker's records alone, versus pooling across makers, and is there a structure under which no amount of one person's data yields a confident estimate? This is the constructed form of ToMpathy's global tab.

**Theory row.** Anti-capture (filed, 14 October errata, E9): error bars stay "unacceptably large until we include a mountain of data", a low displayed confidence "can't be lying", and it should be "a theorem". Values as personal recipes over drives (walkthrough H; recorded, not filed, as of 14 October).

**Pushback first.** Unconditionally, the claim is false: with enough stationary records from a maker whose menus span the value space, individual estimates converge, and capping displayed confidence there would be dishonest the other way. Ghost can show where it holds as a theorem; whether people live there is for real data.

**World generator.**
- **Culture:** each world's recipe distribution π is one of 32 candidates per heterogeneity level, drawn from a symmetric Dirichlet with concentration 0.05 (low heterogeneity), 0.5 or 5 (nearly uniform).
- **Makers:** four drives; 4 singleton and 6 conjunctive pair recipes (an option must serve both drives); one or two recipes per maker, weights {1, 2}: 200 types. Precision {1, 2, 4}; R decisions of three options each.
- **Structures:** **S-id** (menus span all affordance patterns); **S-alias** (a maker's menus never separate one maker-specific drive pair, so recipes such as {a}, {b} and {a, b} predict identically in its own contexts, while other makers' menus separate other pairs); **S-drift** (at each record, with probability δ in {0.01, 0.05}, the type is redrawn from π).
- **Targets:** in-context choice; **out-of-context choice** on menus separating every pair (the behavioral form of value extraction); recipe presence.

**Readers** (all with the true drift model; only population knowledge differs): **exact hierarchical** (sums over the 32 cultures given all makers, then each maker's type; cost cultures × makers × types); **individual-only** (own records, uniform prior); **population-only**; **partial pooling** (culture estimated from other makers, used as the prior); **prototypes** (4 or 16 components); direct (individual and population-summary inputs; fresh-law rival); engine slot: population backbone and personal parameters.

**Controls.** Positive: in S-id at R = 512, individual-only must come within 0.02 nats of exact out of context, ruling out a reader too weak to find signal that exists. Negative: a one-type population. Shuffle: records permuted across makers; no personal reader may beat population-only.

**Metrics and margin.** Per-maker in-context and out-of-context loss; recipe-presence loss and calibration; the **confidence ceiling** (90th percentile of top recipe probability, by R); the **total-risk ratio** (summed out-of-context loss under pooling over the sum under individual-only; below 1 is the **Stein-like gain**, lower total error even though makers far from the population can be worse off); the **harm share** (makers worse off under pooling, by type rarity). Margin 0.02 nats.

**Sweep and batches.** Grid (240): makers {10, 100, 1,000, 10,000} × records {2, 8, 32, 128, 512} × heterogeneity (3) × structure (S-id, S-alias, S-drift 0.01, S-drift 0.05). 16 worlds per cell, 8 where makers × records ≥ 500,000; cases are makers. The stopping rule covers only R = 512 under alias and drift (36 variants), with primary contrast individual-only minus exact out-of-context loss. The rest is a map: two batches, plus worlds wherever an interval is wider than ±0.02.

**Predictions.**
1. S-id: individual-only reaches exact within the margin by R = 128.
2. S-alias at R = 512: individual-only trails exact by more than the margin at low and medium heterogeneity; for aliased makers its top recipe confidence stays at or below the alias share under its uniform prior (one half or one third) at any R, while exact sits at calibrated population odds. In-context prediction is unaffected: your own data predicts you in your own world; only the population says what you would do in a new one.
3. S-drift at δ = 0.05: individual confidence plateaus by about R = 64; pooling gains exceed the margin at every R.
4. The total-risk ratio is below 1 wherever R ≤ 32; the harm share rises with heterogeneity and falls on rare types. With only 10 makers, estimated pooling can lose under high heterogeneity.
5. Population-only beats individual-only at R = 2 under low heterogeneity; 16 prototypes close half the gap to exact by R = 32.
6. **Display honesty:** exact and pooled readers stay calibrated (error ≤ 0.02); a fixed display cap of 0.5 is miscalibrated in S-id from R = 128.

**What outcomes mean.** If 1 to 3 hold, anti-capture is a theorem with stated conditions: when contexts never separate a person's values, or values drift, no volume of that person's records identifies them, and calibrated population odds are the honest best; the global tab is then a necessity. If S-alias converges, the alias leaks; if S-id fails its control, no N1 claim stands. Which structure real makers live in is for Sounding Line (Part T).

### N2. Diagnosticity weights

**Question.** When marks differ in how much they reveal about preferences, which reading rule recovers preferences and stays calibrated, and can response time estimate a mark's informativeness?

**Theory row.** Noise, contested as of 14 October. The curator rejects a noise bin; the agreed replacement is a per-mark **diagnosticity**, d(m) = | ln P(m | A) - ln P(m | B) |, where m is one mark (a choice, a timing, an edit) and P(m | A), P(m | B) are its likelihood if the maker holds preference A or the rival B. Near zero, the mark says nothing about which preference the maker holds.

**Framing.** Ghost cannot test whether noise exists in people. Formalized, the curator's position is the exact reader: every mark has a goal-directed story, and some stories carry no information about the maker's own preferences. The rival is the naive rule that scores every mark as a preference choice at one precision.

**World generator.** Binary preference (X over Y, or the reverse). Mark classes with exact likelihoods and sidecar d(m): **preference choices** (choice precision β in {1, 4}); **near-ties** (utility gap under 0.1); **mixed-strategy draws** whose frequency an opponent's payoffs set (the serve example in the results read), observed or hidden; **spontaneous timing**, independent of preference; **priming and carry-over** (strength 0.3 or 0.6); **high-resolution context** that explains choices when modeled. Response time falls in five bins, decisive choices faster, slope s_RT in {0, 1, 2} (0 is uninformative).

**Readers.** Exact (infers mark class from response time and context); **everything-is-preference** (one precision); **oracle-weighted** (log-likelihood times true d, rescaled; diagnostic only); **response-time-weighted** (bin weights fitted on training makers; deployable); class-aware without response time; exact plus weights (double-count control); direct; engine slot: the diagnosticity estimator.

**Controls.** Negative: all-preference marks, and separately s_RT = 0; weighting adds nothing. Positive: oracle weights at a non-diagnostic share of 0.75 beat everything-is-preference.

**Metrics and margin.** Preference loss; held-out preference-choice loss; calibration error; signed bias toward the opponent's payoffs; rank correlation of estimated with true d. **Primary contrast:** everything-is-preference minus response-time-weighted loss. Margin 0.02 nats.

**Sweep and batches.** Grid (124): dominant class (5) × non-diagnostic share {0.25, 0.75} × s_RT (3) × records {16, 128} × β (2), plus four all-preference controls; cases are makers.

**Predictions.**
1. At a non-diagnostic share of 0.25 or more, everything-is-preference is overconfident (calibration error above 0.05) and trails exact by more than the margin.
2. Oracle weights recover at least half that gap; response-time weights at least a quarter when s_RT ≥ 1, and nothing at 0.
3. With payoffs hidden, everything-is-preference reads the opponent-set frequency as preference; with payoffs observed, exact gives those marks no preference weight.
4. Exact plus weights is worse calibrated than exact: weighting an exact likelihood double counts (B1's lesson).

**What outcomes mean.** If everything-is-preference matches exact everywhere, the no-noise rule costs nothing here, supporting the curator's classifier. If it fails and response-time weights repair it, diagnosticity weighting is the engine's rule, and Sounding Line's decisiveness test gets a calibrated expectation.

### N3. Corrections as half-spaces

**Question.** When makers revise toward their preferences, how much faster does a reader that treats each correction as a constraint on the preference vector learn than one reading final choices only, and what happens when corrections are local fixes?

**Theory row.** "Improvements are corrections: half-space constraints on the preference vector" (filed in v3, 14 October); coactive learning (Shivaswamy & Joachims, cited in the results read); Sounding Line's corrections test.

**World generator.**
- **Preference** w: a nonzero vector in {-1, 0, 1}^d, normalized (26 directions at d = 3, 242 at d = 5).
- **Episodes:** the system proposes x0 in {-1, 0, 1}^d; the maker accepts or picks one of E edits (one or two coordinates, one step) by softmax at precision β_c over (w · Δx - edit cost), where Δx is the edit's change and w · Δx is how far it moves the artifact toward the maker's preference. A correction implies the half-space w · Δx > 0, more reliably as β_c rises.
- **Local fixes:** a share λ of corrections fix process (register, fit to context) along a direction set by an observed context event, not by w.
- **Final choices:** menus of four options.

**Readers.** Exact (models local fixes; marginalizes λ and precisions); **half-space Bayesian** (all corrections as preference); **coactive perceptron** (adds each Δx); **final-choice only**; corrections plus choices; direct; engine slot: the corrections module.

**Controls.** Negative: λ = 1, where the half-space reader must not beat final-choice-only. Positive: λ = 0 and β_c = 4, where it must.

**Metrics and margin.** Final-choice loss; cosine between estimated and true w; calibration; the **sample-efficiency ratio** (records the final-choice reader needs to get within 0.05 nats of exact's asymptote, over those the correction reader needs). **Primary contrast:** final-choice-only minus half-space loss at matched records. Margin 0.02 nats; a ratio of 2 counts as useful.

**Sweep and batches.** Grid (96): d {3, 5} × β_c {1, 4} × λ {0, 0.25, 0.5, 0.75} × E {4, 16} × records {2, 8, 32}.

**Predictions.**
1. At λ = 0 and E = 16, the ratio is at least 2: the best of many edits is a many-way comparison, made where the system erred.
2. The half-space reader falls below final-choice-only at λ ≥ 0.5 in some precision setting; exact stays within the margin of its λ = 0 score.
3. Corrections plus choices beat either alone by more than the margin at 8 records or fewer.
4. The perceptron nears the Bayesian readers on cosine, not on calibration.

**What outcomes mean.** If 1 and 2 hold, corrections are the richest preference record but need classifying, preference versus process, before use; Sounding Line's corrections test then needs a local-fix classifier first.

### N4. Values as recipes over drives

**Question.** Can a reader identify which sparse drive combinations a maker holds, and does modeling SEEKING as pursuit vigor (an intercept) rather than a value dimension matter?

**Theory row.** Drives are Panksepp's systems plus bodily needs (Solms; adopted 14 October). Values are learned, personal, identifiable recipes ("PLAY and LUST"; "FEAR and SEEKING"); SEEKING is "the engine of pursuit for all other drives" (walkthrough H; recorded, not filed); the analyst proposes SEEKING as vigor. No learned ends (filed in v3).

**World generator.**
- **Drive sets:** base (PLAY, LUST, FEAR, CARE, SEEKING, one need) or full (seven systems, two needs).
- **Recipes:** singletons and pairs over non-SEEKING drives, plus SEEKING pairs when SEEKING is a dimension; 1 to 3 per maker in base worlds, at most 2 in full worlds (up to about 12,000 types, which exact inference still handles).
- **Meaning:** **additive** (mean satisfaction over the recipe's drives) or **conjunctive** (pays only when the option serves every drive in it).
- **Utility:** u(o) = the sum over the maker's recipes r of w_r × sat_r(o): how much the maker wants option o, with w_r the recipe's weight and sat_r(o) how far o satisfies r.
- **SEEKING:** as **vigor**, a level {low, mid, high} scaling choice precision and the chance of acting at all; as a **dimension**, an affordance (exploration, escape routes) recipes can include.
- **Needs** vary by episode and scale need-linked recipes. **Menus:** four options, random or **separating** (some serve one conjunct but not both).
- **Optional N4b:** a distorted transition map, a habit and an intrinsic recipe produce the same persistent choice; a contingency-change probe asks whether records tell them apart. This tests whether "no learned ends" is readable, not whether it is true of people.

**Readers.** Exact; **drive weights only** (linear, no recipes); recipe readers with SEEKING as dimension or vigor, crossed with the generator; pooled recipe reader (N1's machinery); direct; engine slot: the values module behind the core values box.

**Controls.** Negative: additive meaning, where recipes and drive weights predict identically. Positive: conjunctive meaning, separating menus, 512 records, where exact recipe-presence loss is under 0.05 nats.

**Metrics and margin.** Choice loss on random and separating menus; recipe-presence, act-or-abstain and SEEKING-level loss; **profile domination** (the share of makers with over half their posterior on SEEKING-containing recipes when SEEKING is vigor). **Primary contrast:** drive-weights-only minus exact on separating menus. Margin 0.02 nats.

**Sweep and batches.** Grid (160, plus 12 for N4b): drive set × meaning (2) × recipes per maker × SEEKING generator (2) × records {8, 32, 128, 512} × menus (2).

**Predictions.**
1. Conjunctive, separating menus: recipe presence within 0.05 nats of the ceiling by 128 records; drive-weights-only trails exact by more than the margin.
2. Additive: no reader tells "PLAY with LUST" from PLAY and LUST held separately; the posterior splits at prior odds, and prediction is unaffected. Recipes are identifiable only through interaction.
3. SEEKING as vigor but read as a dimension: at least 60% of makers show profile domination, and act-or-abstain loss trails the vigor reader by more than the margin. The reverse mismatch costs more on SEEKING-affordance menus.
4. Random menus need at least three times the records of separating menus.

**What outcomes mean.** If 1 and 4 hold, values are readable only from menus that pull drives apart: rich natural menus (chess, Wikipedia) or chosen questions, the engine's acquisition stage. If 3 holds, the core values box shows SEEKING as vigor, not as a value. If 2 describes people, the box can show drive weights only.

### N5. Projection bias: "whose steps"

**Question.** How much does a reader lose by forecasting a maker from its own steps instead of the maker's usual way, and how much do base-rate calibration and partial pooling recover?

**Theory row.** Whose steps (walkthrough H; recorded, not filed): the true signal is departure from the maker's usual way, but the reader runs its own forward model modified for the maker. **Real anchor:** the cloud reader put 43% of its probability on "edit" against 22% observed, and never named "dismiss" (23% observed).

**World generator.** Four handling classes (accept, edit, dismiss, ignore). Maker propensities come from a population Dirichlet (mean matched to CoAuthor-like marginals; concentration low or high) plus a suggestion-quality feature. The reader's own policy is displaced from the population mean by Δ logits, shaped like the anchor (one class over-weighted, one suppressed) or randomly, with quality sensitivity correct or absent.

**Readers.** Exact hierarchical; **projection**; **projection with prior-shift calibration** (class probabilities times the ratio of population to own rates, renormalized: the Saerens-style correction in the results read); projection plus maker frequencies; partial pooling; **adjusted projection** (own policy plus a shrunk maker offset, as the curator describes readers); direct; engine slot: the evidence-layer calibration.

**Controls.** Negative: Δ = 0. Positive: Δ = 2 with no records, where projection must lose to base rates.

**Metrics and margin.** Log loss and half-Brier (Sounding Line's scale); class-marginal bias; **never-named rate** (a class seen in at least 20% of decisions that is never the top prediction); calibration. **Primary contrast:** projection minus calibrated projection. Margin 0.02 nats.

**Sweep and batches.** Grid (128): Δ {0, 0.5, 1, 2} × shape (2) × records {0, 2, 8, 32} × concentration (2) × quality sensitivity (2).

**Predictions.**
1. At Δ ≥ 1 with no records, projection loses to base rates by more than the margin, with one over-predicted and one never-named class.
2. Calibration recovers at least 75% of the gap to base rates with no records, but not the gap to partial pooling from 8 records.
3. Adjusted projection approaches pooling as records grow, and beats it only when its quality sensitivity is right and the population model's is not.

**What outcomes mean.** This sets the expectation for Sounding Line's calibrated-reader test: how much of the 0.078 deficit calibration should recover. If 3 holds, LLM readers belong in the engine as feature sources, never as raw judges.

### N6. Model everyone, gate uptake

**Question.** A human-like reader withdraws modeling from makers judged low in warmth and competence; an engine reader models everyone and gates what it absorbs. What does each cost in prediction and in unwanted uptake?

**Theory row.** **Warmth** reads the maker's willingness to direct CARE at the reader; **competence** judges the maker's transition map; being modeled takes warmth or usefulness, and disgust is the lock with neither (walkthrough H, recorded). Engine principle: model everyone, absorb selectively (results read, E4; open as of 14 October). Extends A3.

**World generator.** Warmth {low, high} is the chance an artifact serves the reader rather than steering against them; competence {low, high} is whether the maker's procedures reach their goals. Artifacts carry claims (as in A1 and A2), procedures and purpose signals. The reader forecasts each maker's next action, with a stakes weight on low-low makers (defense). In one arm, goal-bearing procedures carry unwanted goals (A3's physical lesson).

**Readers.** **Withdraw** (stops updating and absorbing for makers judged low-low); **model-and-gate** (updates everyone; technique gated by competence, purpose and value by warmth); absorb-everything; the exact decision-theoretic reader (payoff ceiling); direct; engine slot: the uptake gate.

**Controls.** Negative: stakes 0 with compute charged, where withdrawal should match or win. Positive: stakes 1 with 32 records, where model-and-gate must forecast low-low makers better.

**Metrics and margin.** Next-action loss by warmth × competence; unwanted uptake; useful technique; false withdrawals; modeling compute; total payoff (prediction and useful technique, minus unwanted uptake, adopted failing procedures and compute). **Primary contrast:** low-low next-action loss, withdraw minus model-and-gate. Margin 0.02 nats.

**Sweep and batches.** Grid (108): low-low share {0.1, 0.25, 0.5} × stakes {0, 0.5, 1} × records {2, 8, 32} × coupling (2) × threshold {0.7, 0.9}.

**Predictions.**
1. Model-and-gate wins on low-low next-action loss by more than the margin when stakes exceed 0 and records reach 8.
2. Unwanted uptake is equal with separable procedures; with goal-bearing ones, model-and-gate leaks when taking technique from competent, cold makers.
3. False withdrawals cost prediction, more with fewer records.
4. At stakes 0, withdrawal is cheaper and pays the same: the human pattern is rational when stakes are low.

**What outcomes mean.** If 1 and 2 hold, modeling everyone dominates withdrawal wherever predictions matter, and the remaining risk is the leak through useful technique, which needs a procedure-level filter. If withdrawal matches it everywhere, the CARE-prosthesis claim has no constructed support.

### N7. Stated versus revealed

**Question.** When makers state values with noise and self-presentation bias, how should a reader combine statements with choices, and when do statements mislead?

**Theory row.** StatedGoal records; wanting to want (filed in v3: an edit made to train a value not yet held). The worlds mirror the shapes of PRISM, MovieLens Beliefs and MultiPref once the coding agent reads their documentation; nothing here claims their contents.

**World generator.** True values are recipes (N1, N4). Statements add noise σ_s {low, high} and a **self-presentation shift** toward desirable drives (CARE overstated, RAGE understated) of strength {0, 0.5, 1, 2}, shared or idiosyncratic. A share {0, 0.3} of makers make **aspirational statements** that their choices drift toward. Scopes follow the ledger: prompt, title, decision summary.

**Readers.** Exact; choices only; statements only; **both, naive**; **both, pooled shift correction** (the shift learned from makers with many choices, applied to makers with few); direct; engine slot: the stated-goal module.

**Controls.** Positive: at shift 0, statements help at 2 records or fewer. Negative: statements independent of values add nothing.

**Metrics and margin.** Value-presence and held-out choice loss; signed bias toward desirable drives; future-choice loss for aspirational makers. **Primary contrast:** choices-only minus pooled-correction loss at 8 records or fewer. Margin 0.02 nats.

**Sweep and batches.** Grid (128): σ_s (2) × shift strength (4) × shift type (2) × records {0, 2, 8, 32} × aspiration (2).

**Predictions.**
1. At shift 0, statements add more than the margin at 8 records or fewer.
2. Under a shared shift of 1 or more, naive statement readers are biased toward desirable drives, and both-naive loses to choices-only from 32 records.
3. Pooled correction keeps the statements' value under a shared shift, not an idiosyncratic one.
4. Aspirational statements predict future choices better than past choices do.

**What outcomes mean.** StatedGoal records enter the engine as noisy measurements corrected by a population-learned shift, and as forecasts of where a maker is heading.

### N8 (optional). Play mode and threat mode

**Question.** If uptake opens in play (fake danger, "claws in") and closes under real threat (walkthrough H; results read, E5), does a mode-aware reader read preferences better? **Design:** a latent mode per round; play actions are self-handicapped threat actions carrying a play signal (reliability 0.6 or 0.9); threat events end play. Readers: mode-aware, mode-blind, always-closed. Grid (24): play share {0, 0.3, 0.6} × signal reliability × threat rate × records {8, 32}; play share 0 is the negative control. **Primary contrast:** preference loss, mode-blind minus mode-aware. **Prediction:** at a play share of 0.3 or more, mode-blind reads play-fighting as RAGE and trails by more than the margin. Run only if budget remains.

---

## 4. Part T: synthetic twins of the real testbeds

A **twin** is a constructed world whose ledger matches a real testbed's shape (record types, counts per maker, timing distributions, class marginals), with planted truth. Twins **validate engine modules** before real data; **set detectability floors** (if a module cannot recover a planted effect of a given size, a real null below that size says nothing); and **supply spike-ins**, planted makers that Sounding Line inserts into real ledgers as positive controls. Each ships a **realism table** comparing twin and real marginals, required before any twin result is cited.

### T1. CoAuthor twin
- **Shape** (Sounding Line's compiled ledger, as of 14 October): about 60 makers (61 usable writers), with about 24 sessions and 280 decisions each (1,445 sessions; 16,959 decisions). About half of handling decisions are accepts; edits are near 22% and dismissals near 23% in the scored sample. Options, latencies and keystrokes come from Sounding Line's summary statistics; no raw text enters Ghost.
- **Planted:** preferences over option features (continues the topic, adds an entity, changes register, contradicts earlier text); handling propensities; decisiveness-linked latency; corrections after acceptance, split into preference and local fix; session momentum, phase and fatigue; the prompt as a StatedGoal with N7's shift.
- **What scores mean:** power at the real shape for Sounding Line's prequential, decisiveness, corrections and source-trust tests; for example, the smallest planted personal effect the pooling backbone detects with 61 writers, against the real 0.016 half-Brier gain.

### T2. Chess twin
- **Shape:** a constructed game with exact values (a few random game trees, depth 6 to 10 and branching 3 to 5, solved once by minimax), so the optimal policy is known; 200 to 2,000 players with 20 to 5,000 games each; a move clock.
- **Planted:** skill (precision on value); style (move-feature preferences paid for in value); clock pressure as context; habit (a personal opening repertoire followed with strength h); blunders as policy error; each move's class and diagnosticity.
- **What scores mean:** deviation-class accuracy, and individual gain over a population model as games accumulate: expectations for a real chess rung, where individual models beat population models only at thousands of games (McIlroy-Young et al., 2022, cited in the results read).

### T3. Wikipedia twin
- **Shape:** 500 to 5,000 editors with heavy-tailed histories (100 to 10,000 edits); multi-maker articles with diffs on an article state vector; edit summaries as decision-scope StatedGoals; reverts.
- **Planted:** goal trees (inform, maintain, promote a view); edit intents (copyedit, add, remove, revert, promote); summary honesty; steering flags that other editors detect by reverting.
- **What scores mean:** purpose recovery from the diff, the summary (R4's top-down cue) and both; next-purpose prediction from deep history; steering detection at scale (A1); attribution in multi-maker artifacts.

### T4. Preference-panel twin
- **Shape:** many raters × many shared items, with ordinal ratings and short explanations, in the crossed design of panels such as Community Alignment and DICES (matched once the coding agent reads their documentation).
- **Planted:** rater recipes and reliability; stated-reason shift (N7); groups sharing recipe distributions (cultures).
- **What scores mean:** N1 at a real shape, where shared items supply the population identification writing logs lack; calibration of the global tab per rater and per group; and a **flood test**, measuring how far the population estimate moves when one maker or a coordinated group contributes 100 times the median records: the constructed form of the curator's flooding concern.

---

## 5. Part B: the bridge

**Emit the ledger from every Ghost world.**
- **Format:** every V23 family and twin writes its public observations as the engine's decision ledger, in exactly these record types: JSON Lines, one file per type, with the schema version and hash in every header.
- **Canonical schema:** `tompathy/engine/tompathy_engine/ledger/schema.py`, pinned by ToMpathy commit for the whole round.

| Record | Fields | Plain meaning |
|---|---|---|
| Maker | maker_id, source, identity | one person or agent; source is the dataset or Ghost world |
| Artifact | artifact_id, maker_ids, kind, revisions | the thing made; may have several makers; revisions in order |
| Session | session_id, maker_id, artifact_id, t_start, t_end, stated_goal_ids | one sitting of work |
| StatedGoal | goal_id, maker_id, scope (session, artifact or decision), text, source (prompt, edit_summary, commit_message, assignment or title), t | what the maker said they were doing |
| Decision | decision_id, maker_id, session_id, t_open, t_resolve, options (option_id; origin: maker, ai, system or env; content; features), taken (option_id or null), action, cost (latency_s, keystrokes, chars_typed, time_left), context, outcome | one choice among options |
| Correction | correction_id, maker_id, decision_id, span, before, after, t, stated_reason | a later change to what a decision produced |
| ContextEvent | event_id, t, kind, payload | anything in the situation that is not a decision: a clock, opponent payoffs, a prime, a threat |
| StatedPreference | pref_id, maker_id, attribute, value, scale, source (survey, profile or self_description), t | what the maker says they value or prefer; added by the engine spec for N7 and for real survey data (PRISM, MovieLens Beliefs) |
| Truth (sidecar) | latent drives, value recipes, set points, habits, goal tree, steering flags, diagnosticity of each mark; also mark class, correction class, mode, warmth and competence | evaluator only; never readable by a reader |

**Mapping rule.** Every mark becomes a Decision (a choice among the options the maker could have produced) or a Correction; artifact-level observations such as claims and selection signatures become Artifact revisions plus Decisions over the alternatives the maker could have made. Each family's one-page mapping is written before implementation.

**Adapters.** Ghost writes one emitter per family. Sounding Line's compilers produce real ledgers: CoAuthor works; Git and Wikipedia intake are built and bounded; a chess adapter would be new. Engine modules read only ledgers, so one module runs unchanged on a twin and on its real testbed.

**Conformance suite** (Ghost ships it; both repositories run it):
1. **Schema:** types and required fields.
2. **Integrity:** references resolve; taken is an option or null; t_open ≤ t_resolve, within the session.
3. **Truth isolation:** sidecars live in separate files and carry a random canary token per maker; the suite fails if any token appears in, or is predictable from, reader-visible records.
4. **Round trip:** a Ghost exact reader gives identical forecasts natively and through the ledger, within 1e-12.
5. **Invariance:** relabeled identifiers and permuted order-free records leave forecasts unchanged.
6. **Known answers:** frequency and partial-pooling modules on ledgers with closed-form answers.
7. **Real-shape smoke:** Sounding Line runs the reference modules on a small CoAuthor sample.
8. **Version pin:** a schema mismatch fails.

**What it closes.** Ghost targets become ledger fields and Ghost readers become engine modules, so Sounding Line can score real records the same way. This closes Sounding Line's retained "missing Ghost target mapping" deficit.

---

## 6. Sequencing and budget

**Order of work.** Day 0 is when V21's final report posts, no later than 16 October, 18:00 PDT. Nothing runs before it.

| When | Work |
|---|---|
| Before day 0 | No V23 process runs |
| Day 0 to 1 | Freeze acceptance. Ledger schema, emitters and conformance suite first, because every family emits. Stopping-rule calibration, known-answer gates, cost pilots |
| Days 1 to 2 | R1 to R4 (cheap; they close V22's open controls). N1 and N2 (the curator's two live disputes). N5 (cheap, with a real anchor) |
| Days 2 to 4 | N3, N4, N7, N6. T1, timed to feed Sounding Line's first real-data tests |
| Days 4 to 6 | T2, T4, T3. N8 if budget remains. Confirmations through e-LOND |
| By day 7 | Final report and closure: at most seven elapsed days, as in V22 |

**CPU budget** (CPU-hours; projected from V22 receipts and replaced by pilots). A base variant-batch costs about 0.5 CPU-minutes for R1, R3, R4, N2, N3, N6 and N7; 0.2 for N5; 0.7 for N4; and 1.5 for R2 (about 40 policy settings). N1's largest cells dominate its cost.

| Item | Variants | Expected | Cap |
|---|---:|---:|---:|
| Setup: ledger, conformance, calibration, gates, pilots, services | | 5 | 8 |
| R1 to R4 | 36, 16, 72, 90 | 7 | 14 |
| N1 | 240 | 9 | 16 |
| N2 | 124 | 4 | 7 |
| N3 | 96 | 3 | 6 |
| N4 (with N4b) | 172 | 7 | 14 |
| N5 | 128 | 1.5 | 3 |
| N6 | 108 | 3 | 6 |
| N7 | 128 | 4 | 7 |
| N8 (optional) | 24 | 1 | 2 |
| Twins T1 to T4: build, realism tables, module runs, spike-in sets | | 7 | 14 |
| Confirmations (about 10 to 15 promotions) | | 7 | 15 |
| Protected: verification, review, closure | | 18 | 18 |
| **Total** | **about 1,200** | **about 76** | **about 130** |

*Expected assumes about 3.5 batches per family; cap, all 7.* The proposed 90 CPU-hour ceiling binds before every family reaches its cap; priority order decides what is deferred, recorded as "not run". The expected 58 unprotected CPU-hours take under a day on three Gear 2 workers when all slots are free; Sounding Line's priority will stretch that. **GPU:** none.

**What the curator sees after each batch:** a one-page, machine-written **batch card** with the family's question in one sentence; the batch number; q before and after, and the decision (continue, stop below margin, nominate, or inconclusive at cap); the specification curve so far; controls passed or failed; CPU used and remaining; and one sentence on anything surprising.

**Reviews.** Model-written review happens at family decisions and the twelve-hour health checks, because review time binds before CPU does (10 October errata, GS-8). A **program board** tracks every family's q and the budget. Public write-through to FINDINGS and the theory store moves to family decisions, which changes the per-batch publication rule and needs the curator's approval.

---

## 7. Pre-mortem: how this round fails

Suppose it is 23 October and the round has failed. The likeliest causes, each with its guard:

1. **Twins too clean:** modules pass on twins and fail on messier real data. **Guard:** realism tables before any twin result is cited; "dirty" variants (missingness, duplicated sessions, label noise); spike-ins inside real ledgers.
2. **Planted effects tuned to pass,** likeliest in N1, where weak individual signal is the wanted message. **Guard:** settings frozen before outcomes; a regime where each mechanism must lose (S-id for N1); every variant on the curve; one adversarial axis per family, chosen by the coding agent.
3. **Exact readers unavailable at scale** (N1 under drift at 10,000 makers, N4's full drive set, the chess twin). **Guard:** supports sized by pilots at the largest cell; otherwise a labelled approximate ceiling, never used in a headroom gate.
4. **The stopping rule kills slow starters.** **Guard:** two-batch burn-in; record axes spanned from batch 1; calibration on simulated slow starters; up to a third of variants pre-declared as late-effect, exempt from halving until batch 4; every kill revivable.
5. **Interface drift between Ghost and the engine.** **Guard:** one versioned, hashed schema owned in one place; the conformance suite in both repositories; a pinned version per round; migration notes and reruns on any change.
6. **Shared-seed correlation.** V22 reused one bootstrap table for every interval, shared fit seeds across cards, and drew eight worlds from two to four laws, so its errors were correlated and a curve built that way looks more coherent than it is. **Guard:** comparison-specific bootstrap seeds; family-specific fit seeds; at least 16 distinct laws per batch, counted in reports; a seed audit that fails if two batches or confirmations share a world.
7. **Multiplicity** across about 1,200 variants. **Guard:** shrinkage within families, e-LOND across promotions, fresh worlds for confirmation.
8. **Privileges mistaken for capability:** exact readers know law menus and class structures a real engine will not. **Guard:** privileges labelled per reader, one privilege ablation per family, engine modules scored without privileges.
9. **Texture drowns the reader.** **Guard:** batch cards, one curve per family, plain-sentence verdicts.
10. **Host contention** with Sounding Line's repair pass and first real-data tests. **Guard:** V23 yields; the ceiling is in CPU, not calendar.

---

## 8. Decisions: resolved, and defaults in force

**Resolved (curator, 14 October, 20:00 PDT).**
1. **Commission:** V23 runs as specified, starting when V21's final report posts.
2. **Ledger ownership:** the ToMpathy engine (`tompathy/engine/`) owns the canonical schema, and Ghost pins it by ToMpathy commit for each round.

**Defaults in force unless the curator overrides.**
3. **Budget:** 90 CPU-hours, 18 of them protected for verification and closure. Gear 2 only, no GPU, at most seven elapsed days.
4. **Stopping rule:**
   - burn-in of 2 batches;
   - kill at *q* < 0.10, which calibration may move to between 0.05 and 0.15;
   - nominate at *q* ≥ 0.90;
   - hard cap of 7 batches;
   - halving factor 3;
   - false discovery target 0.10, with e-LOND spending 1/(j(j + 1)).
5. **N1 framing:** anti-capture as a conditional theorem (alias and drift), with S-id as the positive control. A confidence cap in ToMpathy is honest only where the structure supports it. This matches the ToMpathy spec's section 9: show true shrinkage, never an artificial cap.
6. **Scope and order:** R1 to R4, N1, N2 and N5 first; then N3, N4, N7, N6 and T1; then T2, T4 and T3. N8 runs only if budget remains.
7. **Reporting:**
   - machine-written batch cards;
   - model review at family decisions and at the twelve-hour checks;
   - public write-through at family decisions.
