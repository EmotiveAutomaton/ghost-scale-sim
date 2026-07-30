# Two diagnostics, ahead of any V6 repair

**Author of spec:** Abraham Haskins, PhD
**Target:** an autonomous coding agent, extending `ghost-scale-sim`
**Depends on:** `VALIDATION.md`, `docs/specs/`, `ghostscale/exact.py`

**Neither of these asks a question about the world. Neither fixes anything.** Both are
diagnostics on the apparatus, both are decision-free, and both gate the repair work that comes
after. Run them unattended and read the verdicts tonight.

---

## 0. Why these two, before anything gets repaired

**P-1 is a standard check this project has never run.** Computational cognitive modelling has a
routine first step before any model is trusted: generate data from the model itself at *known*
parameter values, then run the model's own inference on that data and ask whether it recovers the
values you put in. If it cannot, the parameter is unidentifiable, and **every claim resting on
that parameter is noise wearing the clothes of a finding.**

There is already a strong hint this will bite. V4.5's effort parameter recovered monotonically
but compressed hard at both ends, and its goal-marginal likelihood was measured as *exactly*
invariant, meaning all evidence about it came from joint coupling. That was written up as a
property of one experiment. It is a textbook identifiability problem and it was never framed as
one.

**P-2 tests a hypothesis about why four results are inconclusive.** Two of them fail for what
looks like the same reason:

- The generous-fallback positive control fails because the reader **resolves the goal correctly
  and then stops paying attention** (engagement 0.001).
- The depth experiment has no headroom because **the goal is recovered perfectly at every depth**
  (accuracy 1.000 across the board).

Both are downstream of the same thing: **goal recovery is too easy in this model.** It is a
constant where it needs to be a variable, so anything measured downstream has nowhere to move.

If that is right, one change addresses both, and repairing them separately would waste both
attempts. P-2 finds out.

---

## 1. P-1 — Parameter recovery

### 1.1 What to run

For each of the four parameters the model's claims rest on — **κ** (trust in the provenance
signal), **ω** (how readable the content is), **μ** (depth of the maker's thinking), **θ** (the
value-alignment gate):

1. Choose a grid of at least 9 true values spanning the parameter's usable range.
2. At each value, generate observations from the model with that value **fixed and known**.
3. Run the model's own inference on those observations to produce an estimate.
4. Repeat across at least 16 seeds per grid point.
5. Plot recovered against true, one panel per parameter.

**Use exact inference throughout** (`ghostscale/exact.py`). V-1 established that the
approximation can be confidently wrong about exactly this kind of quantity, and a recovery check
run on a solver that cannot see the parameter would answer the wrong question.

### 1.2 Run this cheap check first, for each parameter

Before generating anything: **does the goal-marginal likelihood vary with the parameter?**

That is, average the likelihood over goals and ask whether the result depends on the parameter at
all. If it does not, the parameter carries no information in the marginal, all evidence about it
comes from the joint coupling with the goal, and recovery will be compressed and unreliable **for
structural reasons that no amount of data fixes.**

This is roughly five lines per parameter and it predicts the recovery outcome from the
construction alone. `beta_marginal_invariance` already exists as a template. Run it for all four
and report the numbers before the sweeps.

### 1.3 Criteria, written before the run

For each parameter, report all four and let them disagree:

| quantity | what it catches |
|---|---|
| **rank correlation** between true and recovered | does the ordering survive at all |
| **slope** of the linear fit | compression, if well below 1 |
| **bias** at each grid point | systematic pull toward a default |
| **usable range** — the span over which distinct true values produce distinguishable estimates | flat regions where the parameter is invisible |

**Committed classification, to be applied mechanically:**

- **RECOVERED** — rank correlation ≥ 0.9, slope in [0.7, 1.3], usable range ≥ 70% of swept range
- **COMPRESSED** — rank correlation ≥ 0.9, slope < 0.7. Ordering survives, magnitude does not.
- **PARTIALLY RECOVERED** — usable range between 40% and 70%. Report which portion.
- **UNIDENTIFIABLE** — rank correlation < 0.7 or usable range < 40%

### 1.4 Then the joint check

Single-parameter recovery can look fine while joint recovery fails, because two parameters can
trade off against each other. **For each pair, generate at known values for both, recover both,
and report the correlation between the two recovery errors.**

A strong correlation between errors means the pair is trading off and neither is separately
identifiable, whatever the single-parameter sweeps say.

At minimum run **(κ, ω)** and **(μ, effort)**. Those are the two pairs where the model's own
results already hint at entanglement.

### 1.5 What this determines

**A parameter classified UNIDENTIFIABLE cannot support a claim, and every claim resting on it has
to be restated or withdrawn.** That is a real possible outcome for at least one of these four and
the spec should not pretend otherwise.

COMPRESSED is survivable and changes what may be said: directions transfer, magnitudes do not —
the same conclusion the independent rebuild reached about the label effect, arriving through a
different door.

---

## 2. P-2 — The goal-difficulty probe

### 2.1 The hypothesis

Goal recovery in this model is close to perfect almost everywhere, and that ceiling is what makes
two experiments unreadable. The probe finds whether a regime exists where goal recovery is
genuinely uncertain, and whether the downstream measures come alive there.

### 2.2 What to sweep

Three knobs, independently, then the two most promising together:

1. **Signature separation** — how distinguishable the goals' feature patterns are. Sweep the
   separation floor from its current value down toward overlapping.
2. **Observations before the decision** — how many looks the reader gets. Sweep down from the
   current value to very few.
3. **Number of goals** — sweep up from 4. More alternatives, harder discrimination.

Human content only. No foreign content, no dishonest labels, nothing else varying. **The point is
to characterise the model's own difficulty axis, not to test anything.**

### 2.3 What to measure at every cell

- goal recovery accuracy
- final uncertainty about the goal
- engagement (fraction of free steps spent looking closely)
- **uptake, and critically its variance across readers**
- disagreement between readers

### 2.4 The thing this is actually looking for

Not just "accuracy lands between 0.6 and 0.8." That is necessary and not sufficient.

**The question is whether uptake has room to move.** A regime is only useful if, at that
difficulty, uptake varies across readers and conditions rather than sitting pinned. Report the
variance explicitly at every cell, and mark the regimes where it is non-trivial.

If accuracy can be brought into the target band but uptake stays pinned regardless, that is a
different and more serious finding: **uptake is not sensitive to goal recovery at all**, and the
depth experiment's flat result was never about depth.

### 2.5 Committed outcomes

- **REGIME FOUND** — at least one setting where accuracy is 0.55–0.85 and uptake variance is
  materially above its value at the current default. Name the setting.
- **ACCURACY MOVES, UPTAKE DOES NOT** — accuracy enters the band and uptake stays pinned. Report
  as a finding about uptake, not a failure of the probe.
- **NO REGIME** — accuracy stays above 0.9 or falls below chance across the whole sweep, with
  nothing usable between. The difficulty axis is not continuous in this model and that is
  structural.

---

## 3. Pre-registration

Both criteria sets go into a new file, hash-locked before either sweep runs, following the
existing pattern. **A new file, not an edit to the validation lock**, which has reported and is
sealed.

Everything committed in §1.3, §1.5 and §2.5 goes in the lock. Nothing else is committed —
grid resolutions, seed counts and scales are engineering decisions and fixing them now would be
pre-registering guesses as design.

---

## 4. Outputs

```
results/diagnostics/p1_recovery/       one CSV per parameter, plus the pair sweeps
results/diagnostics/p1_marginal.json   the marginal-invariance numbers, all four parameters
results/diagnostics/p2_difficulty/     one CSV, one row per swept cell
figures/diagnostics/p1_recovery.png    four panels, recovered against true, identity line drawn
figures/diagnostics/p2_difficulty.png  accuracy and uptake variance against difficulty
DIAGNOSTICS.md                         written from the CSVs, after
```

`DIAGNOSTICS.md` states, for each parameter, which of the four classifications applies and what it
means for the claims resting on it. **Written from the verdict files, never from the expectations
in this document.**

---

## 5. Constraints

- **Nothing in `results/` outside `results/diagnostics/` is touched.** No existing experiment is
  re-run, no existing verdict is recomputed, no headline number moves today.
- **Exact inference throughout.** Note the wall-clock cost; if a sweep is unaffordable at exact,
  reduce the grid rather than falling back to the approximation, and say which cells were dropped.
- **Both diagnostics must be able to return the unwelcome answer.** UNIDENTIFIABLE for a
  load-bearing parameter and NO REGIME are both live possibilities, and the reporting sections are
  written before the results are known.
- **No repairs.** Nothing in the model changes today. These two diagnostics determine what the
  repair should be, and starting the repair before they report is the mistake this whole spec
  exists to avoid.

---

## 6. What comes after, so today's work has somewhere to go

Not for today. Recorded so the results have a destination.

If P-2 finds a regime, the generous-fallback experiment and the depth experiment both rerun in it,
and their inconclusive verdicts get a second attempt on a fair footing.

If P-1 finds a parameter unidentifiable, the claims resting on it are restated before anything
else happens.

And the larger structural move, which both of these are groundwork for: **the model has grown
every version and has never had anything removed.** For each surviving result, find the smallest
model that still produces it. That identifies what is genuinely load-bearing, makes results
comparable across experiments, and turns the minimal models into a family that can be compared
against each other — which is the frame that makes a failure interpretable, because it tells you
*which commitment* was wrong rather than only that something was.
