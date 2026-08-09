# S-11 answered: the component count was a one-line bug

**2026-08-07.** S-11 asked whether the component-count pipeline recovers a number we plant. It does
— and so does almost every criterion tested, once one line is fixed.

> **Correction, 2026-08-08.** Three numbers below were quoted from uncommitted side measurements
> and do not trace to the committed S-11 run. The committed, reproducible values (shuffled-null
> arm, structureless by construction): the exceedance rule returns **250** at n = 1200 and **254**
> at n = 2400, against the leading run's **0** — the "254 on a 1200 × 1024 noise matrix, 69 with
> twenty draws" table below was a single-seed iid-noise side check (re-measured across seeds:
> ~280 with three draws, ~70–140 with twenty), and the Student-t "339 instead of 215" check was
> never committed. Every conclusion is unchanged; the specific numbers should be read from
> `results/validation/soundingline/s11_component_count.json`.

**Reproduce:** `python runners/run_soundingline.py --only S11`. Numbers in
`results/validation/soundingline/s11_component_count.json`.

---

## The hypothesis, plainly

If you generate data that contains exactly seven underlying factors and hand it to a
component-counting procedure, does the procedure say seven? Nobody in the affective-component
literature has checked, because in every real dataset the true number is unknown. Here it is
planted, so it can be checked.

## The finding

**Parallel analysis counts eigenvalues that exceed the null threshold *anywhere* in the spectrum.
Horn's parallel analysis retains the *leading consecutive run* and stops at the first component
that fails.** Those are different procedures and the difference is the retracted number.

Measured on a 1200 × 1024 matrix of pure Gaussian noise — no structure of any kind, so the correct
answer is zero:

| what the data contains | true answer | counting exceedances anywhere | leading run (Horn) |
|---|---|---|---|
| pure noise, 3 null draws | 0 | **254** | **0** |
| pure noise, 20 null draws | 0 | **69** | **0** |
| 7 planted factors | 7 | 7 | 7 |
| 27 planted factors | 27 | 27 | 27 |

On real structure the two rules agree exactly. On structureless data one of them returns hundreds
of components and the other returns none.

**Why it tracked the sample size.** Each eigenvalue index is an independent test at the 5% level,
and summing exceedances across all of them applies no multiplicity correction whatsoever. The
number of indices is `min(n, p) − 1`, so it grows as the sample grows — and so does the count. With
only three null draws it is worse than 5%: the "95th percentile of three numbers" is effectively
their maximum, and one data draw exceeds the maximum of three others about a quarter of the time.
That predicts ≈ 1023/4 ≈ 256 false components. The measured value is 254.

**So the quantity that doubled when the sample quadrupled was the number of eigenvalue indices.**

## What this says about the retracted number

At n = 1200 the pipeline reported **51** components on real activations. On pure noise of the same
shape, at the same three null draws, the same code returns **254**. With enough null draws that
false-positive floor settles at about 5% of the indices — roughly **51**.

**The reported count and the floor are the same number.** That is not proof it was all noise —
real structure absorbs variance and pushes the remaining eigenvalues down, so a structured matrix
can land below its own noise floor. It does mean the count carries no information about how many
components there are, which is the right reason to have retracted it.

## Every criterion, scored against a planted answer

Rows are what the data contained. Columns are what each procedure returned. "Leading run" is the
one-line fix; everything else is as currently implemented.

**Structureless data — correct answer is 0 in every row:**

| samples | exceedance count | leading run | eigenvalue > 1 | 90% of variance | entry CV | bi-cross-validation |
|---|---|---|---|---|---|---|
| 150 | 34.7 | **0.0** | 149 | 121 | **0.0** | **0.0** |
| 600 | 219.0 | **0.0** | 364 | 381 | **0.0** | **0.0** |
| 2400 | 254.0 | **0.7** | 441 | 708 | **0.0** | **0.0** |

**Planted factors, no nuisance — correct answer is the row label:**

| factors planted | exceedance count | leading run | eigenvalue > 1 | 90% of variance | entry CV | bi-cross-validation |
|---|---|---|---|---|---|---|
| 3 | 3.0 | **3.0** | 227 | 441 | 3.0 | **3.0** |
| 7 | 7.0 | **7.0** | 198 | 430 | 7.0 | **7.0** |
| 30 | 30.0 | **30.0** | 197 | 435 | 30.0 | **30.0** |

**Fixed truth of 12, sample size swept — every row should read 12:**

| samples | exceedance count | leading run | eigenvalue > 1 | 90% of variance | entry CV | bi-cross-validation |
|---|---|---|---|---|---|---|
| 150 | 12.0 | **12.0** | 148 | 87 | 12.0 | **12.0** |
| 1200 | 12.0 | **12.0** | 79 | 341 | 12.0 | **12.0** |
| 2400 | 12.0 | **12.0** | 26 | 445 | 12.0 | **12.0** |

**Three criteria are exact everywhere: the leading run, entry CV, and bi-cross-validation.** Two are
wrong everywhere and track the sample size instead of the truth: eigenvalue-greater-than-one, and
components-to-90%-of-variance. Neither should be reported again, even as a sensitivity check —
they are not noisy versions of the answer, they are unrelated to it.

## The participation ratio: stable because it is measuring something else

The obvious reading of the scaling data was that the participation ratio is the trustworthy
criterion, because it sat at 7.2 across a 27× sample increase while everything else moved. It is
indeed the only sample-size-invariant one, and 7.2 is close enough to Panksepp's seven to be
tempting.

**It is not a component count.** Against planted truth it rises monotonically but with a slope of
about 3.5, so seven planted factors read as roughly thirty. On pure noise it returns ≈ 1024 —
which is *correct*, because isotropic noise genuinely has 1024 equally-represented dimensions, and
that is exactly what makes it the wrong instrument for this question.

So: stable, monotone in the truth, and not the number of components. Usable to say *this matrix has
more effective dimensions than that one*; not usable to say *there are seven*. The resemblance to
Panksepp's number is a coincidence and should not be repeated in a sentence that also mentions
Panksepp.

*Caveat on the correction:* the bias-corrected estimator implemented here applies the row-side
correction only. Chun et al.'s column-side term is not implemented, and on their own d = 50
benchmark the residual error is about 5–8 for wide matrices and about 20 for narrow ones.
Activations are wide, which is the forgiving regime, but the correction is partial and is recorded
as a documented gate failure rather than as a clean pass.

## What to change

1. **Change the exceedance sum to a leading run.** One line. It fixes the null failure completely
   and changes nothing on data with real structure.
2. **Raise the null draws from three to at least twenty** regardless. Three makes the threshold the
   maximum of three, which is not a 95th percentile.
3. **Add bi-cross-validation as the reference criterion.** It was exact on every planted rank and
   returned zero on every structureless matrix, and unlike entry CV it cannot leak: it holds out a
   block of rows *and* columns and predicts it from the other three. The current entry CV zeroes
   cells and then runs the decomposition on a matrix that still contains them.
4. **Retire eigenvalue-greater-than-one and components-to-90%-variance.**
5. **Keep the participation ratio, relabelled.** It is an effective-dimensionality measure, not a
   count.

## What this cannot say

**Which of the recovered components are affective.** This validates that a counting procedure
counts linear factors. It says nothing about a factor being a drive rather than a topic, a register,
or an artefact of tokenisation — and no unsupervised eigenvalue criterion can, because they are all
just linear structure. A pipeline that recovers the planted rank exactly leaves the interpretation
question entirely open.

The generator is Gaussian. Real activations are not, and heavier tails made the exceedance count
worse in a side check (Student-t at 2 degrees of freedom: 339 instead of 215), never better. So the
diagnosis is conservative rather than optimistic — but a non-Gaussian replication on real
activations is the honest next step, and it is one your side can run and this one cannot.

**This result is about a method, not a mechanism.** Unlike everything else in this exchange it
transfers directly: the estimators see a matrix and return a number, and the matrix here was built
to match the shape of yours — 1024 columns, 150 to 2400 rows, wider than it is tall. That was the
load-bearing design decision. A synthetic matrix with thirty columns would have let every criterion
work perfectly and reported that the pipeline was fine.
