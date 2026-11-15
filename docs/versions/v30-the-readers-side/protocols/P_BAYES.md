# P posterior extension and fresh-page score

This extension retains posterior particles using four independent latent-label
Gibbs/Metropolis chains. Page mixes are sampled conditional on latent true-label
allocations; maker mixes use a joint additive-log-ratio proposal with the correct
Jacobian. Adaptation stops before retained draws. Independent density arithmetic
reconstructs all retained fitted outputs. It is an approximate posterior reader,
never a proven ceiling.

Freeze the excluded ladder (burn-in, retained draws per chain): (512,256),
(1024,512), (2048,1024), (4096,2048), with two transitions per retained draw.
Four excluded makers per PERFECT/HUMAN cell, km=6, kp=3, h=.5, all eight page
levels, must have split R-hat below 1.1 and autocorrelation effective sample
size at least 50 in every mix component. Adjacent fitted means must differ
by less than .03. A no-data Dirichlet known answer, replay and independently
calculated posterior density are additional gates. Stop at a 1,800-CPU-second
validation allowance or the family limit; failure is a retained numerical
limitation, not evidence that correction beats Bayes.

The owner's fresh-page amendment is scored as a sum of sentence-marginal
predictive log scores: posterior-mean label probabilities versus the population
mean on a separately generated page. This is a proper marginal prediction score,
not posterior odds of a continuous point, and not a joint page Bayes factor.
The true-mix oracle score is separately labelled. Inference never receives truth
or fresh-page outcomes. Main and pilot particle diagnostics are retained;
unconverged estimates cannot support the pages-to-threshold comparison.

All 180 m=60 cells are built; admission and finite execution await these checks.
No larger Bayes grid is authorized by this file.
