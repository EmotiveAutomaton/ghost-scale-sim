# A2: equal-size training-only predictive portfolios

Does a learned predictive bank improve because of which questions it preserves,
or because a decoder repairs its numerical errors? The A1 diagnostic and the
different readout rankings motivate an equal-size, fixed-capacity comparison.
This is an old-world artifact-history method screen, not local-goal transfer.

Freeze eight candidate questions: the five original training questions followed
by the three existing new-combination questions. Every portfolio contains five
16-category distributions. The two farther questions remain the common targets
and are excluded from the candidate menu. Compare original five, one seeded
uniform random subset, a response-diversity heuristic, and training-only targeted
forward selection. No outcome chooses the candidate menu or population.

Use original training lineages 193000–193127, first eight development lineages
190000–190007, two original training draws and two feature seeds. The native
generator provides 16 histories per training lineage and 32 per development
lineage. Four old-world classes receive equal weight; retain all cells and the
history-inert class. Development, test and confirmation populations are distinct;
the reserved test/confirmation lineages remain untouched. This is a new explanatory
comparison on existing populations, not seed padding or a confirmation study.

Split training by lineage before any fit: first 64 for fitting, next 32 for
portfolio selection, last 32 for ridge selection. Fit a single shared eight-query
old-outcome head per draw/seed from artifact/context history with the existing
512-padded, 128-tanh-feature basis and ridge 0.01. The old head has 129 by 128
coefficients. Train teachers are the simulator's observable response distributions,
equally available to every portfolio; they are privileged distribution supervision.
Hidden states, world rules and development targets are never learner features.

Random selection uses the declared feature seed plus 700. Diversity starts at
candidate zero, then greedily maximizes minimum squared response distance on the
fitting lineages' observable teachers. Targeted selection greedily minimizes
far-question logarithmic loss on selection lineages using learned-bank features,
fitting a linear ridge head at 0.01 on fitting lineages. This is a predictive proxy
for noise-aware selection: it includes actual learned-bank errors but is not a
proof of an optimal experimental design. Ties choose the lowest candidate index.

After selecting a portfolio, use 81 coefficients per target category: intercept
plus 80 bank entries, with no hidden layer. Compare exact conditional banks,
learned banks, and exact banks with independent Gaussian noise whose per-entry
variance is estimated from learned-minus-exact errors on fitting lineages only.
Exact and noisy banks are labelled evaluator-law diagnostics. They do not alter
portfolio choice. For each bank/portfolio choose ridge from 0.001/0.01/0.1 on the
last 32 training lineages, retaining the first choice on exact loss ties. Fit on
the same first 64 lineages throughout; do not refit on selection data.

Every raw distribution uses the same declared repair: negative entries clipped,
1e-6 added, then category normalization. Report invalid raw probability frequency,
logarithmic loss, squared probability error and complete predictions separately.
This bounded linear leaf does not implement a hull or new softmax decoder; their
existing results remain separate. It does not close the primary learning curve.

Retain eight-entry acquisition/teacher cost during selection, five-entry deployed
bank width, all fitted-head dimensions, candidate evaluations and separate CPU
timings. Selection overhead is not free, and simulated distribution teachers are
not eight naturally acquired observations. Exact matrix spectra at two tolerances
are numerical diagnostics, never exact nonidentification certificates.

Controls precede science: known separating response, constant-target placebo,
probability repair, informative-candidate selection with stable ties, unique
equal-size portfolios, split disjointness and corruption rejection. Preserve
source/plan/environment hashes, inputs, evaluator records, fitted parameters,
raw predictions, scores, adjacent and extracted-source replays. Regroup paired
lineages after averaging draws/seeds, separately by class and equal-class total;
report fit/draw variation separately. Any interval is conditional on the fixed
training choices, not a universal training-population interval.

The original A2 five-CPU-hour cap includes admission, failures and full replays.
The shared two tiny settings remain consumed; this uses NumPy ridge only. The
existing single worker and immutable resource window own execution.

Two independent alternatives remain prepared before outcomes. C1 uses saved tiny
weights and native purpose/belief donor-recipient pairs, including changes that
should affect a target and changes that should leave it fixed; it requires exact
fixture validation before any alignment fit, with no new architecture setting.
F1 compares direct decoding, 1/4/16 forward rollouts and matched extra decoding
from the same retained state, with oracle/wrong-law controls and all query costs;
freeze its native local target and transition teachers before fitting. Neither
handler is claimed implemented or admitted. A2 improvement motivates held-out-law
transfer; a bounded failure leaves these alternatives open.
