# N13 implementation preparation

The October 17 Sounding Line packet, section FX1, supplies the exact frozen
verdicts. This is preparation, not a declared protocol or admitted study.

Use matched observed records, features and tuning budgets. The gain is the
records-only reader's Brier loss minus the goal reader's loss. The margin is
10% of the records-only reader's headroom above the personal floor, fixed before
unblinding. SUPPORTED requires the gain's lower bound above zero and its point
estimate at least the margin. FALSIFIED requires the upper bound below the margin
and every applicable precondition. Other cases are INCONCLUSIVE, with the failed
condition named. Report the two N13 P3 rates separately: at least 13/16 planted
margin blocks with lower bound above zero, and 0/16 habit-only blocks SUPPORTED.

The generator needs a precise score-scale calibration. Do not simply use a Brier
margin as a logit coefficient. Also distinguish the goal-informed oracle from
the complete personal oracle: setting the goal oracle's gain equal to the entire
headroom and simultaneously to 10% of that headroom is circular except at zero.
Declare independent calibration inputs and a generator parameter search on known
probabilities, with no production cases used for that mapping. Match habit-only
self-consistency at each setting. Any privilege given to the goal reader (such
as knowing a nuisance prior) must also reach its records-only comparator.

Generic tempering is selected on separate validation inputs. Each of the sixteen
law blocks supplies 64 independent maker cases; the block's verdict is computed
from those cases, and across-block reported uncertainty resamples blocks. Do not
force the declared power target to pass: a failed power control is a useful
result that leaves Sounding Line's FX1 precondition unmet.

The delivered consistency/headroom aggregates remain optional. Check for them
at declaration; otherwise explicitly freeze the packet's defaults. The eight
conditional studies retain rank order: N13 follows N14 and N9.
