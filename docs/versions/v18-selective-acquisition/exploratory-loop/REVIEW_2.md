# V18.4 review 2: U96 landed, neural slate running

Does uncertainty over changing roles remain useful in longer streams? With independent observations in the 96-step comparison, the mixture improves prediction after skill changes: logarithmic loss is 0.67629 versus 0.68863 for static inference, reversing the shorter packet's ordering. After goal changes, its loss is 0.71549 versus 1.62893 for static inference and 0.75456 for fixed fast-role updating; expected artifact matching is 79.56%, 67.11% and 79.48%, respectively. The skill benefit disappears with copied observations. A small stationary cost remains, and the mixture still does not repair an omitted decision rule. These are descriptive constructed-method results; the horizons use separate generated traces, so their difference is not a paired causal estimate of extra observation time.

U96 generation and aggregation completed at 00:16:19 UTC on 20 September; its
independent replay used 77.25 CPU seconds and passed. All 19,200 finite assigned
evaluations in the initial roster are now verified. They do not close the adaptive
campaign. The first nine-fit main learned-reader packet is active, followed by
five more matched support/supervision packets and their verification. Re-estimate
the remaining budget from its first full fit/packet, preserving the immutable
05:00 PDT minimum target, 06:00 science cutoff and cumulative CPU ceiling.

Next research/design priorities: matched-prefix skill diagnostic; equally sized
aligned versus irrelevant purpose portfolios; uncertainty over decision families
versus paid revision. These distinguish live explanations instead of adding
seeds to a settled contrast. Candidate implementation continues at the next event
review while the frozen neural buffer runs. No useful-work-exhausted state applies.
