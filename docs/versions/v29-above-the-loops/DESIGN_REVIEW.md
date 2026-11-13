# V29 implementation review
## Resolved equation conflict
The owner approved preserving the specified equations and correcting X1's claimed
known answer. With gain 3 and live weight 4, the sell probability is 0.1824255,
not RULE's 0.02. Exact individual aliasing requires weight minus gain at least
log(49)/1.5 = 2.5945469. Individual aliases do not make class-mean log evidence
identical. Report both without an argmax across unequal or nested classes.

Predicted short separation and carryover push rates are sampled expectations,
not algebraic known answers. X2's under-20% freeze cutoff is a design exclusion,
not proof that every world stayed live. These distinctions preserve the equations.

## Prospective implementation choices
- Q2 starts at deficit 2, as supplied; the text omitted its initial deficit.
  Prior-read-right requires log odds 3. A separate argmax accuracy is labelled separately.
- Ledger purchases at stock 10 follow the specified money condition, then stock
  is clipped to 10. The initial draft's additional capacity refusal was removed
  before scientific admission, with its initial control receipt retained.
- X2's late-entry reader assigns a uniform prior to each possible current refusal
  streak and the already-frozen state within a FROZE model. This is an explicit
  reader prior, not the generating world's conditional prehistory distribution.
  No hidden evaluator state enters inference. Early-entry inference starts unfrozen.
- Q3 uses the exact 216-state stationary distribution for scaling and quantiles;
  tied discrete quantiles retain their declared prior multiplicity. Shared service
  picks the lowest indexed maximizer with probability 0.9 and otherwise rests.
- Q2's W episode supplies one observed success per step at half weight, as written.
  It contributes no direct choice likelihood for the focal agent.
- Separation is the one-based first observation above log odds 3 that stays above
  through the record. A final failure is censored, never replaced by the horizon.
- Pointwise uncertainty uses whole blocks. Conditional event measures keep their
  event counts; an absent event is not a measured zero effect.
- The preflight's old Q3 resting-rule comparison is retained as a failed design,
  not treated as the expected output of the replacement stress-reading study.
