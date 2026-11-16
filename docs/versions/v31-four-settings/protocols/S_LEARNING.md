# S.b and S.c prospective extensions

S.b uses the original 40-step waiting task. Unknown reward odds live on a
201-point grid from 0.001 to 0.999. BASE starts with Beta(2,2); SHIFT has the
same concentration and a mean shifted by the original matched optimism.
SHARPEN applies the matched confidence multiplier to its learned log-odds.
SLOW tempers both censored-outcome evidence and within-trial absence evidence
by its matched power. Each updates only the terminal outcome it observed;
quitting never reveals whether an unreceived reward existed.

Core cells cross true odds 0.25/0.5/0.75/0.9, cost 1/2, temperatures 0.5/1,
and BASE/SHIFT/SHARPEN/SLOW, with 400 trials, 200 worlds and eight blocks,
after 40 excluded pilot worlds. The reader conditions on the retained outcome
history and the true experimental odds; it does not see the maker's belief.
Report sustained three-nat separation and error, and SHIFT minus BASE waiting
on the same history at trials 41–50 and 351–400.

The conditional COST-learning extension is triggered only when the absolute
mean SHIFT diagnostic at trials 41–50 plus its 95% block half-width is at most
0.2 steps (five percent of the original four-step lift). It adds COST to the
candidate set on fresh worlds at those settings. This is a prospective
practical-vanishing rule, not a claim that a prior ever becomes exactly equal.

Independent posterior arithmetic, independent terminal-probability products
and a separate dynamic-programme implementation reconstruct every world.
Received-reward, censored-wait and zero-wait controls precede admission.

S.c has uniform arrival on 1–20 and a prospectively solved reward giving BASE
ten waiting steps at odds 0.75 and temperature 0.5. It crosses all ordered
maker pairs, odds 0.75 alone or all four, costs 1 alone or 1/2, temperature
0.5 and lift four. Forty-eight finite cells use the shared texture allocation.
Infeasible matches are recorded without draws. All are descriptive miniatures;
neither learning convention establishes a biological mechanism.
