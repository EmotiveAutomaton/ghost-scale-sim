# Ghost V29 pre-flight, 12 November 2026

Three numpy scripts with fixed seeds, run in the analyst's container. They exist to catch designs that are decided by construction or by an incidental parameter before the version builds them. Nothing here is a result. Build your own X1, X2, Q2 and Q3 from the packet first, then run these and compare (packet section 2.0).

| Script | World and reader | Output |
|---|---|---|
| `x_exchange.py` | The ledger world as the packet states it (the kept good starts at 8; an empty stock draws no sell offer; a freeze is absorbing). Reader: live sums with weights 1 to 16, a rule, hubs, and frozen sums. This is X2's reader in small | `x_exchange_out.txt` |
| `x_buy_side_check.py` | The same world with X1's reader: live sums with weights 1 to 256, rules with buying weight 1, 2, 4, hubs; no frozen sums. Share of buy offers at 0, 0.1 and 0.5 | `x_buy_side_check_out.txt` |
| `q_queue_control.py` | Q3 as first drafted (queue or sum) and Q2 (a deficit with and without control) | `q_queue_control_out.txt` |

## What the pre-flight found, and what changed in the packet

1. **A hub is read in 3 to 5 steps, with or without a market.** HUB(4) separated at a median of 3 to 5 steps in all ten cells tried, including the cells with no buy offers, because it leaves needs unmet to keep its buying power. Naming a strong hub is trivial in this world. The hub cells became X1's known answer, and the open question moved to milder hubs (texture) and to V28 G3.
2. **A rule is told from a large live sum by its buying side.** With no buy offers the rule was never separated, in any of four cells (`x_buy_side_check_out.txt`): on the selling side a rule and a live sum whose weight exceeds the largest temptation are the same record. With half of offers to buy it separated by 1,024 steps in 87 to 100% of worlds (medians 95 to 393). With a tenth, in 96% under mixed prices with plenty (median 268) and in 38 to 50% of the other three cells. The tell is the gap between the price refused and the price paid. X1 was redrawn around it.
3. **The freeze is slow to see, and an early freeze cannot be seen at all.** With frozen sums in the reader's grid, a sum of weight 4 that froze after 20 straight refusals separated in 79% of worlds at 1,024 steps under mixed prices with plenty (median 722), in 24% under low prices with scarcity, and in none of the other two cells. A rule from the start separated from the frozen sums in 0% and 49% of plentiful worlds and 77% and 86% of scarce ones: a rule and a sum that froze after five refusals differ only in the first five sell offers.
4. **The "read-as" column in `x_exchange_out.txt` is not a misreading.** Class evidence is averaged over unequal numbers of members, and a late-freezing hypothesis is nested in a live one until a long refusal run occurs. Where the column shows FROZE for a live truth (98% of worlds for LIVE(4) under mixed prices with scarcity), the log-odds are near zero and the cell is unresolved. The version reports log-odds and never an argmax across classes of unequal size.
5. **Queue or sum, as first drafted, was decided by a nuisance parameter.** The two types separated in 5 to 135 steps in every cell, including cells where they prefer the same act in over 97% of states. The separation came from a different resting propensity and utility scale, which are not the claim. Q3 was redrawn: the types share one rule for whether to act, and the family asks what the stress reading counts (how many needs are unmet, their total, or the loudest).
6. **A deficit without control hides whether the agent does not care or does not believe.** For a passive-by-default agent with sensitivity 1, in an uncontrollable episode alone the prior was read right in 13% of worlds and the sensitivity in 52%. One controllable episode raised those to 56% and 96%, and an uncontrollable episode followed by a controllable one to 83% and 95%. An active-by-default agent with low sensitivity kept its prior unresolved in most worlds whatever the order (read right in 1 to 30%, read wrong in about 1%). Q2 keeps this design and adds the response to the agent's own success as the tell.
7. **Push rates after control.** In an uncontrollable episode that follows a controllable one, the three agents with sensitivity 1 or 2 pushed in 79 to 94% of steps, against 26 to 44% when the uncontrollable episode came first. The low-sensitivity agent went from 22% to 34%. This is fixed by the design: belief carried forward plus a growing deficit.

## Two defects found in the pre-flight itself, and fixed before this bundle

- A first version turned sell offers into buy offers when the stock ran out. That leaked buying evidence into the zero-buy cells (a rule "separated" in 21 to 32% of them). Fixed: an empty stock draws no sell offer.
- A first version seeded cells with Python's string hash, which differs between processes. Fixed: crc32 of the cell's name.

## Costs

`x_exchange.py`: 20 cells of 200 worlds by 1,024 steps, under two minutes on one thread. `x_buy_side_check.py`: 18 cells, under two minutes. `q_queue_control.py`: under one minute.
