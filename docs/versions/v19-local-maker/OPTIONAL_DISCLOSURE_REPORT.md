# Optional metadata requests: independently verified

We tested whether declining low-value metadata requests improves prediction after request costs. At a stipulated price of 0.1 nats per request, optional requests preserve forced-request forecast loss and reduce cost-adjusted loss by 0.07216 nats on original tools and 0.06626 on changed tools. Independent reconstruction, all 1,944 paired estimates and both complete replays verify this constructed-method advantage; supplied laws and reliable replies do not establish learned access or human intent.

The frozen comparison uses 640 retained queries, eight development laws,
original and changed tools, uniform legal-completion and native conditional
laws, native and equal-query populations, and three stipulated request prices.
Four policies make no request, force one entropy-selected request, request only
when expected entropy reduction strictly exceeds price, or request at a constant
probability matched to the optional policy's population request rate. Field
selection is identical across the latter three policies. Selection uses the
visible group's supplied conditional law, never the individual hidden target.

Logarithmic forecast loss measures endpoint prediction error in nats; lower is
better. Cost-adjusted loss adds request price times request probability. The
prices are stipulated comparison units, not measured human burdens. Squared
forecast error, true-endpoint probability, ambiguity and request rates remain
separate. Infinite-loss mass is explicit and zero throughout this finite batch.
Random requests average realized-branch proper scores, not mixed forecasts.

This table uses supplied native laws and native query weights, then weights the
eight laws equally. Rows cross tool and price; the four policy columns give mean
cost-adjusted loss. The last column gives optional requests per query.

| Tool | Request price | No request | Forced request | Optional request | Matched-rate random | Optional request rate |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0 | 0.14366 | 0.05295 | 0.05295 | 0.11156 | 35.36% |
| Original | 0.02 | 0.14366 | 0.07295 | 0.05852 | 0.12394 | 27.84% |
| Original | 0.1 | 0.14366 | 0.15295 | 0.08079 | 0.14621 | 27.84% |
| Changed | 0 | 0.18483 | 0.05295 | 0.05295 | 0.13173 | 40.32% |
| Changed | 0.02 | 0.18483 | 0.07295 | 0.05970 | 0.14704 | 33.74% |
| Changed | 0.1 | 0.18483 | 0.15295 | 0.08669 | 0.17403 | 33.74% |

At either positive price the optional rule makes requests on 27.84% of original
queries and 33.74% of changed-tool queries. Its forecast loss remains 0.05295
nats for both tools, identical to forcing a request on every query. On this
finite roster it skips only requests that do not improve forecasts. The savings
against forced requests are therefore cost savings, not additional predictive
accuracy. Targeting still improves forecasts over requests made at the same
overall rate independently of the visible group. At price 0.1, forcing requests
is worse than no request on the original tool; the optional rule remains better.

The next table gives paired cost-adjusted loss differences with 95% lineage
intervals. Negative values favor optional requests. Each row states tool and
price; columns identify the comparison policy. At price 0.02, savings relative
to forcing requests fall below the declared 0.02-nat practical margin. At 0.1
they exceed it on the stipulated cost scale. These are exploratory conditional
estimates, not confirmation or a universal price recommendation.

| Tool | Price | Optional minus forced | Optional minus matched-rate | Optional minus none |
|---|---:|---:|---:|---:|
| Original | 0.02 | -0.01443 [-0.01455, -0.01431] | -0.06542 [-0.06668, -0.06427] | -0.08514 [-0.08752, -0.08295] |
| Original | 0.1 | -0.07216 [-0.07276, -0.07156] | -0.06542 [-0.06668, -0.06427] | -0.06286 [-0.06477, -0.06112] |
| Changed | 0.02 | -0.01325 [-0.01338, -0.01311] | -0.08734 [-0.08886, -0.08589] | -0.12513 [-0.12876, -0.12192] |
| Changed | 0.1 | -0.06626 [-0.06690, -0.06554] | -0.08734 [-0.08886, -0.08589] | -0.09814 [-0.10118, -0.09537] |

The next table scores each retained query equally while retaining the supplied
native-law policy. Column definitions are unchanged. This is a different target
population, not a reweighting hidden inside the main result.

| Tool | Request price | No request | Forced request | Optional request | Matched-rate random | Optional request rate |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0 | 0.20931 | 0.09958 | 0.09958 | 0.16397 | 41.33% |
| Original | 0.02 | 0.20931 | 0.11958 | 0.10633 | 0.17903 | 33.75% |
| Original | 0.1 | 0.20931 | 0.19958 | 0.13333 | 0.20603 | 33.75% |
| Changed | 0 | 0.24777 | 0.09958 | 0.09958 | 0.17911 | 46.33% |
| Changed | 0.02 | 0.24777 | 0.11958 | 0.10746 | 0.19730 | 39.38% |
| Changed | 0.1 | 0.24777 | 0.19958 | 0.13896 | 0.22880 | 39.38% |

The final table uses a uniform law over mechanically legal completions, scored
under native query weights. It keeps law privilege separate from the policy
comparison; the full record also retains uniform-law/equal-query cells.

| Tool | Request price | No request | Forced request | Optional request | Matched-rate random | Optional request rate |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0 | 0.18546 | 0.07571 | 0.07571 | 0.14204 | 39.51% |
| Original | 0.02 | 0.18546 | 0.09571 | 0.08362 | 0.14995 | 39.51% |
| Original | 0.1 | 0.18546 | 0.17571 | 0.11522 | 0.18155 | 39.51% |
| Changed | 0 | 0.22630 | 0.07571 | 0.07571 | 0.15783 | 45.40% |
| Changed | 0.02 | 0.22630 | 0.09571 | 0.08479 | 0.16691 | 45.40% |
| Changed | 0.1 | 0.22630 | 0.17571 | 0.12111 | 0.20323 | 45.40% |

Independent scalar reconstruction verifies all 7,936 conditional-law records,
40,960 before/after forecast vectors, 768 score cells, 192 decision records and
1,152 reader projections, with maximum discrepancy 3.34e-16. Native query
weights inherit the independently verified parent and are not newly generated.
The checker passed 16 isolated controls including corrupt laws, forecasts,
decisions, prices, request rates, reader packets and denominators. A separate
regroup of original cells reproduces all 1,944 estimates within 5.56e-16 using
bootstrap multiplicities rather than the checker's regroup code. The 10,000
resamples use seed 190977 and paired laws; repeated questions are not independent
worlds. Native and equal-query populations remain distinct.

Strict zero-price decisions are numerically sensitive: independent scalar sums
change decisions in 32 cells, affecting 28–80 queries per cell. Native request
rates decrease by 0.03031–0.14773, while all logarithmic-loss, squared-error,
ambiguity and cost-adjusted-loss changes are zero. True-target probability moves
only at floating-point roundoff. All affected cells are native-law, zero-price
cells; positive-price decisions remain unchanged. Original literal binary64
decisions are retained. The matched-rate control's reported results retain those
original request rates; its rates and losses are not claimed invariant to a
changed zero-price rule. The checker reports no changed field choices, while
the parent's general field-tie qualification remains on record.

All 2,341 deterministic files match across the original, adjacent and complete
extracted-source replays. Their 543 source files and 1,186 inputs verify; the
checker binds 547 sources and 2,344 inputs. Plans, source archives, environments,
raw evidence, execution measurements and exports remain hash bound. No fitted
model, new observation, protected lineage or shared tiny setting was consumed.

**Warrant:** constructed-method cost advantage, miniature — architecture
untested. Supplied mechanics, native laws and reliable binary replies are explicit
privileges. This establishes neither learned access, historical process
correspondence, human intention nor confirmation on new laws. Finite score
contributions must never conceal infinite-loss mass in future comparisons.

**Pursuit:** noisy mechanical replies and retrospective evidence after grouping
future schedules remain independent finite alternatives. The optional-request
half-hour cap includes all controls, replays, reconstruction and failed attempts.
V19 remains active.

Reader inputs expose only anonymous visible artifact/operation queries and a
requested field with its mechanical reply. Hidden completions, endpoint targets,
law tables, weights, policy decisions and review estimates remain separately
labelled scientific/evaluator evidence.

[Protocol](OPTIONAL_DISCLOSURE_PROTOCOL.md),
[independent review](OPTIONAL_REVIEW_PROTOCOL.md),
[evidence](../../../results/v19/G19-F-optional-disclosure-1/README.md).
