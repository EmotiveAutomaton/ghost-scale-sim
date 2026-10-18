# N12: concurrent goals, replacement and final authority

Can one artifact distinguish concurrent goals, replacement and final authority? In N12, the overlay reader improves true-mechanism log loss by 0.774–3.567 nats for emergent, replacement and editor-final mechanisms when recommendations conflict, but worsens it by 0.480–0.511 nats for the unchanged author-final mechanism. At zero conflict all four mechanisms remain exactly aliased. All 12,288 cases, 204 independently reconstructed intervals, ten forecasts per setting and full replay verify. This is a constructed-mechanism result, not evidence about human authorship or authority.

The table crosses four stipulated mechanisms with recommendation-conflict rates
of 0, .2 and .5. Primary gain is the single-top reader's true-mechanism log loss
minus the overlay reader's loss; positive favors the overlay reader. Parentheses
give descriptive 95% whole-law intervals. Adoption error is in decision steps.
Final-authority error is shown only where authority is defined and recommendations
conflict; elsewhere a hard class would turn exact ambiguity into an arbitrary tie.

| Stipulated mechanism | Conflict rate | Overlay gain (95% interval) | Overlay loss | Single-top loss | Change-point loss | Adoption-time error | Final-authority error |
|---|---:|---:|---:|---:|---:|---:|---:|
| emergent A/B mixture | 0.0 | 0.000000 (0.000000, 0.000000) | 1.386294 | 1.386294 | 1.386294 | 1.464844 | unavailable |
| emergent A/B mixture | 0.2 | 1.508291 (1.355479, 1.666063) | 1.235090 | 2.743380 | 1.651290 | 1.413338 | unavailable |
| emergent A/B mixture | 0.5 | 3.567189 (3.319314, 3.813184) | 1.042841 | 4.610030 | 1.945617 | 1.314956 | unavailable |
| replacement by B | 0.0 | 0.000000 (0.000000, 0.000000) | 1.386294 | 1.386294 | 1.386294 | 1.464844 | unavailable |
| replacement by B | 0.2 | 0.868533 (0.779626, 0.960424) | 0.968656 | 1.837189 | 1.093274 | 1.274754 | unavailable |
| replacement by B | 0.5 | 1.483402 (1.391195, 1.578743) | 0.661945 | 2.145347 | 0.908442 | 0.967900 | unavailable |
| author-final A | 0.0 | 0.000000 (0.000000, 0.000000) | 1.386294 | 1.386294 | 1.386294 | 1.464844 | unavailable |
| author-final A | 0.2 | -0.480236 (-0.524845, -0.429647) | 0.986587 | 0.506351 | 0.849034 | 1.509668 | 0.075195 |
| author-final A | 0.5 | -0.510854 (-0.586157, -0.432814) | 0.610416 | 0.099562 | 0.374821 | 1.530332 | 0.069336 |
| editor-final C | 0.0 | 0.000000 (0.000000, 0.000000) | 1.386294 | 1.386294 | 1.386294 | 1.464844 | unavailable |
| editor-final C | 0.2 | 0.774398 (0.685061, 0.863584) | 0.743952 | 1.518350 | 0.697802 | 1.283352 | 0.314453 |
| editor-final C | 0.5 | 1.237763 (1.138309, 1.325807) | 0.463079 | 1.700842 | 0.424127 | 0.962446 | 0.095703 |

The asymmetry is real in this miniature. Emergence, replacement and editor-final
change the response law after the hidden adoption point, so an overlay reader can
recover structure that a single-root reader omits. Author-final follows A before
and after adoption. In that mechanism the single-top A assumption is correct and
the overlay reader's extra alternatives make mechanism attribution worse. The
author-final adoption time consequently remains unidentified; more conflict does
not turn an absent response change into a temporal signal.

At zero conflict A, B and C agree. Every public history, posterior and forecast is
then identical across the four mechanisms, the final-authority posterior is exactly
one half/one half, and all primary gains are zero. The raw hard-error implementation
would choose the first tied authority; this is why the table reports authority as
unavailable at zero conflict rather than misdescribing tie-breaking as recovery.
At conflict .2/.5, final-authority error is .075/.069 for author-final and
.314/.096 for editor-final. Authority is not defined for emergence or replacement.

All ten forecast streams per setting—the overlay, single-top, change-point and
exact readers plus six fixed/fresh direct rivals—use the same six withheld choices.
Independent review recomputed every case-level proper score, primary and secondary
mean, posterior normalization, public/truth separation and the zero-conflict alias.
All source and fit hashes, 12,288 raw cases, full replay and 204 law-block intervals
verify. Positive mechanism-posterior gain does not establish superiority to every
predictive rival. The exact reader knows the stipulated mechanism family, adoption
range and reliability laws; acquisition, prevalence, human interpretation and the
miniature architecture remain untested. This study is outside confirmation.

[Protocol](N12_PROTOCOL.md), [complete summary](../../../results/v23/n12-decision-review/SUMMARY.json),
[secondary review](../../../results/v23/n12-decision-review/SECONDARY_REVIEW.json),
[provenance](../../../results/v23/n12-decision-review/PROVENANCE.json).
