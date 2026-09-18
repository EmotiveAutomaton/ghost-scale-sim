# V17 final reader and evaluator handoff

**Final delivery, 18 September 2026. Outcome-selected illustrations; constructed mechanism, miniature — architecture untested.**

Can artifact evidence support useful predictions about a maker's next choice after a constraint or recipient change?

The final [reader ZIP](../../../results/v17/closeout-1/reader-final.zip) contains six constructed observer cases, each offered at three evidence tiers, for eighteen requests. It is a selected illustrative challenge. The [evaluator ZIP](../../../results/v17/closeout-1/evaluator.zip) contains answers and reference forecasts and must be kept outside blind reader evidence. [ILLUSTRATIONS.json](../../../results/v17/closeout-1/ILLUSTRATIONS.json) also contains evaluator truth and is not reader input.

Extract the reader ZIP into an empty directory. Its standard-library `consumer.py` reads one JSON request per line and writes one prediction per line. In PowerShell:

```powershell
Get-Content -LiteralPath requests.jsonl | python -B consumer.py > predictions.jsonl
```

Use the project's explicit virtual-environment interpreter when running from this repository. The standalone consumer requires no environment synchronization or access to the scientific database.

The versioned request schema is `v17.observer.1`. Each request declares a target kind, finite answer support, permitted evidence tier, observable context and query cost. Output probabilities must cover that exact support and sum to one. Preserve opaque task identifiers to join committed predictions with evaluator answers. Make predictions before revealing the evaluator ZIP; possession of its answers changes the evidence condition.

The evidence tiers are finished artifact, artifact collection and recorded process. Goals, actual maker policies and private histories are hidden from artifact-only requests. Possible native model families and task opportunities are public. The task is a finite policy reconstruction: an unseen choice after an intervention. It does not promise unique recovery of all past stochastic actions.

Final selection takes the lowest scientific case hash within each observed outcome type and each of the four observer families in the completed expansion cohort. It adds no observations. Advantages, reversals, failures, plausible-but-wrong histories and memory examples are represented. Wrong recipient models, wasted computation and surprising downstream edits are available only as evaluator illustrations; their blind-reader types are explicitly absent. Outcome selection means this small bundle cannot estimate real-world or even unbiased in-model prevalence.

The original packaging receipt records an actual extracted run: all eighteen forecasts matched and an attempted private-file read was denied. A second fresh extraction passed at closeout. A third executed the delivered reader-final.zip and matched all eighteen forecasts exactly, with private-read denial. These checks protect the fixed trusted Python consumer and do not constitute a general hostile-code sandbox. The evaluator archive and illustrative examples are separate from the reader archive by content and purpose, not merely filename.

The included `ghostscale.transfer.adapter.1` envelope carries old `v16.transfer.1` payloads unchanged; it does not reinterpret their schema. The [V16 handoff](../v16-acquired-craft/TRANSFER.md) remains independently usable. The earlier [V17 pilot](../../../results/v17/reader-pilot-1/reader.zip) remains a discarded development pilot and is not silently relabeled fresh evaluation.

For a real-text reader, the useful transfer is the task contract: evidence available before prediction, a withheld future choice after a declared change, separate prediction commitment and evaluation, and explicit computation cost. Genuine recorded process and outcomes would be needed in the destination data. Synthetic goals and recipient states cannot supply missing human truth. This repository artifact neither installs anything in Sounding Line nor reports any Sounding Line execution.

The frozen original reader.zip and EXPORT.json incorrectly retained early-pilot selection wording. The documentation-only successor changes README.md and its manifest entry; scientific inputs and consumer code are byte-identical. [TRANSFER_FINAL.json](../../../results/v17/closeout-1/TRANSFER_FINAL.json) binds that repair and actual execution. Use reader-final.zip for delivery. Evaluator SOURCE.json is original lock provenance, not a substitute for the final selection rule above.

The [original pilot handoff](TRANSFER_PILOT_2026-09-12.md) is retained as a dated historical record.
