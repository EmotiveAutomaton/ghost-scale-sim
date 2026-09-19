# V18.3 completed research extension

All eight families A-H are executed. Read the [final report](../../../docs/versions/v18-selective-acquisition/research-extension/FINAL_REPORT.md), [coverage ledger](COVERAGE.json), [scope disposition](SPEC_COVERAGE.json) and [closeout](CLOSEOUT.json).

Current finite evidence comprises 43 packets and 19,136 assigned evaluations. Four retained superseded A packets add 1,088 evaluations. These counts are not independent sample sizes. E has 18 current training fits, frozen-purpose readouts on reused data, and a separately retained scoring repair; G has ten fits. Failed E work and the unexecuted earlier G plan remain retained. Earlier versions remain unchanged.

Each finite packet contains PLAN.json, SOURCE.zip, SUMMARY.json, COMPLETE.json, all raw blocks in RAW.zip, an independently bound REPLAY_DRIVER.py and INDEPENDENT_REPLAY.json. Extract RAW.zip beside PLAN.json; extract SOURCE.zip into an isolated source checkout. Replay validates all retained means and eight fixed units, not full campaign regeneration.

Neural packets contain an ordinary ZIP split into binary parts. BUNDLE.json lists their order and checksums; SCIENTIFIC_MANIFEST.json identifies every uncompressed scientific file. Use the public `runners.export_v18_3_neural.unpack` function, or concatenate parts in the listed order and extract the ZIP. All source, serialized inputs, selected weights and forecasts are retained. Evaluator truth is present for audit and must remain outside reader inputs. Neural replay uses the recorded optional CPU PyTorch environment and selected weights; it does not retrain every fit. Machine-local ownership, PIDs and credentials are excluded.

`review/` is the current source-bound numerical review. `review-initial/` and `review-purpose-repaired/` preserve superseded diagnostics. E_PURPOSE_REPAIR.json maps original and current accuracy scoring; proper losses did not change. `update-costs/` independently binds the final single-observation update benchmark. Figures are standalone PNG/SVG artifacts; the SVGs retain generator formatting and immutable hashes.

All 45 scoped controls pass: 37 Ghost and eight actual optional PyTorch checks. Four Torch modules skip in the base interpreter and execute in the optional run. Every current scientific result has full documentary write-through. The repository-wide V15 C11/M01 gate failure remains historical debt. EXPORT_INTEGRITY.json covers the main packet archive; final closeout additionally binds the update-cost export and report artifacts.
