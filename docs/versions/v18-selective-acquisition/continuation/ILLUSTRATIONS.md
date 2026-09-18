# Eight selected walkthrough illustrations

These are purpose-selected examples from a separate namespace and physical contexts, not the 72-case blind benchmark. All eight histories were independently executed.

The table names the physical family, context condition, true executed strategy and the probability assigned to that strategy from endpoint, context and process evidence.

| Example | Family | Context | Actual strategy | Endpoint probability | Context probability | Process probability |
|---|---|---|---|---:|---:|---:|
| 1 | assembly | positive | routine | 0.25 | 0.50 | 1.00 |
| 2 | assembly | positive | episode | 0.25 | 0.50 | 1.00 |
| 3 | assembly | ambiguous | primitive | 0.25 | 0.25 | 0.25 |
| 4 | assembly | misleading | adaptation | 0.25 | 0.50 | 1.00 |
| 5 | graphic | positive | routine | 0.25 | 0.25 | 1.00 |
| 6 | graphic | positive | adaptation | 0.25 | 0.50 | 1.00 |
| 7 | graphic | ambiguous | primitive | 0.25 | 0.25 | 0.25 |
| 8 | graphic | misleading | episode | 0.25 | 0.25 | 1.00 |

## Example 1: positive assembly context

Initial state `[0, 0, 1]`; final state `[1, 0, 1]`. Actual actions `[4, 5, 6, 1, 2, 9]`.
Earlier artifact `[0, -1, 1]`; observed process `[{"action": 4, "actor": "tool-A", "index": 0}, {"action": 5, "actor": "tool-B", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[1.0, 0.0, 0.0, 0.0]`.
The process identifies the executed method within the supplied finite family.

## Example 2: positive assembly context

Initial state `[1, 0, 1]`; final state `[0, 0, 1]`. Actual actions `[7, 4, 5, 6, 1, 2, 9]`.
Earlier artifact `[1, 1, 1]`; observed process `[{"action": 7, "actor": "tool-B", "index": 0}, {"action": 4, "actor": "tool-A", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.0, 0.0, 1.0, 0.0]`.
The process identifies the executed method within the supplied finite family.

## Example 3: ambiguous assembly context

Initial state `[0, 0, 1]`; final state `[1, 0, 1]`. Actual actions `[5, 4, 6, 1, 2, 9]`.
Earlier artifact `[1, 0, 1]`; observed process `[{"action": 6, "actor": "tool-A", "index": null}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.25, 0.25, 0.25, 0.25]`.
The available evidence remains ambiguous.

## Example 4: misleading assembly context

Initial state `[0, 0, 1]`; final state `[1, 0, 1]`. Actual actions `[8, 7, 5, 4, 6, 1, 2, 9]`.
Earlier artifact `[0, -1, 1]`; observed process `[{"action": 8, "actor": "tool-A", "index": 0}, {"action": 7, "actor": "tool-B", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.0, 0.0, 0.0, 1.0]`.
The indexed process resolves the misleading earlier artifact.

## Example 5: positive graphic context

Initial state `0`; final state `1041`. Actual actions `[10, 0, 4]`.
Earlier artifact `1024`; observed process `[{"action": 10, "actor": "tool-A", "index": 0}, {"action": 0, "actor": "tool-A", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[1.0, 0.0, 0.0, 0.0]`.
The process identifies the executed method within the supplied finite family.

## Example 6: positive graphic context

Initial state `0`; final state `1296`. Actual actions `[8, 12, 28, 10, 4]`.
Earlier artifact `256`; observed process `[{"action": 8, "actor": "tool-A", "index": 0}, {"action": 12, "actor": "tool-A", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.0, 0.0, 0.0, 1.0]`.
The process identifies the executed method within the supplied finite family.

## Example 7: ambiguous graphic context

Initial state `0`; final state `7168`. Actual actions `[11, 12, 10]`.
Earlier artifact `7168`; observed process `[{"action": 10, "actor": "tool-A", "index": null}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.25, 0.25, 0.25, 0.25]`.
The available evidence remains ambiguous.

## Example 8: misleading graphic context

Initial state `0`; final state `32801`. Actual actions `[0, 5, 15]`.
Earlier artifact `32`; observed process `[{"action": 0, "actor": "tool-A", "index": 0}, {"action": 5, "actor": "tool-B", "index": 1}]`.
Full process posterior in routine/primitive/episode/adaptation order: `[0.0, 0.0, 1.0, 0.0]`.
The indexed process resolves the misleading earlier artifact.
