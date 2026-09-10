# V16 public task bundle

Extract reader.zip into a new empty directory. Its payload is the corrected transfer-fixture-2 public whitelist: 44 recorded synthetic cases and 1,574 public reader tasks. Evaluator mappings and committed predictions are retained separately.

From public/consumer, run `python -s -B -u -m consumer` in a Python environment with NumPy. Send one public observation JSON object per input line. The first output is the ready/source receipt; subsequent outputs are predictions. Send `{"operation":"shutdown"}` to stop. Neither Ghost Scale nor Sounding Line needs to be installed.

See docs/versions/v16-acquired-craft/TRANSFER.md in the repository for the evidence interface and real-record limits. This is an executed synthetic task interface, not a deployed Sounding Line reader or evidence of human capability.
