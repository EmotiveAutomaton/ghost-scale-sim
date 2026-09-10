# V16 bounded replay bundle

Extract raw.zip into a new empty directory. Member paths recreate a small source checkout. Use a compatible isolated Python environment with the dependencies declared in pyproject.toml and uv.lock. The original environment must remain unchanged.

From the extracted checkout, run:

```text
python -B -m runners.replay_v16_bundle --root results/v16 --plan results/v16/replay-bundle-1/PLAN.json --output results/v16/portable-replay-attempt-1
```

The command checks every bundled dependency before regenerating the fixed cases. An existing output directory is refused. Timestamps, transport aliases and OS clock/RSS samples may differ; scientific state and scores must match. This bounded subset complements the complete raw archive and is not a claim that every campaign unit was replayed.
