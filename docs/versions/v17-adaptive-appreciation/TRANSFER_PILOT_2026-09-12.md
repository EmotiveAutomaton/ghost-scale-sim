Historical pilot handoff; superseded by [the final delivery](TRANSFER.md) on 18 September 2026. Original text follows.

# V17 transfer status

The [early reader-only pilot](../../../results/v17/reader-pilot-1/reader.zip) is
available: eight new cases generated independently of V16, two from each native A?D
family, exposed through three evidence tiers for 24 requests. This is a discarded
development pilot, not fresh confirmation or the final illustrative challenge.

Extract the reader ZIP and run `python -B consumer.py < requests.jsonl > predictions.jsonl`.
The standard-library consumer predicts the maker's unseen next choice after
the declared intervention. Each request declares finite answer support, permitted
evidence and query cost. Actual goals and maker policies are hidden; possible
native model families and task opportunities are public. Artifact, earlier-object
and recorded-process tiers are distinct assistance levels.

The [separate evaluator ZIP](../../../results/v17/reader-pilot-1/evaluator.zip)
contains private truth and references. Keep it out of blind reader evidence.
[The portability receipt](../../../results/v17/reader-pilot-1/PORTABILITY.json)
records an actual extracted run with an empty PYTHONPATH: all 24 forecasts match,
and a real private-file read was denied after trusted imports. That guard protects
this fixed Python consumer; it is not a general sandbox for hostile native code.
EXPORT.json preserves its original packaging-time status, before this execution.

Pilot selection took the first two completed cases per native family without
outcome ranking. Final selection spanning advantages, reversals, failures and
the five commissioned illustrative types remains to be done; unavailable types
must be marked absent.

The included `ghostscale.transfer.adapter.1` envelope preserves old
`v16.transfer.1` payloads unchanged. The completed [V16 handoff](../v16-acquired-craft/TRANSFER.md)
remains usable. The earlier [V17 evaluator snapshot](../../../results/v17/initial-evaluator-snapshot/snapshot.zip)
contains private construction truth and is unsuitable for blind evidence.

This is a repository artifact, not an outbound communication or a deployment
into Sounding Line. No sibling source, environment, lock or queue was changed.
