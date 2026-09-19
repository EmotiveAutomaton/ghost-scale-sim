# Synthetic maker-state bridge, revision 1

[reader.zip](reader.zip) contains 12 fixed illustration families, each with stable,
changed-goal, reader-correction and maker-correction siblings: 48 JSON requests
and their text renderings. Each includes chronology, declared observation access,
stable identifiers, a source lineage and input hash. All histories replayed.

[evaluator.zip](evaluator.zip) is a separate **truth-bearing evaluator archive**.
Do not send it to a blind reader. The [manifest](MANIFEST.json) binds both archives.
The [equivalence-class addendum](EQUIVALENCE_EVALUATOR.json) is also evaluator-only;
the blind reader archive and its original hash are unchanged.
These selected fixtures are independent of the performance sample and cannot be
counted as 48 independent maker histories. This repository handoff requires no
other operator to wait or change their run.
