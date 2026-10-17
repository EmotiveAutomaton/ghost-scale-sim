# T3 Wikipedia source reference: source-identity review required

`MANIFEST_T3.json` authenticates `T3.json` and `T3_SOURCE.json`. The source file
contains aggregate distributions, article revision anchors and hashes binding
the complete 105-article API metadata snapshot. It contains no raw revision
text, editor IDs or comments. The input has 22,217 revisions and 8,408 observed
editors. Returns to older hashes and the fixed nonautomatic-summary heuristic
are descriptive proxies; hidden comments/hashes remain in the denominator.

**Required companion review:** `source.revision` is the actual 40-character
Wikipedia content SHA-1 of the named Empathy anchor revision. It is not a Git
commit. The full snapshot is separately bound by SHA-256 and per-article API
revision anchors. Verify this source convention is acceptable under the twin
contract; if the consumer requires Git specifically, retain the reference as
pending rather than treating string-length validation as scientific admission.

This supplies the requested source-shape input. It does not heal the failed
profile witness or authorize new computation, spending, module promotion or
a human-mechanism claim. T2 still awaits bounded chess parsing; T4's 104-rater
sensitivity still needs the actual exclusion roster. This file supplements the
earlier T1/T4 delivery without replacing those source bytes or manifests.
