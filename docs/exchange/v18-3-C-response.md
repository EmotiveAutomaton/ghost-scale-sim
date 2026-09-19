# V18.3 C: Corroboration and source dependence

When do repeated reports become misleading corroboration? With two underlying sources, six reports per source and a shared error, treating all reports as independent produced logarithmic loss 3.509 and assigned over 95% probability to the wrong answer in 21.88% of networks. A reader allowing uncertain source groups and shared error reduced loss to 0.643 and had no such initial false-confidence cases; after two noisy independent corrections, its rate was 3.13%. This is descriptive evidence about constructed provenance mechanisms and finite inference methods. Dependence modeling helps in this construction; it does not make uncertain provenance or correction infallible.

The table fixes two underlying roots and six descendants per root, except the independent condition, which instead contains twelve independent roots. It reports initial logarithmic loss; lower is better. The known-graph reader also knows the true bias/selection mechanism and is a privileged comparison.

| Source mechanism | Treat reports independently | Known graph/mechanism | Uncertain graph plus shared error |
|---|---:|---:|---:|
| independent | 0.08988 | 0.08988 | 0.29665 |
| copied | 0.73468 | 0.30541 | 0.42109 |
| shared-error | 3.50937 | 0.64559 | 0.64263 |
| selected | 0.60115 | 0.52137 | 0.48024 |
| partial | 1.59801 | 0.53799 | 0.53776 |
| wrong-provenance | 0.32533 | 0.31759 | 0.50003 |

Six packets retain 1,728 source-network evaluations: six mechanisms, root counts one/two/four, one/three/six reports per root, and 32 network draws per condition. The root truth and independent corrections are paired across descendant counts. Reports descending from a root are not independent samples. Under ordinary copied reports, expanding two roots from one to six reports each changes the independence-assuming loss from 0.309 to 0.735, while the known-graph loss stays near 0.305. More reports can still help recover a noisy root; they do not create fresh root evidence.

False confidence means more than 95% posterior probability on the wrong binary claim. Independent corrections themselves have reliability 0.9, so a correction can be wrong. Graph uncertainty is a finite catalog of block and cyclic partitions, not unrestricted provenance discovery. Partial or false source metadata constrains this catalog; inconsistent cases retain explicit invalid denominators. Selection-aware and cautious readers introduce additional assumptions and do not uniformly win. All retained posterior/loss checks and aggregate means passed; 48 fixed source-extracted whole-network replays passed.

[Complete current record](../versions/v18-selective-acquisition/research-extension/RESULTS.md). V18.3 remains active.
