# N18: play inversion and information, declared before sampling

Sixteen independent constructed laws, each with norm in {-1,+1} and a noise
amplitude drawn uniformly from [0.15,0.45]. Maker parameter theta is equiprobable
{-1,+1}. A non-inverted record has Bernoulli probability
0.5 + amplitude * (s*norm + (1-s)*theta), for s=0.2,0.5,0.8.
The law (including norm/noise) is known to the reader; theta is unknown.

For s=0.2, a fair independent inversion flips the literal binary record. The
detector reports that inversion with symmetric accuracy a=0.6,0.8,0.95,1.0.
The detector reader corrects the record using its report and marginalizes its
remaining error. The ignore-frame reader sees the fair mixture (zero maker
information); the oracle knows the inversion. Nonplay records are not inverted.
This accuracy is a symmetric binary-channel assumption, not arbitrary classifier
accuracy under unknown prevalence. Each record includes all enumerated joint
probabilities, so exact mutual information can be reconstructed independently.

Primary: per-law bisection threshold a* in [0.5,1] at equality between detector
play information and s=0.8 information; strictly above the threshold is the
advantage. Report threshold distribution and 95% bootstrap intervals over laws.
Report nats per record for every strength, accuracy and reader. The detector's
flag is included in the observed channel; its independence from theta makes the
corrected-record marginal sufficient here. Use 16 laws, with 64 repeated
enumeration records per law to fit the inherited allocation contract; these
duplicates provide **no additional inferential sample size**. Only 16 laws
enter intervals. No Monte Carlo substitute for the finite entropy sum.

Known answers: entropy and direct joint sums agree; s=1 gives no theta
information; a=0.5 gives zero play information; a=1 equals oracle; probabilities
normalize; detector information is monotone in a. Pilot/full replay, frozen
analytic reader definitions, raw channels and independent interval reconstruction
are required. Cap 1.5 CPU-hours including all verification; no controls shed.

Weak situations expose theta more by construction. The useful conditional result
is the detector threshold, not evidence that human play necessarily does so.
Route: the engine may increase play weights only if a separately measured,
appropriately calibrated detector exceeds the threshold under defensible laws.
