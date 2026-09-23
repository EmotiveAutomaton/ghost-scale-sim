# Continuous reliability: executable certificate contract

This implements the prepared continuous-reliability question without changing
its 1,800 CPU-second inclusive cap. No result is accepted by this document.
Use all 640 queries, eight existing development laws, both tool rules and both
native/uniform supplied conditional models from the verified disclosure inputs.
Every visible group, both requested fields and both replies are enumerated.
No fit, new episode, outcome-selected population or protected lineage is used.

For one group and reply, let the eight-vector h contain the joint masses of
endpoints whose hidden bit matches the reply; let m contain the nonmatching
masses. Both are nonnegative. Let H and M be their sums. The unnormalized
endpoint vector is r h + (1-r)m; its mass is Z(r)=rH+(1-r)M.
At r=1/2 the conditional vector is p0=(h+m)/(H+M). If H>0, let p1=h/H.
Writing t=2r-1, Z=(1-t)(H+M)/2+tH, so

    p(r) = (1-a) p0 + a p1, where a=tH/Z.

For 1/2 <= r <= 1 with positive Z, 0 <= a <= 1. It varies continuously
from zero to one when H>0. Thus the compatible forecasts form exactly the
closed endpoint segment, and the coordinate extrema and total-variation
diameter are attained at those endpoints. This algebra concerns real-valued
nonnegative masses. Binary64 equality and a 1e-12 numerical tolerance remain
separate implementation checks.

If H=0, r=1 is impossible. Every r<1 gives m/M, whose limit is also m/M.
No fallback at the impossible endpoint enters the compatible set. The half
endpoint is possible because H+M>0. If M=0, all conditionals are h/H.
No endpoint probability envelope is treated as a normalized point forecast.

Retain h, m, H, M, endpoint vectors, compatibility flags, left limits,
coordinate extrema and diameters for all cases. Independently check exact
rational fixtures, constant/disjoint supports, singular endpoints, invalid
inputs and deliberate erroneous endpoint inclusion. Evaluate a fixed grid
r=i/32 for i=16,...,32 using a separately written scalar Bayes formula;
this grid validates numerical implementation and is not the continuum proof.
Store all grid outputs and interpolation errors. Require original, adjacent
and extracted-source replay and independent scalar reconstruction before
scientific acceptance. Failed attempts count against the same cap.

Reader packets contain only initial artifact, proposed operations, requested
field, reply and reliability interval. Laws, weights, true bits and numerical
certificates are evaluator evidence. No new reliability or decision ranking,
learned access, historical process correspondence or human intent is claimed.

The exhaustive retrospective-evidence design remains independently prepared.
The joint-process support-decoder protocol provides a second independent
alternative aimed at the primary readout's poor process compatibility.
