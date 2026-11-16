# M: prospective colored-remainder calibration

Question: how many messages and independent makers expose a clock-time carry
component once nuisance means are removed? COLORED follows an Ornstein–Uhlenbeck
state, WHITE refreshes each message, WHITE_IN_THREAD refreshes only at thread
boundaries, and PROMPTED refreshes with probability 1/8 per message. Positive
message gaps are lognormal with median 25 minutes and log SD 1.5, as in the
supplied EN1 fixture. A new thread begins with probability 0.2 per message;
thread effects have SD 0.5 and a daily sine has amplitude 0.5.

The latent remainder is sqrt(q) times unit-variance state plus sqrt(1-q) fresh
noise. The reader has correlation m with that remainder through independent
unit-variance measurement noise. Thus observable covariance is m²q exp(-gap/tau).
Primary nuisance removal uses the known thread/rhythm contributions: an oracle
calibration ceiling. It does not validate nuisance estimation on real messages.

Retain all message times, thread identifiers, observed values and nuisance terms.
Compute all ordered-in-time pairs, and separately cross-thread pairs, in clock
bins under15,15–60,60–240,240+ minutes and message-distance bins1,2–3,4–10,11+.
Pairs are dependent. Ten delete-group maker blocks estimate each world's
correlation uncertainty; pair counts are exposure counts, never sample sizes.
Report positive short-gap and fading detection rates with null false-positive
rates. The independent reconstruction uses explicit triangular pair arrays.

Also compute exact two-law COLORED-versus-WHITE likelihoods per maker using a
scalar Kalman filter, independently reconstructed in precision form and checked
against dense multivariate Gaussian likelihoods. This oracle knows q,tau,m.
Report three-nat evidence shares and finite-horizon sustained-crossing bounds.
PROMPTED and WHITE_IN_THREAD are adversarial controls, not fitted alternatives
inside this two-law diagnostic. Correlation-versus-clock/message profiles test
their difference descriptively; no false promise of a unique human signature.

The packet Cartesian roster has 1,296 labels. Exact law aliases coalesce:
WHITE is independent of q,tau,m; WHITE_IN_THREAD and PROMPTED ignore tau.
Retain a label-to-canonical map and run 552 distinct laws, with 40 excluded
pilot worlds and 200 worlds in eight blocks per cell. Full replay and independent
reconstruction remain mandatory. Stop at the last complete cell within15CPUh.

Sounding Line EN1 receives pair counts, power/false-positive curves and the
oracle-nuisance qualification. ToMpathy receives per-maker message requirements.
These are finite constructed-method calibrations, not observations about people.
