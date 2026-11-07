"""Per-record interpretation decisions; prior science and curator status persist."""
from .registry import EARLIER

DISPOSITIONS={
 'G-V26-DAY-G4-1':'No changed reading yet: the earlier trade ladder has no decision times. K1 requires its own law choice; no timing covariate can be fabricated in the old data.',
 'G-V26-DAY-G2-1':'No changed reading: earlier framing records contain no decision-time channel, and K1 does not alter their frozen choice evidence.',
 'G-V26-DAY-G1-1':'No changed reading from the static comparisons: G1 measured triggers and feelings under a different consumption law. Dynamic K3 evidence is still pending; revisit this disposition when it lands.',
 'G-V26-DAY-G5-1':'No changed reading: G5 included surplus-spending actions. The static restore-only band comparison tests lower-edge grid coverage and does not contradict that design.',
 'G-V26-DAY-G3-1':'Potentially more informative conditional reading: K5 exposed the role of exactly balanced evidence at closure. K5-R1 is a separately frozen level-one retained-data reread with closure balance as a covariate; it reproduces the earlier scores before stratification and overwrites nothing.',
 'G-V26-DAY-G3-T09-1':'No changed reading: the expanded source-library result remains intact. K5 changes group pooling and priors as well, so its group contrast is not an isolated test of the earlier library effect. No claim is reversed.',
 'G-V26-H2-1':'No changed reading: H2 already integrated the joint hidden causes. Failure of a new iterative approximation to recover the joint gain does not invalidate the earlier exact reader.',
 'G-V26-H1-1':'No changed reading: H1 lacks the new marker channel and shared-goal checksum construction. Laughter measurements cannot be retrospectively fabricated in those records.'}


def dispositions(family):
    return {identity:DISPOSITIONS.get(identity,'Curator position, not a previous empirical result. Its recorded status and wording remain unchanged; this constructed comparison does not establish a human mechanism.') for identity in EARLIER[family]}


def prose(family):
    return ['Earlier-record dispositions (each identifier is a separate record):','']+[
        '- `'+key+'`: '+value for key,value in dispositions(family).items()]
