"""Analytic raw log-gain support for the finite repair generators.

These are mathematical worst-case bounds, never development-data minima. For a
gain log(p_mechanism / p_rival), p_rival <= 1 gives gain >= log(p_mechanism).
Mixtures of probabilities preserve a componentwise lower bound. Loose bounds
may produce weak evidence; they are never tightened by looking at outcomes.
"""
import math
from . import registry


def repair_bound(family):
    if family == 'R1':
        # Q has uniform prior. Its only channels are own (.25..75) and claim
        # (.10..90). Selection/operations update G/law but not Q directly.
        lower = math.log(.25*.10/(.25*.10+.75*.90))
        proof = 'Condition on G/law: own likelihood ratio <=3 and claim ratio <=9; posterior odds <=27. Mixing preserves 1/28.'
    elif family == 'R2':
        # Sum over other heads and law. At most 32 flag plus 32 purpose marks;
        # each factor >= .05 (fractional likelihood powers in [0,1]). All
        # other head evidence cancels from Q conditional on law; prior >=.2.
        lower = math.log(.2)+64*math.log(.05)
        proof = 'Conditional on law, purpose prior >=.2 and at most 64 purpose/flag likelihood factors >=.05; evidence normalizer <=1. Other-head observations are conditionally independent of purpose. Law mixing preserves the bound.'
    elif family == 'R3':
        # Softmax cancels the common pre-choice drive. d<=4, n<=4,m=2,
        # |target-h-effect|<=4.75, after-drive <=sqrt(4*4.75**4).
        lower = -3*(math.sqrt(4*4.75**4)+1)-math.log(3)
        proof = 'Every future-choice logit lies in [-3*(sqrt(4*4.75^4)+1),0] after cancelling common drive; three-option softmax and every posterior mixture preserve exp(lower). Includes irrelevant-role effort weighting.'
    elif family == 'R4':
        # One complete assignment for either root value suffices. 121 nodes,
        # 120 transitions >=.02, 81 leaf emissions >=.1, cue emission >=.01.
        lower = math.log(.5)+120*math.log(.02)+81*math.log(.1)+math.log(.01)
        proof = 'For either root value choose any full assignment: prior .5, at most 120 transitions >=.02, 81 leaf factors >=.1 and one cue >=.01. Joint evidence <=1, so this assignment lower-bounds posterior root mass. All allowed laws/cue positions covered.'
    else: raise ValueError('no support proof for '+family)
    return dict(family=family, contrast=list(registry.PRIMARY[family][:2]), lower_bound=lower,
        target='uncapped raw mean log-loss gain', proof=proof, empirical_floor=False,
        sampler='independent blocks of sixteen laws, one fresh world per law; average complete block before betting',
        limitations='Conservative support can make confirmation underpowered. No clipping, smoothing or data-derived tightening is authorized.')


def verify(plan):
    if plan.get('confirmation_support') != repair_bound(plan['family']):
        raise ValueError('missing or altered family support proof')
