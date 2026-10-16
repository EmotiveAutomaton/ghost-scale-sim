"""Generator-derived raw-gain support; never an observed minimum or score floor."""
import math
from .registry import PRIMARY


def bound(family):
    if family=='N1':
        lower=-16-math.log(3)
        proof='At most two nonnegative recipe weights of at most 2, binary conjunctive satisfaction in [0,1], and precision at most 4. Three choice logits have range at most 16; every posterior mixture preserves exp(-16)/3. A whole-population average preserves the same raw-gain bound. Confirmation must separately balance all 32 culture candidates.'
    elif family=='N2':
        lower=-512-math.log(2)
        proof='Uniform binary prior, at most 128 marks, beta at most 4, and fixed learned weights in [0,1]. Absolute weighted log odds are at most 512, so either posterior probability is at least exp(-512)/2.'
    elif family=='N5':
        lower=-math.log(4)-math.log(10)-3
        proof='Calibrated projection cancels the own-policy offset. Base masses are .05 through .50; public quality is within [-1.5,1.5] and quality coefficients within [-1,1]. Four logits have range at most log(10)+3; every normalized class mass exceeds exp(lower). No approximate hierarchical ceiling enters this contrast.'
    elif family=='N4':
        lower=-72-math.log(4)
        proof='At most three nonnegative recipe weights at most 2, satisfaction at most 2 after need scaling, beta at most 3 and vigor at most 2. Utility lies in [0,12]; four logits have range at most 72. Every finite posterior mixture preserves exp(-72)/4. Conditional binary N4b is excluded.'
    elif family=='N6':
        lower=math.log(.05**2)
        proof='Every four-way future action is the product of two Bernoulli factors in [.05,.95]. Every posterior mixture, including within the predeclared low-low target stratum, assigns at least .0025 to each action.'
    elif family=='N7':
        lower=-96-math.log(3)
        proof='There are at most two recipes of weight 2. Six recipe coordinates have nonzero statement-shift direction, shift magnitude at most 2. Absolute future utility is at most 4+6*2=16; beta at most 3. The range of three logits is at most 96. Every finite posterior mixture preserves exp(-96)/3 for each future choice.'
    elif family=='N8':
        lower=-32*math.log(19)-math.log(2)
        proof='Conditional on a law, preference prior is uniform. Each of at most 32 marks has a likelihood ratio at most .95/.05=19: mixing play classes with the same observed-signal weights preserves this bound. Either preference probability is at least exp(-32*log(19))/2; mixing laws preserves it.'
    elif family=='N3':
        return dict(family=family,available=False,contrast=list(PRIMARY[family][:2]),
            reason='Literal hard constraints can have empty support or exclude the target. Raw primary gain has no finite guaranteed lower bound; unavailable and infinite predictions are retained, never floored.',
            confirmation_available=False,empirical_floor=False)
    else:raise ValueError('no checked support proof for '+family)
    return dict(family=family,available=True,contrast=list(PRIMARY[family][:2]),lower_bound=lower,
        target='uncapped raw mean log-loss gain',proof=proof,empirical_floor=False,
        sampler='independent complete balanced law blocks; fresh worlds in each block',
        limitations='Conservative bound may yield weak evidence; confirmation requires separate source-bound admission and complete frozen nomination order.')


def verify(plan):
    expected=bound(plan['family'])
    if plan.get('confirmation_support')!=expected:raise ValueError('missing or altered N-family support proof')
    if plan.get('task')=='confirmation' and not expected['available']:
        raise ValueError('no finite raw-gain support for confirmation')
