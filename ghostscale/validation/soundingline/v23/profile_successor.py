"""Frozen D5 successor to the failed Stage 15 profile witness."""
from . import profile_validation as base

TAG = 'certified-fit-1'
TAG_OBJECT = 'cebcbd1ac45216ed38810094491da9385ce10c98'
ENGINE_COMMIT = '5f3ad0a304727fff5e1a6beca592e786ee71b181'
ENGINE_IMPLEMENTATION = '25801f9f81ecbce7a4c47397eb957eff528a537942ad86c159906841f44ffb12'
CERTIFICATION_SHA256 = '46ff69cbae7094ee8c41b8006464219bb1de70c29f9b268228f073b666e17253'


def configuration():
    return dict(study='S15-PROFILE-FEATURE-V2', worlds=40, writers=8, rounds=16,
        conditions=list(base.CONDITIONS), seed_namespace='ghost-v23-s15-profile-successor-v1',
        task_dimensions=9, profile_dimensions=32, option_count=4, feature_columns=164,
        signal_probability_range=[.70,.95], profile_noise_words=[0,1,2],
        matched_donor='fixed random writer derangement, equal opportunity/history length',
        severity='independent world draws: signal strength, type/action mapping, hash collisions, nuisance words, task context, resolution delays and writer order',
        fitting='exact sibling predict_all with certified-fit-1 Backbone; resolved strictly before target opens',
        score='mean paired half-Brier gain per independent world; all chronological targets included',
        interval='two-sided Student t 95% interval across 40 independent world means, descriptive finite generator support',
        positive_rule='both planted mean gain lower bounds > 0.005 against baseline and donor',
        null_rule='both own-minus-comparator gain upper bounds <= 0.005 in each of 40 prospectively fixed uninformative and constant worlds',
        null_precision='40 worlds per null condition, fixed before sampling; no optional extension',
        structural_rule='constant own/donor predictions equal <=1e-12; future-label prefix differences <=1e-7; option-row-equivariance differences <=1e-9; exact strict train-ID sets; finite normalized probabilities',
        margin=.005, probability_tolerance=1e-9, future_prefix=48,
        structural_worlds=list(range(40)), option_control_condition='planted',
        pilot_seed_namespace='ghost-v23-s15-profile-successor-cost-v1',
        pilot_excluded=True, no_adaptive_stopping=True, no_empirical_or_values_claim=True)


def verify_configuration(plan):
    if plan['configuration'] != configuration():
        raise ValueError('frozen profile successor configuration differs')
    if plan['sounding_line_plan_sha256'] != base.SL_PLAN:
        raise ValueError('wrong Sounding Line comparison')
    if plan['certified_fit_tag'] != TAG or plan['certified_fit_tag_object'] != TAG_OBJECT:
        raise ValueError('wrong certified fitting tag')
    if plan['engine_commit'] != ENGINE_COMMIT:
        raise ValueError('wrong certified engine commit')
    if plan['implementation_hash'] != ENGINE_IMPLEMENTATION or plan['ledger_schema_hash'] != base.SCHEMA:
        raise ValueError('wrong certified engine interface')
    if plan['certification_sha256'] != CERTIFICATION_SHA256:
        raise ValueError('wrong certification evidence')
    worlds=plan.get('world_indices',[])
    if plan['task']=='profile-successor-pilot':
        if worlds != [1000]: raise ValueError('wrong cost-pilot world')
    elif plan['task']=='profile-successor-part':
        if not worlds or worlds != sorted(set(worlds)) or min(worlds)<0 or max(worlds)>=40:
            raise ValueError('invalid successor partition')
    else: raise ValueError('wrong successor task')
    if plan['budget_group']!='repairs' or plan.get('protected'):
        raise ValueError('successor must use unprotected repairs allocation')
    if plan['cpu_cap_seconds'] > 14400:
        raise ValueError('packet CPU cap exceeded')


def run(root, plan, pulse):
    return base.run(root,plan,pulse)
