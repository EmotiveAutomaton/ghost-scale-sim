import pytest
from ghostscale.validation.soundingline.v19 import crossed_review as R


def test_known_answer_controls():
    assert all(R.controls().values())


def test_support_is_not_finite_loss():
    result = R.score({'a':.25,'b':.75},{'a':1.})
    assert result['coverage'] == .25
    assert result['infinite_loss_mass'] == .75
    assert result['finite_loss_contribution'] == 0
    assert not result['abstain']


def test_population_and_corruption():
    from ghostscale.validation.soundingline.v19 import crossed_rules as producer, local_world
    rr = producer.enumerate_rule(local_world.law(190000),'both')
    groups, packets, error = R.population(rr,190000,'both')
    assert error < 1e-14 and len(groups) == 4 and packets
    rr[0]['probability'] *= 2
    with pytest.raises(ValueError):
        R.population(rr,190000,'both')


def test_union_prior_and_zero_support():
    groups = [{'x':{'a':2.}},{'x':{'b':1.}}]
    assert R.union(groups,'x',(0,1)) == {'a':2/3,'b':1/3}
    assert R.union(groups,'missing',(0,1)) == {}


def test_complete_native_checker_fixture(tmp_path):
    from ghostscale.validation.soundingline.v19 import crossed_rules as producer
    from ghostscale.validation.soundingline.v18_3.io import write, file_digest
    root = tmp_path/'review'; original = root/'inputs/original'; original.mkdir(parents=True)
    design = dict(lineages=[190000], rules=list(producer.RULES), arms=list(producer.ARMS))
    write(original/'PLAN.json', {'design':design})
    summary = producer.run(original, {'design':design}, lambda **kwargs:None)
    write(original/'SUMMARY.json', summary)
    pins = {p.relative_to(root/'inputs').as_posix():file_digest(p) for p in original.rglob('*') if p.is_file()}
    result = R.run(root, {'design':dict(input_files=pins,target_plan_sha256=file_digest(original/'PLAN.json'),bootstrap_seed=190501,bootstrap_resamples=100)}, lambda **kwargs:None)
    assert all(result['controls'].values()) and result['paths'] == 55296
    assert result['rows'] == summary['rows'] and result['cells'] == 96
