"""V24 twin successor contract tests; no scientific sampling."""
import pytest

from ghostscale.validation.soundingline.v23 import registry
from ghostscale.validation.soundingline.v24.twin_successor import check_full_profile,validate_profile_plan


def test_full_twin_profile_refuses_reduced_or_partial_grids():
    expected=[f'T1-{i:02d}' for i in range(24)]
    rows=[dict(variant=dict(id=name),worlds=[dict(makers=3)]) for name in expected]
    result=check_full_profile(dict(variants=rows),expected)
    assert result==dict(state='FULL_PROFILE_COMPLETE',settings=24,reduced_grid=False,
                        confirmation=False)
    with pytest.raises(ValueError,match='complete fixed'):
        check_full_profile(dict(variants=rows[:-1]),expected)
    rows[0]['worlds']=[]
    with pytest.raises(ValueError,match='one complete world'):
        check_full_profile(dict(variants=rows),expected)


def test_full_twin_profile_refuses_empty_world():
    rows=[dict(variant=dict(id='T4-00'),worlds=[dict(makers=0)])]
    with pytest.raises(ValueError,match='partial or empty'):
        check_full_profile(dict(variants=rows),['T4-00'])


@pytest.mark.parametrize('family',['T1','T3','T4'])
def test_twin_profile_plan_requires_every_setting_and_one_excluded_world(family):
    variants=[v['id'] for v in registry.variants(family)]
    plan=dict(kind='pilot',task='v24-twin-profile',family=family,confirmation=False,
              production=False,variant_ids=variants,world_indices=[0],cpu_cap_seconds=7200)
    assert validate_profile_plan(plan)==variants
    with pytest.raises(ValueError,match='complete fixed'):
        validate_profile_plan(dict(plan,variant_ids=variants[:-1]))
    with pytest.raises(ValueError,match='one cost world'):
        validate_profile_plan(dict(plan,world_indices=[0,1]))


def test_common_v24_runner_routes_only_built_twin_successors():
    from runners.run_v24_pilot import implementation
    from ghostscale.validation.soundingline.v24 import twin_successor
    assert implementation({'family':'T1'}) is twin_successor
    assert implementation({'family':'T3'}) is twin_successor
    assert implementation({'family':'T4'}) is twin_successor
    from ghostscale.validation.soundingline.v24 import population_successor
    assert implementation({'family':'N1'}) is population_successor


def test_twin_preparation_rejects_non_twin_family(tmp_path):
    from runners.run_v24_pilot import prepare_twin
    with pytest.raises(ValueError,match='built twin'):
        prepare_twin(tmp_path,tmp_path/'unused.json','N1')
