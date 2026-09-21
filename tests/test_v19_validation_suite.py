import pytest
from ghostscale.validation.soundingline.v19.validation_suite import validate_design


def test_validation_inventory_rejects_arbitrary_or_missing_tests(tmp_path):
    (tmp_path/'tests').mkdir();(tmp_path/'tests/test_v19_fixture.py').write_text('')
    cfg=dict(tests=['tests/test_v19_fixture.py'],successors=[],protected_lineages=[])
    assert validate_design(cfg,tmp_path)
    with pytest.raises(ValueError):validate_design(dict(cfg,tests=['../test_v19_evil.py']),tmp_path)
    with pytest.raises(ValueError):validate_design(dict(cfg,tests=['tests/test_v19_missing.py']),tmp_path)
    with pytest.raises(ValueError):validate_design(dict(cfg,tests=cfg['tests']*2),tmp_path)


def test_validation_cannot_release_conditional_or_protected_science(tmp_path):
    (tmp_path/'tests').mkdir();(tmp_path/'tests/test_v19_fixture.py').write_text('')
    cfg=dict(tests=['tests/test_v19_fixture.py'],successors=[dict(name='G19-D-crossed-rules-1',design=dict(handler='crossed-rules',lineages=[190000]))],protected_lineages=[191000])
    assert validate_design(cfg,tmp_path)
    cfg['successors'][0]['design']['lineages']=[191000]
    with pytest.raises(ValueError):validate_design(cfg,tmp_path)
    cfg['successors'][0]['design'].update(handler='rollout-transfer',lineages=[190000])
    with pytest.raises(ValueError):validate_design(cfg,tmp_path)
