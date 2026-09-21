import gzip,json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import crossed_rules as C,missing_tool as M,local_world as L


def test_controls_and_complete_family_support():
    assert all(C.controls().values())
    target={('new',):1.};groups=[{'e':{'mass':{('old',):1.}}},{},{},{'e':{'mass':target}}]
    assert M.score(target,C.posterior(groups,'e',(0,1,2)))['coverage']==0
    assert M.score(target,C.posterior(groups,'e',(0,1,2,3)))['coverage']==1
    assert C.posterior(groups,'absent',(0,1,2,3))=={}


def test_original_and_single_tool_populations_unchanged():
    world=L.law(190961)
    for rule in ('original','presentation-tool'):
        actual=C.enumerate_rule(world,rule);expected=M.enumerate_rule(world,rule)
        assert actual==expected
        assert len(actual)==13824 and abs(sum(r['probability'] for r in actual)-1)<1e-10


def test_independent_changed_repair_and_privilege():
    for skill in (0,1):
        for belief in (0,1):
            a=(1,1,0);m=(0,skill,belief,0)
            assert C.execute(a,a,'repair-evidence',m,'both')==(1,belief if skill else 0,0)
    with pytest.raises(ValueError):C.execute((0,0,0),(0,0,0),'accept-tool',(0,0,0,0),'both')
    with pytest.raises(ValueError):C.execute((0,0,0),(0,0,0),'inspect',(0,1,0,0),'unknown')
    r=C.enumerate_rule(L.law(190962),'both')[0]
    for tier in M.TIERS:
        p=L.project(r,tier);assert L.validate_public(p)
        assert not {'maker','goal','rule','dependency_edges'} & set(p['inputs'])
        assert all('goal' not in e and 'dependency_edges' not in e for e in p['inputs'].get('observations',[]))


def test_complete_native_fixture(tmp_path):
    result=C.run(tmp_path,{'design':{'rules':list(C.RULES),'arms':list(C.ARMS),'lineages':[190963]}},lambda **kw:None)
    assert result['paths']==55296 and len(result['cells'])==96
    assert all(r['coverage']==1 for r in result['cells'] if r['arm'] in ('complete','known-rule-oracle'))
    points=json.loads(gzip.decompress((tmp_path/'raw/190963-E2-full-scores_points.json.gz').read_bytes()))
    assert {r['truth_rule'] for r in points}==set(C.RULES)
    assert all(-1e-12<=r['coverage']<=1+1e-12 for r in points)
    # Omission need not reduce process support: opposite belief can alias the
    # inverted repair. Missing support is tested on a known fixture above.


def test_inverted_repair_has_a_belief_alias_for_visible_execution():
    from itertools import product
    for a in product(range(2),repeat=3):
        for b in product(range(2),repeat=3):
            for belief in (0,1):
                maker=(0,1,belief,0);alias=(0,1,1-belief,0)
                for op in L.OPERATIONS:
                    assert C.execute(a,b,op,maker,'both')==C.execute(a,b,op,alias,'presentation-tool')
