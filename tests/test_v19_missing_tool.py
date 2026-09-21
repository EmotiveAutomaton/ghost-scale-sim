from copy import deepcopy
import gzip
import json
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import missing_tool as M
from ghostscale.validation.soundingline.v19 import local_world as L


def test_known_answer_controls():
    assert all(M.controls().values())
    with pytest.raises(ValueError):M.execute((0,0,0),(0,0,0),'inspect',(0,0,0,0),'unadmitted')


@pytest.fixture(scope='module')
def populations():
    w=L.law(190971)
    return [M.enumerate_rule(w,r) for r in M.RULES]


def test_original_enumerator_and_natural_mass(populations):
    assert populations[0]==L.enumerate_world(L.law(190971))
    for records in populations:
        assert len(records)==13824 and np.isclose(sum(r['probability'] for r in records),1)
        assert all(set(r['final'])<={0,1} for r in records)


def test_inspect_identity_and_changed_tool_impossibility(populations):
    old,new=populations
    def inspect(rr):return [r for r in rr if all(e['operation']=='inspect' for e in r['steps'])]
    assert inspect(old)==inspect(new)
    groups=[M.grouped(rr,'E2-full')[0] for rr in populations]
    r=next(r for r in new if any(e['operation']=='accept-tool' for e in r['steps']))
    key=M.digest(L.project(r,'E2-full'))
    assert key not in groups[0]
    pred,p=M.posterior(groups,key,'old-exact',1);assert not pred and not p.any()
    pred,p=M.posterior(groups,key,'expanded-exact',1);assert pred and p[1]==1


def test_exact_uniform_and_missing_probability():
    a=(('meaning',)*3,('inspect',)*3);b=(('dependency',)*3,('inspect',)*3)
    groups=[{'x':{'mass':{a:.1,b:.3},'count':{a:1.,b:1.}}},{'x':{'mass':{a:.6},'count':{a:1.}}}]
    p,laws=M.posterior(groups,'x','expanded-exact',1)
    assert np.allclose([p[a],p[b]], [.7,.3]) and np.allclose(laws,[.4,.6])
    p,_=M.posterior(groups,'x','expanded-template',1);assert np.isclose(p[a],2/3)
    s=M.score({a:.3,b:.7},{a:1.});assert s['infinite_loss_mass']==.7 and s['finite_loss_contribution']==0
    with pytest.raises(ValueError):M.score({a:.5},{a:1.})


def test_complete_fixture_and_public_roles(tmp_path):
    cfg=dict(lineages=[190972],rules=list(M.RULES),tiers=list(M.TIERS),arms=list(M.ARMS))
    summary=M.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert all(summary['controls'].values()) and summary['paths']==27648
    for packet in json.loads((tmp_path/'PUBLIC_PACKET.json').read_text())['frames'].values():L.validate_public(packet)
    rows=json.loads(gzip.decompress((tmp_path/'raw/missing-tool_points.json.gz').read_bytes()))
    assert len(rows)==summary['rows']
    assert len(summary['cells'])==32
    assert all(r['coverage']==1 for r in rows if r['arm']=='known-rule-oracle')
