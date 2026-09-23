from fractions import Fraction
from itertools import product
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import continuous_reliability as C
from ghostscale.validation.soundingline.v18_3.io import read
from test_v19_joint_reply import joint_fixture

LEGAL = [(s,b,2,1,4,4) for s,b in product((0,1), repeat=2)]


def scalar(legal, ends, weights, axis, bit, r):
    terms = [float(w)*(float(r) if q[axis] == bit else 1-float(r)) for q,w in zip(legal,weights)]
    mass = math.fsum(terms)
    if mass == 0: return None, mass
    return np.array([math.fsum(w for w,e in zip(terms,ends) if e == j)/mass for j in range(8)]), mass


def test_exact_rational_live_null_and_known_fractional_linear_weights():
    assert all(C.controls().values())
    cert = C.certify(*C.coefficients(LEGAL, [0,1,2,3], [.125,.375,.25,.25], 'skill', 0))
    assert cert['hit_mass'] == .5
    p, mass, a = C.evaluate(cert, Fraction(3,4))
    assert mass == .5 and a == .5
    assert np.array_equal(p[:4], [.1875,.5625,.125,.125])
    # Non-balanced support tests denominator weighting rather than linear r weights.
    cert = C.certify(*C.coefficients(LEGAL, [0,1,2,3], [.125,.125,.25,.5], 'skill', 0))
    p, mass, a = C.evaluate(cert, .75)
    assert mass == .375 and a == pytest.approx(1/3)
    assert np.allclose(p, scalar(LEGAL,[0,1,2,3],[.125,.125,.25,.5],0,0,.75)[0], atol=1e-15)


def test_singular_limit_excludes_wrong_fallback_and_arbitrarily_close_endpoint():
    cert = C.certify(*C.coefficients(LEGAL, [0,1,2,3], [0,0,.25,.75], 'skill', 0))
    assert not cert['high_compatible'] and C.evaluate(cert,1) == (None,0.,None)
    for r in (.5, .75, np.nextafter(1.,0.)):
        p, mass, a = C.evaluate(cert,r)
        assert mass > 0 and a == 0 and np.array_equal(p[2:4],[.25,.75])
    # An arbitrary valid simplex vector at the impossible reply expands the set wrongly.
    wrong = np.zeros(8); wrong[0] = 1
    assert .5*np.abs(wrong-cert['low']).sum() == 1 and cert['diameter'] == 0


def test_zero_miss_mass_and_binary64_vs_tolerance_collapse():
    cert = C.certify(np.eye(8)[0],np.zeros(8))
    assert cert['collapsed_binary64'] and cert['high_compatible']
    h = np.eye(8)[0]*.5; m=h.copy();m[0]-=1e-14;m[1]+=1e-14
    c=C.certify(h,m)
    assert not c['collapsed_binary64'] and c['diameter'] < 1e-12


@pytest.mark.parametrize('defect',['negative','nan','empty','shape','weights','bit','endpoint','reliability'])
def test_invalid_inputs_rejected(defect):
    with pytest.raises(ValueError):
        if defect=='negative': C.certify([-1.]+[0.]*7,np.ones(8))
        elif defect=='nan': C.certify([float('nan')]*8,np.ones(8))
        elif defect=='empty': C.certify(np.zeros(8),np.zeros(8))
        elif defect=='shape': C.certify(np.ones(7),np.ones(8))
        elif defect=='weights': C.coefficients(LEGAL,[0,1,2,3],[.2]*4,'skill',0)
        elif defect=='bit': C.coefficients(LEGAL,[0,1,2,3],[.25]*4,'skill',2)
        elif defect=='endpoint': C.coefficients(LEGAL,[0,1,2,8],[.25]*4,'skill',0)
        else: C.evaluate(C.certify(np.ones(8),np.ones(8)),float('nan'))


def fixture(tmp_path):
    root,cfg=joint_fixture(tmp_path)
    cfg={k:cfg[k] for k in ('lineages','queries','rules','models','input_files')}
    cfg.update(fields=list(C.FIELDS),grid_denominator=32,grid_numerators=list(range(16,33)))
    return root,cfg


def test_full_handler_scalar_grid_complete_roles_and_coefficients(tmp_path):
    root,cfg=fixture(tmp_path); result=C.run(root,dict(design=cfg),lambda **kw:None)
    groups=read(root/'inputs/MEMBERSHIP.json')['omit-both'];ids=sorted(groups)
    lookup={(x['lineage'],x['rule'],x['model'],x['reader_id']):x for x in read(root/'inputs/DISCLOSURE_LAWS.json')}
    assert result['certificates']==len(ids)*16 and result['fits']==0
    assert len(result['cells'])==4 and all(result['controls'].values())
    count=0
    for lin,rule,model in product(cfg['lineages'],cfg['rules'],cfg['models']):
        with np.load(root/'certificates'/f'{lin}-{rule}-{model}_points.npz',allow_pickle=False) as d:
            for gi,key in enumerate(ids):
                law=lookup[lin,rule,model,key]
                for axis,bit in product((0,1),repeat=2):
                    ix=gi,axis,bit
                    for ri,r in enumerate(C.GRID):
                        p,m=scalar(law['legal_completions'],law['endpoints'],law['conditional_weights'],axis,bit,r)
                        j=(*ix,ri)
                        assert d['grid_mass'][j]==pytest.approx(m,abs=1e-15)
                        assert bool(d['grid_compatible'][j])==(m>0)
                        if p is None:
                            assert not d['high_compatible'][ix] and r==1
                            assert np.array_equal(d['grid_forecasts'][j],np.zeros(8))
                        else:
                            assert np.allclose(d['grid_forecasts'][j],p,atol=1e-15,rtol=0)
                            assert np.all(p>=d['lower'][ix]-1e-15) and np.all(p<=d['upper'][ix]+1e-15)
                        count+=1
                    # Independently sum exact binary64 weights as rationals for both partitions.
                    for k,match in [('hit',True),('miss',False)]:
                        expected=[float(sum((Fraction(float(w)) for q,e,w in zip(law['legal_completions'],law['endpoints'],law['conditional_weights']) if (q[axis]==bit)==match and e==ep),Fraction())) for ep in range(8)]
                        assert np.array_equal(d[k][ix],expected)
    assert count==result['certificates']*17
    for packet in read(root/'reader/REPLIES.json').values():
        assert set(packet)=={'initial','operations','requested_field','reply_value','reliability_interval'}
        assert packet['reliability_interval']==[.5,1.]


def test_corrupt_design_and_input_bindings_rejected(tmp_path):
    root,cfg=fixture(tmp_path);cfg['grid_denominator']=16
    with pytest.raises(ValueError,match='design'):C.run(root,dict(design=cfg),lambda **kw:None)
    cfg['grid_denominator']=32
    (root/'inputs/DISCLOSURE_LAWS.json').write_text('[]\n',encoding='utf-8')
    with pytest.raises(ValueError,match='input binding'):C.run(root,dict(design=cfg),lambda **kw:None)
