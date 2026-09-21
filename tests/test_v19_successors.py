import numpy as np
from ghostscale.validation.soundingline.v19 import probability_head as P
from ghostscale.validation.soundingline.v19 import process_sufficiency as S
from ghostscale.validation.soundingline.v19 import local_world as L


def test_probability_head_known_answers_and_gradient():
    assert all(P.controls().values())


def test_multiple_query_heads_are_independently_normalized():
    x=np.array([[1.,-1.],[1.,1.]])
    y=np.array([[1.,0.,0.,.2,.3,.5],[0.,1.,0.,.4,.4,.2]])
    w,_=P.fit(x,y,[3,3]);_,_,logp=P.objective_gradient(x,y,w,[3,3])
    assert np.all(np.isfinite(logp))
    assert np.allclose(np.exp(logp).reshape(2,2,3).sum(2),1)


def test_alias_ruler_rejects_numerical_nonidentification():
    assert all(S.synthetic_controls().values())


def test_path_symmetry_checks_complete_transient_executor():
    world=L.law(0)
    # Fixture law, no campaign lineage or outcomes used at admission.
    for m in L.MAKERS:
        for context in L.CONTEXTS:
            values=[]
            for path in [('replace-presentation','inspect','inspect'),('inspect','replace-presentation','inspect')]:
                a=tuple(context['initial']);previous=a;prob=1.
                for step,op in enumerate(path):
                    prob*=sum(p for g,o,p in L.choices(world,m,a,step,context) if o==op)
                    a,previous=L.execute(a,previous,op,m),a
                values.append(prob/64)
            expected=S.common_path_likelihood(world,m,context)
            assert np.allclose(values,[expected,expected],rtol=1e-14,atol=1e-16)
