import numpy as np
from ghostscale.validation.soundingline.v21.routes import controls,law,posterior,programs,execute
from ghostscale.validation.soundingline.v21.routes_check import reference,execute as scalar_execute

def test_known_answers():
    assert all(controls().values())

def test_every_route_and_witness_against_independent_enumeration():
    p=law(71000792);history=np.array([[0,1],[1,2],[0,3]])
    for goal in (0,1):
        for r,program in enumerate(programs(goal)):
            assert execute(program)==goal==scalar_execute(program)
            for witness in (None,('inspection',int(r==3)),('complete',r)):
                for h in ([],history):
                    q,f=posterior(p,h,goal,witness);a,b=reference(p,h,goal,witness)
                    assert np.max(abs(q-a))<1e-12 and np.max(abs(f-b))<1e-12

def test_history_maker_inference_does_not_identify_unwitnessed_route():
    p=law(71000793);q,_=posterior(p,np.tile([[0,3]],(64,1)),0,None)
    assert np.count_nonzero(q)==4 and q.max()<1
    q,_=posterior(p,[],0,('complete',3));assert q[3]==1
