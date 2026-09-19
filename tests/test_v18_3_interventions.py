import numpy as np
from ghostscale.validation.soundingline.v18_3 import world as W,neural_data as N,intervention_data as D


def test_generator_swaps_preserve_other_roles_and_remain_supported():
    for a in N.NEURAL_STATES:
        for b in N.NEURAL_STATES:
            for role,axes in enumerate(D.ROLES):
                state=D.mixed_state(a,b,role)
                assert state in N.NEURAL_STATES
                assert all(W.STATES[state][i]==W.STATES[b if i in axes else a][i] for i in range(4))
                if a==b:assert state==a


def test_counterfactual_public_features_and_self_pair_exactness():
    data,points=D.cell_data('test',0,1,pilot=True)
    assert len(data['base'])==16*16*3*3
    assert np.allclose(data['target'].sum(1),1)
    for b in (0,3,15):
        p=W.posterior(W.packet(points[0]['world'],points[0]['histories'][b]))[list(N.NEURAL_STATES)];p/=p.sum()
        for q,c in enumerate(D.QUERIES):
            selected=(data['ids'][:,2]==b)&(data['ids'][:,3]==b)&(data['ids'][:,5]==q)
            expected=p@W.artifact_matrix(points[0]['world'],c)[list(N.NEURAL_STATES)]
            assert np.allclose(data['exact'][selected],expected,atol=1e-12,rtol=0)

