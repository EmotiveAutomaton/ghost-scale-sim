import numpy as np
from ghostscale.validation.soundingline.v18_4 import neural_data as D


def test_support_count_pairing_and_far_query_exclusion():
    worlds=[];lengths=[]
    for support in ('even','odd','all'):
        data,truth=D.make_split('pilot',16,support=support)
        assert len(data['history'])==256 and len(data['query'])==1280
        states={x['state'] for x in truth}
        assert len(states)==(16 if support=='all' else 8)
        assert all(D.parity(s)==(support=='odd') for s in states) if support!='all' else True
        worlds.append([x['world'] for x in truth]);lengths.append(data['length'])
        assert np.allclose(data['target'].sum(1),1,atol=1e-6)
    assert worlds[0]==worlds[1]==worlds[2]
    assert np.array_equal(lengths[0],lengths[2])
    assert not any(c in D.TRAIN_QUERIES+D.NEW_QUERIES for c in D.FAR_QUERIES)


def test_diverse_supervision_preserves_count_and_public_schema():
    # Cross the first 16-history rotation boundary; its first block is old-only.
    old,_=D.make_split('pilot',32,support='all')
    diverse,_=D.make_split('pilot',32,support='all',query_mode='diverse')
    assert np.array_equal(old['history'],diverse['history'])
    assert old['target'].shape==diverse['target'].shape
    assert not np.array_equal(old['query'],diverse['query'])
    assert np.allclose(diverse['target'].sum(1),1,atol=1e-6)


def test_each_state_gets_all_eight_queries_at_equal_counts():
    from collections import Counter
    import json
    for states in (8,16):
        counts=[Counter() for _ in range(states)]
        for i in range(128):
            for query in D.training_queries(i,'diverse'):counts[i%states][json.dumps(query,sort_keys=True)]+=1
        assert all(len(c)==8 and len(set(c.values()))==1 for c in counts)
    assert { (8,16,32)[(i//16)%3] for i in range(96)}=={8,16,32}
