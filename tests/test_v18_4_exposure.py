"""Exact exposure matching and nested histories for the L4 discrimination."""
from collections import Counter
import json
import numpy as np
from ghostscale.validation.soundingline.v18_4 import neural_data as D


def exposure(n, mode):
    counts=Counter()
    for i in range(n):
        for q in D.training_queries(i,mode):
            counts[(i%16,(8,16,32)[(i//16)%3],json.dumps(q,sort_keys=True))]+=1
    return counts


def test_matched_old_question_exposure_in_every_state_and_length():
    old=exposure(480,'old');diverse=exposure(768,'diverse')
    assert len(old)==16*3*5 and len(diverse)==16*3*8
    assert set(old.values())=={10} and set(diverse.values())=={10}
    assert all(diverse[key]==value for key,value in old.items())
    assert sum(diverse.values())==768*5
    assert set(exposure(768,'old').values())=={16}


def test_nested_histories_and_matched_total_label_control():
    kw=dict(split='pilot',support='all',namespace='v18.4-exposure')
    small,st=D.make_split(per_cell=16,query_mode='old',**kw)
    large,lt=D.make_split(per_cell=32,query_mode='old',**kw)
    diverse,dt=D.make_split(per_cell=32,query_mode='diverse',**kw)
    assert lt==dt
    assert np.array_equal(large['history'],diverse['history'])
    assert large['target'].shape==diverse['target'].shape
    assert not np.array_equal(large['query'],diverse['query'])
    for cell in range(16):
        assert np.array_equal(small['history'][cell*16:(cell+1)*16],large['history'][cell*32:cell*32+16])
        assert st[cell*16:(cell+1)*16]==lt[cell*32:cell*32+16]
    other,_=D.make_split(per_cell=16,query_mode='old',**dict(kw,namespace='v18.4-neural'))
    assert not np.array_equal(small['history'],other['history'])


def test_development_and_test_holdout_identity():
    # Development is common across all arms, so model selection has equal access.
    assert set(exposure(384,'diverse').values())=={5}
    assert not any(q in D.TRAIN_QUERIES+D.NEW_QUERIES for q in D.FAR_QUERIES)


def test_common_development_capsule_and_resume_identity(tmp_path):
    from ghostscale.validation.soundingline.v18_3.io import file_digest
    arrays=[]
    for mode in ('old','diverse'):
        root=tmp_path/mode
        spec=dict(train_per_cell=16,dev_per_cell=32,test_per_cell=8,support='all',
            query_mode=mode,dev_query_mode='diverse',namespace='v18.4-exposure')
        manifest=D.prepare(root,**spec)
        before=file_digest(root/'reader/TRAIN.npz')
        assert D.prepare(root,**spec)==manifest
        assert file_digest(root/'reader/TRAIN.npz')==before
        with np.load(root/'reader/TRAIN.npz') as z:
            arrays.append({k:z[k].copy() for k in z.files if k.startswith('dev_')})
    assert all(np.array_equal(arrays[0][k],arrays[1][k]) for k in arrays[0])
