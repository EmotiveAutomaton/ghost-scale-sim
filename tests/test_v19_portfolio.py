import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import portfolio as P

def test_known_answer_and_null_before_science():
    assert all(P.controls().values())

def test_lineage_split_rejects_overlap_and_unknowns():
    ids=np.array([[i,0,0] for i in range(128)])
    masks=P.splits(ids,list(range(128)))
    assert [m.sum() for m in masks]==[64,32,32]
    with pytest.raises(ValueError):P.splits(ids,[0]*128)
    with pytest.raises(ValueError):P.splits(np.array([[129,0,0]]),list(range(128)))

def test_candidate_menu_excludes_far_targets():
    assert len(P.CANDIDATES)==8
    assert not any(q in P.CANDIDATES for q in P.N.FAR_QUERIES)

def test_probability_repair_reports_corrupt_raw_prediction():
    p,bad=P.predict(np.array([[1.]]),np.array([[0.]*32,[-1.,2.]*16]))
    assert bad.all() and np.all(p>0)
    assert np.allclose(p.reshape(1,2,16).sum(2),1)
