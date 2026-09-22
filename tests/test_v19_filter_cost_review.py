import pytest
from ghostscale.validation.soundingline.v19 import filter_cost_review as R
from test_v19_filter_cost import row, test_complete_retained_fixture as fixture


def test_independent_known_counts_and_copies():
    s=dict(length=128,kind='skill',observations=[row(i) for i in range(1,129)])
    v, hs=R.reconstruct(s,128,'unknown-time-type')
    assert len(hs)==3632 and v['weight_bytes']==29056
    assert v['likelihood_factor_applications']==464896
    v,_=R.reconstruct(s,128,'reset-16')
    assert v['retained_sources']==16 and v['likelihood_factor_applications']==256
    s['observations']=[row(1),row(2,1)]
    assert R.reconstruct(s,2,'static')[0]['supplied_sources']==1
    s['observations'][1]['endpoint']=1
    with pytest.raises(ValueError,match='conflict'):R.reconstruct(s,2,'static')


def test_independent_comparison_rejects_bad_values():
    with pytest.raises(ValueError):R.equal(dict(x=float('nan')),dict(x=0.))
    with pytest.raises(ValueError):R.equal(dict(x=1),dict(x=0))
    with pytest.raises(ValueError):R.equal(dict(x=0),dict(y=0))


def test_full_fixture_independent_rows(tmp_path):
    fixture(tmp_path)
    rows=R.zipped(tmp_path/'fixture_points.json.gz')
    for r in rows:
        s=dict(length=32,kind='purpose',observations=[row(i) for i in range(1,33)])
        c,_=R.reconstruct(s,r['step'],r['arm'])
        R.equal({k:r[k] for k in c},c)
    rows[0]['weight_bytes']+=8
    with pytest.raises(ValueError):R.equal({k:rows[0][k] for k in c},R.reconstruct(s,rows[0]['step'],rows[0]['arm'])[0])
