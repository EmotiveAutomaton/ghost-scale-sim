import numpy as np
from ghostscale.validation.soundingline.v21 import persistence as P,world as W
from ghostscale.validation.soundingline.v21.persistence_check import reference,check_tables


def test_persistence_controls_and_scalar_reference():
    assert all(P.controls().values())
    rows=W.enumerate_world(71000721);t=P.tables(rows);check_tables(rows,t)
    chosen=W.sampled(rows,16,'persistence-control');obs=P.observations(rows,chosen,4,'control')
    context=None
    for condition,d in obs.items():
        if context is None:context=d['contexts']
        assert np.array_equal(context,d['contexts'])
        x=d['evidence'];p=P.predict(t,x);r=reference(t,x)
        for k in r['by_rate']:assert np.allclose(np.exp(p['by_rate'][k]),r['by_rate'][k],atol=1e-12)
        assert np.allclose(np.exp(p['inferred']),r['inferred'],atol=1e-12)
        assert np.allclose(p['by_rate']['0.0'],p['no_history'],atol=1e-12)
        assert np.all(d['same_source']) if condition=='1.0' else True
        assert not np.any(d['same_source']) if condition=='0.0' else True
        if condition=='1.0':assert not np.any(p['conflicts']['1.0'])
        assert np.all(np.isfinite(np.exp(p['by_rate']['1.0'])))


def test_informative_persistence_differs_from_no_history():
    rows=W.enumerate_world(71000722);t=P.tables(rows)
    obs=P.observations(rows,W.sampled(rows,32,'positive'),4,'positive')['1.0']
    p=P.predict(t,obs['evidence'])
    assert np.max(np.abs(np.exp(p['by_rate']['1.0'])-np.exp(p['no_history'])))>1e-4
