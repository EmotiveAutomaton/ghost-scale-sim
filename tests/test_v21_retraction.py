import numpy as np
from ghostscale.validation.soundingline.v21.retraction import Ledger,controls
from ghostscale.validation.soundingline.v21.retraction_check import posterior,updated

def test_known_answers():
    assert all(controls().values())

def test_long_mixed_update_sequence_and_deterministic_support_recovery():
    for reliability in (.6,.8,1.):
        records={i:(i%2,reliability) for i in range(64)};prior=[.31,.69];ledger=Ledger(prior,records)
        updates=[(i,None,reliability) for i in range(64)]+[(0,1,1.),(0,0,1.),(0,0,1.),(9,-1,.8),(0,None,1.)]
        for update in updates:
            before=ledger.factor_visits;ledger.update(update);records=updated(records,update);q,conflict=posterior(prior,records)
            assert ledger.factor_visits-before<=2
            assert np.max(abs(q-ledger.posterior()[0]))<1e-12 and conflict==ledger.posterior()[1]

def test_incremental_path_never_calls_raw_decoder(monkeypatch):
    from ghostscale.validation.soundingline.v21 import retraction_check
    def forbidden(*args):raise AssertionError('raw decoder used')
    monkeypatch.setattr(retraction_check,'posterior',forbidden)
    a=Ledger([.5,.5],{0:(1,.8),1:(0,.6)});a.update((1,None,.6))
    assert np.allclose(a.posterior()[0],[.2,.8],atol=1e-15)
