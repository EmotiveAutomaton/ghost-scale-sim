import numpy as np
from ghostscale.validation.soundingline.v21.memory import controls,encode,decode,READERS
from ghostscale.validation.soundingline.v21.memory_check import reference
from ghostscale.validation.soundingline.v21.drift import channels

def test_known_answer_controls():
    assert all(controls().values())

def test_payload_limits_and_independent_decoder():
    rng=np.random.default_rng(14);records=np.column_stack([np.arange(512),rng.integers(2,size=(512,4))]);emission=channels(71000771)['emission']
    for kind in READERS:
      for budget in (256,1024,4096,16384):
        blob=encode(kind,records,emission,budget,'test');assert len(blob)<=budget
        q,ledger=decode(blob,emission,'test');r,h,rows=reference(blob,emission,[0,511],'test')
        assert np.allclose(q,r,atol=1e-12) and len(ledger)==len(rows)
        if 511 in ledger:assert h[1,2*records[511,3]+records[511,4]]==1

def test_prediction_preserved_but_historical_witness_lost():
    records=np.array([[i,i%2,(i//2)%2,i%2,(i//3)%2] for i in range(128)]);emission=channels(71000772)['emission'];twin=records.copy();twin[:,3:]=1-twin[:,3:]
    for kind in ('maker-posterior','predictive-head'):
        assert encode(kind,records,emission,256,'test')==encode(kind,twin,emission,256,'test')
        q,ledger=decode(encode(kind,records,emission,256,'test'),emission,'test');assert not ledger
    q,_=decode(encode('maker-posterior',records,emission,256,'test'),emission,'test')
    r,_=decode(encode('predictive-head',records,emission,256,'test'),emission,'test');assert np.allclose(q,r,atol=1e-12)
