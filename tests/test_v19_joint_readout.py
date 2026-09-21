import json
import numpy as np
from ghostscale.validation.soundingline.v19 import joint_readout as J

def test_controls():assert all(J.controls().values())

def test_full_universe_score_and_compatibility():
    alphabet=[0,5,100];p=np.array([.4,.3,.3])*32/33;y={0:.4,6:.6}
    full=np.full(J.UNIVERSE,1/33/(J.UNIVERSE-len(alphabet)));full[alphabet]=p
    truth=np.zeros(J.UNIVERSE)
    for k,v in y.items():truth[k]=v
    m=J.metrics(p,alphabet,list(y.items()),32)
    assert np.isclose(full.sum(),1)
    assert np.isclose(m['loss'],-truth@np.log(full))
    assert np.isclose(m['squared_error'],sum((full-truth)**2))
    assert np.isclose(m['compatible_mass'],full[0]+full[6])
    assert m['abstain'] and not m['top_incompatible']

def test_training_only_and_complete_fixture(tmp_path):
    cfg=dict(arms=list(J.ARMS),train_lineages=[190973],development_lineages=[190974],training_draws=[190201],fit_seeds=[190101],paths_per_lineage=32,budgets=[32],tiers=['E0'])
    s=J.run(tmp_path,dict(design=cfg),lambda **kw:None)
    assert all(s['controls'].values()) and len(s['fits'])==3 and s['rows']==4
    labels=np.load(tmp_path/'training/190201-E0.npz')['joint_label']
    alphabet=json.loads((tmp_path/'models/190201-E0-190101-32-alphabet.json').read_text())
    assert alphabet==sorted(set(labels))
    assert len({r['head_parameters'] for r in s['fits']})==1
    reader=json.loads((tmp_path/'reader/PACKETS.json').read_text())
    assert all(set(p['inputs'])=={'artifact'} for p in reader['packets'].values())
