"""Decision-law and approximation checks independent of scientific outcomes."""
import copy
import numpy as np
from ghostscale.validation.soundingline.v27 import timing,setpoints,setpoint_resolution as resolution


def test_locked_choices_are_independent_of_diffusion_error_probability():
    for kind in ('L','Ls','N'):
        for speed,boundary,t0 in timing.PARAMETERS:
            _,prob=timing.probability(kind,8,True,(speed,boundary,t0),'locked_choices')
            assert prob==.99
    _,unlocked=timing.probability('N',8,False,(.5,.8,.2),'locked_choices')
    assert unlocked!=.99


def test_hidden_particle_reader_cannot_observe_planted_state():
    s=dict(needs=2,records=12,dynamic='S',beta=0.,width=0,consumption=3.,probe=1/64,lapse=.02,
           inference='checked_particles',particles=256)
    raw=setpoints.generate(s,['excluded-test-no-state-leak'])
    poisoned=copy.deepcopy(raw)
    poisoned['point']=[100,100];poisoned['dynamic']='invented'
    for row in poisoned['records']:
        for field in ('stock','after','edge','precision','streak'):row[field]=[float('nan')]*2
    a,_=setpoints.likelihood(raw,s,(8,11),'S',0,0.)
    b,_=setpoints.likelihood(poisoned,s,(8,11),'S',0,0.)
    np.testing.assert_array_equal(a,b)
    observed=resolution.hidden_observations(poisoned)
    assert all(set(row)=={'action','size','shock'} for row in observed['records'])


def test_resolution_rejects_bar_crossing_even_when_path_error_is_small():
    a=dict(posterior=np.full((150,5),.2),margins={law:np.r_[np.zeros(127),np.full(23,1.01)] for law in ('S','P-','G')})
    b=dict(posterior=a['posterior'].copy(),margins={law:np.r_[np.zeros(127),.99,np.full(22,1.01)] for law in ('S','P-','G')})
    assert resolution.compare(a,a)['passed']
    assert not resolution.compare(a,b)['passed']


def test_retained_preflight_is_not_reexecuted(monkeypatch):
    from ghostscale.validation.soundingline.v27 import registry
    s=registry.primary_settings('K3')[0]|dict(records=4)
    identity=['retained-control'];rows=[]
    for count in (2048,8192,32768):
        for replicate in range(2):
            rows.append(dict(particles=count,replicate=replicate,posterior=np.full((4,5),.2).tolist(),
                             margins={law:[0.]*4 for law in ('S','P-','G')}))
    def unexpected(*args,**kwargs):raise AssertionError('retained likelihood recomputed')
    monkeypatch.setattr(resolution,'paths',unexpected)
    result=resolution.fixture(s,identity,retained=dict(identity=identity,configuration=s,diagnostic_paths=rows))
    assert result['passed'] and result['diagnostic_paths']==rows
