"""Complete commissioned roster and prospective provisional choices."""
from itertools import product
from .contract import ORDER,FAMILY_HOURS

NAMES=dict(W1='PIECES_ENDS',W2='HIDDEN_SETPOINTS',W3='SELECTIVE_UPTAKE',W4='POPULATION_DEVIATION',W5='LIBRARY_MISSPEC',W6='EMOTION_SIGNATURES',W7='ATTENTION_DIVERGENCE',W8='VOLATILE_WANTS',W9='BETRAYAL_TYPES',W10='ABSORPTION',W11='CARE_GROUPS',W12='SACRED_STRUCTURES',W13='PERSISTENCE_OR_PROXY')

def grid(**factors):return [dict(zip(factors,values)) for values in product(*factors.values())]

def settings(family):
    if family=='W1':return grid(overlap=['none','partial','heavy'],beta=[1,3])
    if family=='W2':return grid(sigma=[[.25,.25],[.25,1.],[1.,.25],[1.,1.]],decay=[.9,.8,.95])
    if family=='W3':return grid(alignment=['aligned','partial','opposed'],bundling=['separable','bundled'])
    if family=='W4':return grid(alpha=[10,3,30],placebo=[False,True])
    if family=='W5':return grid(overlap=['partial'],beta=[1,3])
    if family=='W6':return grid(overlap=['partial','heavy'],block_probability=[0.,.1,.3],kappa=[3.,1.],generator=['value','rival'],temperament=[0.,.2])
    if family=='W7':return grid(habit=[0,1,2],epsilon=[0.,.1])
    if family=='W8':return [dict(process=k,rate=v) for k,values in [('walk',[0.,.02,.05]),('jump',[0.,1/64,1/16]),('trend',[.02]) ] for v in values]
    if family=='W9':return grid(n=[5,20,80],k=[1,3,8],m=[0,2,8],hyperprior=[[.1,.4],[1.,4.],[4.,16.]],typical=[True,False],epsilon=[.05,.1])
    if family=='W10':return grid(growth=['linear','saturating','normalized'],exposure_gain=[0.,.25,1.],attention=[.2,.8],exposure=['repetition','forced'])
    if family=='W11':return grid(care=[0.,.1,.3,.15],contact=[.1,.5],withheld=[0.,.5],stereotype=[False,True])
    if family=='W12':return grid(habit_affect=[False,True],end_distress=[False,True],weight=[.5,1.,2.])
    if family=='W13':return grid(shift=['cue-only','restoration-only','harmful-cue','delayed'],learning_rate=[0.,.05,.2])
    raise ValueError('unknown family')

def card(family):
    return dict(id=family,name=NAMES[family],ceiling_seconds=FAMILY_HOURS[family]*3600,
        blocks=64 if family=='W11' else 16,settings=settings(family),makers_per_block=8 if family in ('W1','W3','W5','W6','W7','W8','W10','W12','W13') else 'full declared roster',
        independent_unit='whole-world block',architecture='miniature, architecture untested',
        reader='exact finite-library posterior; W11 exact individual inference in simulated populations',
        inference='descriptive; no confirmation or human claim',dependency=['W1'] if family=='W5' else [],
        source_packet_section=family,provisional_choices=True)

def selected_settings(family,shed=0):
    rows=settings(family)
    if family=='W1':
        if shed>=1:rows=[r for r in rows if r['beta']==3]
        if shed>=2:rows=[dict(r,prefixes=[8,32,128]) for r in rows]
        if shed>=3:rows=[dict(r,training_count=2) for r in rows]
    if family=='W2':
        if shed>=1:rows=[r for r in rows if r['decay']!=.95]
        if shed>=2:rows=[dict(r,gains=[.9]) for r in rows]
    if family=='W3':
        if shed>=1:rows=[dict(r,prefixes=[8,32]) for r in rows]
        if shed>=2:rows=[r for r in rows if r['alignment']!='opposed']
    if family=='W4':
        if shed>=1:rows=[r for r in rows if r['alpha']==10]
        if shed>=2:rows=[dict(r,prefixes=[8,32,128,512]) for r in rows]
        if shed>=3:rows=[dict(r,worlds=8) for r in rows]
    if family=='W5':
        if shed>=1:rows=[dict(r,omit_split=True) for r in rows]
        if shed>=2:rows=[dict(r,worlds=8) for r in rows]
    if family=='W6':
        if shed>=1:rows=[r for r in rows if r['kappa']==3]
        if shed>=2:rows=[r for r in rows if r['block_probability']!=.3]
    if family=='W7':
        if shed>=1:rows=[dict(r,omit_correct_random=(r['epsilon']==.1)) for r in rows]
        if shed>=2:rows=[dict(r,omit_revision=True) for r in rows]
    if family=='W8':
        if shed>=1:rows=[dict(r,omit_windowed_512=True) for r in rows]
        if shed>=2:rows=[r for r in rows if r['process']!='trend']
    if family=='W9':
        if shed>=1:rows=[r for r in rows if r['n']!=80]
        if shed>=2:rows=[r for r in rows if r['hyperprior']!=[4.,16.]]
        if shed>=3:rows=[dict(r,omit_deception=True) for r in rows]
    if family=='W10' and shed>=1:rows=[r for r in rows if r['growth']!='normalized']
    if family=='W11':
        if shed>=1:rows=[r for r in rows if not r['stereotype']]
        if shed>=2:rows=[r for r in rows if r['contact']!=.5]
    if family=='W12' and shed>=1:rows=[r for r in rows if r['habit_affect']]
    if family=='W13' and shed>=1:rows=[r for r in rows if r['shift']!='delayed']
    return rows
