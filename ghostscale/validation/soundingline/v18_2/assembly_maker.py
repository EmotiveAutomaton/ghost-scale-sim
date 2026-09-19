"""G4 new physical family: reuse V16 assembly, acquired fragments and finite policy.

A new adapter, not transfer of board-trained weights. Train/dev/test dependency
graphs are respectively fork/independent/chain. All action routes really execute.
"""
from functools import lru_cache
from itertools import permutations
import json
import math
import random
import time
import numpy as np
from ..v16 import assembly as a
from ..v16.learning import encoding_cost
from ..v16.records import write
from . import model as m,learned as n
from .verify import interval


def world(split,index):
    return dict(parents=list({'train':(-1,0,0),'dev':(-1,-1,-1),'test':(-1,0,1)}[split]),
                defaults=[(index>>i)&1 for i in range(3)],max_code=5,price=.25,temperature=.7)


def physical(w):return a.World(tuple(w['parents']),tuple(w['defaults']))


def replay(w,program):
    """Independent dictionary/set executor, including stopped-state rejection."""
    state={};stopped=False
    for action in program:
        if stopped or action not in range(10):raise ValueError('invalid stopped action')
        if action==9:stopped=True;continue
        part=action%3;parent=w['parents'][part]
        child=any(p==part and i in state for i,p in enumerate(w['parents']))
        if action<3:
            if part in state or (parent>=0 and parent not in state):raise ValueError('invalid attachment')
            state[part]=w['defaults'][part]
        elif action<6:
            if part not in state or child:raise ValueError('invalid removal')
            del state[part]
        else:
            if part not in state or child:raise ValueError('invalid rotation')
            state[part]=1-state[part]
    if not stopped:raise ValueError('missing stop')
    return sum(state[i]<<i for i in range(3)) if len(state)==3 else 8


@lru_cache(maxsize=64)
def catalog(encoded):
    w=json.loads(encoded);pw=physical(w);programs=[(9,)];outputs=[8]
    for target in range(8):
        for ordering in permutations(range(3)):
            program=[]
            for part in ordering:
                program.append(part)
                if ((target>>part)&1)!=w['defaults'][part]:program.append(6+part)
            program.append(9);result=a.execute(pw,program)
            if result['legal'] and result['successfully_stopped']:
                assert replay(w,program)==target
                programs.append(tuple(program));outputs.append(target)
    libraries=[()]
    for target in (0,7):
        route=programs[outputs.index(target)]
        training=[dict(program=list(route),target=[(target>>i)&1 for i in range(3)])]*4
        libraries.append(tuple(map(tuple,a.fragments(training,pw)['library'])))
    return tuple(programs),tuple(outputs),tuple(libraries)


@lru_cache(maxsize=8192)
def kernel(encoded,state,goal,signal,noticed):
    w=json.loads(encoded);programs,outputs,libraries=catalog(encoded)
    skill,pref,default,prior=state;task=default if goal is None else goal
    beliefs=[(prior,1.)] if signal is None else [(signal,noticed),(prior,1-noticed)]
    cost=np.array([encoding_cost(p,libraries[skill]) for p in programs]);answer=np.zeros(len(programs))
    for belief,mass in beliefs:
        target=7 if task^belief else 0
        values=[]
        for output,c in zip(outputs,cost):
            error=3 if output==8 else (output^target).bit_count()
            tradeoff=0 if output==8 else output.bit_count()-1.5
            utility=-1.6*error+.65*(pref-1)*tradeoff-w['price']*c
            values.append(math.exp(utility/w['temperature']) if c<=w['max_code'] else 0.)
        p=np.array(values);answer+=mass*p/p.sum()
    return tuple(answer)


def distribution(w,state,context):
    return np.array(kernel(m.canonical(w).decode(),tuple(state),context['goal'],context['signal'],context['noticed']))


def marginal(w,q):
    _,outputs,_=catalog(m.canonical(w).decode());p=np.zeros(16)
    for output,mass in zip(outputs,q):p[output]+=mass
    return p


def draw(w,state,ctx,rng):
    q=distribution(w,state,ctx);programs,outputs,_=catalog(m.canonical(w).decode())
    j=rng.choices(range(len(q)),weights=q)[0]
    return dict(context=ctx,program=list(programs[j]),artifact=outputs[j])


def make_case(namespace,index,split):
    rng=random.Random(m.seed(namespace,index,split));state=m.STATES[rng.randrange(36)];w=world(split,index)
    history=[draw(w,state,m.context(rng.choice([0,1,None]),rng.choice([0,1,None])),rng) for _ in range(8)]
    probes=[draw(w,state,m.context([None,0,1,None][j],[None,None,None,1][j]),rng) for j in range(4)]
    return dict(case_id=f'{namespace}:{split}:{index}',split=split,world=w,history=history,probes=probes,truth=list(state))


def packet(case,j):
    return m.canonical(dict(schema='v18.2.assembly-public.1',world=case['world'],history=case['history'],current=case['probes'][j]['context']))


def parse(payload):
    p=json.loads(payload)
    if set(p)!={'schema','world','history','current'} or p['schema']!='v18.2.assembly-public.1':raise ValueError('assembly public boundary')
    if set(p['world'])!={'parents','defaults','max_code','price','temperature'}:raise ValueError('private assembly world fields')
    for obs in p['history']:
        if set(obs)!={'context','program','artifact'} or replay(p['world'],obs['program'])!=obs['artifact']:raise ValueError('bad observation')
    for ctx in [p['current'],*[obs['context'] for obs in p['history']]]:
        if set(ctx)!={'goal','signal','reader_fact','noticed'}:raise ValueError('private context field')
    return p


def infer(payload,method='persistent'):
    p=parse(payload);w=p['world'];programs,_,_=catalog(m.canonical(w).decode());weights=np.ones(36)/36
    if method=='raw':
        q=np.ones(16)*.025
        for obs in p['history']:
            distance=sum(obs['context'][k]!=p['current'][k] for k in ('goal','signal','noticed'))
            q[obs['artifact']]+=math.exp(-distance)
        return q/q.sum()
    if method!='direct':
        for obs in p['history']:
            index=programs.index(tuple(obs['program']))
            weights*=np.array([distribution(w,s,obs['context'])[index] for s in m.STATES]);weights/=weights.sum()
    return weights@np.array([marginal(w,distribution(w,s,p['current'])) for s in m.STATES])


def features(payload):
    p=parse(payload);x=np.zeros(n.HISTORY+n.CURRENT,np.float32)
    for i,obs in enumerate(p['history']):
        offset=i*n.OBS;x[offset:offset+11]=n.context_features(obs['context']);x[offset+11+obs['artifact']]=1
        for action in obs['program']:x[offset+27+action]+=1/7
        x[offset+43]=1
    offset=n.HISTORY;x[offset:offset+11]=n.context_features(p['current'])
    for part,parent in enumerate(p['world']['parents']):x[offset+11+part*3+(parent+1)]=1
    x[offset+20:offset+23]=p['world']['defaults']
    return x


def run(root,design,limited,heartbeat):
    from .runtime import keep
    start=time.monotonic();train=[];target=[];dev=[];dy=[]
    for split,count,xx,yy in [('train',1250,train,target),('dev',64,dev,dy)]:
        for i in range(count):
            if limited():raise RuntimeError('resource cutoff')
            case=make_case(design['namespace'],i,split)
            for j in range(4):xx.append(features(packet(case,j)));yy.append(case['probes'][j]['artifact'])
            if i%64==0:heartbeat(phase='assembly-training-data',split=split,makers=i)
    x=np.array(train);y=np.array(target);dx=np.array(dev);dy=np.array(dy)
    models,fit=n.train_pair(x,y,dx,dy,design.get('seed',0),limited,heartbeat,epochs=80)
    for key,net in models.items():net.save(root/f'{key}.npz')
    write(root/'FIT.json',dict(**fit,training_makers=1250,training_history_episodes=10000,supervised_examples=5000,
        development_makers=64,scope='new adapter fit; board-trained weights are not transferred'))
    cells={};verified=0;rows_count=0
    for first in range(0,128,8):
        if limited():return
        block=[];cpu=time.process_time();wall=time.monotonic()
        for i in range(first,first+8):
            case=make_case(design['namespace'],i,'test');rows=[];per={}
            for obs in case['history']+case['probes']:
                assert replay(case['world'],obs['program'])==obs['artifact'];verified+=1
            for j,probe in enumerate(case['probes']):
                payload=packet(case,j);truth=marginal(case['world'],distribution(case['world'],case['truth'],probe['context']))
                for method in ('direct','raw','persistent','split','flat','oracle'):
                    if method in models:q=models[method].forward(features(payload)[None,:])[0][0]
                    elif method=='oracle':q=truth
                    else:q=infer(payload,method)
                    scores=m.score(q,truth,probe['artifact']);rows.append(dict(method=method,probe=j,probabilities=q.tolist(),scores=scores))
                    for metric,value in scores.items():per.setdefault((method,metric),[]).append(value)
                    rows_count+=1
            for (method,metric),values in per.items():cells.setdefault(method,{}).setdefault(metric,[]).append(float(np.mean(values)))
            block.append(dict(case=case,rows=rows))
        keep(root,f'test-{first:05d}',block,time.process_time()-cpu,time.monotonic()-wall)
        heartbeat(phase='assembly-test',completed=first+8)
        if first==8:write(root/'FORECAST-test.json',dict(completed_units=16,observed_remaining_seconds=(time.monotonic()-wall)/8*112,twice_as_fast_remaining_seconds=(time.monotonic()-wall)/16*112))
    write(root/'SUMMARY.json',dict(units=128,rows=rows_count,independent_executions=verified,
        cells={k:{metric:interval(values) for metric,values in v.items()} for k,v in cells.items()},
        scope='new assembly family, independent maker lineages and held-out dependency topology; discovery miniature, architecture untested'))
