"""G5 held-out physical dependency using V16's assembly learner and planner."""
import random
import time
import numpy as np
from ..v16 import assembly as a
from ..v16.records import write
from . import model as m
from .verify import interval


def evaluate(namespace,index):
    rows=[]
    for topology,parents in [('fork',(-1,0,0)),('heldout-chain',(-1,0,1))]:
        w=a.World(parents,(0,(index>>1)&1,(index>>2)&1))
        for dependent in (False,True):
            records=[]
            for j in range(12):
                procedure=j%2==0
                program=([0,6,1,2,9] if dependent else [0,1,9]) if procedure else [0,6,9]
                rng=random.Random(m.seed(namespace,index,'demonstration',j))
                if rng.random()<.1:program=[0,rng.randrange(9),9]
                result=a.execute(w,program)
                records.append(dict(source=j,topic='procedure' if procedure else 'goal',program=program,
                                    target=result['state'],execution=result))
            for attention in ('all','procedure'):
                selected=[r for r in records if attention=='all' or r['topic']=='procedure']
                for weight in (0.,.25,1.):
                    retained=[r for r in selected if random.Random(m.seed(namespace,index,'update',r['source'])).random()<weight]
                    acquisition=a.fragments(retained,w)
                    hits=sum(r['execution']['state'][0]==1 for r in selected)
                    alpha=1+weight*hits;beta=9+weight*(len(selected)-hits);uptake=alpha/(alpha+beta)
                    acquisition_cost=sum(r['execution']['primitive_cost'] for r in selected)+len(selected)
                    for apply in (False,True):
                        library=acquisition['library'] if apply else []
                        # Each probe has the same total work envelope, including
                        # observation, fit, search and reserved final execution.
                        budget=max(0,256-acquisition_cost-acquisition['training_primitives']-7)
                        outputs=[]
                        for target in ([0,0,0],[int(uptake>=.5),0,0]):
                            plan=a.plan(w,target,library=library,budget=budget,max_steps=7)
                            result=a.execute(w,plan['program'])
                            charged=acquisition_cost+acquisition['training_primitives']+plan['successor_evaluations']+result['primitive_cost']
                            assert charged<=256
                            outputs.append(dict(target=target,plan=plan,execution=result,charged_total=charged))
                        own,free=outputs
                        rows.append(dict(condition=f'{topology}-'+('dependent' if dependent else 'separable'),
                            method=f'{attention}-weight-{weight}-apply-{int(apply)}',world=w.public(),
                            observations=selected,retained_sources=[r['source'] for r in retained],library=acquisition['library'],
                            beta_parameters=[alpha,beta],outputs=outputs,
                            scores=dict(task_transfer=float(own['execution']['successfully_stopped'] and own['execution']['state']==[0,0,0]),
                                unwanted_goal_uptake=uptake,unwanted_goal_selection=float(uptake>=.5),
                                unwanted_goal_execution=float(free['execution']['successfully_stopped'] and free['execution']['state'][0]==1),
                                retained_variance=alpha*beta/((alpha+beta)**2*(alpha+beta+1)),
                                own_cost=own['charged_total'],free_cost=free['charged_total'])))
    return dict(case_id=f'{namespace}:maker-{index}',rows=rows)


def run(root,design,limited,heartbeat):
    from .runtime import keep
    cells={};count=0;start=time.monotonic()
    for first in range(0,128,8):
        if limited():return
        cpu=time.process_time();wall=time.monotonic();units=[]
        for index in range(first,first+8):
            unit=evaluate(design['namespace'],index);units.append(unit)
            for r in unit['rows']:
                key=r['condition']+'|'+r['method']
                for metric,value in r['scores'].items():cells.setdefault(key,{}).setdefault(metric,[]).append(value)
                count+=1
        keep(root,f'test-{first:05d}',units,time.process_time()-cpu,time.monotonic()-wall)
        heartbeat(phase='selective-assembly',completed=first+8)
        if first==8:
            elapsed=time.monotonic()-start
            write(root/'FORECAST-test.json',dict(completed_units=16,observed_remaining_seconds=elapsed/16*112,twice_as_fast_remaining_seconds=elapsed/32*112))
    write(root/'SUMMARY.json',dict(units=128,rows=count,
        cells={k:{metric:interval(values) for metric,values in metrics.items()} for k,metrics in cells.items()},
        scope='paired fork/chain physical dependency, shared update weights; discovery constructed learner, not human attention/trust'))
