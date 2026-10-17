import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, math, time
from collections import defaultdict
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--batch',type=int,choices=(1,2),required=True)
args=parser.parse_args()
campaign=args.campaign.resolve();snapshot=args.source.resolve();work=args.output.resolve()
work.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(snapshot),str(campaign/'dependencies')]
from ghostscale.validation.soundingline.v23.common import read,write,digest,interval
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v23.family import rows,verify_packet
from ghostscale.validation.soundingline.v23.makers import recipe_library
runtime.below_normal(); started=time.process_time(); batch=args.batch
output=work/f'SECONDARY_BATCH_{batch}.json'
try:
    root=campaign/f'packets/v23-n7-batch-{batch}'
    plan,summary=verify_packet(root)
    assert plan['family']=='N7' and plan['worlds']==16 and len(plan['variants'])==128
    recipes,_,_=recipe_library()
    direction=[int(1 in recipe)-int(2 in recipe) for recipe in recipes]
    receipt={}; total=0
    for item in summary['variants']:
        identity=item['variant']['id']; folder=root/'original'/identity
        groups=defaultdict(lambda:defaultdict(list)); seen=set()
        for post,truth in zip(rows(folder/'posterior_points.jsonl.gz'),rows(folder/'truth_points.jsonl.gz'),strict=True):
            world,case=post['world'],post['case']; key=(world,case)
            assert key not in seen and world in range(16) and case in range(plan['cases'])
            assert truth['identity']==['N7',identity,batch,world,case]
            seen.add(key); target=truth['truth']['presence']
            for name,detail in post['detail'].items():
                if name=='training': continue
                presence=detail['recipe_presence']
                probabilities=[p if t else 1-p for p,t in zip(presence,target,strict=True)]
                loss=math.fsum(-math.log(p) for p in probabilities)/len(probabilities) if all(p>0 for p in probabilities) else None
                bias=math.fsum((p-t)*d for p,t,d in zip(presence,target,direction,strict=True))/len(target)
                groups[name+':recipe_presence_loss'][world].append(loss)
                groups[name+':desirable_bias'][world].append(bias)
            groups['aspirational_maker'][world].append(float(truth['truth']['aspiration']))
        assert len(seen)==plan['worlds']*plan['cases']
        assert set(groups)==set(item['secondary'])
        for name,worlds in groups.items():
            assert set(worlds)==set(range(plan['worlds']))
            values=[math.fsum(worlds[w])/len(worlds[w]) if all(v is not None for v in worlds[w]) else None for w in range(plan['worlds'])]
            expected=item['secondary'][name]
            if any(v is None for v in values):
                assert expected['finite'] is False
                assert all(expected[k] is None for k in ('mean','lower','upper','se'))
                assert expected['unavailable']==sum(v is None for v in values)
            else:
                estimate=interval(values,[identity,batch,'secondary',name])
                assert expected['finite'] is True
                for k in ('mean','lower','upper','se'):
                    assert math.isclose(estimate[k],expected[k],abs_tol=1e-11,rel_tol=0),(identity,name,k)
        receipt[identity]=dict(cases=len(seen),metrics=len(groups),summary_sha256=digest(folder/'SUMMARY.json'))
        total+=len(seen)
    write(output,dict(state='PASSED',batch=batch,cases=total,variants=receipt,
        complete_sha256=digest(root/'COMPLETE.json'),posterior_secondary_formulas_independently_rebuilt=True,
        whole_world_intervals_reconstructed=True,unavailable_preserved=True,
        scope='Every retained N7 recipe-presence loss, signed desirable-drive bias, and aspiration indicator; no new draws or changed criterion.'))
    print('Verified all secondary metrics:',batch,total,flush=True)
finally:
    write(work/f'RECONSTRUCTION_COST_BATCH_{batch}.json',dict(cpu_seconds=time.process_time()-started,
        state='COMPLETE' if output.exists() else 'FAILED'))
