"""Prespecified retained-data persistence and whole-world bootstrap; no new makers."""
import argparse
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v26.common import read,write,digest,canonical,rng,close
from ghostscale.validation.soundingline.v26 import friday
from ghostscale.validation.soundingline.v23 import runtime
from runners.run_v26_followups import verify_complete


def analyze(parents,independent=False):
    rows=[];bindings={}
    for packet in parents:
        verify_complete(packet);plan=read(packet/'PLAN.json');bindings[packet.name]=digest(packet/'COMPLETE.json')
        index=read(packet/'original/RAW_INDEX.json');worlds=defaultdict(list)
        for name,bound in index.items():
            path=packet/'original'/name
            if digest(path)!=bound['sha256']:raise ValueError('retained raw changed')
            with gzip.open(path,'rt',encoding='utf-8') as stream:world=json.load(stream)
            if plan['family']=='E1':
                for maker in world['data']['truth_and_observations']:
                    typ=world['data']['public']['hypotheses'][maker['truth']]['kind']
                    if typ not in ('A','B'):continue
                    path=[]
                    for obs in maker['records']:
                        a,b=obs['type_posterior'][:2]
                        # No probability flooring. These retained A/B trajectories
                        # have positive marginals; a violation is an explicit hold.
                        if a<=0 or b<=0:raise ValueError('retained marginal underflow; stable reconstruction required')
                        path.append(math.log(b)-math.log(a))
                    m=friday.crossing_metrics(path)
                    if independent:
                        hits=np.flatnonzero(np.array(path)>=1)+1
                        last=np.flatnonzero(np.array(path)<1)+1
                        assert m['first_crossing']==(int(hits[0]) if len(hits) else None)
                        assert m['sustained_crossing']==(int(last[-1]+1) if len(last) and last[-1]<len(path) else 1 if not len(last) else None)
                    worlds[(world['setting'],typ)].append(m)
        if plan['family']=='E1':
            for (setting,truth),ms in sorted(worlds.items()):
                item=dict(packet=packet.name,setting=setting,configuration=plan['settings'][setting],truth=truth,worlds=len(ms),
                    ever_crossed=sum(x['ever_crossed'] for x in ms)/len(ms),final_above=sum(x['final_above'] for x in ms)/len(ms),
                    recrossed_below=sum(x['recrossed_below'] for x in ms)/len(ms),sustained_unresolved=sum(x['sustained_crossing'] is None for x in ms),world_metrics=ms)
                rows.append(item)
        else:
            summary=read(packet/'original/SUMMARY.json');curve=sorted((c for c in summary['cells'] if c['metric']=='excess_capture' and c['labels']['truth']==0),key=lambda c:c['labels']['history'])
            mat=np.array([c['whole_world_values'] for c in curve]).T;indices=rng('friday-retained-bootstrap').integers(0,len(mat),(2000,len(mat)));boot=[]
            for draw in indices:
                means=np.array([math.fsum(float(mat[i,j]) for i in draw)/len(draw) for j in range(64)]) if independent else mat[draw].mean(0)
                inside=abs(means)<=.05;first=next((i+1 for i,x in enumerate(inside) if x),None)
                sustained=next((i+1 for i in range(64) if all(inside[i:])),None)
                boot.append([first,sustained])
            rows.append(dict(packet=packet.name,bootstrap_resamples=2000,worlds=len(mat),seed_domain='friday-retained-bootstrap',
                             first_crossing_histogram={str(k):sum(b[0]==k for b in boot) for k in [None,*range(1,65)]},
                             sustained_crossing_histogram={str(k):sum(b[1]==k for b in boot) for k in [None,*range(1,65)]},
                             limitation='Descriptive whole-world bootstrap on 16 worlds; not a confidence-certified required sample size.'))
    return dict(bindings=bindings,rows=rows,no_new_scientific_draws=True)


def main(campaign):
    runtime.below_normal();started=time.process_time()
    parents=[campaign.parent/'v26/packets/v26-e1-study-1',campaign.parent/'v26-followups-20261103/packets/v26-e1-followup-20261103-study',campaign.parent/'v26-followups-20261103/packets/v26-u1-followup-20261103-study']
    root=campaign/'retained-analysis';root.mkdir()
    write(root/'PROTOCOL.json',dict(source_sha256=digest(Path(__file__)),inputs={p.name:digest(p/'COMPLETE.json') for p in parents},
        question='Do first evidence crossings persist, and how variable is the empirical source-training crossing?',
        rules='A/B type odds threshold one nat, finite-horizon sustained crossing; no probability floors; 2000 fixed-seed whole-world bootstrap resamples of U1; original, replay, scalar reconstruction.'))
    try:
        original=analyze(parents);write(root/'ORIGINAL.json',original)
        replay=analyze(parents);write(root/'REPLAY.json',replay)
        assert canonical(original)==canonical(replay)
        scalar=analyze(parents,True);close(original,scalar,tol=1e-12)
        write(root/'VALIDITY.json',dict(passed=True,byte_replay=True,independent_crossing_and_scalar_bootstrap=True,source_sha256=digest(Path(__file__)),result_sha256=digest(root/'ORIGINAL.json')))
        repo=Path(read(campaign/'LOCAL_PATHS.json')['repo']);dest=repo/'results/v26/friday-20261104/retained-analysis';dest.mkdir()
        for name in ('PROTOCOL.json','ORIGINAL.json','VALIDITY.json'):write(dest/name,read(root/name))
        print(json.dumps({k:v for k,v in original.items() if k!='rows'}));print('verified retained analysis',len(original['rows']))
    finally:write(campaign/'accounting'/'retained-analysis.json',dict(pool='studies',family='E1',kind='study',cpu_seconds=time.process_time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();main(a.campaign.resolve())
