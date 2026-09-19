"""CPU encoding/decoding costs of frozen selected E readers; no fitting."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import platform
import time
import uuid
from ghostscale.validation.soundingline.v18_3 import torch_worker as T
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
import numpy as np
import torch


def measure(call,count=500,items=64):
    for _ in range(5):call()
    values=[];cpu_values=[]
    for _ in range(5):
        started=time.perf_counter();cpu=time.process_time()
        for _ in range(count):call()
        cpu_values.append((time.process_time()-cpu)/(count*items))
        values.append((time.perf_counter()-started)/(count*items))
    if np.median(cpu_values)<=0:raise ValueError('CPU benchmark below timer resolution')
    return dict(median_seconds=float(np.median(values)),range_seconds=[min(values),max(values)],replicates=values,
        median_cpu_seconds=float(np.median(cpu_values)),cpu_replicates=cpu_values,calls_per_replicate=count)


def run(root,campaign,output):
    with local_owner(campaign/'scientific-worker-owner'):
        acceptance=read(campaign/'ACCEPTANCE.json');attempt=campaign/'attempts'/('benchmark-'+uuid.uuid4().hex+'.json')
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('science still active')
        old=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        def pulse(state):
            write(attempt,dict(packet='E-frozen-reader-costs',state=state,cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
            if state=='running' and (old+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])):raise TimeoutError('cost audit cutoff')
        try:
            pulse('running');complete=read(root/'neural/COMPLETE.json');inputs=read(root/'data/reader/INPUTS.json')
            environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
            if environment!=complete['environment']:raise ValueError('cost environment mismatch')
            data=T.dataset(root/'data/reader'/inputs['tests']['in-support']['name'])
            # The same deterministic 64 histories are used for every method.
            h=data['history'][:64];length=torch.full((64,),32,dtype=torch.long);q=data['query'][::5][:64]
            rows=[]
            with torch.no_grad():
                for reader,forecasts in complete['predictions'].items():
                    selected=forecasts['in-support']['selected'];path=root/'neural'/selected/'BEST.pt';saved=torch.load(path,weights_only=True,map_location='cpu')
                    if file_digest(path)!=complete['fits'][selected]['best_sha256']:raise ValueError('frozen weight changed')
                    model=T.Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval();state=model.encode(h,length)
                    encoded=measure(lambda:model.encode(h,length));decoded=measure(lambda:model.decode(state,q),count=2000)
                    kind,seed=reader.rsplit('-seed',1)
                    training=sum(v['final_attempt_cpu_seconds'] for k,v in complete['fits'].items() if k.startswith(reader+'-width'))
                    rows.append(dict(reader=reader,method=kind,seed=int(seed),selected=selected,weight_sha256=file_digest(path),
                        parameters=T.count(model),state_floats=model.width,raw_history_floats=int(np.prod(h.shape[1:])),
                        public_world_floats=17,full_encoding=encoded,cached_query_decoding=decoded,capacity_search_fit_cpu_seconds=training))
                    pulse('running')
            if torch.cuda.is_initialized():raise ValueError('unexpected GPU')
            write(output,dict(rows=rows,environment=environment,parent_complete_sha256=file_digest(root/'COMPLETE.json'),
                source_sha256=file_digest(__file__),cpu_seconds=time.process_time(),history_envelope=32,batch_histories=64,
                scope='five local timing replicates, each 500 encoding or 2000 decoding calls after five warmups; CPU and wall time retained separately; all readers may cache; fit CPU includes both capacity candidates but excludes setup and failed prior packet work, which is charged separately'))
            pulse('complete')
        except BaseException:pulse('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.campaign,a.output)
