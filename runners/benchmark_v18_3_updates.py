"""Measure one cached recurrent update against direct history recomputation."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import uuid
from ghostscale.validation.soundingline.v18_3 import torch_worker as T
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
from runners.benchmark_v18_3_torch import measure
import torch


def cached_update(model,state,token):
    if model.kind=='direct':raise ValueError('direct history needs its full envelope')
    encoders=model.encoders if model.kind=='split' else [model.encoder]
    chunks=state.chunk(len(encoders),dim=1)
    return torch.cat([encoder(token,chunk.unsqueeze(0).contiguous())[1][0] for encoder,chunk in zip(encoders,chunks)],dim=1)


def run(root,campaign,output):
    with local_owner(campaign/'scientific-worker-owner'):
        acceptance=read(campaign/'ACCEPTANCE.json');previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('science active')
        old=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        attempt=campaign/'attempts'/('update-costs-'+uuid.uuid4().hex+'.json')
        def pulse(state):
            write(attempt,dict(packet='E-one-observation-update-costs',state=state,cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
            if state=='running' and (old+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])):raise TimeoutError('update benchmark cutoff')
        try:
            pulse('running');complete=read(root/'neural/COMPLETE.json');inputs=read(root/'data/reader/INPUTS.json')
            data=T.dataset(root/'data/reader'/inputs['tests']['in-support']['name']);n=int(data['length'][0]);chosen=torch.where(data['length']==n)[0][:64]
            if len(chosen)!=64 or n<2:raise ValueError('insufficient same-length histories')
            h=data['history'][chosen];length=data['length'][chosen];prefix_length=length-1;token=h[:,n-1:n];rows=[]
            with torch.no_grad():
                for reader,forecasts in complete['predictions'].items():
                    selected=forecasts['in-support']['selected'];path=root/'neural'/selected/'BEST.pt';saved=torch.load(path,weights_only=True,map_location='cpu')
                    assert file_digest(path)==complete['fits'][selected]['best_sha256']
                    model=T.Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval()
                    if model.kind=='direct':
                        call=lambda:model.encode(h,length);error=0.
                    else:
                        prior=model.encode(h,prefix_length);call=lambda:cached_update(model,prior,token)
                        error=float((call()-model.encode(h,length)).abs().max())
                        if error>1e-6:raise ValueError('incremental state differs from full history')
                    timing=measure(call,count=1000)
                    rows.append(dict(reader=reader,method=model.kind,selected=selected,weight_sha256=file_digest(path),actual_history_length=n,full_envelope_length=h.shape[1],max_state_error=error,update=timing))
                    pulse('running')
            assert not torch.cuda.is_initialized() and torch.get_num_threads()==torch.get_num_interop_threads()==1
            write(output,dict(rows=rows,parent_complete_sha256=file_digest(root/'COMPLETE.json'),source_sha256=file_digest(__file__),environment=complete['environment'],
                scope='same 64 retained histories; one last observed token from cached prefix for recurrent readers, full fixed-envelope recomputation for direct; existing state preparation and subsequent query excluded; five replicates after five warmups; no new fitting or scientific data'))
            pulse('complete')
        except BaseException:pulse('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.campaign,a.output)
