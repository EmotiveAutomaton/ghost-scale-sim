"""Independent mean reconstruction and bounded source-extracted weight replay.

Run with the pinned CPU Torch interpreter from extracted scientific source plus
this separately hashed driver. This verifies forecasts, not complete retraining.
"""
import argparse
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path
import platform
import time
import uuid
from ghostscale.validation.soundingline.v18_3 import torch_worker as T
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
import numpy as np
import torch


def subset(data,indices,pair=False):
    index=torch.as_tensor(indices,dtype=torch.long)
    keys=('base','source') if pair else ('sample',)
    used=torch.unique(torch.cat([data[k][index] for k in keys]),sorted=True)
    inverse=torch.full((len(data['history']),),-1,dtype=torch.long);inverse[used]=torch.arange(len(used))
    result={k:v[index] for k,v in data.items() if k not in ('history','length','world',*keys)}
    result.update(history=data['history'][used],length=data['length'][used])
    for key in keys:result[key]=inverse[data[key][index]]
    return result


def independently_average(root):
    raw=json.loads(gzip.decompress((root/'neural_points.json.gz').read_bytes()));summary=read(root/'SUMMARY.json')
    family=summary['family'];conditioned=family in ('E','E-purpose')
    metrics=('expected_loss','brier','total_variation') if conditioned else ('counterfactual_loss','behavior_loss','wrong_mapping_loss','incompatible_partition_loss','brier','total_variation')
    if family=='E-purpose':metrics+=('role_accuracy','whole_state_accuracy')
    grouped={}
    for row in raw['rows']:
        key=((row['condition'],row['method']) if conditioned else (row['method'],))+(row['lineage'],)
        for metric in metrics:
            value=row[metric]
            if isinstance(value,dict):value=None if value['infinite'] else value['value']
            grouped.setdefault(key,{}).setdefault(metric,[]).append(value)
    def mean(values):return None if any(x is None or not math.isfinite(x) for x in values) else math.fsum(values)/len(values)
    clusters={key:{m:mean(values) for m,values in metrics_.items()} for key,metrics_ in grouped.items()}
    for row in raw['clusters']:
        key=((row['condition'],row['method']) if conditioned else (row['method'],))+(row['lineage'],)
        for metric,value in clusters[key].items():
            if value is None:assert row[metric] is None
            else:assert abs(value-row[metric])<1e-10
    checked=0
    for key,reported in summary['cells'].items():
        condition,method=key.split('|') if conditioned else (None,key)
        selected=[values for tags,values in clusters.items() if tags[:-1]==((condition,method) if conditioned else (method,))]
        for metric in metrics:
            expected=mean([v[metric] for v in selected]);actual=reported[metric]
            assert actual['n']==len(selected)
            if expected is None:assert actual['mean'] is None
            else:assert abs(expected-actual['mean'])<1e-10
            checked+=1
    return dict(raw_rows=len(raw['rows']),lineage_clusters=len(clusters),independent_means_checked=checked)


def replay(root,output,campaign):
    source=Path(__file__).resolve().parents[1];started=time.monotonic();attempt='neural-audit-'+uuid.uuid4().hex
    with local_owner(campaign/'scientific-worker-owner'):
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(r['state']=='running' for r in previous):raise ValueError('unreconciled scientific owner')
        acceptance=read(campaign/'ACCEPTANCE.json')
        old=acceptance['prior_cpu_seconds']+sum(max(r['cpu_seconds'],r.get('native_cpu_seconds',0),r.get('uncertainty_cpu_seconds',0))+r.get('child_cpu_seconds',0) for r in previous)
        def emit(state):write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name+'-neural-audit',state=state,cpu_seconds=time.process_time(),child_cpu_seconds=0,wall_seconds=time.monotonic()-started),immutable=False)
        def pulse(**kw):
            emit('running')
            if old+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):raise TimeoutError('audit resource cutoff')
        emit('running')
        try:
            plan=read(root/'PLAN.json')
            assert all(file_digest(source/name)==value for name,value in plan['sources'].items())
            assert file_digest(root/'SOURCE.zip')==plan['source_archive_sha256']
            manifest=read(root/'SCIENTIFIC_MANIFEST.json') if (root/'SCIENTIFIC_MANIFEST.json').exists() else read(root/'COMPLETE.json')
            for name,value in manifest['files'].items():assert file_digest(root/name)==value,name
            checks=independently_average(root);pulse()
            child=root/'neural';complete=read(child/'COMPLETE.json');family=read(root/'SUMMARY.json')['family']
            environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
            if environment!=complete['environment']:raise ValueError('forecast replay environment differs')
            scratch=output.parent/'forecast-replay';scratch.mkdir(exist_ok=False);replays=[]
            if family=='E':
                inputs=read(root/'data/reader/INPUTS.json');tests={}
                for condition,entry in inputs['tests'].items():
                    data=T.dataset(root/'data/reader'/entry['name']);indices=np.linspace(0,len(data['sample'])-1,128,dtype=np.int64)
                    tests[condition]=(subset(data,indices),indices)
                for name,predictions in complete['predictions'].items():
                    for condition,entry in predictions.items():
                        data,indices=tests[condition];path=scratch/f'{name}-{condition}.npz'
                        T.forecast(child/entry['selected']/'BEST.pt',data,path)
                        with np.load(path,allow_pickle=False) as z:actual=z['probabilities']
                        with np.load(child/entry['file'],allow_pickle=False) as z:expected=z['probabilities'][indices]
                        error=float(np.max(abs(expected-actual)));assert error<=2e-6,(name,condition,error)
                        replays.append(dict(reader=name,condition=condition,probes=len(indices),max_absolute_error=error));pulse()
                for name,receipt in complete['fits'].items():
                    saved=torch.load(child/name/'BEST.pt',weights_only=True,map_location='cpu');model=T.Reader(**saved['spec'])
                    assert T.count(model)==receipt['parameters']
            elif family=='E-purpose':
                from ghostscale.validation.soundingline.v18_3 import purpose_worker as P,purpose_readout as R
                inputs=read(root/'data/reader/INPUTS.json')
                tests={}
                for condition,entry in inputs['tests'].items():
                    full=P.dataset(root/'data/reader'/entry['name'],public=True)
                    indices=np.linspace(0,len(full['history'])-1,128,dtype=np.int64)
                    tests[condition]=({k:v[indices] for k,v in full.items()},indices)
                for name,outputs in complete['predictions'].items():
                    with np.load(child/name/'READOUT.npz',allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
                    assert model['coefficients'].size+len(model['intercept'])==complete['fits'][name]['readout_parameters']
                    best=None if name=='raw-history' else root/'data/reader'/inputs['encoders'][name]['name']
                    for condition,entry in outputs.items():
                        data,indices=tests[condition];actual=R.predict(model,P.features(data,best))
                        with np.load(child/entry['file'],allow_pickle=False) as z:expected=z['probabilities'][indices]
                        error=float(np.max(abs(expected-actual)));assert error<=2e-6,(name,condition,error)
                        replays.append(dict(reader=name,condition=condition,probes=len(indices),max_absolute_error=error));pulse()
            else:
                from ghostscale.validation.soundingline.v18_3 import intervention_worker as I
                inputs=read(root/'data/reader/INPUTS.json');data=I.dataset(root/'data/reader'/inputs['test']['name'],public=True)
                indices=np.linspace(0,len(data['base'])-1,128,dtype=np.int64);data=subset(data,indices,pair=True)
                for name,entry in complete['predictions'].items():
                    path=scratch/f'{name}.npz';I.forecast(child/name/'BEST.pt',data,path,pulse)
                    maximum=0.
                    with np.load(path,allow_pickle=False) as z,np.load(child/entry['file'],allow_pickle=False) as expected:
                        forecast_types=list(z.files)
                        for key in z.files:maximum=max(maximum,float(np.max(abs(z[key]-expected[key][indices]))))
                    assert maximum<=2e-6,(name,maximum)
                    replays.append(dict(reader=name,probes=len(indices),forecast_types=forecast_types,max_absolute_error=maximum));pulse()
                    saved=torch.load(child/name/'BEST.pt',weights_only=True,map_location='cpu');model=I.Model(**saved['spec'])
                    assert T.count(model)==complete['fits'][name]['parameters']
            assert not torch.cuda.is_initialized()
            proof=dict(passed=True,plan_sha256=file_digest(root/'PLAN.json'),summary_sha256=file_digest(root/'SUMMARY.json'),
                verifier_sha256=file_digest(Path(__file__)),checks=checks,forecast_replays=replays,parameters_checked=len(complete['fits']),
                cpu_seconds=time.process_time(),scope='every retained lineage mean; fixed evenly spaced source-extracted weight forecasts; no complete retraining claim')
            write(output,proof);emit('complete');print(json.dumps(dict(passed=True,**checks,forecast_files_replayed=len(replays))))
        except BaseException:emit('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();replay(a.root,a.output,a.campaign)
