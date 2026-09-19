"""Repair numerical tie credit without changing any learned model or forecast."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
from ghostscale.validation.soundingline.v18_3 import purpose_data as P
from ghostscale.validation.soundingline.v18_3.runtime import REPO,fingerprint
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now
from ghostscale.validation.soundingline.v16.runtime import local_owner
from runners.export_v18_3_neural import selected


def run(root,campaign):
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        plan=read(root/'PLAN.json');old=campaign/plan['design']['parent_packet'];acceptance=read(campaign/'ACCEPTANCE.json')
        if file_digest(old/'COMPLETE.json')!=plan['design']['parent_complete_sha256']:raise ValueError('scoring parent changed')
        if fingerprint()!=plan['environment'] or any(file_digest(REPO/name)!=value for name,value in plan['sources'].items()):raise ValueError('scoring source/runtime changed')
        if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('scoring source archive changed')
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('active science owner')
        charged=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        attempt=campaign/'attempts'/('purpose-rescore-'+uuid.uuid4().hex+'.json')
        def pulse(state='running',**detail):
            write(attempt,dict(packet=root.name,state=state,cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
            if state=='running' and (charged+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])):raise TimeoutError('scoring repair cutoff')
        try:
            pulse();copied={}
            for source in selected(old):
                name=source.relative_to(old).as_posix()
                if name in ('PLAN.json','SOURCE.zip','SUMMARY.json','neural_points.json.gz'):continue
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                if file_digest(target)!=file_digest(source):raise ValueError('reused scientific input changed')
                copied[name]=file_digest(target)
            P.score(root,pulse)
            summary=read(root/'SUMMARY.json');original=read(old/'SUMMARY.json')
            for key,value in summary['cells'].items():
                for metric in ('expected_loss','brier','total_variation'):
                    if value[metric]!=original['cells'][key][metric]:raise ValueError('repair changed a proper score')
            # Independent scalar tie credit for every retained per-case forecast.
            raw=json.loads(gzip.decompress((root/'neural_points.json.gz').read_bytes()))['rows'];lookup={}
            inputs=read(root/'data/EVALUATOR.json');complete=read(root/'neural/COMPLETE.json')
            for condition,entry in inputs['truth'].items():
                with np.load(root/'data'/entry['name'],allow_pickle=False) as z:
                    truth=z['target'];ids=z['ids'];forecasts={m:z[k].copy() for m,k in (('exact','exact'),('passive-summary','passive'),('intervention-summary','intervention'))}
                for reader,outputs in complete['predictions'].items():
                    with np.load(root/'neural'/outputs[condition]['file'],allow_pickle=False) as z:forecasts[reader]=z['probabilities'].copy()
                for reader,p in forecasts.items():
                    method,seed=reader.rsplit('-seed',1) if '-seed' in reader else (reader,None)
                    for i,(cell,lineage) in enumerate(ids):
                        credit=[]
                        for lo,hi in ((0,3),(3,5),(5,7),(7,9)):
                            target=max(range(lo,hi),key=lambda j:truth[i,j]);maximum=max(float(p[i,j]) for j in range(lo,hi))
                            choices=[j for j in range(lo,hi) if float(p[i,j])>=maximum-1e-10]
                            credit.append(1/len(choices) if target in choices else 0.)
                        lookup[(condition,method,None if seed is None else int(seed),int(cell),int(lineage))]=(math.fsum(credit)/4,math.prod(credit))
            for row in raw:
                a,b=lookup[(row['condition'],row['method'],row['seed'],row['cell'],row['lineage'])]
                if abs(a-row['role_accuracy'])>1e-12 or abs(b-row['whole_state_accuracy'])>1e-12:raise ValueError('independent tie credit differs')
            summary['scoring_repair']=dict(parent_packet=old.name,parent_summary_sha256=file_digest(old/'SUMMARY.json'),unchanged_proper_scores=True,
                unchanged_inputs_weights_forecasts=copied,independent_scalar_accuracy_rows=len(raw),scope='scoring-only repair; no retraining or fresh lineages')
            write(root/'SUMMARY.json',summary,immutable=False)
            files={p.relative_to(root).as_posix():file_digest(p) for p in selected(root)}
            write(root/'COMPLETE.json',dict(files=files,completed_at=now(),validity='unchanged complete scientific forecasts; scalar-verified numerical-tie scoring repair; source replay pending'))
            pulse('complete')
        except BaseException:pulse('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();run(a.root,a.campaign)
