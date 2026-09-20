"""Run inside extracted packet source; independently regroup and replay outputs."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,canonical


def budget():
    from datetime import datetime,timezone
    if 'GHOST_VERIFY_DEADLINE' in os.environ and datetime.now(timezone.utc)>=datetime.fromisoformat(os.environ['GHOST_VERIFY_DEADLINE']):raise TimeoutError('absolute verification deadline')
    if time.process_time()>=float(os.environ.get('GHOST_VERIFY_CPU_LIMIT','inf')):raise TimeoutError('verification CPU allowance')


def finite(root):
    from ghostscale.validation.soundingline.v18_4 import runtime as R
    complete=read(root/'COMPLETE.json');summary=read(root/'SUMMARY.json');groups={};units=[]
    for block in complete['blocks']:
        budget()
        for unit in R.load(root,block):
            units.append(unit)
            dimensions=('family','cell','rule','tilt') if unit['family']=='P' else ('family','cell','condition','length','copy_span')
            metrics=('old_loss','new_loss','entropy_nats','old_optimum_ties') if unit['family']=='P' else ('expected_loss','pre_change_loss','post_change_loss','expected_match','abstention_loss')
            if unit['family']=='P2':
                dimensions=('family','cell','rule')
                metrics=('old_loss','new_loss','entropy_nats','old_optimum_ties','training_loss')
            if unit['family']=='U2':
                dimensions=('family','cell','condition','length','copy_span','change_at')
                metrics=('expected_loss','pre_change_loss','post_change_loss','expected_match','abstention_loss',
                         'prefix32_loss','prefix64_loss','prefix96_loss','age0_16_loss','age16_32_loss',
                         'age48_64_loss','prefix32_match','prefix64_match','prefix96_match')
            if unit['family']=='R1':
                dimensions=('family','cell','kind','order','copy_span','length')
                metrics=('expected_loss','final_loss','expected_match','expanded','purchase_step',
                         'likelihood_evaluations','net_match_low','net_match_high')
            if unit['family']=='R2':
                dimensions=('family','cell','kind','order','copy_span','length','prior_mode')
                metrics=('expected_loss','final_loss','expected_match','expanded','purchase_step',
                         'likelihood_evaluations','net_match_low','net_match_high','model_entropy')
            if unit['family']=='S1':
                dimensions=('family','shared','audit_true','audit_assumed','stake','audit_price')
                metrics=('expected_loss','expected_match','net_utility','audit_count','evidence_count','cost','graph_brier','false_confidence')
            if unit['family']=='L2c':
                dimensions=('family','cell','support')
                metrics=('expected_loss','brier','bank_residual','projection_gap','raw_invalid')
            for row in unit['rows']:
                tags={k:unit.get(k) for k in dimensions};tags['method']=row['method']
                if 'cardinality' in row:tags['cardinality']=row['cardinality']
                key=canonical(tags).decode()
                for metric in metrics:groups.setdefault((key,metric),[]).append(float(row[metric]))
    for (key,metric),values in groups.items():
        item=summary['cells'][key][metric]
        if item['n']!=len(values) or abs(math.fsum(values)/len(values)-item['mean'])>1e-10:raise ValueError('independent mean differs')
    indices=sorted({int(i*(len(units)-1)/7) for i in range(8)})
    for i in indices:
        budget()
        if json.loads(canonical(R.dispatch(units[i]['request'])))!=units[i]:raise ValueError('whole-unit replay differs')
    return dict(units=len(units),independent_means=len(groups),whole_unit_replays=indices)


def target_audit(root):
    if (root/'data/BUILD_PLAN.json').exists() and read(root/'data/BUILD_PLAN.json').get('state_sampling')=='iid':
        from ghostscale.validation.soundingline.v18_4.conditional_targets import audit
        actual=audit(root/'data')
        saved=read(root/'data/TARGET_AUDIT.json')
        if not actual['passed'] or not saved['passed']:raise ValueError('failed target audit')
        for key in actual:
            if key=='max_teacher_error':
                if max(actual[key],saved[key])>1e-12:raise ValueError('target numerical reconstruction failed')
            elif actual[key]!=saved[key]:raise ValueError('independent target audit changed')
        return actual
    return None


def neural(root,output):
    from ghostscale.validation.soundingline.v18_4 import torch_worker as T
    import torch
    supervision=target_audit(root)
    raw=json.loads(gzip.decompress((root/'neural_points.json.gz').read_bytes()));summary=read(root/'SUMMARY.json')
    grouped={};metrics=('expected_loss','brier','total_variation')
    for row in raw['rows']:
        key=(row['condition'],row['method'],row['lineage'])
        for metric in metrics:
            value=row[metric]
            if isinstance(value,dict):value=None if value['infinite'] else value['value']
            grouped.setdefault(key,{}).setdefault(metric,[]).append(value)
    def mean(v):return None if any(x is None for x in v) else math.fsum(v)/len(v)
    clusters={k:{m:mean(v) for m,v in a.items()} for k,a in grouped.items()};checked=0
    for key,reported in summary['cells'].items():
        condition,method=key.split('|');selected=[a for k,a in clusters.items() if k[:2]==(condition,method)]
        for metric in metrics:
            expected=mean([a[metric] for a in selected]);actual=reported[metric]
            if actual['n']!=len(selected):raise ValueError('lineage count differs')
            if expected is None:
                if actual['mean'] is not None:raise ValueError('missing mean differs')
            elif abs(expected-actual['mean'])>1e-10:raise ValueError('independent neural mean differs')
            checked+=1
    bank_means=0
    if read(root/'PLAN.json')['design']['study']=='E-bank':
        diagnostics=read(root/'data/BANK_DIAGNOSTICS.json')
        for key,reported in summary['bank_diagnostics'].items():
            condition,mode=key.split('|')
            selected=[r for r in diagnostics['rows'] if r['condition']==condition and r['mode']==mode]
            for metric,values in reported.items():
                actual=[r[metric] for r in selected]
                expected=dict(minimum=min(actual),maximum=max(actual),mean=math.fsum(actual)/len(actual))
                if any(not math.isclose(values[k],v,rel_tol=1e-12,abs_tol=1e-10) for k,v in expected.items()):raise ValueError('independent bank-law summary differs')
                bank_means+=1
    complete=read(root/'neural/COMPLETE.json');inputs=read(root/'data/reader/INPUTS.json');replays=[]
    scratch=output.parent/'forecast-replay';scratch.mkdir(exist_ok=True)
    for reader,predictions in complete['predictions'].items():
        for condition,entry in predictions.items():
            budget()
            data=T.dataset(root/'data/reader'/inputs['tests'][condition]['name'])
            indices=torch.as_tensor(np.linspace(0,len(data['sample'])-1,64,dtype=np.int64))
            used=torch.unique(data['sample'][indices],sorted=True)
            inverse=torch.full((len(data['history']),),-1,dtype=torch.long);inverse[used]=torch.arange(len(used))
            subset=dict(history=data['history'][used],length=data['length'][used],query=data['query'][indices],sample=inverse[data['sample'][indices]])
            path=scratch/f'{reader}-{condition}.npz'
            if read(root/'PLAN.json')['design']['study']=='E-decoder':
                from ghostscale.validation.soundingline.v18_4.decoder_worker import forecast
                fit=root/'neural'/entry['selected']
                forecast(fit/'READOUT.npz',subset,path,root/'data/reader',read(fit/'COMPLETE.json')['identity'])
            elif read(root/'PLAN.json')['design']['study']=='E-bank':
                from ghostscale.validation.soundingline.v18_4.bank_worker import forecast,load_maps
                # Full-file batches preserve floating arithmetic before ill-conditioned maps.
                maps=load_maps(root/'data/reader'/inputs['maps'][condition]['name'],len(data['sample'])//len(data['history']))
                weight=inputs['encoders'][entry['selected']]
                forecast(root/'data/reader'/weight['name'],data,maps,path,entry['mode'])
            else:T.forecast(root/'neural'/entry['selected']/'BEST.pt',subset,path)
            with np.load(path,allow_pickle=False) as z,np.load(root/'neural'/entry['file'],allow_pickle=False) as original:
                bank_study=read(root/'PLAN.json')['design']['study']=='E-bank'
                replay=z['probabilities'][indices.numpy()] if bank_study else z['probabilities']
                error=float(np.max(abs(replay-original['probabilities'][indices.numpy()])))
            if error>2e-6:raise ValueError('forecast replay differs')
            if read(root/'PLAN.json')['design']['study']=='E-bank':
                with np.load(path,allow_pickle=False) as z,np.load(root/'neural'/entry['file'],allow_pickle=False) as original:
                    raw_error=float(np.max(abs(z['raw_probabilities']-original['raw_probabilities'])))
                    raw=original['raw_probabilities'];p=original['probabilities']
                    invalid=int(np.sum(np.any(raw< -1e-8,axis=1)|np.any(raw>1+1e-8,axis=1)|(abs(raw.sum(1)-1)>1e-6)))
                    if invalid!=entry['raw_invalid_rows'] or abs(float(np.mean(abs(raw-p).sum(1)))-entry['mean_repair_l1'])>1e-10:raise ValueError('bank invalidity accounting differs')
                if raw_error>2e-6:raise ValueError('raw bank replay differs')
            replays.append(dict(reader=reader,condition=condition,probes=64,max_absolute_error=error,
                **(dict(raw_forecast_rows=len(raw),max_raw_absolute_error=raw_error,invalidity_reconstructed=True) if bank_study else {})))
    if torch.cuda.is_initialized():raise ValueError('unexpected CUDA')
    return dict(independent_means=checked,lineage_clusters=len(clusters),forecast_replays=replays,
        **(dict(independent_target_audit=supervision) if supervision else {}),
        **(dict(independent_bank_diagnostic_means=bank_means) if bank_means else {}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root;source=Path(__file__).resolve().parents[1]
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);handle=ctypes.c_void_p(-1)
        if not kernel.SetPriorityClass(handle,0x4000) or kernel.GetPriorityClass(handle)!=0x4000:
            raise ctypes.WinError(ctypes.get_last_error())
    budget()
    write(args.output.parent/'REPLAY-STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid()),immutable=False)
    plan=read(root/'PLAN.json');complete=read(root/'COMPLETE.json')
    if any(file_digest(source/name)!=sha for name,sha in plan['sources'].items()):raise ValueError('extracted scientific source differs')
    if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('source archive differs')
    if plan['design']['engine']=='neural':
        manifest=read(root/'SCIENTIFIC_MANIFEST.json') if (root/'SCIENTIFIC_MANIFEST.json').exists() else complete
        for name,sha in manifest['files'].items():
            if file_digest(root/name)!=sha:raise ValueError('retained neural artifact differs')
        checked=neural(root,args.output)
    else:
        if complete['summary_sha256']!=file_digest(root/'SUMMARY.json'):raise ValueError('summary changed')
        checked=finite(root)
    write(args.output,dict(passed=True,plan_sha256=file_digest(root/'PLAN.json'),summary_sha256=file_digest(root/'SUMMARY.json'),
        verifier_sha256=file_digest(Path(__file__)),checks=checked,cpu_seconds=time.process_time(),
        scope='all retained means; eight evenly spaced whole-unit replays or 64 forecast rows per reader/test from extracted source; not full retraining'))


if __name__=='__main__':main()
