"""Retained discovery-score design calibration. No new worlds or confirmation."""
import gzip
import json
import math
import time
from pathlib import Path
import numpy as np
from ..v23.common import digest, read, write
from . import evidence, n16


def calibrate_n16(raw_root, raw_index, original_summary, public_proof):
    started=time.process_time();raw_root=Path(raw_root);index=read(raw_index)
    variants=n16.model.variants();rows=[];sources={}
    for variant in variants:
        relative=variant['id']+'/evidence_points.jsonl.gz';path=raw_root/relative
        actual=digest(path)
        if actual!=index[relative]['sha256'] or path.stat().st_size!=index[relative]['bytes']:
            raise ValueError('retained raw source mismatch')
        sources[relative]=actual
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);family,identity,phase,law,case=row['identity']
                if family!='N16' or identity!=variant['id'] or phase!='discovery' or not 0<=case<64:
                    raise ValueError('wrong discovery identity')
                rows.append(dict(identity=[case,law,identity,0],raw_gain=float(row['metrics']['primary'])))
    # Check all 64 retained case indices, all 16 laws and all 48 settings.
    for row in rows:row['mixture_gain']=row['raw_gain']
    raw_blocks=n16.aggregate(rows,1)
    if len(raw_blocks)!=64 or len(rows)!=49152:raise ValueError('incomplete retained calibration')
    choice=evidence.choose_epsilon([r['raw_gain'] for r in rows])
    summary=read(original_summary)
    if digest(original_summary)!=index['SUMMARY.json']['sha256']:
        raise ValueError('original summary source mismatch')
    if not math.isclose(choice['raw_mean'],summary['study_readout']['primary']['interval']['mean'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('retained primary was not reproduced before reuse')
    epsilon=choice['epsilon'];result=dict(
        state='DESIGN_CALIBRATION_ONLY',new_scientific_worlds=0,confirmation=False,
        retained_cases=len(rows),settings=48,laws=16,
        grouping='Complete five-case groups across all settings and laws; preserves N9 paired mechanism draws',
        epsilon_choice=choice,source_streams=sources,
        raw_index_sha256=digest(raw_index),original_summary_sha256=digest(original_summary),
        public_proof_sha256=digest(public_proof),
        limitations=['Discovery-selected epsilon and plug-in expected evidence are not power guarantees.',
                    'Future pilot must measure complete CPU costs and check the forecast on new excluded seeds.',
                    'The assumed first-family e-LOND threshold is not a frozen V24 confirmation order.'])
    if epsilon is not None:
        for row in rows:row['mixture_gain']=float(evidence.mixture_gain(row['raw_gain'],epsilon))
        complete=[dict(identity=[r['identity'][0]//5,r['identity'][1],r['identity'][2],r['identity'][0]%5],
                       mixture_gain=r['mixture_gain']) for r in rows if r['identity'][0]<60]
        blocks=n16.aggregate(complete,5)
        # Retained batch's complete original + replay + independent review cost.
        cpu=9513.109375/64*5
        result.update(mixture_mean=float(np.mean([r['mixture_gain'] for r in rows])),lower_bound=math.log(epsilon),
            retained_complete_groups=12,cases_per_group=5*16*48,
            forecast_excluded_tail_cases=4*16*48,
            forecast_exclusion_reason='Indices 60-63 cannot form the fifth complete N9 mechanism cycle; retained in epsilon calibration and all source records',
            assumed_log_threshold=math.log(20),assumed_alpha=.1,assumed_first_gamma=.5,
            conservative_cpu_seconds_per_balanced_replicate=cpu,
            cpu_source='N16_REPORT.md: 9513.109375 CPU seconds / 64 cases per cell * 5 cases per complete group',
            forecast=evidence.expected_sample_plan(blocks,math.log(epsilon),math.log(20),cpu),
            retained_balanced_mixture_means=blocks)
    result['process_cpu_seconds']=time.process_time()-started
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    for name in ('raw-root','raw-index','original-summary','public-proof','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    write(args.output,calibrate_n16(args.raw_root,args.raw_index,args.original_summary,args.public_proof))
