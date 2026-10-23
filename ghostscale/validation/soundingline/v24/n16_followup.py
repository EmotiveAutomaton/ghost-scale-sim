"""Descriptive retained-data misspecification map; no new worlds or selection."""
import math
from pathlib import Path
from ..v23.common import read, write, digest
from ..v23.family import rows
from ..v23.pipeline import Stream
from .theory_pilot import close

BUDGETS=(2,4,6)
SCENARIOS=('correct','high-reliability','truth-omitted','other-omitted')
POLICIES=('P2','P3','P4','P5')


def configuration():
    return dict(blocks=4,cases=15360,budgets=list(BUDGETS),scenarios=list(SCENARIOS),policies=list(POLICIES),
                epsilon=.1,reference='correct-model fixed-order P1 at the same budget',inference='retained descriptive, no new worlds')


def validate_design(plan):
    if (plan['kind']!='pilot' or plan['family']!='N16' or plan['task']!='v24-n16-retained'
            or plan['identity']!='v24-n16-retained-stress-1' or plan['configuration']!=configuration()
            or plan['cpu_cap_seconds']!=1800 or plan['confirmation'] is not False):
        raise ValueError('unknown retained-data design')
    for name,sha in plan['input_files'].items():
        if digest(Path(plan['input_root'])/name)!=sha:raise ValueError('retained input changed')


def calculate(evidence,truth,independent=False):
    lookup={(r['scenario'],r['policy']):r['mass'] for r in evidence['posteriors']}
    mechanism=truth['mechanism'];out=[]
    mechanisms=len(lookup['correct','P1'][0])
    for scenario in SCENARIOS:
        names=([scenario] if scenario in ('correct','high-reliability') else
               [f'omit-{mechanism}'] if scenario=='truth-omitted' else
               [f'omit-{i}' for i in range(mechanisms) if i!=mechanism])
        for policy in POLICIES:
            for budget in BUDGETS:
                q=lookup['correct','P1'][budget-1][mechanism]
                if q<=0:raise ValueError('correct comparator lacks support')
                values=[]
                for name in names:
                    probability=lookup[name,policy][budget-1][mechanism]
                    value=math.log(.9*(probability/q)+.1) if independent else math.log(.9*probability+.1*q)-math.log(q)
                    if value<math.log(.1)-1e-11:raise ValueError('mixture support failed')
                    values.append(value)
                out.append(math.fsum(values)/len(values))
    primary=math.log(.9*lookup['correct','P4'][1][mechanism]+.1*lookup['correct','P1'][1][mechanism])-math.log(lookup['correct','P1'][1][mechanism])
    return out,primary


def summarize(records):
    keys=[tuple(r['identity']) for r in records]
    if len(keys)!=15360 or len(set(keys))!=15360:raise ValueError('incomplete retained case roster')
    # Each source has 12 complete settings, 16 laws and five cases per block.
    sources=('N4d','N9','N11','N15');cells=[]
    for source in sources:
        for index,(scenario,policy,budget) in enumerate((s,p,b) for s in SCENARIOS for p in POLICIES for b in BUDGETS):
            means=[]
            for block in range(4):
                selected=[r['gains'][index] for r in records if r['identity'][0]==block and r['identity'][2].split('-')[1]==source]
                if len(selected)!=960:raise ValueError('incomplete source-balanced block')
                means.append(math.fsum(selected)/len(selected))
            cells.append(dict(source=source,scenario=scenario,policy=policy,budget=budget,
                balanced_block_means=means,mean=math.fsum(means)/4,minimum_block=min(means),maximum_block=max(means)))
    return dict(state='RETAINED_DESCRIPTIVE_STRESS_MAP',cases=len(records),independent_blocks=4,new_worlds=0,cells=cells,
                confirmation=False,limitations=['Reference is the correct-model fixed order for every stress scenario.',
                'Four retained blocks do not provide a powered decision; no hypothesis selection or confirmation.',
                'Truth-omitted mixtures receive comparator support; this does not recover the omitted mechanism.'])


def preflight(plan):
    validate_design(plan)
    source=Path(plan['input_root'])
    evidence=next(rows(source/'evidence_points.jsonl.gz'));truth=next(rows(source/'truth_points.jsonl.gz'))
    a,primary=calculate(evidence['evidence'],truth['truth']);b,_=calculate(evidence['evidence'],truth['truth'],True)
    close(a,b);close(primary,evidence['score']['mixture_gain'])
    return dict(passed=True,new_scientific_worlds=0,retained_inputs_bound=True,independent_ratio_identity=True)


def run(root,plan,pulse,independent=False):
    validate_design(plan);root=Path(root);source=Path(plan['input_root']);records=[]
    if not independent:root.mkdir(exist_ok=False)
    stream=Stream(root/'scores_points.jsonl.gz') if not independent else None
    saved=iter(rows(root/'scores_points.jsonl.gz')) if independent else None
    try:
        for e,t in zip(rows(source/'evidence_points.jsonl.gz'),rows(source/'truth_points.jsonl.gz'),strict=True):
            if e['identity']!=t['identity']:raise ValueError('source alignment changed')
            gains,primary=calculate(e['evidence'],t['truth'],independent);close(primary,e['score']['mixture_gain'])
            record=dict(identity=e['identity'],gains=gains);records.append(record)
            if independent:close(record,next(saved))
            else:stream.write(record)
            if len(records)%256==0:pulse(cases=len(records))
    finally:
        if stream:stream.close()
    result=summarize(records)
    if independent:
        if next(saved,None) is not None:raise ValueError('extra derived rows')
        close(result,read(root/'SUMMARY.json'))
        return dict(state='N16_RETAINED_INDEPENDENT_REVIEW_PASSED',cases=len(records),primary_reproduced=True,new_worlds=0,confirmation=False)
    write(root/'SUMMARY.json',result)
    write(root/'RAW_INDEX.json',{'scores_points.jsonl.gz':dict(sha256=digest(root/'scores_points.jsonl.gz'),bytes=(root/'scores_points.jsonl.gz').stat().st_size)})


def execute(root,plan,pulse):return run(root,plan,pulse)


def reconstruct(root,plan,pulse):
    index=read(Path(root)/'RAW_INDEX.json')
    for name,item in index.items():
        if digest(Path(root)/name)!=item['sha256']:raise ValueError('derived raw corruption')
    return run(root,plan,pulse,independent=True)
