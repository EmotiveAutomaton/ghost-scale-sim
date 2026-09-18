"""Independent attachment-key reconstruction over retained forecast evidence."""
from collections import defaultdict
import json
from statistics import mean
from pathlib import Path
from ..v16.records import read,write,file_digest,digest
from .runtime import load_block
from .verify import physical


def report(root,output):
    plan=read(root/'PLAN.json');complete=read(root/'COMPLETE.json')
    assert complete['plan_sha256']==file_digest(root/'PLAN.json')
    groups=defaultdict(lambda:defaultdict(list));bindings={};count=0
    for name in complete['blocks']:
        block,receipt=load_block(root,name);bindings[name]=receipt['raw_sha256']
        for unit in block['units']:
            case=unit['case'];source,sr=load_block(root.parent/case['origin_run'],case['origin_block'])
            assert sr['raw_sha256']==case['origin_raw_sha256']
            original=source['units'][case['origin_index']];p=original['case']['public'];law=original['case']['private']['true_world']
            for row in unit['rows']:
                old=next(r for r in original['rows'] if digest(r)==row['origin_row_sha256'])
                assert row['evidence_sha256']==digest(dict(p,observations=old['observation_record']))
                lookup={};retrieval=0;checking=0;selection=0
                for obs in old['observation_record']:
                    retrieval+=1
                    if obs['query']['kind']=='context':continue
                    stopped=False
                    for t in obs['outcome']['trace']:
                        key=(frozenset(i for i,v in enumerate(t['before']) if v>=0),stopped,t['action'])
                        lookup.setdefault(key,[]).append(int(t['legal']));retrieval+=1;checking+=len(t['before']);stopped=t['stopped']
                probabilities=[];truths=[];covered=0
                for q in old['forecast_probes']:
                    key=(frozenset(i for i,v in enumerate(q['initial']) if v>=0),False,q['program'][0])
                    values=lookup.get(key,[]);checking+=len(q['initial']);selection+=max(1,len(values));covered+=bool(values)
                    probabilities.append(mean(values) if values else 0.5)
                    truths.append(physical(json.dumps(law,sort_keys=True),json.dumps(q['initial']),tuple(q['program']),p['max_steps'])['legal'])
                assert truths==old['forecast_truths'] and probabilities==row['forecast_probabilities']
                score=mean((prob-int(t))**2 for prob,t in zip(probabilities,truths))
                assert abs(score-row['forecast_brier'])<1e-12
                assert row['covered_probes']==covered
                assert row['forecast_operations']==checking+selection+retrieval==row['costs']['total_online']<=row['forecast_budget']
                metrics={'brier':score,'operations':row['forecast_operations'],'coverage':covered/len(truths)}
                for method,rival in row['paired'].items():
                    matched=next(r for r in original['rows'] if r['method']==method and r['budget']==row['construction_budget'] and r['query_policy']==row['query_policy'] and r['requested_queries']==row['requested_queries'])
                    assert rival==dict(forecast_brier=matched['forecast_brier'],forecast_operations=matched['forecast_operations'])
                    metrics[method+'_brier']=rival['forecast_brier'];metrics[method+'_operations']=rival['forecast_operations']
                dimensions=('origin_run','n','family','selection','donor','truth_excluded','query_policy','requested_queries')
                combined={**case,**row};key=tuple(combined[k] for k in dimensions)
                groups[key][case['structural_unit']].append(metrics);count+=1
    cells=[]
    for key,units in sorted(groups.items()):
        names=next(iter(units.values()))[0]
        cells.append(dict(zip(dimensions,key),**{'mean_'+m:mean(mean(r[m] for r in rs) for rs in units.values()) for m in names},distinct_context_units=len(units),rows=sum(map(len,units.values()))))
    result=dict(branch='g2-mask',rows_checked=count,cells=cells,raw_bindings=bindings,source_plan_sha256=file_digest(root/'PLAN.json'),
                scope='descriptive paired forecast diagnostic on reused evidence; no new physical support or task execution',aggregation='equal physical-context means with histories nested')
    write(output/'COMPARISONS.json',result)
    write(output/'VERIFICATION.json',dict(passed=True,rows_checked=count,summary_sha256=file_digest(output/'COMPARISONS.json'),verifier_sha256=file_digest(Path(__file__))))
    return result
