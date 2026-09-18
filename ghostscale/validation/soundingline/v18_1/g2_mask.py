"""Orientation-invariant legality table on already retained G2 evidence.

This is a forecast-only rival. It never receives evaluator laws, acquires another
observation, or re-executes an existing scientific task. The public operation
semantics license discarding orientation for legality; attachment and stopping
remain part of each key. Cold table building and all lookups are charged.
"""
from collections import defaultdict
from functools import lru_cache
from .common import Work,Exhausted
from .g2 import contract
from ..v16.records import canonical,digest


def predict(payload,probes,budget=1000000):
    public=contract(payload);work=Work(budget);table=defaultdict(list)
    probabilities=[];covered=0
    def key(state,stopped,action):
        work.charge('checking',len(state))
        return (tuple(v!=-1 for v in state),stopped,action)
    try:
        for observation in public['observations']:
            work.charge('retrieval')
            if observation['query']['kind']=='context':continue
            stopped=False
            for transition in observation['outcome']['trace']:
                work.charge('retrieval')
                table[key(transition['before'],stopped,transition['action'])].append(transition['legal'])
                stopped=transition['stopped']
        for probe in probes:
            if probe['kind']!='action' or len(probe['program'])!=1:raise ValueError('one-action forecast required')
            values=table.get(key(probe['initial'],False,probe['program'][0]),[])
            work.charge('selection',max(1,len(values)))
            probabilities.append(sum(values)/len(values) if values else 0.5)
            covered+=bool(values)
    except Exhausted:
        return dict(probabilities=None,costs=work.receipt(),missing_output=True,covered_probes=covered)
    return dict(probabilities=probabilities,costs=work.receipt(),missing_output=False,covered_probes=covered)


@lru_cache(maxsize=1)
def reference(campaign,origin_run,origin_block):
    from .runtime import load_block
    return load_block(campaign/origin_run,origin_block)


def evaluate(case,campaign):
    block,receipt=reference(campaign,case['origin_run'],case['origin_block'])
    assert receipt['raw_sha256']==case['origin_raw_sha256']
    unit=block['units'][case['origin_index']];original=unit['case']
    assert original['case_id']==case['origin_case_id']
    rows=[];budget=max(row['budget'] for row in unit['rows'])
    for old in unit['rows']:
        if old['method']!='dependencies' or old['budget']!=budget:continue
        evidence=dict(original['public'],observations=old['observation_record'])
        answer=predict(canonical(evidence),old['forecast_probes'])
        assert not answer['missing_output']
        truth=old['forecast_truths'];probabilities=answer['probabilities']
        rivals=[r for r in unit['rows'] if r['budget']==budget and r['query_policy']==old['query_policy'] and r['requested_queries']==old['requested_queries']]
        rows.append(dict(query_policy=old['query_policy'],requested_queries=old['requested_queries'],acquired_queries=old['acquired_queries'],
            construction_budget=budget,forecast_budget=1000000,method='attachment-mask',
            forecast_brier=sum((p-int(t))**2 for p,t in zip(probabilities,truth))/len(truth),
            forecast_operations=answer['costs']['total_online'],forecast_probabilities=probabilities,covered_probes=answer['covered_probes'],
            forecast_probe_count=len(truth),costs=answer['costs'],missing_output=False,
            evidence_sha256=digest(evidence),origin_row_sha256=digest(old),
            paired={r['method']:dict(forecast_brier=r['forecast_brier'],forecast_operations=r['forecast_operations']) for r in rivals},
            scope='forecast-only cold deployment; original task outcomes unchanged; fixed/decision/uniform evidence retained'))
    return rows
