"""Audit a completed packet using its pinned checkout, preserving raw records."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def audit(root,source):
    sys.path.insert(0,str(source))
    import numpy as np
    from ghostscale.validation.soundingline.v18_2 import model as m,verify as v,branches as b,learned as n
    from ghostscale.validation.soundingline.v16.records import read,write,file_digest,now
    from ghostscale.validation.soundingline.v18_2.runtime import load
    cpu=time.process_time();wall=time.monotonic()
    plan=read(root/'PLAN.json');complete=read(root/'COMPLETE.json');design=plan['design'];branch=design['branch']
    assert complete['plan_sha256']==file_digest(root/'PLAN.json')
    assert complete['summary_sha256']==file_digest(root/'SUMMARY.json')
    assert {p:file_digest(source/p) for p in plan['sources']}==plan['sources']
    checks=0;replayed=0;cases=[];cells={};dispositions={};score_rows=0;reference_checks=0
    if branch=='g6':
        summary=read(root/'SUMMARY.json');lookup={}
        for cell in summary['cells']:
            se=math.sqrt((cell['heterogeneity']**2+1/cell['within'])/cell['persons'])
            assert abs(cell['population_width']-3.92*se)<1e-12
            assert abs(cell['individual_width']-3.92/math.sqrt(cell['within']))<1e-12
            expected_bias=cell['selection_bias']*cell['heterogeneity']
            normal=lambda z:.5*(1+math.erf(z/math.sqrt(2)))
            coverage=normal(1.96-expected_bias/se)-normal(-1.96-expected_bias/se)
            tolerance=6*math.sqrt(coverage*(1-coverage)/256)+1/256
            assert abs(cell['population_coverage']-coverage)<tolerance
            key=tuple(cell[k] for k in ('persons','within','heterogeneity','selection_bias'))
            if key in lookup:
                assert lookup[key]['population_mse']==cell['population_mse']
                assert lookup[key]['population_coverage']==cell['population_coverage']
            lookup[key]=cell;checks+=4
        record=dict(passed=True,analytic_checks=checks,cells=len(summary['cells']),scope='Gaussian widths, coverage and duplicate dependence; distinct estimands')
    else:
        model_objects={}
        if branch=='learned':
            for kind in ('split','flat'):
                net=n.Network(kind)
                with np.load(root/f'{kind}.npz') as saved:net.parameters=[saved[f'p{i}'] for i in range(len(net.parameters))]
                model_objects[kind]=net
        for block_name in complete['blocks']:
            for unit in load(root,block_name):
                case=unit['case'];cases.append(case['case_id']);assert v.check_case(case)['passed'];checks+=len(case['history'])+4
                per={}
                for row in unit['rows']:
                    # Correct the early learned report's grouping, never its raw
                    # scores: compare the same three base / one held-out probes.
                    condition=row.get('condition','base')
                    if branch=='learned' and row['method'] in ('direct','raw','persistent','oracle'):
                        condition='crossed-heldout' if row['probe']==3 else 'base'
                    key='|'.join([case['split'],condition,row['tier'],row['method']])
                    disposition=row.get('instrument','missing');counts=dispositions.setdefault(key,{})
                    counts[disposition]=counts.get(disposition,0)+1
                    if disposition!='valid':continue
                    for metric,value in row['scores'].items():
                        assert math.isfinite(value);per.setdefault((key,metric),[]).append(value)
                    score_rows+=1
                for (key,metric),values in per.items():cells.setdefault(key,{}).setdefault(metric,[]).append(sum(values)/len(values))
                # Selection uses case ids/positions, independent of score outcomes.
                if len(cases) in (1,64,128):
                    if branch=='g0':actual=m.evaluate(case)
                    elif branch=='learned':
                        actual=None
                        for row in unit['rows']:
                            if row['method'] not in model_objects or row.get('condition','base') not in ('base','crossed-heldout'):continue
                            payload=m.public_packet(case,row['tier'],row['probe'])
                            if hasattr(n,'predict'):q=n.predict(model_objects[row['method']],payload,design.get('aligned',False))
                            else:q=model_objects[row['method']].forward(n.features(payload)[None,:])[0][0]
                            assert np.allclose(q,row['result']['probabilities'],atol=2e-7)
                            replayed+=1
                    else:actual=b.evaluate_branch(case,design)
                    if actual is not None:
                        assert m.canonical(actual)==m.canonical(unit['rows']);replayed+=len(actual)
                    for tier in m.TIERS:
                        payload=m.public_packet(case,tier)
                        assert np.allclose(v.posterior(json.loads(payload)),m.infer(payload)['posterior'],atol=1e-12);reference_checks+=1
        write(root/'AUDITED_SUMMARY.json',dict(units=len(cases),unique_lineages=len(set(cases)),score_rows=score_rows,
            dispositions=dispositions,cells={k:{metric:v.interval(values) for metric,values in metrics.items()} for k,metrics in cells.items()},
            regrouping='learned baseline and network rows share the identical base/withheld probe subsets; original raw scores preserved'),immutable=False)
        record=dict(passed=True,units=len(cases),unique_lineages=len(set(cases)),score_rows=score_rows,
                    independent_execution_checks=checks,replayed_rows=replayed,independent_posterior_checks=reference_checks,
                    selection='lineage positions 1,64,128; selected before comparing their scores')
    record.update(at=now(),source_archive_sha256=file_digest(root/'SOURCE.zip'),plan_sha256=file_digest(root/'PLAN.json'),
        raw_checksums={p.name:file_digest(p) for p in sorted((root/'raw').glob('*.gz'))},
        auditor_sha256=file_digest(Path(__file__)),cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        scientific_eligibility='withheld; see SUPERSEDED.json' if (root/'SUPERSEDED.json').exists() else 'discovery')
    write(root/'VERIFICATION.json',record,immutable=False)
    campaign=root.parent
    write(campaign/'audit/attempts'/f'{root.name}.json',dict(cpu_seconds=record['cpu_seconds'],wall_seconds=record['wall_seconds'],child_cpu_seconds=0),immutable=False)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.root,args.source),sort_keys=True))
