"""Independent receipts and physical checks; no acceptance of cached success fields."""
from collections import defaultdict,Counter
import gzip
import json
from pathlib import Path
from statistics import mean
import zipfile

from ..v16.records import digest,file_digest,read,write
from ..v16 import craft,world,assembly
from ..v18.verify import verify_case,verify_row


def g0_report(root,archive,output):
    plan=read(root/'PLAN.json')
    assert file_digest(archive)==plan['archive_sha256']
    completed=read(root/'COMPLETE.json')
    assert completed['plan_sha256']==file_digest(root/'PLAN.json')
    reference=defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        original=json.loads(z.read('report/COMPARISONS.json')) if 'report/COMPARISONS.json' in z.namelist() else None
        if original is None:
            original=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/COMPARISONS.json'))))
        for name in sorted(z.namelist()):
            if not name.startswith('run/raw/') or not name.endswith('.json.gz'):continue
            block=json.loads(gzip.decompress(z.read(name)))
            for unit in block['units']:
                verify_case(unit['case'])
                for row in unit['rows']:
                    verify_row(unit['case'],row)
                    reference[(row['budget'],row['stratum'],row['method'])].append(row['success'])
    assert len(reference)==28
    for cell in original['cell_table']:
        assert mean(reference[(cell['budget'],cell['stratum'],cell['method'])])==cell['success_fraction']
    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    hashes={};rows_checked=0
    for name in completed['blocks']:
        receipt=read(root/'blocks'/(name+'.json'))
        path=root/'raw'/(name+'_points.json.gz')
        assert file_digest(path)==receipt['raw_sha256']
        block=json.loads(gzip.decompress(path.read_bytes()))
        assert digest(block)==receipt['content_sha256'] and block['plan_sha256']==file_digest(root/'PLAN.json')
        hashes[name]=file_digest(path)
        for unit in block['units']:
            case=unit['case'];verify_case(case)
            for row in unit['rows']:
                processed=case['acquisitions'][row['allocation']]['request']['processed']
                counts=Counter(tuple(t['program']) for t in processed if t['feedback'])
                ranked=sorted((p for p,n in counts.items() if n>=3),key=lambda p:(-counts[p],p))
                library=[list(p) for p in ranked[:row['capacity']]]
                assert library==row['request']['library']
                target=case['transfer_targets'][row['stratum']][row['target_index']]
                active=[];check_cost=0
                for fragment in library:
                    state=0;bad=False
                    for action in fragment:
                        after=world.step(state,action)
                        if (after^target).bit_count()>(state^target).bit_count():bad=True
                        state=after
                        if row['checking']:check_cost+=1
                    if not row['checking'] or not bad:active.append(fragment)
                answer=craft.construct(target,active,primitive_budget=row['budget']-check_cost)
                actual=world.execute(answer['program'])
                assert row['program']==answer['program']
                assert row['success']==(actual.legal and actual.artifact==target)
                assert row['costs']==dict(checking=check_cost,search=answer['search_primitives'],actual=actual.primitive_cost,
                                          total_online=check_cost+answer['search_primitives']+actual.primitive_cost)
                key=(row['budget'],row['stratum'],row['allocation'],row['capacity'],row['checking'])
                grouped[key][case['constructor_index']][case['history_index']].append(row)
                rows_checked+=1
    table=[]
    for key,constructors in sorted(grouped.items()):
        budget,stratum,allocation,capacity,checking=key
        successes=[mean(mean(r['success'] for r in records) for records in histories.values()) for histories in constructors.values()]
        costs=[mean(mean(r['costs']['total_online'] for r in records) for records in histories.values()) for histories in constructors.values()]
        table.append(dict(budget=budget,stratum=stratum,allocation=allocation,capacity=capacity,checking=checking,
                          success_fraction=mean(successes),mean_total_primitives=mean(costs),constructor_means=successes,
                          original_configurations=len(constructors),histories=sum(map(len,constructors.values()))))
    result=dict(schema='v18.1.g0-original-summary.1',cells=table,rows_checked=rows_checked,
                original_cells_reproduced=28,source_plan_sha256=file_digest(root/'PLAN.json'),raw_bindings=hashes,
                inference='exposed descriptive diagnostic; original cases reused; no fresh confirmation',
                new_independent_law_context_units=0)
    write(output/'COMPARISONS.json',result)
    write(output/'VERIFICATION.json',dict(passed=True,rows_checked=rows_checked,original_cells_reproduced=28,
         summary_sha256=file_digest(output/'COMPARISONS.json'),verifier_sha256=file_digest(Path(__file__)),
         scope='original independent trace verification; new-library count and native enumeration reexecution; exact reaggregation'))
    return result
