"""Full raw reconstruction of descriptive action-evidence cell means."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

root,output=map(Path,sys.argv[1:])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
plan=json.loads((root/'PLAN.json').read_bytes());complete=json.loads((root/'COMPLETE.json').read_bytes())
assert plan['branch']=='g2-cyclic-action' and complete['plan_sha256']==sha(root/'PLAN.json')
groups=defaultdict(lambda:defaultdict(list));families=defaultdict(set);bindings={};rows=0
for name in complete['blocks']:
    receipt=json.loads((root/'blocks'/f'{name}.json').read_bytes())
    path=root/'raw'/f'{name}_points.json.gz';bindings[name]=sha(path)
    block=json.loads(gzip.decompress(path.read_bytes()))
    assert bindings[name]==receipt['raw_sha256'] and digest(block)==receipt['content_sha256']
    assert block['plan_sha256']==sha(root/'PLAN.json')
    for unit in block['units']:
        case=unit['case'];public=case['public'];evidence={}
        families[case['family']].add(case['structural_unit'])
        for row in unit['rows']:
            rows+=1;policy=row['query_policy'];count=row['requested_queries']
            key=(policy,count);value=digest([row['observation_record'],row['query_indices'],row['acquisition_operations']])
            assert key not in evidence or evidence[key]==value
            evidence[key]=value
            if policy=='target-action':
                assert not row['direct_parent_cues_used']
                for observation in row['observation_record'][len(public['observations']):]:
                    query=observation['query'];state=query['initial']
                    assert query['kind']=='action' and len(query['program'])==1
                    assert query['program'][0]%7 in case['target_parts']
                    assert 'parent' not in observation['outcome']
                    for model in public['models']:
                        assert all(v==-1 or p==-1 or state[p]!=-1 for v,p in zip(state,model['parents']))
            groups[row['method'],policy,count][case['structural_unit']].append(row)
cells=[]
metrics=('success','query_exhausted','query_compatible_laws','query_isolates_truth','acquisition_operations','direct_parent_cues_used')
for (method,policy,count),units in sorted(groups.items()):
    assert len(units)==192 and all(len(v)==4 for v in units.values())
    cell=dict(method=method,query_policy=policy,requested_queries=count,contexts=192,rows=768)
    cell.update({f'mean_{m}':mean(mean(r[m] for r in rs) for rs in units.values()) for m in metrics})
    cell['mean_online_operations']=mean(mean(r['costs']['total_online'] for r in rs) for rs in units.values())
    cells.append(cell)
assert rows==18432 and {k:len(v) for k,v in families.items()}==dict(chain=64,fork=64,groups=64)
finding=[c for c in cells if c['method'] in ('dependencies','conditioned-direct','candidate-set-primitive') and c['query_policy']=='target-action']
result=dict(schema='v18.1.cyclic-action-diagnostic.1',scope='descriptive constructed mechanism; miniature architecture untested; no third primary',
    plan_sha256=sha(root/'PLAN.json'),rows_checked=rows,distinct_context_units=192,histories_per_context=4,
    family_context_units={k:len(v) for k,v in families.items()},cells=cells,raw_bindings=bindings,
    aggregation='equal structural-context means; four histories nested; action methods share evidence',
    limits=['A supplied public candidate family and a prepared menu are part of the apparatus; menu construction and physical setup are not counted in online acquisition operations.',
            'Common-valid query states do not establish free real-world access to those states; no arbitrary-menu or architecture transfer claim.'])
output.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
print(json.dumps(finding,sort_keys=True))
