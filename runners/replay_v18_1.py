"""Bounded decision replay in an extracted, checksum-verified branch source tree."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-zip',type=Path,required=True);parser.add_argument('--extracted',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads((args.root/'PLAN.json').read_text());completed=json.loads((args.root/'COMPLETE.json').read_text())
    destination=args.extracted.resolve()
    if destination.exists():raise ValueError('fresh extraction required')
    with zipfile.ZipFile(args.source_zip) as archive:
        for member in archive.namelist():
            if not (destination/member).resolve().is_relative_to(destination):raise ValueError('unsafe source archive')
        archive.extractall(destination)
    for path,expected in plan['sources'].items():
        assert hashlib.sha256((destination/path).read_bytes()).hexdigest()==expected
    sys.path.insert(0,str(destination))
    records=importlib.import_module('ghostscale.validation.soundingline.v16.records')
    runtime=importlib.import_module('ghostscale.validation.soundingline.v18_1.runtime')
    assert Path(runtime.__file__).resolve().is_relative_to(destination)
    branch=plan['branch'];module=importlib.import_module('ghostscale.validation.soundingline.v18_1.'+{'g2-mask':'g2_mask','g2-permuted':'permuted','g2-cyclic':'cyclic_union','g2-cyclic-cost':'cyclic_union','g2-cyclic-target':'cyclic_union','g2-cyclic-action':'cyclic_union','g2-cyclic-physical':'cyclic_union','g2-cyclic-misspecified':'cyclic_union','g2-cyclic-misspecified-physical':'cyclic_union','g3-stitch':'g3_stitch','structural-direct':'direct'}.get(branch,branch.split('-')[0]))
    selected={};raw_bindings={}
    for name in completed['blocks']:
        block,receipt=runtime.load_block(args.root,name);raw_bindings[name]=receipt['raw_sha256']
        for unit in block['units']:
            case=unit['case']
            if branch=='g1-native':key=tuple(case[k] for k in ('stratum','information','representation'))
            elif branch in ('g2-native','g2-transfer'):key=tuple(case.get(k) for k in ('n','family','selection','donor','truth_excluded'))
            elif branch=='g2-mask':key=tuple(case[k] for k in ('origin_run','n','family','selection','donor','truth_excluded'))
            elif branch=='g2-permuted':key=tuple(case[k] for k in ('n','family','selection','donor'))
            elif branch in ('g2-cyclic','g2-cyclic-cost','g2-cyclic-target','g2-cyclic-action','g2-cyclic-physical','g2-cyclic-misspecified','g2-cyclic-misspecified-physical'):key=tuple(case[k] for k in ('n','family'))
            elif branch in ('g3-representation','g3-stitch'):key=tuple(case[k] for k in ('n','family','condition'))
            elif branch=='g4-history':key=(case['case_id'],)
            elif branch=='structural-direct':key=tuple(case[k] for k in ('origin_branch','n','family','condition','stratum'))
            else:key=(int(case['case_id'][:8],16)%8,)
            if key not in selected or case['case_id']<selected[key]['case']['case_id']:selected[key]=unit
    args.output.mkdir(parents=True,exist_ok=True)
    records.write(args.output/'SELECTION.json',dict(rule='lowest opaque case id per predeclared stratum; G0 eight fixed hash buckets; G4 all72',
        case_ids=sorted(u['case']['case_id'] for u in selected.values()),outcome_independent=True,raw_bindings=raw_bindings))
    count=0
    for unit in selected.values():
        case=unit['case'];design=plan.get('design',{})
        if branch=='g0-original':rows=module.original_matrix(case,plan['budgets'])
        elif branch=='g0-common':rows=module.common_matrix(case,design['budgets'],design['boundary_budgets'],design['balanced'])
        elif branch=='g1-native':
            rows=module.evaluate(case,design['budgets']);assert records.digest(module.gate_diagnostics(case))==records.digest(unit['diagnostics'])
        elif branch in ('g2-native','g2-transfer'):rows=module.evaluate(case,design['budgets'],design['query_counts'])
        elif branch=='g3-representation':rows=module.evaluate(case,design['budgets'],design['storage_caps'])
        elif branch=='g3-stitch':rows,_=module.evaluate(case,args.output/'stitch-replay',design['budgets'],design['storage_caps'])
        elif branch=='g4-history':rows=module.evaluate(case)
        elif branch=='structural-direct':rows=module.evaluate(case,case['budgets'])
        elif branch=='g2-mask':rows=module.evaluate(case,args.root.parent)
        elif branch=='g2-permuted':rows=module.evaluate(case,design['budgets'],design['query_counts'])
        elif branch=='g2-cyclic':rows=module.evaluate(case,design['budgets'],design['query_counts'])
        elif branch=='g2-cyclic-cost':rows=module.evaluate_cached_decision(case,design['online_budget'],design['selector_budget'])
        elif branch=='g2-cyclic-target':rows=module.evaluate_target_aware(case,design['budget'],design['query_counts'])
        elif branch=='g2-cyclic-action':rows=module.evaluate_target_action(case,design['budget'],design['query_counts'])
        elif branch=='g2-cyclic-physical':rows=module.evaluate_physical_action(case,design['budget'],design['query_counts'])
        elif branch=='g2-cyclic-misspecified':rows=module.evaluate_misspecified(case,design['budget'],design['query_counts'])
        elif branch=='g2-cyclic-misspecified-physical':rows=module.evaluate_physical_misspecified(case,design['budget'],design['query_counts'])
        else:raise ValueError('unknown branch')
        assert records.digest(rows)==records.digest(unit['rows']);count+=len(rows)
    records.write(args.output/'REPLAY.json',dict(passed=True,cases=len(selected),rows=count,branch=branch,
        source_archive_sha256=records.file_digest(args.source_zip),plan_sha256=records.file_digest(args.root/'PLAN.json'),
        selection_sha256=records.file_digest(args.output/'SELECTION.json'),replayer_sha256=records.file_digest(Path(__file__)),
        scope='exact method decisions/costs/evidence regenerated in fresh extracted source; independent full physics/reaggregation separate'))
    print(json.dumps(dict(branch=branch,cases=len(selected),rows=count,passed=True)),flush=True)


if __name__=='__main__':main()
