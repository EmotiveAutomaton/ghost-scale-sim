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
    branch=plan['branch'];module=importlib.import_module('ghostscale.validation.soundingline.v18_1.'+branch.split('-')[0])
    selected={};raw_bindings={}
    for name in completed['blocks']:
        block,receipt=runtime.load_block(args.root,name);raw_bindings[name]=receipt['raw_sha256']
        for unit in block['units']:
            case=unit['case']
            if branch=='g1-native':key=tuple(case[k] for k in ('stratum','information','representation'))
            elif branch=='g2-native':key=tuple(case[k] for k in ('selection','donor','truth_excluded'))
            elif branch=='g3-representation':key=tuple(case[k] for k in ('n','family','condition'))
            elif branch=='g4-history':key=(case['case_id'],)
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
        elif branch=='g4-history':rows=module.evaluate(case)
        else:raise ValueError('unknown branch')
        assert records.digest(rows)==records.digest(unit['rows']);count+=len(rows)
    records.write(args.output/'REPLAY.json',dict(passed=True,cases=len(selected),rows=count,branch=branch,
        source_archive_sha256=records.file_digest(args.source_zip),plan_sha256=records.file_digest(args.root/'PLAN.json'),
        selection_sha256=records.file_digest(args.output/'SELECTION.json'),replayer_sha256=records.file_digest(Path(__file__)),
        scope='exact method decisions/costs/evidence regenerated in fresh extracted source; independent full physics/reaggregation separate'))
    print(json.dumps(dict(branch=branch,cases=len(selected),rows=count,passed=True)),flush=True)


if __name__=='__main__':main()
