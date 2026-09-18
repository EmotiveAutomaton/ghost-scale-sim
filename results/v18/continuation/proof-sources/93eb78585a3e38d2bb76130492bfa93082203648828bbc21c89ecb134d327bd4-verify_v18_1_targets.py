"""Independent target-realization checks on already frozen inputs, not new science."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
from ghostscale.validation.soundingline.v18_1.verify import physical
from ghostscale.validation.soundingline.v16.records import read,write,digest,file_digest


def reachable(world,initial,target,bound,monotone=False):
    queue=deque([(tuple(initial),False,0)]);seen={(tuple(initial),False)};law=json.dumps(world,sort_keys=True)
    while queue:
        state,stopped,depth=queue.popleft()
        if list(state)==target and stopped:return True
        if stopped or depth==bound:continue
        before=sum(a!=b for a,b in zip(state,target))
        for action in range(3*len(state)+1):
            actual=physical(law,json.dumps(state),(action,),1)
            if not actual['legal']:continue
            after=tuple(actual['state']);key=(after,actual['stopped'])
            if monotone and sum(a!=b for a,b in zip(after,target))>before:continue
            if key not in seen:seen.add(key);queue.append((after,actual['stopped'],depth+1))
    return False


def verify(root,output):
    plan=read(root/'PLAN.json');path=root/'INPUTS.json.gz';assert file_digest(path)==plan['inputs_sha256']
    cases=json.loads(gzip.decompress(path.read_bytes()));seen={};evaluated=0;unreachable=0;detours=0
    for case in cases:
        if plan['branch']=='g0-common':
            for targets in case['transfer_targets'].values():
                for target in targets:
                    assert 0<=target<16 and target.bit_count()<=3
                    seen[target]=True
            continue
        p=case['public'];world=case['private']['true_world'];law=json.dumps(world,sort_keys=True)
        key=digest([world,p['initial'],p['target'],p['max_steps'],p.get('menu')])
        if key in seen:continue
        seen[key]=True;evaluated+=1
        if plan['branch']=='g1-native':
            possible=reachable(world,p['initial'],p['target'],p['max_steps'])
            monotone=reachable(world,p['initial'],p['target'],p['max_steps'],True)
            assert possible==case['private']['reference']['reachable']
            assert monotone==case['private']['monotone_reference']['reachable']
            unreachable+=not possible;detours+=possible and not monotone
        elif plan['branch']=='g3-representation':
            actual=physical(law,json.dumps(p['initial']),tuple(case['private']['target_witness']),p['max_steps'])
            assert actual['legal'] and actual['stopped'] and actual['state']==p['target']
        elif plan['branch'] in ('g2-native','g2-transfer'):
            outcomes=[physical(law,json.dumps(q['initial']),tuple(q['program']),p['max_steps']) for q in p['menu'] if q['kind']=='routine']
            assert any(a['legal'] and a['stopped'] and a['state']==p['target'] for a in outcomes)
        else:raise ValueError('unsupported target-verification family')
    result=dict(passed=True,branch=plan['branch'],input_cases=len(cases),distinct_target_or_menu_checks=len(seen),
        evaluated_model_targets=evaluated,unreachable_targets=unreachable,
        required_detour_targets=detours if plan['branch']=='g1-native' else None,
        inputs_sha256=file_digest(path),plan_sha256=file_digest(root/'PLAN.json'),script_sha256=file_digest(Path(__file__)),
        scope=('independent exhaustive native reachability and monotone reachability' if plan['branch']=='g1-native' else
               'independent target realization for every distinct frozen target/menu; detour necessity not scored'))
    write(output,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.root,args.output)),flush=True)


if __name__=='__main__':main()
