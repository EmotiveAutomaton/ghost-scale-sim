"""Independent receipts and physical checks; no acceptance of cached success fields."""
from collections import defaultdict,Counter
import gzip
import json
from pathlib import Path
from statistics import mean
from functools import lru_cache
import zipfile

from ..v16.records import digest,file_digest,read,write
from ..v16 import craft,world,assembly
from ..v18.verify import verify_case,verify_row


def independent_assembly(law,initial,program,max_steps):
    """Separate dictionary/set executor for the parameterized assembly check."""
    n=len(law['parents']);present={i:v for i,v in enumerate(initial) if v!=-1};stopped=False;cost=0
    legal=True;trace=[]
    for action in program:
        if cost==max_steps:legal=False;break
        before=[present.get(i,-1) for i in range(n)];cost+=1
        valid=not stopped and type(action) is int and 0<=action<=3*n and action not in law.get('forbidden',[])
        if valid and action==3*n:
            stopped=True
        elif valid:
            operation,part=divmod(action,n)
            if operation==0:
                parent=law['parents'][part]
                valid=part not in present and (parent==-1 or parent in present)
                if valid:present[part]=law['defaults'][part]
            else:
                blockers={i for i in present if law['parents'][i]==part}
                valid=part in present and not blockers
                if valid:
                    if operation==1:del present[part]
                    else:present[part]=int(not present[part])
        after=[present.get(i,-1) for i in range(n)]
        trace.append(dict(before=before,action=action,after=after,legal=valid,stopped=stopped))
        if not valid:legal=False;break
    return dict(state=[present.get(i,-1) for i in range(n)],legal=legal,stopped=stopped,primitive_cost=cost,trace=trace)


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


@lru_cache(maxsize=100000)
def physical(law_json,initial_json,program_tuple,max_steps):
    law=json.loads(law_json);initial=json.loads(initial_json)
    if law['kind']=='assembly':return independent_assembly(law,initial,program_tuple,max_steps)
    marked={i for i in range(law['cells']) if initial&(1<<i)};legal=True;cost=0
    for action in program_tuple:
        if cost==max_steps:legal=False;break
        cost+=1
        if type(action) is not int or not 0<=action<2*law['cells'] or action in law.get('forbidden',[]):
            legal=False;break
        if action<law['cells']:marked.add(action)
        else:marked.discard(action-law['cells'])
    return dict(state=sum(1<<i for i in marked),legal=legal,stopped=True,primitive_cost=cost)


def branch_report(root,output):
    """Full raw reconstruction and independent physics, without reader re-search.

    Same-source method replay is a separate bounded proof; it is not implied by
    this complete reaggregation/physical-execution verification.
    """
    from .runtime import load_block
    plan=read(root/'PLAN.json');completed=read(root/'COMPLETE.json')
    assert completed['plan_sha256']==file_digest(root/'PLAN.json')
    branch=plan['branch'];groups=defaultdict(lambda:defaultdict(list));bindings={};count=0
    contexts=set();laws=set();endpoints=set();signatures=set();examples={};diagnostics=defaultdict(list)
    dimensions={
      'g1-native':('stratum','information','representation','gate','budget'),
      'g2-native':('selection','donor','truth_excluded','method','query_policy','requested_queries','budget'),
      'g2-transfer':('n','family','selection','donor','truth_excluded','method','query_policy','requested_queries','budget'),
      'g2-permuted':('n','family','selection','donor','truth_excluded','method','query_policy','requested_queries','budget'),
      'g2-cyclic':('n','family','selection','donor','truth_excluded','method','query_policy','requested_queries','budget'),
      'g2-cyclic-cost':('n','family','method','query_policy','budget','selector_budget'),
      'g2-cyclic-target':('n','family','method','query_policy','requested_queries','budget'),
      'g2-cyclic-action':('n','family','method','query_policy','requested_queries','budget'),
      'g2-cyclic-physical':('n','family','method','query_policy','requested_queries','budget'),
      'g2-cyclic-misspecified':('n','family','method','query_policy','requested_queries','budget'),
      'g3-representation':('n','family','condition','method','storage_cap','budget'),
      'g3-stitch':('n','family','condition','method','storage_cap','budget'),
      'g0-common':('stratum','allocation','capacity','representation','checking','planner','action_order_stratum','budget'),
      'g4-history':('family','condition','tier','method'),
      'structural-direct':('origin_branch','n','family','condition','stratum','selection','donor','truth_excluded','budget','method')}[branch]
    for name in completed['blocks']:
        block,receipt=load_block(root,name);bindings[name]=receipt['raw_sha256']
        assert block['plan_sha256']==file_digest(root/'PLAN.json')
        assert receipt['rows']==sum(len(u['rows']) for u in block['units'])
        for unit in block['units']:
            case=unit['case'];uid=case['structural_unit']
            p=case.get('public',{});truth=case.get('private',{}).get('true_world')
            if branch=='g4-history':
                base=p['tiers'][0];alternatives=case['private']['alternatives']
                # The frozen first packet used a draw-index cluster ID. Distinct
                # physical support must be counted from the actual public context.
                uid=digest([base['world'],base['initial'],base['method_family'],case['condition']])
                laws.add(digest(base['world']));endpoints.add(digest([case['family'],base['endpoint']]))
                for history in alternatives:
                    actual=physical(json.dumps(base['world'],sort_keys=True),json.dumps(base['initial']),tuple(history['program']),12)
                    assert actual['legal'] and actual['stopped'] and actual['state']==base['endpoint']==history['endpoint']
                    for transition in history['trace']:
                        assert any(r['actor']==transition['actor'] and transition['action'] in r['available_actions'] for r in base['roles'])
            contexts.add(uid)
            if truth:
                laws.add(digest(truth));endpoints.add(digest([truth['kind'],p['target']]))
            if case.get('topology_signature'):signatures.add((case.get('n'),case['topology_signature']))
            evidence={}
            for row in unit['rows']:
                data={**case,**row};key=tuple(data[k] for k in dimensions);count+=1
                if branch in ('g1-native','g2-native','g2-transfer','g2-permuted','g2-cyclic','g2-cyclic-cost','g2-cyclic-target','g2-cyclic-action','g2-cyclic-physical','g2-cyclic-misspecified','g3-representation','g3-stitch','structural-direct'):
                    assert row['costs']['total_online']<=row['budget']
                    assert sum(v for k,v in row['costs'].items() if k not in ('total_online','envelope','unit'))==row['costs']['total_online']
                    if row['program'] is None:
                        assert row['missing_output'] and not row['success'] and not row['invalid']
                    else:
                        actual=physical(json.dumps(truth,sort_keys=True),json.dumps(p['initial']),tuple(row['program']),p['max_steps'])
                        assert actual['state']==row['execution']['state']
                        assert actual['legal']==row['execution']['legal'] and actual['stopped']==row['execution']['stopped']
                        assert row['success']==(actual['legal'] and actual['stopped'] and actual['state']==p['target'])
                        assert row['costs']['actual_execution']==actual['primitive_cost']
                    if branch in ('g3-representation','g3-stitch'):assert row['acquisition']['storage_tokens']<=row['storage_cap']
                if branch in ('g2-native','g2-transfer','g2-permuted','g2-cyclic','g2-cyclic-cost','g2-cyclic-target','g2-cyclic-action','g2-cyclic-physical','g2-cyclic-misspecified') and 'observation_record' in row:
                    ek=(row['query_policy'],row['requested_queries'],row['budget'])
                    if ek in evidence:assert evidence[ek]==digest(row['observation_record'])
                    evidence[ek]=digest(row['observation_record'])
                if branch in ('g2-native','g2-transfer','g2-permuted','g2-cyclic','g2-cyclic-cost','g2-cyclic-target','g2-cyclic-action','g2-cyclic-physical','g2-cyclic-misspecified') and 'forecast_probes' in row:
                    predicted_truth=[]
                    for q in row['forecast_probes']:
                        actual=physical(json.dumps(truth,sort_keys=True),json.dumps(q['initial']),tuple(q['program']),p['max_steps'])
                        predicted_truth.append(actual['legal'])
                    assert predicted_truth==row['forecast_truths']
                    assert sum((prob-int(t))**2 for prob,t in zip(row['forecast_probabilities'],predicted_truth))/len(predicted_truth)==row['forecast_brier']
                    for observation in row['observation_record']:
                        q=observation['query'];o=observation['outcome']
                        if q['kind']=='context':assert o['parent']==truth['parents'][q['part']]
                        else:
                            actual=physical(json.dumps(truth,sort_keys=True),json.dumps(q['initial']),tuple(q['program']),p['max_steps'])
                            assert actual['state']==o['state'] and actual['legal']==o['legal']
                            assert actual['primitive_cost']==o['primitive_cost']
                if branch=='g2-cyclic-cost':
                    assert row['query_policy']=='decision-cached' and row['requested_queries']==1
                    assert row['selector_costs']['total_online']==row['selector_operations']
                    assert row['combined_operations']==row['selector_operations']+row['costs']['total_online']
                    assert row['comparison_role'].startswith('exposed descriptive diagnostic')
                if branch=='g2-cyclic-target':
                    assert row['query_policy'] in ('fixed','decision','target-aware')
                    assert row['requested_queries'] in (1,2)
                    assert row['acquisition_costs']['total_online']==row['acquisition_operations']
                    assert row['acquisition_operations']<=row['costs']['total_online']<=row['budget']
                    assert row['target_changed_parts']==[1,2]
                    assert row['comparison_role'].startswith('descriptive target-aware')
                if branch=='g2-cyclic-action':
                    assert row['query_policy'] in ('fixed','target-aware','target-action')
                    assert row['requested_queries'] in (1,2)
                    assert row['acquisition_costs']['total_online']==row['acquisition_operations']
                    assert row['acquisition_operations']<=row['costs']['total_online']<=row['budget']
                    assert row['target_changed_parts']==[1,2]
                    assert row['comparison_role'].startswith('descriptive target-relevant')
                    paid=row['observation_record'][len(p['observations']):]
                    if row['query_policy']=='target-action':
                        assert not row['direct_parent_cues_used']
                        assert all(o['query']['kind']=='action' for o in paid)
                        assert row['target_action_parts']==[o['query']['program'][0]%len(p['initial']) for o in paid]
                        assert set(row['target_action_parts'])<=set(row['target_changed_parts'])
                if branch=='g2-cyclic-physical':
                    assert row['query_policy']=='target-action-physical'
                    assert row['requested_queries'] in (1,2)
                    assert row['acquisition_costs']['total_online']==row['acquisition_operations']
                    assert row['acquisition_operations']<=row['costs']['total_online']<=row['budget']
                    assert row['physical_setup_operations']==sum(map(len,row['physical_setup_paths']))
                    assert not row['setup_uses_evaluator_truth'] and row['candidate_family_supplied']
                    paid=row['observation_record'][len(p['observations']):]
                    assert len(paid)==len(row['physical_setup_paths'])==row['acquired_queries']
                    prior=list(p['observations'])
                    for observation,setup in zip(paid,row['physical_setup_paths']):
                        compatible=[]
                        for model in p['models']:
                            agrees=True
                            for seen in prior:
                                q=seen['query'];expected=(dict(parent=model['parents'][q['part']],primitive_cost=0)
                                    if q['kind']=='context' else independent_assembly(
                                        model,q['initial'],q['program'],p['max_steps']))
                                if expected!=seen['outcome']:
                                    agrees=False;break
                            if agrees:compatible.append(model)
                        terminals=[]
                        for model in compatible:
                            prepared=independent_assembly(model,p['initial'],setup,p['max_steps'])
                            assert prepared['legal'] and not prepared['stopped']
                            terminals.append(prepared['state'])
                        assert terminals and all(state==observation['query']['initial'] for state in terminals)
                        prior.append(observation)
                    final=[]
                    for model in p['models']:
                        if all((dict(parent=model['parents'][o['query']['part']],primitive_cost=0)
                                if o['query']['kind']=='context' else independent_assembly(
                                    model,o['query']['initial'],o['query']['program'],p['max_steps']))==o['outcome']
                               for o in row['observation_record']):
                            final.append(model)
                    assert len(final)==row['query_compatible_laws']
                    assert row['query_isolates_truth']==(final==[truth])
                if branch=='g2-cyclic-misspecified':
                    assert case['truth_excluded'] and truth not in p['models']
                    assert row['query_policy']=='misspecification-action'
                    assert row['requested_queries'] in (1,2)
                    assert row['acquisition_costs']['total_online']==row['acquisition_operations']
                    assert row['acquisition_operations']<=row['costs']['total_online']<=row['budget']
                    assert row['target_changed_parts']==[1,2]
                    assert row['truth_in_candidate_family'] is False and row['candidate_family_supplied']
                    assert row['comparison_role'].startswith('descriptive candidate-family misspecification')
                    paid=row['observation_record'][len(p['observations']):]
                    assert all(o['query']['kind']=='action' for o in paid)
                    assert row['target_action_parts']==[
                        o['query']['program'][0]%len(p['initial']) for o in paid]
                    compatible=[]
                    for model in p['models']:
                        if all(independent_assembly(model,o['query']['initial'],o['query']['program'],
                                p['max_steps'])==o['outcome'] for o in row['observation_record']):
                            compatible.append(model)
                    inconsistent=not compatible
                    assert len(compatible)==row['candidate_compatible_laws']
                    assert row['evidence_inconsistent_with_candidate_family']==inconsistent
                    if row['candidate_aware'] and inconsistent:
                        assert row['misspecification_detected'] and row['abstained_on_inconsistency']
                        assert row['program'] is None and not row['unsafe_action_attempted_after_inconsistency']
                    if row['method']=='forced-candidate-direct' and inconsistent and row['program'] is not None:
                        assert row['unsafe_action_attempted_after_inconsistency']
                    assert row['method_receives_evaluator_truth']==(row['method']=='known-law')
                if branch=='g0-common':
                    request=row['request'];target=request['target']
                    if row['planner']=='state':assert row['costs']['total_online']<=row['budget']
                    else:assert row['costs']['checking']+row['costs']['search']<=row['budget']
                    if row['program'] is not None:
                        actual=physical(json.dumps(dict(kind='graphic',cells=4,forbidden=[])),json.dumps(0),tuple(row['program']),3)
                        assert row['success']==(actual['legal'] and actual['state']==target)
                if branch=='g4-history':
                    selected=case['private']['selected'];posterior=row['posterior']
                    assert row['strategy_brier']==sum((value-int(i==selected))**2 for i,value in enumerate(posterior))
                    assert row['success']==(row['prediction']==case['private']['strategy'])
                    request=next(t for t in p['tiers'] if t['tier']==row['tier']);weights=[]
                    for i,history in enumerate(alternatives):
                        weight=1.
                        if row['tier']!='endpoint':
                            artifact=request['evidence']['earlier_artifact']
                            if case['condition']=='ambiguous':weight=float(artifact==history['endpoint'])
                            else:weight=sum(artifact==alternatives[j%4]['trace'][0]['after'] for j in (i,i+1))/2
                        for observation in request['evidence'].get('process',[]):
                            eligible=[t for t in history['trace'] if observation['index'] is None or t['index']==observation['index']]
                            if not any(t['action']==observation['action'] and t['actor']==observation['actor'] for t in eligible):weight=0.
                        weights.append(weight)
                    assert posterior==[weight/sum(weights) for weight in weights]
                metrics={k:row[k] for k in ('success','invalid','missing_output','forecast_brier','dependency_recovered',
                    'acquired_queries','query_exhausted','abstained_on_inconsistency','strategy_brier','abstained',
                    'first_actor_probability','first_operation_probability','source_fraction_absolute_error') if k in row}
                if 'costs' in row:metrics['online_operations']=row['costs'].get('total_online',row['costs'].get('total_per_case'))
                groups[key][uid].append(metrics)
                # Outcome-selected illustrations, explicitly separate from blind benchmarks.
                example_key=(case.get('stratum',case.get('condition',branch)),row.get('method',row.get('gate')),row['success'])
                if example_key not in examples:
                    examples[example_key]=dict(case_id=case['case_id'],block=name,selection='first observed outcome in cell; illustrative, not representative',
                        program=row.get('program'),success=row['success'],method=row.get('method',row.get('gate')),condition=example_key[0])
            for diagnostic in (unit.get('diagnostics') or []) if branch=='g1-native' else []:
                diagnostics[(case['stratum'],case['information'],case['representation'],diagnostic['gate'])].append(diagnostic)
    table=[]
    for key,units in sorted(groups.items()):
        metrics=next(iter(units.values()))[0]
        averages={f'mean_{m}':mean(mean(r[m] for r in records) for records in units.values()) for m in metrics}
        table.append(dict(zip(dimensions,key),**averages,distinct_context_units=len(units),rows=sum(map(len,units.values()))))
    result=dict(schema='v18.1.branch-summary.1',branch=branch,cells=table,rows_checked=count,
        distinct_context_units=len(contexts),distinct_law_configurations=len(laws),distinct_endpoints=len(endpoints),
        distinct_unlabeled_topologies=len(signatures),aggregation='equal context means, histories and repeated conditions nested; no independence claim for label permutations',
        inference='descriptive constructed-world discovery; not fresh confirmation',source_plan_sha256=file_digest(root/'PLAN.json'),
        raw_bindings=bindings,examples=list(examples.values()),
        gate_diagnostics=[dict(stratum=k[0],information=k[1],representation=k[2],gate=k[3],
             false_rejection_fraction=mean(r['false_rejection'] for r in rows),
             accepted_non_solution_fraction=mean(r['acceptance_of_non_solution'] for r in rows),
             scope='separate intact-routine evaluator probe, no free planner input') for k,rows in sorted(diagnostics.items())])
    write(output/'COMPARISONS.json',result)
    write(output/'VERIFICATION.json',dict(passed=True,rows_checked=count,summary_sha256=file_digest(output/'COMPARISONS.json'),
        verifier_sha256=file_digest(Path(__file__)),scope='complete receipt/reaggregation; every submitted program independently executed; G2 forecasts and identical observations checked',
        method_replay='separate bounded source-specific replay still required'))
    return result
