from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import canonical,digest
from ghostscale.validation.soundingline.v18_1 import g4,g3
from ghostscale.validation.soundingline.v18_1.verify import independent_assembly


def test_history_known_answers_and_same_evidence_rival():
    readers,evaluators=g4.make_packets('development-history-answers',replicates=1)
    for public,truth in zip(readers,evaluators):
        posterior=[]
        for tier in public['tiers']:
            result=g4.predict(canonical(tier));posterior.append(result['posterior'])
            assert result['posterior']==g4.predict(canonical(tier),'direct-template')['posterior']
            assert result['posterior']==g4.predict(canonical(tier),'finite-history-ceiling')['posterior']
            assert sum(result['posterior'])==1
        assert posterior[0]==[0.25]*4
        if truth['condition']=='ambiguous':assert posterior==[[0.25]*4]*3
        else:
            # Routine and adaptation share the first graphic mark. Keep that
            # extra ambiguity rather than forcing an artificially unique cue.
            assert sorted(posterior[1]) in ([0,0,0.5,0.5],[0,0.25,0.25,0.5])
            assert posterior[2][truth['private']['selected']]==1
            if truth['condition']=='misleading':
                assert 0<posterior[1][truth['private']['selected']]<=0.5


def test_history_physics_actors_and_endpoint_matching():
    readers,evaluators=g4.make_packets('development-history-physics',replicates=1)
    for public,truth in zip(readers,evaluators):
        p=public['tiers'][0];alternatives=truth['private']['alternatives']
        assert len({tuple(h['program']) for h in alternatives})==4
        for history in alternatives:
            if truth['family']=='assembly':
                actual=independent_assembly(p['world'],p['initial'],history['program'],12)
                assert actual['legal'] and actual['stopped'] and actual['state']==p['endpoint']
            with pytest.raises(ValueError,match='actor'):
                g4.enact(p['world'],p['initial'],history['program'],
                         [role for role in p['roles'] if role['actor']!=history['trace'][0]['actor']])
        assert truth['private']['relationship']['exact_source_sequence']==(truth['private']['selected']==0)


def test_history_public_only_and_private_swap():
    readers,evaluators=g4.make_packets('development-history-separation',replicates=1)
    frozen=canonical(readers[0]['tiers'][2]);answer=g4.predict(frozen)
    evaluators[0]['private']=deepcopy(evaluators[-1]['private'])
    assert frozen==canonical(readers[0]['tiers'][2]) and answer==g4.predict(frozen)
    for location in ('top','world','role','method','evidence','process'):
        p=deepcopy(readers[0]['tiers'][2])
        target={'top':p,'world':p['world'],'role':p['roles'][0],'method':p['method_family'],
                'evidence':p['evidence'],'process':p['evidence']['process'][0]}[location]
        target['private_selected_strategy']=0
        with pytest.raises(ValueError):g4.predict(canonical(p))


def test_g3_familiar_offers_and_nested_no_oracle():
    cases=g3.make_cases('development-familiar-guard',per_stratum=4,histories=1,sizes=(7,),families=('chain',))
    for case in cases:
        if case['condition']=='familiar':
            changed=[i for i,(a,b) in enumerate(zip(case['public']['initial'],case['public']['target'])) if a!=b]
            assert len(changed)==1 and changed[0]<4
        p=deepcopy(case['public']);p['world']['target_witness']=case['private']['target_witness']
        with pytest.raises(ValueError,match='nested'):g3.predict(canonical(p),'primitive',512,32)


def test_blind_consumer_has_no_evaluator_input(tmp_path):
    import json
    from runners.read_v18_1_history import consume
    readers,evaluators=g4.make_packets('development-blind-consumer',replicates=1)
    reader=tmp_path/'READER.json';reader.write_bytes(canonical(dict(schema='v18.1.blind-reader-packet.1',cases=readers)))
    (tmp_path/'EVALUATOR.json').write_text('intentionally invalid and unreadable as JSON')
    result=consume(reader,tmp_path/'output','direct-template')
    assert result['requests']==72 and not result['evaluator_access']


def test_history_structural_support_excludes_draw_identity():
    readers,evaluators=g4.make_packets('development-context-dedup',replicates=3)
    for r,e in zip(readers,evaluators):
        p=r['tiers'][0]
        assert e['structural_unit']==digest([p['world'],p['initial'],p['method_family'],e['condition']])
    assert len({e['sampling_cluster'] for e in evaluators})==18
