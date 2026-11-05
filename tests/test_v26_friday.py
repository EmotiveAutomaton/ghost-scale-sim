"""New estimator/knob controls and resource boundary failures, excluded from science."""
from datetime import datetime,timezone
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v26 import friday,friday_reporting,values,alias_repair,contract
from ghostscale.validation.soundingline.v26.common import rng,close,write


def test_first_crossing_is_not_sustained_crossing():
    r=friday.crossing_metrics([0,1.2,.9,2,3])
    assert r['first_crossing']==2 and r['sustained_crossing']==4 and r['recrossed_below']==1
    assert friday.crossing_metrics([0,1.2,0])['sustained_crossing'] is None
    assert friday.crossing_metrics([1,1])['sustained_crossing']==1
    assert friday.crossing_metrics([0,0])['first_crossing'] is None
    with pytest.raises(ValueError):friday.crossing_metrics([math.nan])


def test_evidence_metrics_do_not_change_trajectory_and_scalar_review_matches():
    s=dict(environment='forced',eta=.005,theta=.6,max_history=16,histories=[8,16],truth_types=['A','B'])
    a=values.block(s,['friday-excluded-metrics']);b=values.block(s|dict(evidence_path=True),['friday-excluded-metrics'])
    assert a['public']==b['public']
    assert a['readouts']==[r for r in b['readouts'] if r['labels']['reader']!='B-over-A-persistence']
    for x,y in zip(a['truth_and_observations'],b['truth_and_observations']):assert x['records']==y['records']
    close(b,values.block(s|dict(evidence_path=True),['friday-excluded-metrics'],True),tol=1e-9)


def test_real_switch_strength_and_zero_control():
    a=alias_repair.library(rng('friday-excluded-gap'),'varying')
    assert np.array_equal(a,alias_repair.library(rng('friday-excluded-gap'),'varying',1))
    for gap in (0,.125,.5,1):
        e=alias_repair.library(rng('friday-excluded-gap'),'varying',gap)
        assert e[5,0]-e[5,1]==gap
        hs=alias_repair.hypotheses(3);pairs=alias_repair.split_pairs(3,hs)
        law=alias_repair.laws(e,alias_repair.SWITCH[0],hs)
        assert all((np.max(abs(law[i]-law[j]))>1e-12)==(gap>0) for i,j in pairs)


def fixture():
    a=contract.acceptance('2026-11-02T15:00:00+00:00','spec')
    s=dict(total_cpu_seconds=110*3600,pools_seconds=dict(studies=86*3600,preparation=3*3600,followup_healing=15*3600,protected=6*3600),family_seconds=dict(E1=26*3600),review_target='2026-11-06T14:00:00+00:00')
    p=dict(family='E1',kind='study',pool='studies',cpu_cap_seconds=1000,pass_caps=dict(original=300,replay=300,review=300),numerical_threads=1,priority='below-normal',design_check_sha256='bound',supplement_sha256='bound',projected_wall_seconds=1000)
    return a,s,p


def test_expansion_includes_prior_charges_and_enforces_gear_and_friday():
    a,s,p=fixture();now=datetime(2026,11,4,15,tzinfo=timezone.utc)
    assert friday.admission(p,a,s,now,{'studies':60*3600})
    with pytest.raises(ValueError,match='family ceiling'):friday.admission(p,a,s,now,{'family:E1':26*3600-999})
    with pytest.raises(ValueError,match='two shared'):friday.admission(p,a,s,now,{},2)
    with pytest.raises(ValueError,match='Friday review margin'):friday.admission(p,a,s,datetime(2026,11,6,13,59,tzinfo=timezone.utc),{})
    with pytest.raises(ValueError,match='pool ceiling'):friday.admission(p,a,s,now,{'studies':86*3600-999})


def test_verification_healing_uses_shared_pool_and_declared_attempt():
    a,s,p=fixture();now=datetime(2026,11,4,15,tzinfo=timezone.utc)
    healing=p|dict(kind='healing',pool='followup_healing',healing_number=1)
    assert friday.admission(healing,a,s,now,{'followup_healing':5000})
    assert friday.admission(healing|dict(healing_number=2),a,s,now,{'followup_healing':5000})
    with pytest.raises(ValueError,match='healing limit'):friday.admission(healing|dict(healing_number=3),a,s,now,{})
    with pytest.raises(ValueError,match='pool ceiling'):friday.admission(healing,a,s,now,{'followup_healing':15*3600-999})


def test_outside_counts_same_campaign_manual_scientist(monkeypatch,tmp_path):
    from runners import run_v26_friday as runner
    campaign=tmp_path/'campaign'
    write(campaign/'packets/owned/STATUS.json',dict(state='RUNNING',parent_pid=1))
    write(campaign/'packets/manual/STATUS.json',dict(state='RUNNING',parent_pid=2))
    monkeypatch.setattr(runner.base,'outside_scientists',lambda _:0)
    monkeypatch.setattr(runner,'alive',lambda _:True)
    assert runner.outside(campaign,{'owned'})==1
    assert runner.outside(campaign)==2


def test_public_plan_omits_machine_local_execution_paths():
    from runners.run_v26_friday import public_plan
    plan=dict(card='U1-X02',source_root='private-source',interpreter='private-python',accounting_root='private-campaign')
    assert public_plan(plan)=={'card':'U1-X02'}


def test_friday_qualification_does_not_split_multiline_synthesis(tmp_path):
    path=tmp_path/'READING_INTENT.md'
    e1=friday_reporting.hypothesis_note('E1')
    h1=friday_reporting.hypothesis_note('H1')
    path.write_text(
        '| G-V26-E1-1 | parent |\n'
        '| G-V26-H1-1 | parent |\n'
        f'**What these add up to.** Starting from oneself helped nearby makers {e1}\n'
        f'and hurt elsewhere; fair rematching narrowed that advantage. {h1}\n',
        encoding='utf-8',
    )
    friday_reporting.repair_hypothesis_notes(path)
    repaired=path.read_text(encoding='utf-8')
    assert repaired.count(e1)==repaired.count(h1)==1
    assert repaired.index(e1)<repaired.index(h1)<repaired.index('**What these add up to.**')
    assert ('**What these add up to.** Starting from oneself helped nearby makers\n'
            'and hurt elsewhere; fair rematching narrowed that advantage.\n') in repaired


def test_registry_has_no_nominal_shift_alias_and_fixed_samples():
    cards=friday.cards();assert len({c['id'] for c in cards})==len(cards)==37
    for c in cards:
        assert c['blocks'] in (64,128) and c['question'] and c['case']
        if c['family']=='E1':assert all(s['environment']!='shift' or s['specialists']<1 for s in c['settings'])


def test_refill_cards_are_prospective_bounded_and_design_available():
    cards=friday.refill_cards()
    assert len(cards)==8
    assert len({c['id'] for c in cards})==8
    assert {c['id'] for c in cards}=={
        'E2-X02R1','E2-X03R1','E2-X04R1','U1-X03R1','U1-X04R1',
        'H1-X10','H2-X03','H3-X04',
    }
    successors=[c for c in cards if 'resource_successor_of' in c]
    assert len(successors)==5
    assert all(c['held_packet'].endswith('-probe') for c in successors)
    focused=next(c for c in cards if c['id']=='H1-X10')
    assert focused['blocks']==48 and len(focused['settings'])==4
    assert {s['episodes'] for s in focused['settings']}=={3,12}
    assert all(friday.design_check(c)['passed'] for c in cards)


def test_continuation_refill_is_two_first_generation_r1_evidence_doses():
    cards=friday.continuation_refill_cards()
    assert [card['id'] for card in cards]==['R1-X03','R1-X04']
    assert [card['blocks'] for card in cards]==[64,64]
    assert [len(card['settings']) for card in cards]==[20,20]
    assert [{setting['training_count'] for setting in card['settings']} for card in cards]==[{2},{1}]
    for card in cards:
        assert {setting['structure'] for setting in card['settings']}=={'constant','varying'}
        assert {setting['beta'] for setting in card['settings']}=={1,3}
        assert {setting['switch_gap'] for setting in card['settings']}=={0.,.125,.25,.5,1.}
        assert friday.design_check(card)['passed']


def test_final_refill_is_two_narrow_hierarchy_endpoint_maps():
    cards=friday.final_refill_cards()
    assert [card['id'] for card in cards]==['H1-X11','H1-X12']
    assert [card['held_predecessor'] for card in cards]==['H1-X07','H1-X09']
    assert [card['blocks'] for card in cards]==[32,32]
    assert [{setting['pieces'] for setting in card['settings']} for card in cards]==[{8},{20}]
    for card in cards:
        assert len(card['settings'])==4
        assert {setting['tops'] for setting in card['settings']}=={10}
        assert {setting['episodes'] for setting in card['settings']}=={3,12}
        assert {setting['overlap'] for setting in card['settings']}=={'partial','heavy'}
        assert friday.design_check(card)['passed']


def test_rare_event_refill_freezes_two_higher_precision_boundary_maps():
    cards=friday.rare_event_refill_cards()
    assert [card['id'] for card in cards]==['H2-X04','H3-X05']
    assert [card['blocks'] for card in cards]==[128,128]
    h2,h3=cards
    assert len(h2['settings'])==12 and {setting['d'] for setting in h2['settings']}=={.05}
    assert {setting['units'] for setting in h2['settings']}=={16,32,64}
    assert len(h3['settings'])==12 and {setting['r'] for setting in h3['settings']}=={.05}
    assert {setting['units'] for setting in h3['settings']}=={12,48}
    assert all(friday.design_check(card)['passed'] for card in cards)


def test_lower_rate_refill_freezes_complete_pause_and_endpoint_revision_maps():
    cards=friday.lower_rate_refill_cards()
    assert [card['id'] for card in cards]==['H2-X05','H3-X06']
    assert [card['blocks'] for card in cards]==[256,256]
    h2,h3=cards
    assert len(h2['settings'])==12 and {setting['d'] for setting in h2['settings']}=={.025}
    assert {setting['units'] for setting in h2['settings']}=={16,32,64}
    assert len(h3['settings'])==8 and {setting['r'] for setting in h3['settings']}=={.025}
    assert {setting['partial'] for setting in h3['settings']}=={0.,.75}
    assert {setting['discovery_fraction'] for setting in h3['settings']}=={.25,.75}
    assert {setting['units'] for setting in h3['settings']}=={12,48}
    assert all(friday.design_check(card)['passed'] for card in cards)


def test_revision_length_refill_freezes_two_intermediate_endpoint_maps():
    cards=friday.revision_length_refill_cards()
    assert [card['id'] for card in cards]==['H3-X07','H3-X08']
    assert [card['blocks'] for card in cards]==[256,256]
    complete,incomplete=cards
    assert all(len(card['settings'])==4 for card in cards)
    assert all({setting['r'] for setting in card['settings']}=={.025} for card in cards)
    assert all({setting['units'] for setting in card['settings']}=={24,36} for card in cards)
    assert all({setting['discovery_fraction'] for setting in card['settings']}=={.25,.75} for card in cards)
    assert {setting['partial'] for setting in complete['settings']}=={0.}
    assert {setting['partial'] for setting in incomplete['settings']}=={.75}
    assert all(friday.design_check(card)['passed'] for card in cards)


def test_midpoint_and_half_rate_refill_freezes_two_budgeted_maps():
    cards=friday.midpoint_and_half_rate_refill_cards()
    assert [card['id'] for card in cards]==['H2-X06','H3-X09']
    h2,h3=cards
    assert h2['blocks']==512 and len(h2['settings'])==12
    assert {setting['d'] for setting in h2['settings']}=={.0125}
    assert {setting['units'] for setting in h2['settings']}=={16,32,64}
    assert h3['blocks']==224 and len(h3['settings'])==4
    assert {setting['r'] for setting in h3['settings']}=={.025}
    assert {setting['units'] for setting in h3['settings']}=={18}
    assert {setting['partial'] for setting in h3['settings']}=={0.,.75}
    assert {setting['discovery_fraction'] for setting in h3['settings']}=={.25,.75}
    assert all(friday.design_check(card)['passed'] for card in cards)


def test_transition_and_quarter_rate_refill_freezes_two_budgeted_maps():
    cards=friday.transition_and_quarter_rate_refill_cards()
    assert [card['id'] for card in cards]==['H2-X07','H3-X10']
    h2,h3=cards
    assert h2['blocks']==1024 and len(h2['settings'])==12
    assert {setting['d'] for setting in h2['settings']}=={.00625}
    assert {setting['units'] for setting in h2['settings']}=={16,32,64}
    assert h3['blocks']==224 and len(h3['settings'])==4
    assert {setting['r'] for setting in h3['settings']}=={.025}
    assert {setting['units'] for setting in h3['settings']}=={15}
    assert {setting['partial'] for setting in h3['settings']}=={0.,.75}
    assert {setting['discovery_fraction'] for setting in h3['settings']}=={.25,.75}
    assert all(friday.design_check(card)['passed'] for card in cards)


def test_deterministic_replay_reserves_the_slower_measured_execution_twin():
    from runners.run_v26_friday import measured_study_caps
    cost=dict(
        pass_cpu_seconds=dict(original=3.125,replay=3.0,review=2.984375),
        setting_seconds=dict(
            original=[.046875]*11+[.0625],
            replay=[.046875]*12,
            review=[.03125]*12,
        ),
    )
    caps=measured_study_caps(cost,[{}]*12,128)
    assert caps['original']==caps['replay']
    assert caps['replay']>121.6796875
    assert caps['review']<caps['replay']


def test_monitor_handles_initial_build_and_missing_running_owner(tmp_path):
    from runners.watch_v26 import observe
    from ghostscale.validation.soundingline.v26.common import write
    write(tmp_path/'QUEUE_STATUS.json',dict(state='BUILDING'))
    assert observe(tmp_path)==[] and not list((tmp_path/'events').glob('*.json'))
    write(tmp_path/'QUEUE_STATUS.json',dict(state='RUNNING'),immutable=False)
    assert observe(tmp_path)==[] and len(list((tmp_path/'events').glob('*.json')))==1
