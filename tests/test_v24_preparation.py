"""Engineering fixtures only: these do not enter a scientific pilot ledger."""
import itertools
import math
import numpy as np
import pytest
from datetime import datetime
from ghostscale.validation.soundingline.v24 import evidence, search, n16
from ghostscale.validation.soundingline.v24 import contract, pilot


def test_mixture_is_a_probability_reader_with_analytic_support():
    p=np.array([0.,.2,.8]);q=np.array([.3,.4,.3]);eps=.02
    candidate=evidence.mixture(p,q,eps)
    assert candidate.sum()==pytest.approx(1)
    score=np.log(candidate/q)
    assert score.min()>=math.log(eps)
    raw=np.array([-np.inf,math.log(.2/.4),math.log(.8/.3)])
    np.testing.assert_allclose(score,evidence.mixture_gain(raw,eps),atol=1e-14)
    np.testing.assert_array_equal(evidence.mixture_gain(raw,1),np.zeros(3))
    with pytest.raises(ValueError):evidence.mixture([1,1],q[:2],eps)
    with pytest.raises(ValueError):evidence.mixture_gain(raw,0)


def test_epsilon_choice_has_strict_loss_bar_and_never_confirms():
    result=evidence.choose_epsilon(np.full(16,.1))
    assert result['epsilon']==.02
    assert result['confirmation'] is False
    assert evidence.choose_epsilon([-1.,-2.])['epsilon'] is None
    with pytest.raises(ValueError):evidence.choose_epsilon([math.nan])


def test_adaptive_stake_cannot_see_the_current_or_future_outcome():
    a=evidence.adaptive_bets([.2,.3,-1],-1)
    b=evidence.adaptive_bets([.2,-1,10],-1)
    assert a['stakes'][:2]==b['stakes'][:2]
    assert evidence.adaptive_bets([],-1)['log_e']==0
    with pytest.raises(ValueError,match='do not clip'):evidence.adaptive_bets([-1.000001],-1)
    with pytest.raises(ValueError):evidence.adaptive_bets([0],-1,math.nan)


def test_known_null_has_unit_expected_wealth_and_anytime_error_bound():
    # Exhaustive independent fair signs: no Monte Carlo tolerance or fitted bar.
    wealth=[];crossed=[];threshold=5
    for path in itertools.product((-1.,1.),repeat=10):
        record=evidence.adaptive_bets(path,-1,margin=0)
        prefix=np.cumprod(record['factors'])
        wealth.append(prefix[-1]);crossed.append(bool(np.any(prefix>=threshold)))
    assert np.mean(wealth)==pytest.approx(1,abs=1e-12)
    assert np.mean(crossed)<=1/threshold
    assert evidence.adaptive_bets([1.]*30,-1,margin=0)['log_e']>math.log(20)


def test_sample_forecast_is_not_power_or_adaptive_guarantee():
    result=evidence.expected_sample_plan([.2]*8,math.log(.02),math.log(20),100)
    assert result['state']=='EXPECTED_EVIDENCE_FORECAST'
    assert result['independent_blocks']*result['expected_log_e_per_block']>=1.5*math.log(20)
    assert not result['power_guarantee'] and not result['adaptive_strategy_guaranteed_to_match_forecast']
    assert evidence.expected_sample_plan([0]*8,-1,math.log(20),100)['state']=='FIXED_SAMPLE_DESIGN_REQUIRED'
    assert evidence.expected_sample_plan([.2]*8,-1,math.log(20),1e6)['state']=='FIXED_SAMPLE_DESIGN_REQUIRED'


def test_complete_search_roster_and_ties_and_latest_implausibility_cut():
    schedule=search.halving_schedule(list(range(240)))
    assert [r['cells'] for r in schedule]==[240,80,27,9,3,1]
    assert [r['cumulative_worlds_per_cell'] for r in schedule]==[1,3,9,27,81,243]
    assert search.survivors(dict(a=1,b=1,c=0,d=0),['b','a','c','d'])==['b','a']
    with pytest.raises(ValueError):search.survivors(dict(a=1),['a','b'])
    with pytest.raises(ValueError):search.halving_schedule(['a'],1.5)
    assert search.history_match(2,0,1,0,0)['retain']
    assert not search.history_match(3,0,1,0,0)['retain']
    assert search.history_match(0,0,0,0,0)['retain']
    assert not search.history_match(1,0,0,0,0)['retain']


def test_seed_extension_requires_one_complete_look_and_caps_at_double():
    planned=[2,7,10]
    extension=search.seed_extension(planned,planned,True)
    assert len(extension)==len(planned) and not set(extension)&set(planned)
    assert search.seed_extension(planned,planned,False)==[]
    for observed in ([2,7],[2,7,10,10],planned+extension):
        with pytest.raises(ValueError):search.seed_extension(planned,observed,True)


@pytest.mark.parametrize('source',['N4d','N9','N11','N15'])
def test_n16_excluded_engineering_fixture_replays_and_reconstructs(source):
    variant=next(v for v in n16.model.variants() if v['source']==source)
    args=(variant,'v24-engineering-excluded',1,15,0,.02)
    public,truth,record,score=n16.case(*args)
    assert truth['law']==15
    assert (public,truth,record,score)==n16.case(*args)
    reconstructed=n16.check(variant,public,truth,record,.02)
    for k in score:assert score[k]==pytest.approx(reconstructed[k],abs=1e-11)
    last=record['posteriors'].pop()
    with pytest.raises(ValueError,match='roster'):n16.check(variant,public,truth,record,.02)
    record['posteriors'].append(last)
    record['posteriors'][0]['mass'][0][0]+=.01
    with pytest.raises(ValueError):n16.check(variant,public,truth,record,.02)


def test_n16_balancing_rejects_pseudoreplication_missing_cells_and_nonfinite():
    rows=[dict(identity=[b,l,v['id'],0],mixture_gain=b+1.)
          for b in range(2) for l in range(16) for v in n16.model.variants()]
    assert n16.aggregate(rows,1)==[1,2]
    for bad in (rows[:-1],rows+[rows[0]],rows[768:]):
        with pytest.raises(ValueError):n16.aggregate(bad,1)
    rows[0]['mixture_gain']=math.nan
    with pytest.raises(ValueError):n16.aggregate(rows,1)


def pilot_plan():
    return dict(kind='pilot',family='N16',identity='v24-n16-pilot-1',blocks=4,
        cases_per_cell=5,epsilon=.1,variant_ids=[v['id'] for v in n16.model.variants()],
        numerical_threads=1,priority='below-normal',cpu_cap_seconds=5400)


def test_pilot_admission_enforces_window_total_reserve_and_family_cap():
    plan=pilot_plan();opening=datetime.fromisoformat(contract.OPEN)
    assert contract.admission(plan,opening,0)
    for clock in ('2026-10-19T05:00:00-07:00',contract.CLOSE,contract.PRODUCTION_OPEN):
        with pytest.raises(ValueError):contract.admission(plan,datetime.fromisoformat(clock),0)
    with pytest.raises(ValueError):contract.admission(dict(plan,kind='production'),opening,0)
    with pytest.raises(ValueError):contract.admission(plan,opening,32*3600)
    with pytest.raises(ValueError):contract.admission(plan,opening,0,family_charged_seconds=5*3600)
    assert sum(contract.FAMILY_SECONDS.values())+contract.VERIFICATION_RESERVE+contract.acceptance()['preparation_seconds']==contract.TOTAL_SECONDS


def test_n9_pilot_block_has_each_mechanism_for_every_law():
    model=n16.model.SOURCES['N9'];variant=model.variants()[0]
    for block in (0,1,3):
        for law in range(16):
            labels=[model.sample(variant,'v24-engineering-excluded',16*block+law,c)[1]['mechanism'] for c in range(5)]
            assert sorted(labels)==list(range(5))
    plan=pilot_plan();pilot.validate_design(plan)
    with pytest.raises(ValueError):pilot.validate_design(dict(plan,cases_per_cell=1))
    with pytest.raises(ValueError):pilot.validate_design(dict(plan,variant_ids=plan['variant_ids'][:-1]))


def test_prepared_runner_refuses_early_before_child_launch(tmp_path,monkeypatch):
    from runners import run_v24_pilot as runner
    from ghostscale.validation.soundingline.v23.common import write
    packet=tmp_path/'packets/v24-n16-pilot-1';packet.mkdir(parents=True)
    write(packet/'PLAN.json',pilot_plan())
    monkeypatch.setattr(runner,'now',lambda:datetime.fromisoformat('2026-10-19T05:00:00-07:00'))
    def unexpected(*args,**kwargs):raise AssertionError('process launched before opening')
    monkeypatch.setattr(runner.subprocess,'Popen',unexpected)
    with pytest.raises(ValueError,match='outside'):runner.execute(packet)
    with pytest.raises(ValueError,match='outside'):runner.child(packet,'original')
    assert not (tmp_path/'OWNER.lock').exists()


def test_rank_instrument_is_uniform_on_exchangeable_positions_and_discrete_ties():
    from ghostscale.validation.soundingline.v24 import rank_calibration as rc
    ranks=[rc.rank(i,[j for j in range(101) if j!=i],.5) for i in range(101)]
    tied=[rc.rank(0,[0]*100,(i+.5)/101) for i in range(101)]
    assert ranks==tied==list(range(101))
    result=rc.histogram(ranks)
    assert result['counts']==[1]*101 and result['calibrated'] is None
    assert rc.rank(-1,[0]*100,.5)==0
    assert rc.rank(1,[0]*100,.5)==100
    with pytest.raises(ValueError):rc.rank(0,[0]*99,.5)
    with pytest.raises(ValueError):rc.rank(0,[0]*100,1)


def test_complete_pilot_transport_with_known_zero_score_fixture(tmp_path,monkeypatch):
    # Full transport/coverage test, not a scientific N16 pilot: replace only the
    # generator and numerical checker with a declared known-zero instrument.
    import os
    from pathlib import Path
    from ghostscale.validation.soundingline.v23 import ledger
    from ghostscale.validation.soundingline.v23.common import read,digest
    directory=os.environ.get('V24_SCHEMA_DIRECTORY')
    if not directory:pytest.skip('canonical schema not supplied; transport validation unavailable')
    schema,hashes=ledger.canonical_schema(directory)
    plan=pilot_plan();plan.update(schema_directory=directory,schema_pin=dict(files=hashes,
        schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()))
    score=dict(raw_gain=0.,mixture_gain=0.,lower_bound=math.log(.1))
    def fixture(v,identity,b,l,c,epsilon):
        m=(16*b+l+c)%5 if v['source']=='N9' else v['original']['settings']['mechanism']
        return {'fixture':'known-zero-score'},{'law':l,'mechanism':m},{'fixture':True},dict(score)
    monkeypatch.setattr(n16,'case',fixture)
    monkeypatch.setattr(n16,'check',lambda *args:dict(score))
    for label in ('original','replay'):pilot.execute(tmp_path/label,plan,lambda **progress:None)
    assert {p.name:digest(p) for p in (tmp_path/'original').iterdir()}=={p.name:digest(p) for p in (tmp_path/'replay').iterdir()}
    reviewed=pilot.reconstruct(tmp_path/'original',plan,lambda **progress:None)
    assert reviewed['cases']==15360 and reviewed['blocks']==4
    assert read(tmp_path/'original/SUMMARY.json')['balanced_block_means']==[0.]*4
    path=tmp_path/'original/evidence_points.jsonl.gz'
    with path.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='hash'):pilot.reconstruct(tmp_path/'original',plan,lambda **progress:None)
