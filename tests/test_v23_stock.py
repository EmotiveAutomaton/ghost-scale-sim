from pathlib import Path
from datetime import datetime,timezone,timedelta
from runners import stock_v23 as stock
from ghostscale.validation.soundingline.v23.common import write,read


def test_failed_pilot_gets_a_retained_disposition_without_repair(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()))
    pilot=tmp_path/'packets/pilot';write(pilot/'STATUS.json',dict(state='failed'))
    def forbidden(*args,**kwargs):raise AssertionError('failed pilot was restarted')
    monkeypatch.setattr(stock.admission,'admit',forbidden)
    result=stock.step(tmp_path,tmp_path/'state',dict(families={'N4':dict(pilot=str(pilot),fit_cpu_cap=1000)}))
    assert result['state']=='REVIEW_BOUNDARY' and result['families']['N4']=='PILOT_FAILED'
    assert read(tmp_path/'state/N4-DISPOSITION.json')['repair_requires_review']


def test_verified_fits_advance_to_full_first_batch_without_schedule_wait(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),max_scientific_workers=3))
    pilot=tmp_path/'packets/pilot';write(pilot/'PLAN.json',{});write(pilot/'STATUS.json',dict(state='complete'));write(pilot/'COMPLETE.json',{})
    for label in ('original','replay'):write(pilot/label/'pilot-cost.json',{})
    fit=tmp_path/'packets/v23-n5-fits-1';write(fit/'PLAN.json',{});write(fit/'COMPLETE.json',{})
    write(fit/'STATUS.json',dict(state='complete'))
    monkeypatch.setattr(stock,'verify_packet',lambda p:({},dict(state='PILOT_VALID')))
    monkeypatch.setattr(stock.admission,'profile_estimate',lambda *args:1000.2)
    monkeypatch.setattr(stock.queue,'inventory',lambda *args:[])
    calls=[]
    monkeypatch.setattr(stock.admission,'admit',lambda *args,**kwargs:calls.append((args,kwargs)))
    monkeypatch.setattr(stock.queue,'step',lambda *args:dict(families={'N5':'RUNNING_OR_ADMITTED'}))
    result=stock.step(tmp_path,tmp_path/'state',dict(families={'N5':dict(pilot=str(pilot),fit_cpu_cap=1800)}))
    assert result['state']=='RUNNING'
    assert calls[0][0][1:]==('N5','family-batch',1001) and calls[0][1]['launch']


def test_existing_priority_family_is_serviced_before_lower_priority_fits(tmp_path,monkeypatch):
    write(tmp_path/'ACCEPTANCE.json',dict(report_start=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),max_scientific_workers=3))
    write(tmp_path/'packets/v23-r4-batch-1/PLAN.json',{})
    pilot=tmp_path/'pilot';write(pilot/'PLAN.json',{});write(pilot/'COMPLETE.json',{})
    for label in ('original','replay'):write(pilot/label/'pilot-cost.json',{})
    monkeypatch.setattr(stock,'verify_packet',lambda p:({},dict(state='PILOT_VALID')))
    monkeypatch.setattr(stock.admission,'profile_estimate',lambda *args:1000.)
    monkeypatch.setattr(stock.queue,'inventory',lambda *args:[])
    calls=[]
    def queue_step(campaign,config):
        name=next(iter(config['families']));calls.append(name);return dict(families={name:'RUNNING_OR_ADMITTED'})
    monkeypatch.setattr(stock.queue,'step',queue_step)
    monkeypatch.setattr(stock.admission,'admit',lambda *args,**kwargs:calls.append((args[1],kwargs)))
    config=dict(families={'N2':dict(pilot=str(pilot),fit_cpu_cap=1000,fit_repair=True),'R4':{}})
    stock.step(tmp_path,tmp_path/'state',config)
    assert calls[0]=='R4' and calls[1][0]=='N2' and calls[1][1]['repair']
