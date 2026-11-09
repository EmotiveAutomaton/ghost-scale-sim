from ghostscale.validation.soundingline.v27.common import digest,read,verify_complete,write
from runners import queue_v27 as queue


def test_long_preparation_counts_until_terminal_even_if_pid_is_reused(tmp_path,monkeypatch):
    campaign=tmp_path/'v27'
    write(campaign/'COUNTED_PREPARATION.json',dict(batches=['particle-check']))
    status=campaign/'particle-check/STATUS.json'
    write(status,dict(state='RUNNING',pid=42))
    monkeypatch.setattr(queue,'alive',lambda pid:pid==42)
    assert queue.outside(campaign)==1
    write(status,dict(state='COMPLETE',pid=42),immutable=False)
    assert queue.outside(campaign)==0


def test_controller_restart_preserves_fresh_host_backoff(tmp_path):
    samples=[[1000.,.95],[1030.,.94]]
    write(tmp_path/'HOST_LOAD.json',dict(samples=samples,worker_limit=2))
    assert queue.resume_load(tmp_path,clock=1040.)==(2,samples)
    assert queue.resume_load(tmp_path,clock=1200.)==(4,[])


def test_gear_one_survives_host_load_recovery(tmp_path):
    write(tmp_path/'GEAR.json',dict(worker_limit=1))
    assert queue.requested_limit(tmp_path,2)==1
    assert queue.requested_limit(tmp_path,4)==1


def test_split_seed_uses_logical_setting_index():
    from ghostscale.validation.soundingline.v27.study import global_setting_index
    plan=dict(settings=[{},{}],setting_indices=[3,4])
    assert global_setting_index(plan,0)==3
    assert global_setting_index(plan,1)==4
    assert global_setting_index(dict(settings=[{},{}]),1)==1


def test_split_aggregation_maps_parts_without_copying_raw_bytes(tmp_path):
    campaign=tmp_path/'v27';card='K1-P2';settings=[{'truth':'L'},{'truth':'A100'}]
    write(campaign/'protocols'/f'{card}.json',dict(settings=settings,blocks=1))
    write(campaign/'controls'/f'{card}.json',dict(passed=True))
    write(campaign/'split-builds'/f'{card}.json',dict(card=card,frozen_partition=[
        dict(part=1,setting_indices=[0]),dict(part=2,setting_indices=[1])]))
    for part in (1,2):
        packet=campaign/'packets'/f'v27-{card.lower()}-study-part{part}'
        plan=dict(card=card,family='K1',kind='precision',scientific_kind='precision',lane='precision',pool='followup_healing',
            settings=[settings[part-1]],setting_indices=[part-1],blocks=1,split_part=part,split_total=2,
            logical_aggregation_only=True,identity=packet.name,science_identity=f'v27-{card.lower()}-study',
            projected_wall_seconds=10.,cpu_cap_seconds=20.,pass_caps=dict(original=5.,replay=5.,review=5.))
        write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='COMPLETE'))
        write(packet/'original/SUMMARY.json',dict(family='K1',card=card,kind='precision',settings=plan['settings'],blocks=1,
            cells=[dict(setting=0,metric='timing_ratio',labels={'reader':'contrasts'},statistic='median_ratio')]))
        write(packet/'original/RAW_INDEX.json',{'world-0000-0000_points.json.gz':dict(sha256=str(part)*64,bytes=part)})
        write(packet/'PASS_COSTS.json',dict(pass_cpu_seconds=dict(original=1.,replay=1.,review=1.),
            setting_seconds=dict(original=[1.],replay=[1.],review=[1.])))
        write(packet/'REPLAY.json',dict(state='MATCH'))
        write(packet/'INDEPENDENT_REVIEW.json',dict(state='INDEPENDENT_RECONSTRUCTION_PASSED'))
        write(packet/'COMPLETE.json',dict(plan_sha256=digest(packet/'PLAN.json'),summary_sha256=digest(packet/'original/SUMMARY.json'),
            replay_sha256=digest(packet/'REPLAY.json'),review_sha256=digest(packet/'INDEPENDENT_REVIEW.json')))
    logical=queue.aggregate_split(campaign,card)
    verify_complete(logical)
    assert [row['setting'] for row in read(logical/'original/SUMMARY.json')['cells']]==[0,1]
    assert read(logical/'original/RAW_INDEX.json')=={}
    aggregation=read(logical/'AGGREGATION.json')
    assert sorted(aggregation['part_raw_indexes'])==[f'v27-{card.lower()}-study-part1',f'v27-{card.lower()}-study-part2']
    assert aggregation['no_new_scientific_draws'] is True
