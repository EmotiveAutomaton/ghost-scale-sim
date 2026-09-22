import pytest
from ghostscale.validation.soundingline.v19 import filter_cost as C


def row(i, source=None):
    return dict(step=i, source_step=i if source is None else source, source_id=str(i if source is None else source), context=0, endpoint=0)


def test_live_positive_placebo():
    assert all(C.controls().values())


def test_hand_count_and_reset():
    obs=[row(i) for i in range(1,33)]
    for arm,h in [('static',16),('reset-16',16),('known-time-type',32),('unknown-time',288),('unknown-time-type',560)]:
        c=C.counts(obs,arm,32,'purpose',32)
        assert c['weight_bytes']==8*h
        assert c['likelihood_factor_applications']==h*(16 if arm=='reset-16' else 32)
        assert c['endpoint_law_bytes']==16*4*8*8
    assert C.hypotheses('known-time-type',32,'skill',8)[16]==('skill',8,0)


def test_copies_and_conflicting_sources():
    obs=[row(1),row(2,1),row(3)]
    assert C.counts(obs,'static',32,'purpose',3)['retained_sources']==2
    obs[1]['endpoint']=1
    with pytest.raises(ValueError,match='conflicting'):C.sources(obs)


def test_symbolic_costs_are_not_timings():
    c=C.counts([row(1)],'static',32,'purpose',1)
    for s in C.scenarios(c):
        assert s['reread_units']-s['cached_units']==(s['queries']-1)*16
        assert s['first_strictly_cheaper_integer_queries']==2
        assert 'cpu_seconds' not in s


def test_timing_full_roster_and_corruption():
    streams={1:[dict(length=32,kind='purpose',switched=False,duplicates=True)]}
    rows=[dict(lineage=1,stream=0,arm=a,cpu_seconds=i/10) for i,a in enumerate(C.ARMS)]
    out=C.timing_records(rows,streams)
    assert sum(x['calls'] for x in out)==5
    assert sum(x['total_cpu_seconds'] for x in out)==pytest.approx(1)
    for bad in (rows[:-1],rows+[rows[0]],[dict(rows[0],cpu_seconds=-1)]+rows[1:]):
        with pytest.raises(ValueError):C.timing_records(bad,streams)


def test_complete_retained_fixture(tmp_path):
    import gzip,json
    from ghostscale.validation.soundingline.v18_3.io import canonical,write,file_digest
    p=tmp_path/'inputs/fixture';p.mkdir(parents=True)
    write(p/'PLAN.json',dict(design=dict(lineages=[1])))
    write(p/'COMPLETE.json',dict(plan_sha256=file_digest(p/'PLAN.json')))
    ss=[dict(draw=2,maker=m,length=32,kind='purpose',switched=False,duplicates=False,observations=[row(i) for i in range(1,33)]) for m in range(16)]
    rr=[];mapping={}; cells=[]
    for a in C.ARMS:
        mapping[a]=dict(hypotheses=C.hypotheses(a,32,'purpose'),rows=[])
        for si in range(16):
            for step in (8,16,17,20,32):
                j=len(mapping[a]['rows']);mapping[a]['rows'].append([si,step])
                rr.append(dict(stream=si,step=step,arm=a,joint_array=a,joint_row=j,**{m:1. for m in C.METRICS}))
        for step in (8,16,17,20,32):
            cells.append(dict(lineage=1,draw=2,length=32,kind='purpose',switched=False,duplicates=False,step=step,arm=a,**{m:1. for m in C.METRICS}))
    (p/'raw').mkdir()
    for n,v in [('observations',ss),('forecasts',rr)]:
        (p/'raw'/f'1-{n}_points.json.gz').write_bytes(gzip.compress(canonical(v),mtime=0))
    write(p/'evaluator/1-joint-map.json',mapping);write(p/'SUMMARY.json',dict(cells=cells))
    (p/'timings').mkdir()
    timing=[dict(lineage=1,stream=i,arm=a,cpu_seconds=.01) for i in range(16) for a in C.ARMS]
    for e in ('original','replay','portable'):(p/'timings'/f'{e}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in timing))
    pins={f.relative_to(tmp_path/'inputs').as_posix():file_digest(f) for f in p.rglob('*') if f.is_file()}
    out=C.run(tmp_path,dict(design=dict(input_files=pins,packets=[dict(name='fixture')])),lambda **_:None)
    assert out['rows']==400 and out['cells']==25 and out['timing_calls']==240
