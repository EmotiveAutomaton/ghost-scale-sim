"""Independent retained-output audit; never runs the scientific handler."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal,localcontext
import os,sys,time,json,gzip,zipfile,hashlib,struct,math,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
repo=Path.cwd();c=repo.parent/'.local/v19';sys.path.insert(0,str(repo))
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest as sha,now
from ghostscale.validation.soundingline.v19 import runtime as R
R.below_normal();started=time.process_time();state='failed'
name='G19-B-law-rounding-envelope-1';root=c/name;out=repo/'results/v19'/name
try:
    assert read(c/'QUEUE-STATUS.json')['state']=='awaiting_review_refill'
    plan=read(root/'PLAN.json');prepared=read(c/'ROUNDING_PREPARED_QUEUE_237900.json')
    assert plan['environment']==R.fingerprint()
    original=read(root/'COMPLETE.json');replays=[]
    for job in prepared['jobs']:
        r=Path(job['root']);receipt=read(r/'COMPLETE.json')
        assert job['plan_sha256']==sha(r/'PLAN.json')==sha(root/'PLAN.json')==receipt['plan_sha256']
        assert sha(r/'SOURCE.zip')==plan['source_archive_sha256']
        assert all(sha(Path(job['source'])/n)==h for n,h in plan['sources'].items())
        with zipfile.ZipFile(r/'SOURCE.zip') as z:assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==plan['sources']
        assert all(sha(r/n)==h for n,h in receipt['files'].items())
        assert receipt['files']==original['files']
        replays.append(dict(job=r.name,complete_sha256=sha(r/'COMPLETE.json'),plan_sha256=sha(r/'PLAN.json'),files=len(receipt['files']),all_output_bytes_equal=True,source_verified=True))
    rows=json.loads(gzip.decompress((root/'raw/law_rounding_summary_points.json.gz').read_bytes()));assert len(rows)==192
    row_lookup={(r['lineage'],r['storage_dtype'],r['reconstruction'],r['context']):r for r in rows}
    metrics=[k for k in rows[0] if k not in ('lineage','storage_dtype','reconstruction','context')]
    scalar_rows=[];count=0;max_loss_difference=0.;normalization_discrepancy=0.
    for lineage in range(190000,190008):
        law=np.array(read(root/'inputs'/f'{lineage}-law.json'))
        for dtype,fmt in [('float64','d'),('float32','f'),('float16','e')]:
            cast=np.array([struct.unpack('<'+fmt,struct.pack('<'+fmt,float(v)))[0] for v in law.flat]).reshape(16,4,8)
            for mode in ('direct','row-normalized'):
                with np.load(root/'raw'/f'{lineage}-{dtype}-{mode}_points.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
                assert np.array_equal(raw['original'],law) and raw['stored'].dtype==np.dtype(dtype)
                assert np.array_equal(raw['stored'].astype(float),cast)
                mass=cast.sum(-1);expected=cast if mode=='direct' else cast/mass[...,None]
                assert np.array_equal(raw['reconstructed'],expected)
                assert np.array_equal(raw['delta'],expected-law)
                assert np.array_equal(raw['lost_support'],(law>0)&(cast==0))
                assert np.array_equal(raw['cast_row_mass_drift'],mass-1)
                assert np.array_equal(raw['row_mass_drift'],expected.sum(-1)-1)
                for ctx in range(4):
                    d=[[float(expected[s,ctx,e])-float(law[s,ctx,e]) for e in range(8)] for s in range(16)]
                    squared=[math.fsum(x*x for x in dd) for dd in d]
                    loss=[]
                    for s in range(16):
                        normalization_discrepancy=max(normalization_discrepancy,abs(float(mass[s,ctx])-math.fsum(float(v) for v in cast[s,ctx])))
                        assert math.isclose(raw['row_squared'][s,ctx],squared[s],rel_tol=2e-15,abs_tol=1e-45)
                        lost=math.fsum(float(law[s,ctx,e]) for e in range(8) if cast[s,ctx,e]==0 and law[s,ctx,e]>0)
                        assert math.isclose(raw['lost_support_mass'][s,ctx],lost,rel_tol=2e-15,abs_tol=1e-45)
                        with localcontext() as dec:
                            dec.prec=65;source=[Decimal(float(v)) for v in law[s,ctx]];estimate=[Decimal(float(v)) for v in expected[s,ctx]]
                            value=sum(source[e]*sum((estimate[j]-int(e==j))**2-(source[j]-int(e==j))**2 for j in range(8)) for e in range(8))
                        loss.append(float(value));diff=abs(raw['expected_one_hot_loss_change'][s,ctx]-loss[-1]);max_loss_difference=max(max_loss_difference,diff)
                        assert diff<5e-19,(dtype,mode,diff)
                    assert math.isclose(raw['squared_bound'][ctx],max(squared),rel_tol=2e-15,abs_tol=1e-45)
                    assert raw['squared_attainers'][:,ctx].tolist()==[v==max(raw['row_squared'][:,ctx]) for v in raw['row_squared'][:,ctx]]
                    for e in range(8):
                        vals=[dd[e] for dd in d];lo,hi=min(vals),max(vals)
                        assert raw['lower'][ctx,e]==lo and raw['upper'][ctx,e]==hi
                        assert raw['lower_attainers'][:,ctx,e].tolist()==[x==lo for x in vals]
                        assert raw['upper_attainers'][:,ctx,e].tolist()==[x==hi for x in vals]
                    source_row=row_lookup[(lineage,dtype,mode,ctx)]
                    rebuilt=dict(maximum_absolute_coordinate_error=max(abs(v) for dd in d for v in dd),squared_forecast_bound=max(squared),maximum_row_mass_drift=max(abs(v) for v in raw['row_mass_drift'][:,ctx]),cast_maximum_row_mass_drift=max(abs(v) for v in raw['cast_row_mass_drift'][:,ctx]),lost_support_count=int(((law[:,ctx]>0)&(cast[:,ctx]==0)).sum()),maximum_lost_support_mass=max(raw['lost_support_mass'][:,ctx]),maximum_vertex_expected_loss_change=max(loss),stored_law_bytes=16*4*8*struct.calcsize(fmt))
                    for k,v in rebuilt.items():
                        tol=5e-19 if k=='maximum_vertex_expected_loss_change' else 1e-45
                        assert math.isclose(source_row[k],v,rel_tol=2e-15,abs_tol=tol),(k,source_row[k],v)
                    scalar_rows.append(dict(source_row));count+=1
    groups=defaultdict(list)
    for r in rows:groups[(r['storage_dtype'],r['reconstruction'],r['context'])].append(r)
    cells=[]
    for key,rr in sorted(groups.items()):
        assert len(rr)==8 and sorted(r['lineage'] for r in rr)==list(range(190000,190008))
        cells.append(dict(storage_dtype=key[0],reconstruction=key[1],context=key[2],laws=8,**{k:math.fsum(r[k] for r in rr)/8 for k in metrics}))
    variants=[]
    for dtype in ('float64','float32','float16'):
        for mode in ('direct','row-normalized'):
            rr=[r for r in rows if r['storage_dtype']==dtype and r['reconstruction']==mode]
            variants.append(dict(storage_dtype=dtype,reconstruction=mode,contexts=4,laws=8,**{k:dict(mean=math.fsum(r[k] for r in rr)/32,minimum=min(r[k] for r in rr),maximum=max(r[k] for r in rr)) for k in metrics}))
    result=dict(at=now(),passed=True,numerical_acceptance=True,classification='finite supplied-law algebraic ruler;constructed method;no population or historical-correspondence claim',plan_sha256=sha(root/'PLAN.json'),source_archive_sha256=sha(root/'SOURCE.zip'),environment=plan['environment'],environment_verified=True,source_files=len(plan['sources']),replays=replays,raw_arrays=48,original_strata=count,equal_law_cells=cells,variant_summaries=variants,maximum_independent_loss_difference=max_loss_difference,maximum_scalar_row_sum_difference=normalization_discrepancy,checks='struct scalar casts;float64 normalization;independent scalar extrema and every attaining mask;support masks;65digit direct expected one-hot losses;original-row equal-law regroup',limitations='Bounds range over all normalized nonnegative maker-state mixtures;reachability and realized-history error are not measured. Both replays share original mathematics;the independent scalar audit separates arithmetic implementation but not a different world. No learned law,process correspondence or human intent.',reader_inputs='none')
    write(out/'NUMERICAL_REVIEW.json',result)
    write(out/'INDEPENDENT_REGROUP.json',dict(passed=True,raw_sha256=sha(root/'raw/law_rounding_summary_points.json.gz'),rows=rows,cells=cells,equal_law_weights=True,independent_laws=8,new_population_inference=False))
    for n in ('COMPLETE.json','SUMMARY.json','CONTROLS.json','EVIDENCE_ROLES.json'):shutil.copyfile(root/n,out/n)
    for i,suffix in enumerate(('-replay','-portable'),1):shutil.copyfile(c/(name+suffix)/'COMPLETE.json',out/('ADJACENT_REPLAY.json' if i==1 else 'PORTABLE_REPLAY.json'))
    with zipfile.ZipFile(out/'SCIENTIFIC.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in original['files']:z.write(root/n,n)
        z.write(root/'COMPLETE.json','COMPLETE.json')
    shutil.copyfile(Path(__file__),out/'independent_review.py')
    write(c/'ROUNDING_REVIEW_6356.json',dict(at=now(),passed=True,numerical_review_sha256=sha(out/'NUMERICAL_REVIEW.json'),original_strata=count,equal_law_cells=len(cells),card_cpu_before=R.card_consumed(c,plan['design'],'')))
    print(json.dumps(dict(passed=True,rows=count,cells=len(cells),max_loss_difference=max_loss_difference,variants=variants)));state='complete'
finally:write(c/'attempts/rounding-independent-review-6356.json',dict(packet='rounding-independent-review-6356',accounting_card='G19-B-law-rounding-envelope',state=state,cpu_seconds=time.process_time()-started))
