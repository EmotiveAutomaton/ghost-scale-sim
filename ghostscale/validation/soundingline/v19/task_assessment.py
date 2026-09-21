"""E2 exact objective/skill assessment on retained full transient trajectories."""
from collections import defaultdict
import gzip, json, math
import numpy as np
from ..v18_3.io import canonical, write, file_digest, digest


def project(r, context, degraded):
    packet={'artifact': [r['final'][0], None if degraded else r['final'][1], r['final'][2]],
            'hidden_units': [1] if degraded else []}
    if context: packet.update(initial=r['initial'], requested_purpose=r['requested_purpose'])
    return packet


def aggregate(records, context, degraded):
    groups={}
    for r in records:
        packet=project(r,context,degraded);key=canonical(packet).decode()
        if key not in groups:groups[key]=dict(packet=packet,mass=np.zeros(2),counts=np.zeros(2))
        g=groups[key];purpose=r['maker'][0]
        g['mass'][purpose]+=r['probability'];g['counts'][purpose]+=1
    scores=[];evidence=[]
    for key,g in sorted(groups.items()):
        mass=g['mass'];total=float(mass.sum());exact=mass/total;template=g['counts']/g['counts'].sum()
        for arm,p in [('exact',exact),('uniform-template',template)]:
            supported=mass>0
            scores.append(dict(arm=arm,mass=total,loss=float(-np.sum(mass[supported]*np.log(p[supported]))),
                brier=float(sum(mass[i]*np.sum((np.eye(2)[i]-p)**2) for i in range(2))),
                coverage=float(mass[p>0].sum()),abstention=total*int(np.count_nonzero(p)>1)))
        evidence.append(dict(packet=g['packet'],packet_sha256=digest(g['packet']),mass=total,
            purpose_mass=mass.tolist(),compatible_trajectory_counts=g['counts'].astype(int).tolist(),
            exact=exact.tolist(),uniform_template=template.tolist()))
    cells=[dict(arm=arm,**{k:math.fsum(r[k] for r in scores if r['arm']==arm) for k in ('mass','loss','brier','coverage','abstention')}) for arm in ('exact','uniform-template')]
    return cells,evidence


def controls():
    def r(p,a):return dict(maker=[p,0,0,0],probability=.5,final=a,initial=[0,0,0],requested_purpose=0)
    noinfo=[r(0,[0,0,0]),r(1,[0,0,0])];identified=[r(0,[0,0,0]),r(1,[0,1,0])]
    a,_=aggregate(noinfo,False,False);b,_=aggregate(identified,False,False);d,_=aggregate(identified,False,True)
    return dict(placebo_no_information=abs(a[0]['loss']-math.log(2))<1e-12,
        live_identifiable_purpose=abs(b[0]['loss'])<1e-12,
        positive_degradation=abs(d[0]['loss']-math.log(2))<1e-12,
        projection_coarsening=project(identified[0],False,True)==project(identified[1],False,True),
        requested_not_adopted=project(noinfo[1],True,False)['requested_purpose']!=noinfo[1]['maker'][0])


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();assert all(checks.values())
    raw=[];outcomes=[];packets={}
    for lineage in cfg['lineages']:
        pulse(phase='task-assessment',lineage=lineage)
        name=f'lineage-{lineage}_points.json.gz';path=root/'inputs'/name
        assert file_digest(path)==cfg['input_files'][name]
        records=json.loads(gzip.decompress(path.read_bytes()))
        assert len(records)==13824 and abs(math.fsum(r['probability'] for r in records)-1)<1e-10
        for purpose in range(2):
            for skill in range(2):
                rr=[r for r in records if r['maker'][:2]==[purpose,skill]]
                mass=math.fsum(r['probability'] for r in rr);assert abs(mass-.25)<1e-10
                values=defaultdict(float)
                for r in rr:
                    a=r['final'];functional=bool(a[0]==1 and a[1]==a[0]);presentation=bool(a[2]==1)
                    for key,v in [('functional',functional),('presentation',presentation),('both',functional and presentation),('disagreement',functional!=presentation)]:values[key]+=r['probability']*v/mass
                outcomes.append(dict(lineage=lineage,purpose=purpose,skill=skill,mass=mass,trajectories=len(rr),**values))
        for context in (False,True):
            values={}
            for degraded in (False,True):
                cells,evidence=aggregate(records,context,degraded);values[degraded]=cells[0]['loss']
                raw.extend(dict(lineage=lineage,context=context,degraded=degraded,**x) for x in cells)
                for e in evidence:packets[e['packet_sha256']]=dict(inputs=e['packet'],input_sha256=e['packet_sha256'],passage_anchors=['unit-0','unit-1','unit-2'])
                write(root/'evaluator'/f'{lineage}-{int(context)}-{int(degraded)}.json',dict(groups=evidence))
            if values[True]+1e-12<values[False]:raise ValueError('exact coarsening violated')
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/task_points.json.gz').write_bytes(gzip.compress(canonical(dict(inference=raw,outcomes=outcomes)),mtime=0))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.task.reader.1',cases=[packets[k] for k in sorted(packets)],
        semantics='three units: claim, evidence, display; a request is observed context, adoption unknown',
        hidden_fields='No maker, lineage, skill, actual purpose or inferred posterior supplied.'))
    write(root/'INPUT_SCHEMA.json',dict(reader='PUBLIC_PACKET.json only',evaluator='inputs, raw and evaluator groups; scientific scores are not reader input'))
    cells=[]
    for context in (False,True):
        for degraded in (False,True):
            for arm in ('exact','uniform-template'):
                rr=[r for r in raw if (r['context'],r['degraded'],r['arm'])==(context,degraded,arm)]
                cells.append(dict(context=context,degraded=degraded,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ('loss','brier','coverage','abstention')}))
    return dict(controls=checks,lineages=len(cfg['lineages']),trajectories=13824*len(cfg['lineages']),cells=cells,outcomes=outcomes,
        scope='complete probability-weighted eight-world diagnostic; oracle-family and uniform legal-template readers; no learned production competence',
        warrant='exploratory constructed method/mechanism; miniature — architecture untested',tiny_settings=0)
