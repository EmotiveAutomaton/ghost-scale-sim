"""Freeze the October29 generation-one follow-ups, with measured probe gates."""
import argparse,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v25 import contract,followups
from ghostscale.validation.soundingline.v25.reporting import verified_primary
from runners.run_v25 import frozen_source,verify_source,charges

REPO=Path(__file__).resolve().parents[1]
CARDS={
 'W4F1':dict(family='W4',identity='v25-w4-followup-1',blocks=32,settings=[dict(followup='population-size',population_size=n,alpha=a,placebo=False) for n in (8,32,64,256) for a in (3,30)]+[dict(followup='population-size',population_size=64,alpha=10,placebo=True)],probe_indices=[0,7],reason='Separate finite population-prior estimation from oracle shrinkage at matched specificity.'),
 'W6F2':dict(family='W6',identity='v25-w6-followup-2',blocks=64,settings=[dict(followup='signal-dose',overlap='heavy',block_probability=.1,kappa=k,generator='value',temperament=0.) for k in (.5,1.,3.,6.)],probe_indices=[0,3],reason='Fresh larger replication and signal-dose map around the barely-above-margin parent primary.'),
 'W7F1':dict(family='W7',identity='v25-w7-followup-1',blocks=64,settings=[dict(followup='attention-noise',habit=h,epsilon=e) for h in (0,1,2) for e in (0.,.1,.3,.6)],probe_indices=[0,11],reason='Locate finite-history information from attention as unrelated attention grows.'),
 'W12F1':dict(family='W12',identity='v25-w12-followup-1',blocks=64,settings=[dict(followup='affect-channels',habit_affect=h,end_distress=e,noise_sd=s) for h in (False,True) for e in (False,True) for s in (.25,.5,1.,2.)],probe_indices=[0,15],reason='Paired channel ablations identify which planted affect signal breaks the end/habit alias.')}


def prepare_probe(campaign,card):
    started=time.process_time();campaign=Path(campaign).resolve();design=CARDS[card];family=design['family'];parent=verified_primary(campaign,family)
    source=frozen_source(campaign);controls=followups.controls()
    if not controls.get('passed'):raise ValueError('known-answer failure')
    protocol=dict(card=card,**design,generation=1,parent_packet=parent.name,parent_complete_sha256=digest(parent/'COMPLETE.json'),
        declared_at=datetime.now(timezone.utc).isoformat(),protocol_document_sha256=digest(REPO/'docs/versions/v25-pieces-and-ends/TEXTURE_PROTOCOL_2026-10-29.md'),
        world_identity='SHA256 v25, disjoint packet science identity, family, setting index, world index',
        inference='Exploratory/descriptive constructed method; no confirmation. Whole worlds, not makers or prefixes, are independent units.',
        controls=controls,provisional=True,maximum_generation=1)
    identity=design['identity']+'-probe-1';packet=campaign/'packets'/identity
    if packet.exists():raise ValueError('preserve prepared or failed probe')
    plan=dict(family=family,kind='followup',scientific_kind='probe',pool='followup_healing',generation=1,identity=identity,science_identity=identity,
        texture_card=card,texture_phase='cost_probe',settings=[design['settings'][i] for i in design['probe_indices']],blocks=1,shed_level=0,
        source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),source_archive_sha256=digest(source/'SOURCE.zip'),
        acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),environment_sha256=digest(campaign/'ENVIRONMENT.json'),
        numerical_threads=1,priority='below-normal',cpu_cap_seconds=600.,pass_caps=dict(original=200.,replay=200.,review=200.),
        wall_cap_seconds=2*3600,interpreter=str(Path(sys.executable).resolve()),parent=parent.name,parent_complete_sha256=digest(parent/'COMPLETE.json'),
        production=False,confirmation=False)
    contract.admission(plan,read(campaign/'ACCEPTANCE.json'),datetime.now(timezone.utc),charges(campaign))
    protocol_path=campaign/'protocols'/(card+'_TEXTURE.json');write(protocol_path,protocol)
    packet.mkdir(parents=True);write(packet/'CONTROLS_BEFORE_DATA.json',controls)
    plan.update(protocol_file=protocol_path.name,protocol_sha256=digest(protocol_path),controls_sha256=digest(packet/'CONTROLS_BEFORE_DATA.json'))
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=datetime.now(timezone.utc).isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


def prepare_study(campaign,parent,source=None):
    started=time.process_time();campaign=Path(campaign);parent=Path(parent);prior=read(parent/'PLAN.json');card=prior['texture_card'];protocol_path=campaign/'protocols'/prior['protocol_file']
    if prior['texture_phase']!='cost_probe' or digest(protocol_path)!=prior['protocol_sha256']:raise ValueError('frozen texture protocol mismatch')
    protocol=read(protocol_path);complete=read(parent/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('summary_sha256','original/SUMMARY.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if complete[key]!=digest(parent/name):raise ValueError('cost probe binding changed')
    verify_source(prior)
    if source is not None and Path(source).resolve()!=Path(prior['source_root']).resolve():raise ValueError('source changed')
    measurement=read(campaign/'accounting'/(parent.name+'.json'));timing=read(parent/'PASS_COSTS.json')
    if measurement['state']!='COMPLETE' or not timing.get('includes_world_io'):raise ValueError('complete original/replay/review cost required')
    count=len(protocol['settings'])*protocol['blocks'];caps={}
    for label,used in measurement['pass_cpu_seconds'].items():
        rows=timing['setting_seconds'][label];unit=max(max(rows),1/64);overhead=max(0,used-sum(rows))
        caps[label]=1.5*(overhead+unit*count)+10
    identity=protocol['identity'];packet=campaign/'packets'/identity
    plan=dict(prior,identity=identity,science_identity=identity,scientific_kind='followup',texture_phase='study',settings=protocol['settings'],blocks=protocol['blocks'],
        pass_caps=caps,cpu_cap_seconds=sum(caps.values()),wall_cap_seconds=8*3600,parent=parent.name,parent_complete_sha256=digest(parent/'COMPLETE.json'))
    contract.admission(plan,read(campaign/'ACCEPTANCE.json'),datetime.now(timezone.utc),charges(campaign))
    packet.mkdir();shutil.copyfile(parent/'CONTROLS_BEFORE_DATA.json',packet/'CONTROLS_BEFORE_DATA.json')
    write(packet/'PLAN.json',plan);write(packet/'STATUS.json',dict(state='PREPARED',at=datetime.now(timezone.utc).isoformat()))
    write(campaign/'accounting'/(identity+'-preparation.json'),dict(pool='preparation',kind='preparation',cpu_seconds=time.process_time()-started,state='COMPLETE'))
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--card',choices=CARDS,required=True)
    args=parser.parse_args();print(prepare_probe(args.campaign,args.card))
