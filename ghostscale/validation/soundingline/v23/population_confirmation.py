"""Reserved N1 worlds: whole populations and complete 32-culture blocks."""
from pathlib import Path
import math
import numpy as np
from . import population_campaign as population,population_pipeline as pipe,registry,ledger,checker,evidence
from .common import Direct,read,write,digest,interval,seed


def audit(variant,order_hash,worlds):
    if worlds%32:raise ValueError('N1 confirmation requires all 32 cultures per block')
    reserved={seed(variant,['confirmation',order_hash],w,'population') for w in range(worlds)}
    if len(reserved)!=worlds:raise ValueError('population confirmation collision')
    for phase in ('discovery','precision'):
        for batch in range(1,8):
            prior={seed(variant,[phase,batch],w,'population') for w in range(144)}
            if reserved&prior:raise ValueError('population reserved namespace overlaps')
    return dict(state='passed',reserved_population_identities=worlds,laws_per_block=32,independent_blocks=worlds//32,
        cases='all assigned makers within each population; never independent replications')


def reconstruct(root,plan):
    root=Path(root);identity=plan['variants'][0];values=[];raw=[]
    for world in range(plan['worlds']):
        folder=root/identity/f'world-{world:04d}'
        for name,expected in read(folder/'RAW_INDEX.json').items():
            if digest(folder/name)!=expected:raise ValueError('population confirmation raw binding changed')
        record=read(folder/'IDENTITY.json')
        if record['identity']!=['N1',identity,['confirmation',plan['nomination_order_sha256']],world]:
            raise ValueError('population reserved identity differs')
        with np.load(folder/'truth_points.npz') as t,np.load(folder/'prediction_points.npz') as p:
            if int(t['law'])!=world%32 or len(t['y'])!=plan['cases']:raise ValueError('whole-population balanced culture coverage differs')
            individual=checker.scores(p['individual-only'],t['y'])['log_loss'];exact=checker.scores(p['law-marginal'],t['y'])['log_loss']
            values.append(math.fsum(map(float,individual-exact))/len(individual))
        raw.append(digest(folder/'RAW_INDEX.json'))
    result=evidence.betting_evalue(values,plan['confirmation_support']['lower_bound'],laws_per_block=32,independent_balanced_blocks=True)
    return dict(result,world_gains=values,raw_population_indices=raw,independent_unit='whole population',
        primary_readers=['individual-only','law-marginal'],makers_per_world=plan['cases'])


def run(root,plan,pulse):
    from .confirmation import verify_definition,run_fits
    entry=verify_definition(plan);root=Path(root)
    if plan.get('world_indices') is not None:raise ValueError('population confirmation uses a bounded complete packet')
    if not plan.get('fit_directory'):raise ValueError('population confirmation requires separately verified variant-only refits')
    root.mkdir(parents=True,exist_ok=False);schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin'])
    fitroot=Path(plan['fit_directory']);meta=read(fitroot/'FITS.json');fitted={name:Direct.load(fitroot/(name+'.npz')) for name in meta}
    v=next(v for v in registry.variants('N1') if v['id']==entry['variant']);folder=root/v['id'];folder.mkdir();worlds=[]
    for world in range(plan['worlds']):
        worlds.append(pipe.retained_world(folder/f'world-{world:04d}',v,['confirmation',plan['nomination_order_sha256']],
            world,schema,plan['canary_key'],fitted,pulse,law=world%32))
    gains=[r['primary_mean'] for r in worlds]
    item=dict(variant=v,worlds=len(worlds),cases_per_world=v['settings']['makers'],primary=interval(gains,[v['id'],'confirmation']),
        world_gains=gains,independent_unit='whole population')
    write(folder/'SUMMARY.json',item)
    write(root/'SUMMARY.json',dict(state='POPULATION_CONFIRMATION_CHECKED',family='N1',batch='confirmation',variants=[item],fits=meta))
    write(root/'SEED_AUDIT.json',audit(v['id'],plan['nomination_order_sha256'],plan['worlds']))
    write(root/'EVIDENCE.json',reconstruct(root,plan))
