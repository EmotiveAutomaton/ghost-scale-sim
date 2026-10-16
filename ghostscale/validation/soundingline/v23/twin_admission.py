"""Authenticate measured shape, implementation controls, fits and full costs."""
from pathlib import Path
from . import twin_pipeline,registry
from .common import digest,read
from .family import verify_packet


MODULES=('twin_pipeline','twin_graph','twin_ledger','twin_engineering','twin_writing','twin_chess','twin_wiki','twin_panel',
    'twin_contract','fit_rivals','ledger','common','registry','engine_bridge','twin_partition')


def estimate(packet,family,partitioned=False):
    rows=[read(Path(packet)/label/'pilot-cost.json') for label in ('original','replay')]
    ids={v['id'] for v in registry.variants(family)}
    if any({row['variant'] for row in part['profile']}!=ids for part in rows):raise ValueError('full twin grid cost required')
    batch=max(300,3*16*max(sum(row['cpu_seconds'] for row in part['profile']) for part in rows))
    labels=4096+2048+8192+2*8192*120;fit_labels=(2*2048+4*8192)*120
    fit=max(300,3*max(sum(row['training_generation_cpu_seconds']/row['training_labels']*labels+
        row['fit_cpu_seconds']/row['fit_epochs']/row['training_labels']*fit_labels for row in part['training']) for part in rows))
    result=dict(fits=fit,batch=batch)
    if partitioned and batch>14400:
        from .partition import layout
        split=layout(16,batch,min_worlds=1)
        result.update(batch=split['total_cpu_cap'],partition_layout=split)
    return result


def validate(task,family,extras,repo):
    if family not in ('T1','T2','T3','T4'):raise ValueError('unknown twin family')
    twin_pipeline.reference(dict(extras,family=family))
    receipt=Path(extras['control_receipt']);control=read(receipt)
    if digest(receipt)!=extras['control_receipt_sha256'] or control['exit_code']!=0:raise ValueError('twin controls unresolved')
    for name in MODULES:
        relative='ghostscale/validation/soundingline/v23/'+name+'.py'
        if digest(Path(repo)/relative)!=control['implementation_source_hashes'].get(relative):raise ValueError('twin controls source changed')
    if task=='twin-profile':return 0.,0.
    pilot=Path(extras['pilot_packet']);pp,ps=verify_packet(pilot)
    if (pp['task']!='twin-profile' or pp['family']!=family or ps['state']!='PILOT_VALID' or
        digest(pilot/'COMPLETE.json')!=extras['pilot_complete_sha256']):raise ValueError('twin profile unresolved')
    if pp['empirical_reference_sha256']!=extras['empirical_reference_sha256']:raise ValueError('twin empirical reference changed')
    costs=estimate(pilot,family,partitioned=extras.get('partitioned_twin',False))
    if task=='twin-fits':
        if extras.get('burnin_reservations_seconds')!=[costs['batch']]:raise ValueError('reserve complete fixed twin grid')
        return costs['fits'],costs['batch']
    if task!='twin-batch' or extras.get('worlds')!=16 or extras.get('batch')!=1:raise ValueError('twin fixed grid allocation changed')
    fit=Path(extras['fit_packet']);fp,fs=verify_packet(fit)
    if fp['task']!='twin-fits' or fp['family']!=family or fs['state']!='FITS_VERIFIED' or digest(fit/'COMPLETE.json')!=extras['fit_complete_sha256']:
        raise ValueError('twin fitting unresolved')
    folder=fit/'original/fits'
    if Path(extras['fit_directory']).resolve()!=folder.resolve() or extras['fit_hashes']!={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}:
        raise ValueError('twin fitted artifacts changed')
    return costs['batch'],0.
