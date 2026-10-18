"""Measured continuation costs and exact, separately scoped extension contracts."""
from pathlib import Path
from .common import read,digest
from . import population_campaign,conditional_values,registry

CONDITIONAL_MODULES = {
    'N4b': ('conditional_values','fit_rivals','values','checker_recipes','common','registry','ledger'),
    'N4c': ('conditional_devaluation','devaluation','fit_rivals','common','ledger'),
    'N14': ('conditional_covariation','covariation','conditional_small','fit_rivals','common','ledger'),
    'N9': ('conditional_consideration','consideration','conditional_small','fit_rivals','common','ledger'),
    'N13': ('conditional_falsifier','falsifier','falsifier_calibration','finite_common','conditional_small','fit_rivals','common','ledger'),
    'N10': ('conditional_corrections','corrections_soft','finite_common','conditional_small','fit_rivals','common','ledger'),
    'N11': ('conditional_precision','precision_habits','finite_common','conditional_small','fit_rivals','common','ledger'),
    'N12': ('conditional_overlays','goal_overlays','finite_common','conditional_small','fit_rivals','common','ledger'),
    'N4d': ('conditional_sacrifice','sacrifice','finite_common','conditional_small','fit_rivals','common','ledger'),
    'N15': ('conditional_trust','trust_attention','finite_common','conditional_small','fit_rivals','common','ledger'),
}


def conditional_module(family):
    import importlib
    if family not in CONDITIONAL_MODULES:raise ValueError('undeclared conditional family')
    return importlib.import_module('.'+CONDITIONAL_MODULES[family][0],__package__)


def cost_estimate(task,packet):
    packet=Path(packet)
    rows=[read(packet/label/'pilot-cost.json') for label in ('original','replay')]
    if task.startswith('population-'):
        cells={v['id']:population_campaign.base_worlds(v) for v in registry.variants('N1')}
        if any({r['variant'] for r in x['profile']}!=set(cells) for x in rows):
            raise ValueError('complete population cost coverage required')
        batch=3*max(sum(r['cpu_seconds']*cells[r['variant']] for r in x['profile']) for x in rows)
    else:
        module=conditional_module(read(packet/'PLAN.json')['family']) if (packet/'PLAN.json').exists() else conditional_values
        cells={v['id'] for v in module.variants()}
        if any({r['variant'] for r in x['profile']}!=cells for x in rows):
            raise ValueError('complete conditional cost coverage required')
        batch=3*max(sum((r['cpu_seconds']+1/64)/r['cases']*16*64 for r in x['profile']) for x in rows)
    # Maximum registered epochs, both fresh fits, all fixed/validation draws,
    # complete replay and a 50% timing margin. This may conservatively defer work.
    labels=4096+2048+8192+2*8192*120
    fit_labels=(2*2048+4*8192)*120
    fit=max(300,3*max(x['training_generation_cpu_seconds']/x.get('training_generation_labels',x.get('training_labels'))*labels+
        x['fit_cpu_seconds']/x['fit_epochs']/x.get('fit_labels',x.get('training_labels'))*fit_labels for x in rows))
    return dict(fits=fit,batch=max(300,batch),two_burnins=task.startswith('population-'))


def bound_packet(extras,prefix,task,family):
    from .family import verify_packet
    root=Path(extras[prefix+'_packet']);plan,summary=verify_packet(root)
    if (digest(root/'COMPLETE.json')!=extras[prefix+'_complete_sha256'] or
            plan['task']!=task or plan['family']!=family):
        raise ValueError('extension prerequisite binding differs: '+prefix)
    return root,plan,summary


def validate(task,family,extras,repo):
    """Return minimum cap/reservation; no registry row alone admits a worker."""
    if task.startswith('population-'):
        if family!='N1':raise ValueError('population task belongs to N1')
        if task=='population-controls':return 0.,0.
        _,cp,controls=bound_packet(extras,'controls','population-controls','N1')
        if not controls.get('convergence_passed') or not controls.get('shuffle_passed'):
            raise ValueError('population known-answer controls failed')
        modules=('makers','checker_population','population_pipeline','population_campaign','common','registry','ledger')
        for name in modules:
            rel='ghostscale/validation/soundingline/v23/'+name+'.py'
            if digest(Path(repo)/rel)!=cp['sources'].get(rel):raise ValueError('population controls source changed')
        if task=='population-profile':return 0.,0.
        pilot,_,summary=bound_packet(extras,'pilot','population-profile','N1')
        if summary.get('state')!='PILOT_VALID' or not summary.get('complete_grid_profile'):
            raise ValueError('complete population resource profile required')
        costs=cost_estimate(task,pilot)
        if task=='population-fits':
            if extras.get('burnin_reservations_seconds')!=[costs['batch']]*2:raise ValueError('both complete map burn-ins must be reserved')
            return costs['fits'],2*costs['batch']
        fit,_,fs=bound_packet(extras,'fit','population-fits','N1')
        if fs['state']!='FITS_VERIFIED':raise ValueError('population fits unresolved')
        expected=population_campaign.design(extras['batch'],extras['variants'],precision=task=='population-precision')
        if extras['population_design']!=expected:raise ValueError('population allocation differs')
        if task not in ('population-batch','population-precision'):raise ValueError('unknown population task')
    elif task.startswith('conditional-'):
        if family not in CONDITIONAL_MODULES:raise ValueError('undeclared conditional family')
        if task=='conditional-pilot':return 0.,0.
        pilot,pp,summary=bound_packet(extras,'pilot','conditional-pilot',family)
        if summary.get('state')!='PILOT_VALID' or not read(pilot/'original/CONTROLS.json').get('passed'):
            raise ValueError('conditional controls and full grid profile required')
        modules=CONDITIONAL_MODULES[family]
        if family not in ('N4b','N4c'):
            protocol='docs/versions/v23-research-division/'+family.upper()+'_PROTOCOL.md'
            if digest(Path(repo)/protocol)!=pp['sources'].get(protocol):raise ValueError('conditional protocol changed after pilot')
        for name in modules:
            rel='ghostscale/validation/soundingline/v23/'+name+'.py'
            if digest(Path(repo)/rel)!=pp['sources'].get(rel):raise ValueError('conditional pilot source changed')
        costs=cost_estimate(task,pilot)
        if task=='conditional-fits':
            if extras.get('burnin_reservations_seconds')!=[costs['batch']]:raise ValueError('conditional full batch must be reserved')
            return costs['fits'],costs['batch']
        fit,_,fs=bound_packet(extras,'fit','conditional-fits',family)
        if task!='conditional-batch' or fs['state']!='FITS_VERIFIED' or (extras['worlds'],extras['cases'])!=(16,64):
            raise ValueError('conditional full allocation or frozen fits differs')
    else:raise ValueError('unknown extension task')
    if Path(extras['fit_directory']).resolve()!=(fit/'original/fits').resolve():raise ValueError('extension fit location changed')
    expected={p.name:digest(p) for p in (fit/'original/fits').iterdir() if p.is_file()}
    if extras['fit_hashes']!=expected:raise ValueError('extension fit files changed')
    if task.startswith('population-'):
        timing={row['variant']:row['cpu_seconds'] for row in read(pilot/'original/pilot-cost.json')['profile']}
        replay={row['variant']:row['cpu_seconds'] for row in read(pilot/'replay/pilot-cost.json')['profile']}
        cost=max(300,3*sum(max(timing[row['variant']['id']],replay[row['variant']['id']])*row['worlds'] for row in expected_design(extras,task)))
    else:cost=costs['batch']
    return cost,0.


def expected_design(extras,task):
    return population_campaign.design(extras['batch'],extras['variants'],precision=task=='population-precision')
