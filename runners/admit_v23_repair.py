"""Admit and optionally launch a bounded repair-family pilot or first batch.

All entry conditions are checked from receipts. A work-order row is never a job.
"""
import argparse
from pathlib import Path
import secrets,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import registry,runtime,ledger,support
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners.watch_v23_events import alive,owner
from ghostscale.validation.soundingline.v23 import resource_policy

LIMITS=dict(setup=8,repairs=14,n1=16,n2=7,n3=6,n4=14,n5=3,n6=6,n7=7,n8=2,twins=14)


def calibration_paths(campaign, record):
    """Resolve the original receipt against its stable campaign, never cwd."""
    paths=[]
    for key in ('summary','complete'):
        path=Path(record[key])
        paths.append(path if path.is_absolute() else (Path(campaign)/path).resolve())
    return tuple(paths)


def budget_limits(campaign):
    return resource_policy.group_caps(campaign)


def reservations_released(campaign,family):
    path=Path(campaign)/'reservations'/(family+'-RELEASE.json')
    if not path.exists():return False
    release=read(path);card=Path(campaign)/'packets'/release['packet']/'MACHINE_CARD.json'
    if digest(card)!=release['machine_card_sha256']:raise ValueError('reservation release binding changed')
    if read(card)['state'] in ('BURN_IN','CONTINUE'):raise ValueError('unfinished family reservations cannot be released')
    return True


def burnin_reservation(campaign, plan, batch, amount):
    """Owner-authorized transfer of an unstarted successor after a failed batch.

    No running packet reservation is released. A subsequent batch admission must
    pay its full cap again through the ordinary global/group capacity checks.
    """
    path=Path(campaign)/'reservations'/(plan.get('family','')+'-DEFERRED-BURNIN.json')
    if not path.exists():return amount
    receipt=read(path)
    if receipt['batch']!=batch:return amount
    if digest(Path(campaign)/'CONTINUATION_AUTHORITY_20261016.json')!=receipt['authority_sha256']:
        raise ValueError('reservation transfer authority changed')
    failed=Path(campaign)/'packets'/receipt['failed_packet']
    if (digest(failed/'PLAN.json')!=receipt['failed_plan_sha256'] or
            digest(failed/'STATUS.json')!=receipt['failed_status_sha256'] or
            read(failed/'STATUS.json')['state']!='failed'):
        raise ValueError('reservation transfer requires preserved failed predecessor')
    if read(failed/'PLAN.json')['family']!=plan.get('family') or batch!=2:
        raise ValueError('reservation transfer scope mismatch')
    return 0.


def verify_profile_sources(plan,family,repository):
    modules=dict(R3=['repairs'],N2=['diagnosticity','checker_new'],N3=['corrections','checker_new'],
        N4=['values','makers','checker_recipes','checker_new'],N5=['projection','checker_projection','checker_new'],
        N6=['uptake','payoff','checker_new'],N7=['statements','makers','checker_recipes','checker_new'],N8=['play','checker_new'])
    names=['common','registry','dispatch','ledger','checker']+modules[family]
    if family=='R3':names.remove('dispatch') # repair pipeline calls repairs directly
    for module in names:
        relative='ghostscale/validation/soundingline/v23/'+module+'.py'
        if digest(repository/relative)!=plan['sources'].get(relative):
            frozen=Path(plan['source_archive']).with_suffix('')/relative
            if family=='R3' and module=='repairs' and digest(frozen)==plan['sources'][relative]:
                import ast
                class WarningOnly(ast.NodeTransformer):
                    def visit_With(self,node):
                        self.generic_visit(node)
                        if len(node.items)==1:
                            call=node.items[0].context_expr
                            if (isinstance(call,ast.Call) and not call.args and len(call.keywords)==1 and
                                ast.dump(call.func)==ast.dump(ast.parse('np.errstate',mode='eval').body) and
                                call.keywords[0].arg=='divide' and isinstance(call.keywords[0].value,ast.Constant) and
                                call.keywords[0].value.value=='ignore'):
                                return node.body
                        return node
                before=WarningOnly().visit(ast.parse(frozen.read_text(encoding='utf-8')))
                after=WarningOnly().visit(ast.parse((repository/relative).read_text(encoding='utf-8')))
                if ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False):continue
            if module=='registry' and digest(frozen)==plan['sources'][relative]:
                import importlib.util
                from ghostscale.validation.soundingline.v23.common import canonical
                def load_registry(path):
                    spec=importlib.util.spec_from_file_location('ghostscale.validation.soundingline.v23._pilot_registry',path)
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
                previous=load_registry(frozen);current=load_registry(repository/relative)
                before=previous.variants(family);after=current.variants(family)
                if (canonical(before)==canonical(after) and previous.PRIMARY[family]==current.PRIMARY[family] and
                    previous.ORDER.index(family)==current.ORDER.index(family) and
                    all(canonical(previous.variant_features(v))==canonical(current.variant_features(v)) and
                        previous.selection_eligible(v)==current.selection_eligible(v) for v in after)):
                    continue
            # N4 and N7 have independent functions in one checker file. A change
            # to N4 cannot invalidate an otherwise identical N7 cost profile.
            if (family in ('N4','N7') and module=='checker_recipes') or (family=='N7' and module=='makers'):
                import ast
                def contract(path):
                    tree=ast.parse(path.read_text(encoding='utf-8'))
                    # N7 imports precisely these two independent recipe functions.
                    # Population likelihood caching is outside its call graph.
                    names=(('recipe_library','satisfaction') if module=='makers' else
                           ('dictionary','affordance',family.lower(),'verify_'+family.lower()))
                    selected=[node for node in tree.body if getattr(node,'name',None) in names or
                              (not isinstance(node,(ast.FunctionDef,ast.ClassDef)) and not
                               (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and
                                isinstance(node.value.value,str)))]
                    return [ast.dump(node,include_attributes=False) for node in selected]
                if digest(frozen)==plan['sources'][relative] and contract(frozen)==contract(repository/relative):continue
            raise ValueError('cost/fit evidence source changed: '+relative)


def profile_estimate(costs, family, new_family):
    from ghostscale.validation.soundingline.v23.pipeline_new import variants as new_variants
    variants=new_variants(family) if new_family else registry.variants(family)
    if not all('profile' in row for row in costs):raise ValueError('complete-grid cost profile required')
    if any({p['variant'] for p in row['profile']}!={v['id'] for v in variants} for row in costs):
        raise ValueError('cost profile does not cover complete family grid')
    rates=[]
    for v in variants:
        cells=[next(p for p in row['profile'] if p['variant']==v['id']) for row in costs]
        # Charge one timer quantum to each measured group, then divide by its
        # measured case count. Repair-family historical estimates stay unchanged.
        rates.append(max((p['cpu_seconds']+1/64)/p['cases'] for p in cells) if new_family else
                     max(1/64,*(p['cpu_seconds']/p['cases'] for p in cells)))
    return sum(rates)*16*64*2*1.5+(2*max(row['auxiliary_cpu_seconds'] for row in costs) if new_family else 0.)


def optional_readiness(campaign):
    """Optional work follows explicit, evidence-bound mandatory dispositions."""
    from ghostscale.validation.soundingline.v23 import confirmation,registry
    path=campaign/'MANDATORY_DISPOSITIONS.json'
    if not path.exists():raise ValueError('optional N8 admission waits for mandatory family dispositions')
    rows=read(path)['families']
    if set(rows)!=set(registry.ORDER)-{'N8'}:raise ValueError('all mandatory families need dispositions')
    for name,row in rows.items():
        if row['state'] not in confirmation.DISPOSITIONS or not row.get('reason'):
            raise ValueError('unresolved mandatory family')
        evidence=Path(row['evidence'])
        if digest(evidence)!=row['evidence_sha256']:raise ValueError('mandatory disposition evidence changed')
        if row['state'] in {'NOMINATED','STOPPED_BELOW_MARGIN','INCONCLUSIVE_AT_CAP','PRIMARY_UNAVAILABLE'}:
            review_path=Path(row['review']);review=read(review_path);card=read(evidence)
            if (digest(review_path)!=row['review_sha256'] or card['family']!=name or card['state']!=row['state'] or
                review['family']!=name or review['decision']!=row['state'] or review['state']!='verified_discovery' or
                not review['full_fresh_process_replay'] or review['receipts'][-1]['machine_card_sha256']!=digest(evidence)):
                raise ValueError('mandatory scientific disposition lacks full review')
    for root in (campaign/'packets').iterdir():
        if not (root/'PLAN.json').exists():continue
        plan=read(root/'PLAN.json');status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
        if plan.get('family')!='N8' and status.get('state') not in ('complete','failed'):
            raise ValueError('mandatory admitted work still unfinished')
    return digest(path)


def admit(campaign,family,task,cpu_cap,launch=False,repair=False,cost_recheck=0,partitioned=False,completion_pilot=False):
    with owner(Path(campaign).resolve()/'admission'):
        return _admit(campaign,family,task,cpu_cap,launch,repair,cost_recheck,partitioned,completion_pilot)


def _admit(campaign,family,task,cpu_cap,launch=False,repair=False,cost_recheck=0,partitioned=False,completion_pilot=False):
    campaign=Path(campaign).resolve();acceptance=resource_policy.resolve(campaign)
    policy_binding=resource_policy.binding(campaign)
    from ghostscale.validation.soundingline.v23.pipeline_new import BUILT
    new_family=family in BUILT
    if family not in ('R1','R2','R3','R4') and not (family=='N1' and task=='pilot') and not new_family:
        raise ValueError('only independently implemented repair batches or the N1 resource pilot can be admitted here')
    optional_binding=optional_readiness(campaign) if family=='N8' else None
    budget_group='n8' if family=='N8' else 'setup' if task=='pilot' else family.lower() if new_family else 'repairs'
    if completion_pilot:
        if family!='N4' or task!='pilot' or repair or cost_recheck or not policy_binding:
            raise ValueError('only the authorized N4 completion pilot is admitted')
        failed=campaign/'packets/v23-n4-pilot-1-repair-1'
        if read(failed/'STATUS.json')['state']!='failed':raise ValueError('preserve failed N4 cost profile')
        budget_group='n4'
    limits=budget_limits(campaign)
    if task=='fits' and not new_family and family!='R3':raise ValueError('separate fits require an admitted family pipeline')
    if not 0<cpu_cap or (cpu_cap>14400 and not (partitioned and task=='family-batch')):
        raise ValueError('repair packet cap must be positive and at most four CPU hours')
    if partitioned and not (campaign/'RESOURCE_REALLOCATION.json').exists():raise ValueError('partition amendment required')
    if not acceptance['opened']<=datetime.now(timezone.utc).isoformat()<acceptance['report_start']:
        raise ValueError('outside discovery/setup window')
    calibration=read(campaign/'CALIBRATION_ACCEPTANCE.json')
    summary,completion=calibration_paths(campaign,calibration)
    if digest(summary)!=calibration['summary_sha256'] or read(summary)['state']!='CALIBRATION_PASSED':
        raise ValueError('accepted calibration unavailable')
    if digest(completion)!=calibration['complete_sha256']:raise ValueError('calibration completion changed')
    schema_pin=read(campaign/'SCHEMA_PIN.json')
    if len(schema_pin['commit'])!=40 or schema_pin['committed_files_verified'] is not True:
        raise ValueError('a committed canonical dependency is required')
    ledger.canonical_schema(schema_pin['directory'],schema_pin)
    if runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('environment changed')
    active=[];reserved=0.;group_reserved=0.;group_spent=0.;packet_names=set()
    for root in (campaign/'packets').iterdir():
        if not (root/'PLAN.json').exists():continue
        old=read(root/'PLAN.json');status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {}
        packet_names.add(root.name)
        unfinished=status.get('state') not in ('complete','failed')
        if old.get('logical_parent'):
            if unfinished:active.append(root.name)
            continue
        if unfinished:reserved+=max(0,old['cpu_cap_seconds']-status.get('cpu_seconds',0.)) if old.get('scheduler_only') else old['cpu_cap_seconds']
        released=reservations_released(campaign,old.get('family',''))
        protected_burnin=burnin_reservation(campaign,old,2,old.get('second_burnin_reservation_seconds',0.)) if status.get('state')!='failed' and not released else 0.
        second=campaign/'packets'/('v23-'+old.get('family','').lower()+'-batch-2')
        if protected_burnin and not (second/'PLAN.json').exists():reserved+=protected_burnin
        pending_burnins=0.
        for batch,amount in enumerate(old.get('burnin_reservations_seconds',[]),1):
            target=campaign/'packets'/f"v23-{old['family'].lower()}-batch-{batch}"
            being_admitted=family==old['family'] and task=='family-batch' and batch==1
            if status.get('state')!='failed' and not released and not (target/'PLAN.json').exists() and not being_admitted:
                pending_burnins+=burnin_reservation(campaign,old,batch,amount)
        reserved+=pending_burnins
        if unfinished and (not old.get('scheduler_only') or status.get('uses_scientific_slot')):active.append(root.name)
        if old['budget_group']==budget_group:
            group_reserved+=pending_burnins
            if unfinished:group_reserved+=old['cpu_cap_seconds']
            else:group_spent+=status.get('cpu_seconds',0.)
            if protected_burnin and not (second/'PLAN.json').exists():group_reserved+=protected_burnin
    if len(active)>=acceptance['max_scientific_workers'] and not (partitioned and task=='family-batch'):
        raise ValueError('Gear 2 scientist slots are full')
    total=runtime.cost(campaign)
    if task=='pilot' and budget_group=='setup':
        for path in (campaign/'attempts').glob('*.json'):
            row=read(path)
            if path.stem not in packet_names:
                group_spent+=max(row.get('cpu_seconds',0),row.get('native_cpu_seconds',0),row.get('uncertainty_cpu_seconds',0))
    if total+reserved+cpu_cap>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
        raise ValueError('global verification reserve would be invaded')
    group_limit=limits[budget_group]*3600
    if group_spent+group_reserved+cpu_cap>group_limit:raise ValueError('group cap exhausted')
    identity=f'v23-{family.lower()}-'+({'pilot':'pilot-1','fits':'fits-1','family-batch':'batch-1'}[task])
    if completion_pilot:identity='v23-n4-completion-pilot-1'
    if repair:
        failed=campaign/'packets'/identity
        if read(failed/'STATUS.json')['state']!='failed':raise ValueError('repair requires a retained failed original')
        identity+='-repair-1'
    if cost_recheck:
        if repair or task!='pilot' or cost_recheck not in (1,2):raise ValueError('cost recheck is a separate bounded pilot')
        verify_packet(campaign/'packets'/identity)
        if cost_recheck==2:verify_packet(campaign/'packets'/(identity+'-cost-recheck-1'))
        identity+='-cost-recheck-'+str(int(cost_recheck))
    root=campaign/'packets'/identity
    if root.exists():raise ValueError('preserve existing identity; reviewed repairs need a new identity')
    from ghostscale.validation.soundingline.v23 import support_new
    extras=(dict(canary_key=secrets.token_hex(32)) if family=='N1' else
            dict(confirmation_support=support_new.bound(family) if new_family else support.repair_bound(family)))
    extras.update(policy_binding)
    if completion_pilot:
        extras.update(profile_cases=4,completion_profile_of=str(failed),
            failed_profile_plan_sha256=digest(failed/'PLAN.json'),failed_profile_status_sha256=digest(failed/'STATUS.json'))
    if (campaign/'RESOURCE_REALLOCATION.json').exists():
        extras['resource_amendment_sha256']=digest(campaign/'RESOURCE_REALLOCATION.json')
        extras['resource_amendment_files']={p.name:digest(p) for p in sorted(campaign.glob('RESOURCE_REALLOCATION*.json'))}
    if task=='pilot' and new_family and not completion_pilot:extras['profile_cases']=4 if family=='N4' and repair else 16 if cost_recheck or repair else 1
    if optional_binding:extras['mandatory_dispositions_sha256']=optional_binding
    if family=='N5':
        anchor=Path(__file__).resolve().parents[1]/'results/v23/preparation/N5_SOUNDING_LINE_ANCHOR.json'
        record=read(anchor)
        if not record.get('working_blob_matches_committed_blob') or len(record.get('source_commit',''))!=40:
            raise ValueError('authenticated aggregate anchor required')
        extras.update(empirical_anchor=record,empirical_anchor_sha256=digest(anchor))
    if task in ('family-batch','fits'):
        pilot=campaign/'packets'/f'v23-{family.lower()}-pilot-1'
        repaired=pilot.with_name(pilot.name+'-repair-1')
        if (repaired/'COMPLETE.json').exists():pilot=repaired
        for pass_number in (1,2):
            optimized=campaign/'packets'/f'v23-{family.lower()}-pilot-1-cost-recheck-{pass_number}'
            if (optimized/'COMPLETE.json').exists():pilot=optimized
        completion_pilot_root=campaign/'packets'/f'v23-{family.lower()}-completion-pilot-1'
        if (completion_pilot_root/'COMPLETE.json').exists():pilot=completion_pilot_root
        pp,ps=verify_packet(pilot)
        if ps['state']!='PILOT_VALID':raise ValueError('family pilot not accepted')
        if new_family or family=='R3':verify_profile_sources(pp,family,Path(__file__).resolve().parents[1])
        costs=[read(pilot/label/'pilot-cost.json') for label in ('original','replay')]
        lower_cost=profile_estimate(costs,family,new_family)
        if partitioned:
            from ghostscale.validation.soundingline.v23.partition import layout
            auxiliary=2*max(row['auxiliary_cpu_seconds'] for row in costs) if new_family else 0.
            split=layout(16,lower_cost,auxiliary,min_worlds=1 if policy_binding else 4);lower_cost=split['total_cpu_cap']
            if task=='family-batch':
                extras.update(scheduler_only=True,partition_layout=split)
                if policy_binding:extras['max_parallel_parts']=3
        elif lower_cost>14400:raise ValueError('largest-cell extrapolation exceeds four-hour packet cap')
        if task=='fits':
            extra_reserve=2*lower_cost
            extras.update(burnin_reservations_seconds=[lower_cost,lower_cost],
                pilot_complete_sha256=digest(pilot/'COMPLETE.json'),pilot_estimated_scientific_cpu_seconds=lower_cost)
        else:
            if lower_cost>cpu_cap:raise ValueError('largest-cell extrapolation exceeds the admitted batch cap')
            extra_reserve=0. if new_family or family=='R3' else lower_cost
        if total+reserved+cpu_cap+extra_reserve>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
            raise ValueError('second mandatory burn-in batch cannot be reserved')
        if group_spent+group_reserved+cpu_cap+extra_reserve>group_limit:
            raise ValueError('second burn-in reservation would invade the family group cap')
        if task=='family-batch':
            if new_family or family=='R3':
                fitpacket=campaign/'packets'/f'v23-{family.lower()}-fits-1'
                repaired_fit=fitpacket.with_name(fitpacket.name+'-repair-1')
                if (repaired_fit/'COMPLETE.json').exists():fitpacket=repaired_fit
                fp,fs=verify_packet(fitpacket)
                if fs.get('state')!='FITS_VERIFIED':raise ValueError('verified frozen fits required before discovery')
                verify_profile_sources(fp,family,Path(__file__).resolve().parents[1])
                fitroot=fitpacket/'original/fits'
                extras.update(fit_directory=str(fitroot),fit_hashes={p.name:digest(p) for p in fitroot.iterdir() if p.is_file()},
                    fit_complete_sha256=digest(fitpacket/'COMPLETE.json'))
            from ghostscale.validation.soundingline.v23.pipeline_new import variants as new_variants
            extras.update(batch=1,variants=[v['id'] for v in (new_variants(family) if new_family else registry.variants(family))],worlds=16,cases=64,
                canary_key=secrets.token_hex(32),pilot_complete_sha256=digest(pilot/'COMPLETE.json'),
                pilot_estimated_scientific_cpu_seconds=lower_cost,second_burnin_reservation_seconds=extra_reserve)
    source=campaign/'sources'/identity;source.mkdir(parents=True)
    repository=Path(__file__).resolve().parents[1]
    names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
           'runners/run_v23.py','runners/launch_background.py']
    names += [p.relative_to(repository).as_posix() for p in sorted((repository/'ghostscale/validation/soundingline/v23').glob('*.py'))]
    for name in names:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repository/name,target)
    if digest(source/'ghostscale/validation/soundingline/v23/adaptive.py')!=calibration['adaptive_source_sha256']:
        raise ValueError('family policy differs from calibrated source')
    archive=source.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(source/name,name)
    dependency_files={str(p):digest(p) for p in (campaign/'dependencies').rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
    plan=dict(task=task,family=family,budget_group=budget_group,cpu_cap_seconds=cpu_cap,
        wall_cap_seconds=cpu_cap*4,protected=False,acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
        source_archive=str(archive),source_archive_sha256=digest(archive),sources={name:digest(source/name) for name in names},
        dependency_files=dependency_files,schema_directory=schema_pin['directory'],schema_pin=schema_pin,**extras)
    write(root/'PLAN.json',plan)
    write(root/'ADMISSION.json',dict(at=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(root/'PLAN.json'),
        calibration_complete_sha256=calibration['complete_sha256'],canonical_commit=schema_pin['commit'],
        existing_active=active,charged_cpu_seconds=total,reserved_cpu_seconds=reserved,new_cpu_cap_seconds=cpu_cap))
    if launch and not extras.get('scheduler_only'):
        python=Path(sys.executable)
        if python.name.lower()!='python.exe' or '.venv' not in str(python):raise ValueError('explicit scientific venv interpreter required')
        env=__import__('os').environ.copy();env['PYTHONPATH']=str(campaign/'dependencies')
        argv=[str(python.with_name('pythonw.exe')),str(source/'runners/launch_background.py'),'--cwd',str(source),
              '--log',str(root/'worker.log'),'--',str(python),'-B','-m','runners.run_v23','--root',str(root),'--campaign',str(campaign)]
        child=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        write(root/'LAUNCH.json',dict(at=datetime.now(timezone.utc).isoformat(),launcher_pid=child.pid))
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True,type=Path);p.add_argument('--family',required=True)
    p.add_argument('--task',choices=['pilot','fits','family-batch'],required=True);p.add_argument('--cpu-cap',required=True,type=float)
    p.add_argument('--launch',action='store_true');p.add_argument('--repair',action='store_true')
    p.add_argument('--cost-recheck',type=int,choices=[1,2],default=0)
    a=p.parse_args();print(admit(a.campaign,a.family,a.task,a.cpu_cap,a.launch,a.repair,a.cost_recheck))
