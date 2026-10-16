"""Bounded execution partitions of one unchanged complete scientific batch.

Partition boundaries never alter a random identity, fitting set, sample count,
selection look or law balance. Only the fully assembled batch can be reviewed.
"""
import gzip,math,shutil
from pathlib import Path
from .common import read,write,digest,interval


def encode_nonfinite(value):
    """Preserve diagnostic infinities without invalid JSON or score flooring."""
    if isinstance(value,dict):return {k:encode_nonfinite(v) for k,v in value.items()}
    if isinstance(value,list):return [encode_nonfinite(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else 'positive_infinity' if value>0 else 'negative_infinity'}
    return value


def decode_nonfinite(value):
    if isinstance(value,dict):
        if set(value)=={'nonfinite'}:
            return {'nan':math.nan,'positive_infinity':math.inf,'negative_infinity':-math.inf}[value['nonfinite']]
        return {k:decode_nonfinite(v) for k,v in value.items()}
    if isinstance(value,list):return [decode_nonfinite(v) for v in value]
    return value


def world_indices(plan):
    worlds=plan.get('world_indices',list(range(plan['worlds'])))
    if (not worlds or worlds!=sorted(set(worlds)) or
        any(type(w) is not int or not 0<=w<plan['worlds'] for w in worlds)):
        raise ValueError('invalid world partition')
    if plan.get('world_indices') is not None and not plan.get('fit_directory'):
        raise ValueError('partitions require preverified frozen fits')
    return worlds


def layout(worlds,estimated_total,auxiliary_per_part=0.,assembly_cap=300.):
    if worlds%4:raise ValueError('partition admission requires complete four-world blocks')
    chunks=[list(range(w,w+4)) for w in range(0,worlds,4)]
    # The baseline already contains one auxiliary pass pair. Repeated training
    # reconstruction is charged for every additional part; no hidden free work.
    cap=math.ceil((estimated_total-auxiliary_per_part)/len(chunks)+auxiliary_per_part+60)
    if not 0<cap<=14400:raise ValueError('partition still exceeds individual packet limit')
    return dict(chunks=chunks,part_cpu_cap=cap,assembly_cpu_cap=assembly_cap,
                total_cpu_cap=cap*len(chunks)+assembly_cap)


def part_paths(root, plan):
    root=Path(root)
    retained=plan.get('retained_parts',{})
    return [Path(retained[str(i)]['path']) if str(i) in retained else
            root.with_name(f'{root.name}-part-{i+1}')
            for i in range(len(plan['partition_layout']['chunks']))]


def verify_part_binding(root, plan, part, child, index):
    retained=plan.get('retained_parts',{}).get(str(index))
    parent=Path(root)
    if retained:
        parent=Path(plan['repair_parent'])
        if (digest(parent/'PLAN.json')!=plan['repair_parent_plan_sha256'] or
                digest(parent/'STATUS.json')!=plan['repair_parent_status_sha256'] or
                read(parent/'STATUS.json')['state']!='failed' or
                digest(Path(part)/'COMPLETE.json')!=retained['complete_sha256'] or
                Path(retained['path']).resolve()!=Path(part).resolve()):
            raise ValueError('retained partition provenance changed')
        original=read(parent/'PLAN.json')
        if any(original[k]!=plan[k] for k in ('family','batch','worlds','cases','variants','fit_hashes','canary_key','sources')):
            raise ValueError('partition recovery changed scientific definition')
    if (child.get('logical_parent')!=str(parent.resolve()) or
            child.get('logical_plan_sha256')!=digest(parent/'PLAN.json') or
            child.get('world_indices')!=plan['partition_layout']['chunks'][index]):
        raise ValueError('partition provenance does not match this frozen logical batch')


def merge_inputs(parts,total_worlds):
    if [w for part in parts for w in part['worlds']]!=list(range(total_worlds)):
        raise ValueError('partitions overlap, are reordered or omit worlds')
    out={'worlds':list(range(total_worlds))}
    for key in ('gains','target_counts'):
        if key in parts[0]:out[key]=[x for p in parts for x in p[key]]
    for key in ('reader_worlds','headroom','secondary_worlds','strata'):
        names=set(parts[0][key])
        if any(set(p[key])!=names for p in parts):raise ValueError('partition metric membership changed')
        out[key]={name:[x for p in parts for x in p[key][name]] for name in sorted(names)}
    if 'unavailable' in parts[0]:
        out['unavailable']={name:sum(p['unavailable'][name] for p in parts) for name in parts[0]['unavailable']}
    for key in ('max_check','max_roundtrip'):out[key]=max(p[key] for p in parts)
    return out


def summarize(base,data,plan,fitmeta):
    from .pipeline import headroom_groups
    from .pipeline_new import aggregate
    v=base['variant'];identity=v['id'];batch=plan['batch'];new=plan['family'].startswith('N')
    agg=(lambda values,identity:aggregate(values,identity,stable=bool(plan.get('numerical_validation_packet')))) if new else interval
    result=dict(base,primary=agg(data['gains'],[identity,batch,'primary']),world_gains=data['gains'],
        readers={name:{metric:agg([r[metric] for r in rows],[identity,batch,name,metric]) for metric in rows[0]}
                 for name,rows in data['reader_worlds'].items()},
        secondary={name:agg(values,[identity,batch,'secondary',name]) for name,values in data['secondary_worlds'].items()},
        strata={name:agg(values,[identity,batch,'stratum',name] if new else [identity,batch,name]) for name,values in data['strata'].items()},
        independent_worlds=plan['worlds'],laws=min(16,plan['worlds']),
        law_counts={str(i):sum(w%16==i for w in data['worlds']) for i in range(16)},
        validity=dict(independent_posterior_max_error=data['max_check'],ledger_forecast_max_error=data['max_roundtrip']))
    if new:
        result.update(primary_counts=data['target_counts'],unavailable_predictions=data['unavailable'],
            primary_status='AVAILABLE' if result['primary']['finite'] else 'UNAVAILABLE_NO_CASE_DELETION')
    else:
        result['headroom']={name:dict(interval(values,[identity,batch,'headroom',name]),
            validation_stopped=fitmeta[name]['early_stopped']) for name,values in data['headroom'].items()}
    result['matched_headroom_groups']=(headroom_groups(data['headroom'],fitmeta,v,batch)
        if plan['family']!='N5' and all(all(x is not None for x in vals) for vals in data['headroom'].values()) else {})
    return result


def assemble(root,parts,pulse=lambda:None):
    """Verify all parts, assemble both passes, then require full raw reconstruction."""
    from . import family,runtime
    root=Path(root);plan=read(root/'PLAN.json');parts=[Path(p) for p in parts]
    if not plan.get('scheduler_only') or plan.get('world_indices') is not None:raise ValueError('not a logical batch')
    if (root/'COMPLETE.json').exists():raise ValueError('logical batch already accepted')
    verified=[]
    if parts!=part_paths(root,plan):raise ValueError('partition paths differ from frozen layout')
    for index,part in enumerate(parts):
        pp,summary=family.verify_packet(part)
        verify_part_binding(root,plan,part,pp,index)
        if any(pp[k]!=plan[k] for k in ('family','batch','worlds','cases','variants','fit_hashes','canary_key','sources')):
            raise ValueError('partition does not belong to this frozen logical batch')
        verified.append((pp,summary));pulse()
    if [w for pp,_ in verified for w in world_indices(pp)]!=list(range(plan['worlds'])):
        raise ValueError('incomplete or overlapping partition coverage')
    streams={kind+'_points.jsonl.gz' for kind in ('public','truth','prediction','posterior','ledger')}
    for label in ('original','replay'):
        destination=root/label;destination.mkdir()
        fitmeta=verified[0][1]['fits'];summaries=[];index={}
        write(destination/'DESIGN.json',{k:plan[k] for k in ('family','batch','variants','worlds','cases','schema_pin')})
        for v_index,identity in enumerate(plan['variants']):
            folders=[p/label/identity for p in parts];folder=destination/identity;folder.mkdir()
            inputs=merge_inputs([decode_nonfinite(read(p/'WORLD_AGGREGATES.json')) for p in folders],plan['worlds'])
            for name in streams:
                # Concatenated gzip members retain every original scientific byte.
                with (folder/name).open('xb') as target:
                    for p in folders:
                        with (p/name).open('rb') as source:shutil.copyfileobj(source,target)
            auxiliary={p.name for p in folders[0].iterdir()}-streams-{'SUMMARY.json','WORLD_AGGREGATES.json'}
            for p in folders:
                if {q.name for q in p.iterdir()}-streams-{'SUMMARY.json','WORLD_AGGREGATES.json'}!=auxiliary:
                    raise ValueError('partition auxiliary membership differs')
            for name in auxiliary:
                expected=digest(folders[0]/name)
                if any(digest(p/name)!=expected for p in folders[1:]):raise ValueError('partition fitted evidence differs')
                shutil.copyfile(folders[0]/name,folder/name)
            base=read(folders[0]/'SUMMARY.json')
            summary=summarize(base,inputs,plan,fitmeta)
            write(folder/'SUMMARY.json',summary);summaries.append(summary)
            index[identity]={p.name:digest(p) for p in folder.iterdir() if p.is_file()};pulse()
        write(destination/'RAW_INDEX.json',index)
        base=verified[0][1];write(destination/'SUMMARY.json',dict(base,variants=summaries))
        if plan.get('task')=='confirmation':
            from . import confirmation
            confirmation.verify_definition(plan)
            write(destination/'SEED_AUDIT.json',confirmation.seed_audit(plan['variants'][0],plan['nomination_order_sha256'],plan['worlds'],plan['cases']))
            write(destination/'EVIDENCE.json',confirmation.reconstruct_evidence(destination,plan))
    replay=runtime.compare(root/'original',root/'replay');write(root/'REPLAY.json',replay)
    # Completion is staged privately so independent reconstruction can use the
    # ordinary verifier. A failure removes no evidence and never publishes COMPLETE.
    completion=dict(state='verified',at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
        replay_sha256=digest(root/'REPLAY.json'),summary_sha256=digest(root/'original/SUMMARY.json'),
        partitions={p.name:digest(p/'COMPLETE.json') for p in parts})
    write(root/'ASSEMBLY_VERIFICATION.json',completion)
    review=family.reconstruct(root,assembly_pending=True);write(root/'NUMERICAL_REVIEW.json',review);pulse()
    return completion
