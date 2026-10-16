"""Freeze reviewed, exhaustive discovery dispositions before reserved exposure.

The supplied manifest is an operator review product, not an inferred null for
unbuilt or unaffordable work. This command performs no sampling or publication.
"""
import argparse
from pathlib import Path
import time
from ghostscale.validation.soundingline.v23 import confirmation, registry, runtime
from ghostscale.validation.soundingline.v23.common import read, write, digest
from runners.queue_v23_families import inventory
from runners.watch_v23_events import owner


def build_order(manifest):
    rows = manifest['families']
    if set(rows) != set(registry.ORDER):
        raise ValueError('every family needs a reviewed disposition')
    dispositions, nominations, bindings = {}, [], {}
    for name in registry.ORDER:
        row = rows[name]; state = row['state']
        if state not in confirmation.DISPOSITIONS or not row.get('reason'):
            raise ValueError('explicit valid disposition and reason required')
        evidence = Path(row['evidence'])
        if digest(evidence) != row['evidence_sha256']:
            raise ValueError('disposition evidence changed')
        dispositions[name] = state
        bindings[name] = dict(state=state, reason=row['reason'], evidence_sha256=digest(evidence))
        if state in {'NOMINATED','STOPPED_BELOW_MARGIN','INCONCLUSIVE_AT_CAP','PRIMARY_UNAVAILABLE'}:
            card = read(evidence); review_path = Path(row['review']); reviewed = read(review_path)
            if (card['family'] != name or card['state'] != state or
                    reviewed['family'] != name or reviewed['decision'] != state or
                    reviewed['state'] != 'verified_discovery' or not reviewed['full_fresh_process_replay'] or
                    reviewed['receipts'][-1]['machine_card_sha256'] != digest(evidence) or
                    digest(review_path) != row['review_sha256']):
                raise ValueError('scientific disposition lacks complete bound review')
            bindings[name]['review_sha256'] = digest(review_path)
            if state == 'NOMINATED':
                if reviewed['nominated'] != card['best_id']:
                    raise ValueError('reviewed nomination changed')
                selected = next(r for r in card['curve'] if r['variant']['id'] == card['best_id'])
                nominations.append(dict(family=name, variant=card['best_id'],
                    shrunk_mean=selected['shrunk_mean'], machine_card_sha256=digest(evidence)))
    order = dict(state='DISCOVERY_FROZEN', dispositions=dispositions, evidence=bindings,
        nominations=sorted(nominations, key=lambda r:(-r['shrunk_mean'], registry.ORDER.index(r['family']))))
    confirmation.validate_order(order)
    return order


def freeze(campaign, manifest_path, output):
    campaign, manifest_path, output = map(lambda p:Path(p).resolve(), (campaign,manifest_path,output))
    started = time.process_time(); runtime.below_normal()
    with owner(campaign/'admission'):
        if output.exists(): raise ValueError('never replace a frozen discovery order')
        entries = inventory(campaign)
        if any(p.get('task') == 'confirmation' for _,p,_ in entries):
            raise ValueError('reserved exposure has already been admitted')
        if any(s.get('state') not in ('complete','failed') for _,_,s in entries):
            raise ValueError('discovery or preparation remains unfinished')
        if runtime.now() >= read(campaign/'ACCEPTANCE.json')['report_start']:
            raise ValueError('scientific admission window closed')
        result = build_order(read(manifest_path))
        result.update(at=runtime.now(), manifest_sha256=digest(manifest_path),
            acceptance_sha256=digest(campaign/'ACCEPTANCE.json'))
        write(output, result)
        write(campaign/'attempts/discovery-order-freeze.json',dict(cpu_seconds=time.process_time()-started,state='complete'))
        return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('campaign','manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(freeze(args.campaign,args.manifest,args.output))
