"""Select scientific studies explicitly; a repaired probe is never a primary."""
from pathlib import Path
from ..v23.common import read,digest

def verified_primary(campaign,family):
    candidates=[]
    for suffix in ('study-1','healing-1','healing-2'):
        packet=Path(campaign)/'packets'/f'v25-{family.lower()}-{suffix}'
        if not (packet/'COMPLETE.json').exists():continue
        plan=read(packet/'PLAN.json')
        if plan.get('scientific_kind',plan['kind'])!='study':continue
        if read(packet/'COMPLETE.json')['plan_sha256']!=digest(packet/'PLAN.json'):raise ValueError('primary plan binding changed')
        if read(packet/'STATUS.json')['state']!='COMPLETE':raise ValueError('inconsistent primary completion')
        candidates.append(packet)
    if not candidates:raise ValueError('No verified primary study for '+family)
    return candidates[-1]
