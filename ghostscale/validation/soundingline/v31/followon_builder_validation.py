"""Execute already-declared follow-on preparation inside a Gear 2 worker slot."""
from pathlib import Path
from .common import read


def validate(setting,dest,pulse=lambda **kw:None):
    from runners.prepare_v31_followons import prepare
    pulse(phase='declared_followon_preparation')
    repo=Path(setting['repo']);campaign=Path(setting['campaign'])
    prepare(repo,campaign)
    rows=read(campaign/'TRIGGER_STATUS.json')['rows']
    return dict(passed=True,trigger_count=len(rows),eligible=sum(x['eligible'] for x in rows),
                scope='Preparation of declared conditional follow-ons only; each admission retains its own profile proof',
                accounting='Native preparation CPU is additionally charged conservatively beside per-profile charges')
