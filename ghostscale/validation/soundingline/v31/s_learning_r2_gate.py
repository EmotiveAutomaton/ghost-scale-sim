"""Run the approved R2 numerical gate in an existing Gear 2 worker slot."""
from pathlib import Path
from .common import read


def validate(setting, dest, pulse):
    from runners.prepare_v31_s_learning_r2 import validate as check
    campaign = Path(setting['campaign'])
    pulse(phase='approved-S.b-R2-validation')
    check(campaign)
    return read(campaign / 'preparation/S_LEARNING_R2/VALIDATION.json')
