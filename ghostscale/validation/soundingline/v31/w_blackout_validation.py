"""Conditional blackout matching plus separately gated numerical accuracy."""
from .w_matching import match
from .w_hidden_validation import validate as validate_reader
from .common import write

def validate(s,dest,pulse=lambda **kw:None):
    matching=match(s,0,1,s['fraction']);write(dest/'MATCH.json',matching)
    if not matching['passed']:return dict(passed=True,matched=False,reason='blackout comparison cannot match',production_admitted=False)
    numerical=validate_reader(s,dest/'reader',pulse)
    return dict(passed=numerical['passed'],matched=True,matching=matching,numerical=numerical,production_particles=numerical.get('production_particles'))
