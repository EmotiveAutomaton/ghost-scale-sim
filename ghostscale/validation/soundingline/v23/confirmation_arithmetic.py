"""Curator's October 17 expected-evidence screen; no reserved worlds or fitted bets."""
import math
from . import confirmation,support,support_new
from .common import read,digest


def calculate(order,cards):
    entries=confirmation.validate_order(order);rows=[]
    for j,entry in enumerate(entries,1):
        family=entry['family'];path=cards[family]
        if digest(path)!=entry['machine_card_sha256']:raise ValueError('discovery card changed')
        card=read(path);selected=next(r for r in card['curve'] if r['variant']['id']==entry['variant'])
        proof=support_new.bound(family) if family.startswith('N') else support.repair_bound(family)
        if not proof.get('available',True):raise ValueError('finite support unavailable')
        lower=proof['lower_bound'];margin=.02;fraction=.5/(margin-lower)
        worlds=256 if abs(entry['shrunk_mean']-margin)<=.02 else 128
        mu=selected['raw_mean'];bound=(worlds//16)*math.log1p(fraction*(mu-margin))
        # All earlier j-1 confirmations successful: lowest possible threshold at index j.
        gamma=1/(j*(j+1));threshold=-math.log(.1*gamma*j)
        rows.append(dict(family=family,variant=entry['variant'],order=j,discovery_raw_mean=mu,
            discovery_shrunk_mean=entry['shrunk_mean'],support=proof,worlds=worlds,blocks=worlds//16,
            betting_fraction=fraction,expected_log_e_upper_bound_if_discovery_mean_transfers=bound,
            minimum_log_threshold=threshold,all_earlier_confirmed=j-1,
            state='RUN_AS_FROZEN' if bound>=threshold else 'CONFIRMATION_INFEASIBLE_AS_FROZEN'))
    return dict(state='ARITHMETIC_COMPLETE',reserved_worlds_drawn=0,discovery_nominations_unchanged=True,
        qualification='Plug-in expectation screen, not a proof of impossibility or a power bound. Jensen applies if the unknown confirmation mean equals the supplied discovery estimate.',families=rows)
