"""Finite retained engineering spike-ins, explicitly not real-shape science."""
from copy import deepcopy
import hashlib,hmac
from pathlib import Path
import numpy as np
from . import registry,ledger,twin_ledger,engine_bridge,checker
from . import twin_writing,twin_chess,twin_wiki,twin_panel
from .common import canonical,write


def sample(family,dirty,index=0):
    v=deepcopy(registry.variants(family)[0]);v['settings']['dirty']=dirty
    if family=='T1':
        # Select an existing effect cell; all dimensions remain declared.
        v=next(x for x in registry.variants(family) if x['settings']['personal_effect']==.6 and x['settings']['dirty']==dirty)
        shape=dict(records_per_maker=[16],sessions_per_maker=[2],options_per_decision=4,
            latency_quantiles_s=dict(probabilities=[0,1],values=[1,4]),
            keystroke_quantiles=dict(probabilities=[0,1],values=[0,12]),handling_marginals=[.5,.22,.23,.05])
        p,t=twin_writing.maker(v,'engineering-spikein',index,0,shape)
    elif family=='T2':p,t=twin_chess.player(v,'engineering-spikein',index,0,games=3,clock_quantiles=dict(probabilities=[0,1],seconds=[1,120]))
    elif family=='T3':
        rows=list(twin_wiki.world(v,'engineering-spikein',index,makers=3,records=6))
        p=dict(maker=0,history=[x for x,_ in rows]);t=dict(marks=[y for _,y in rows],y=[rows[0][1]['purpose']])
    else:p,t=twin_panel.rater(v,'engineering-spikein',index,0,semantics='nominal_yes_no_unsure',repeat=1,records=8)
    return v,p,t


def evaluate(v,p):
    family=v['family']
    if family=='T1':
        prediction,detail=twin_writing.evaluate(v,p);validity=twin_writing.verify(v,p,detail)
    elif family=='T2':
        prediction,detail=twin_chess.evaluate(v,p);validity=twin_chess.verify_posterior(v,p,prediction,detail)
    elif family=='T3':
        prediction,detail=twin_wiki.evaluate(v,p['history'],p['maker']);validity=twin_wiki.verify(v,p['history'],p['maker'],detail)
    else:
        prediction,detail=twin_panel.evaluate(v,p);validity=twin_panel.verify(v,p,detail)
    return prediction,detail,validity


def run(root,plan,pulse):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan['schema_directory'],plan['schema_pin']);receipts=[]
    for family in ('T1','T2','T4','T3'):
        for dirty in (False,True):
            identity=[family,'dirty' if dirty else 'clean'];folder=root/(family+'-'+identity[1]);folder.mkdir()
            v,p,t=sample(family,dirty);prediction,detail,validity=evaluate(v,p)
            canary=hmac.new(bytes.fromhex(plan['canary_key']),canonical(identity),hashlib.sha256).hexdigest()
            bundle=twin_ledger.emit(v,p,schema,identity);ledger.validate(bundle,schema,canaries=[canary])
            decoded=twin_ledger.decode(v,bundle,schema)
            if canonical(decoded)!=canonical(p):raise ValueError('twin observation round trip changed')
            reconstructed=evaluate(v,decoded)[0]
            for name in prediction:
                if not np.allclose(prediction[name],reconstructed[name],atol=1e-12,rtol=0):raise ValueError('twin forecast round trip changed')
            write(folder/'PUBLIC.json',p);write(folder/'TRUTH.json',dict(t,canary=canary))
            write(folder/'PREDICTION.json',dict(probabilities=prediction,detail=detail,
                scores={name:checker.scores(q,t['y']) for name,q in prediction.items()}))
            bridge=engine_bridge.verify_export(folder/'engine',v,bundle,schema,canaries=[canary])
            write(folder/'VALIDITY.json',dict(validity,engine=bridge,variant=v,real_shape=False,
                warning='Deliberately small engineering fixture. Timing and counts are illustrative; no empirical detectability or human mechanism result.'))
            receipts.append(dict(family=family,dirty=dirty,engine=bridge));pulse(phase='twin-engineering',family=family,dirty=dirty)
    write(root/'SUMMARY.json',dict(state='TWIN_ENGINEERING_VERIFIED',fixtures=receipts,
        scientific_family_result=False,real_shape_citation_allowed=False,full_twin_pipelines_admitted=False))
