"""GS-2 explicit quadrupled-label calibration of the compatibility instrument."""
import numpy as np
from ..v20.readers import Readers
from .readers import Direct
from .studies import data,retain,controls
from ..v18_3.io import write

def run(root,d,pulse):
    population=[];fits=[]
    for fit in d.get('fit_seeds',[0,1]):
        for budget in d.get('budgets',[512,2048,8192]):
            dd=dict(d,training_cases=budget);x,y=data(dd,'training',fit);vx,vy=data(dd,'validation',fit)
            direct=Direct(x,y,vx,vy,fit_seed=fit,pulse=pulse);fixed=Readers(x,y,fit_seed=fit,pulse=pulse)
            fits.append(dict(fit=fit,budget=budget,direct=direct.metadata,fixed=fixed.metadata()))
            for name,lp in [('converged-direct',direct.predict(vx)),('fixed-schedule-mlp',np.log(fixed.predict(vx,'direct-mlp')))]:
                population.append(dict(fit=fit,budget=budget,reader=name,metrics=retain(root,f'{fit}-{budget}-{name}',lp,vy)))
    transitions=[]
    for lo,hi in zip(d.get('budgets',[512,2048,8192])[:-1],d.get('budgets',[512,2048,8192])[1:]):
        if hi!=4*lo:raise ValueError('optimizer calibration requires quadrupling')
        improvements={}
        for name in ('converged-direct','fixed-schedule-mlp'):
            get=lambda budget:np.mean([r['metrics']['log_loss'] for r in population if r['budget']==budget and r['reader']==name])
            improvements[name]=float(get(lo)-get(hi))
        failed=improvements['converged-direct']>.02 and improvements['fixed-schedule-mlp']<.02
        transitions.append(dict(labels=[lo,hi],improvements=improvements,fixed_schedule_disposition='INSTRUMENT FAILED' if failed else 'NO FAILURE BY THIS CONTROL',practical_improvement_nats=.02))
    write(root/'VALIDATION_CURVES.json',fits)
    return dict(card='G21-A01',phase='GS-2 optimization calibration',population=population,fit_records=fits,
        transitions=transitions,controls=controls(),scope='Validation-only ruler check; no development or confirmation observations used')
