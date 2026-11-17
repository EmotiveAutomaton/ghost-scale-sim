"""Q moving-group reader: retained paths, exact restrictions, resolution checks."""
import numpy as np
from .q_rank import generate,particle_path,exact_nomove,WEIGHTS
from .common import write,save_npz

def validate(s,dest,pulse=lambda **kw:None):
    rows=[]
    for steps in (128,512,2000):
        for truth in (0.,.5,1.):
            for restriction in ('nomove','moving'):
                setting=dict(s,weight=truth,move_rate=0. if restriction=='nomove' else s['move_rate'],steps=steps,worlds=1)
                path=generate(setting,'excluded-hidden-resolution')['records'][0]
                save_npz(dest/f'{steps}-{truth}-{restriction}_points.npz',dict(records=path))
                for weight in (0.,.5,1.):
                    a=particle_path(path,dict(setting,particles=2048),weight,3106)
                    b=particle_path(path,dict(setting,particles=8192),weight,4106)
                    ref=particle_path(path,dict(setting,particles=2048),weight,3106,True)
                    row=dict(steps=steps,truth=truth,candidate=weight,restriction=restriction,resolution_error=float(np.max(abs(a-b))),reconstruction_error=float(np.max(abs(a-ref))))
                    if restriction=='nomove':row['exact_error']=float(np.max(abs(b-exact_nomove(path,setting,weight))))
                    row['passed']=row['resolution_error']<=.25 and row['reconstruction_error']<=1e-8 and row.get('exact_error',0)<=.25;rows.append(row)
                    write(dest/'PROGRESS_VALIDATION.json',rows,immutable=False);pulse(phase='hidden_validation',steps=steps,truth=truth,candidate=weight)
                    if not row['passed']:return dict(passed=False,rows=rows,production_admitted=False)
    return dict(passed=True,rows=rows,production_particles=8192)
