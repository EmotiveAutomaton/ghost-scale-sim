"""One bounded optimizer contrast; identical probability-head objective and inputs."""
import numpy as np
from scipy.optimize import minimize
from . import probability_head as P

def fit(x,y,sizes,pulse=lambda **kw:None):
    shape=(x.shape[1],y.shape[1]);trace=[];iteration=0
    def evaluate(v):
        objective,g,_=P.objective_gradient(x,y,v.reshape(shape),sizes)
        return objective,g.ravel()
    def record(v,step,**extra):
        objective,g=evaluate(v)
        trace.append(dict(step=step,objective=objective,gradient_norm=float(np.linalg.norm(g)),gradient_max=float(abs(g).max()),**extra))
    def callback(v):
        nonlocal iteration
        iteration+=1
        if iteration%25==0:record(v,iteration)
        pulse(optimization_step=iteration)
    zero=np.zeros(np.prod(shape));record(zero,0)
    result=minimize(evaluate,zero,jac=True,method='L-BFGS-B',callback=callback,
        options=dict(maxiter=1000,maxfun=2000,gtol=1e-7,ftol=1e-12,maxls=30))
    record(result.x,int(result.nit),solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),function_evaluations=int(result.nfev))
    return result.x.reshape(shape),trace

def controls():
    x=np.column_stack((np.ones(20),np.tile([-1.,1.],10)))
    y=np.column_stack((x[:,1]<0,x[:,1]>0)).astype(float)
    w,trace=fit(x,y,[2]);_,g,logp=P.objective_gradient(x,y,w,[2])
    null=np.full((20,2),.5);nw,_=fit(x,null,[2])
    return dict(live_known_separation=bool(np.min(np.exp(logp)[y.astype(bool)])>.97),
        placebo_constant_target=bool(np.max(abs(nw))<1e-12),positive_small_fixture_gradient=bool(abs(g).max()<1e-6),
        positive_objective_decreased=trace[-1]['objective']<trace[0]['objective'])

def run(root,plan,pulse):
    summary=P.run(root,plan,pulse,fit_function=fit)
    summary['controls'].update(controls())
    summary['interpretation']='bounded same-objective optimizer comparison; predecessor unchanged; convergence diagnosed separately from prediction'
    summary['scoring']='unfloored normalized softmax as in predecessor; comparator floors retained; not a new scoring repair'
    return summary
