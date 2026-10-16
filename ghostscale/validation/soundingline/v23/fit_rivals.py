"""Fixed/fresh direct rivals for separately declared conditional/twin targets."""
from pathlib import Path
import hashlib,math,time
import numpy as np
from .common import Direct,canonical,write
from . import checker


def verify(root,models,metadata,x,y):
    checked={}
    for name,model in models.items():
        prediction=model.predict(x)
        score=float(np.mean(checker.scores(prediction.reshape(-1,prediction.shape[-1]),y.ravel())['log_loss']))
        if not math.isclose(score,metadata[name]['validation_log_loss'],abs_tol=1e-8,rel_tol=0):
            raise ValueError('independent rival validation score differs')
        if not np.array_equal(Direct.load(Path(root)/(name+'.npz')).predict(x),prediction):
            raise ValueError('saved rival changes predictions')
        checked[name]=dict(validation_log_loss=score,early_stopped=metadata[name]['early_stopped'])
    write(Path(root)/'INDEPENDENT_VALIDATION.json',checked)
    return checked


def profile(root,family,variants,sampler,features,classes,pulse,namespace,count=128):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);started=time.process_time();x=[];y=[]
    for i in range(count):
        v=variants[i%len(variants)]
        p,t=sampler(v,[namespace,'cost-profile'],i//len(variants),i)
        x.append(features(v,p));y.append(t['y'])
        if i%32==0:pulse(phase='rival-cost-profile',family=family,labels=i)
    x,y=np.asarray(x),np.asarray(y);generation=time.process_time()-started
    np.savez_compressed(root/'training-profile.npz',x=x,y=y)
    model=Direct(x,y,x,y,classes,namespace+'-profile',0,max_epochs=2,pulse=pulse)
    model.save(root/'profile.npz');verify(root,{'profile':model},{'profile':model.metadata},x,y)
    result=dict(training_generation_cpu_seconds=generation,training_generation_labels=count,
        fit_cpu_seconds=model.cpu_seconds,fit_epochs=model.metadata['epochs'],fit_labels=count)
    write(root/'training-cost.json',result)
    return result


def fit(root,family,variants,sampler,features,classes,pulse,namespace,training_builder=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    def training(label,count):
        if training_builder is not None:
            return training_builder(root/('draw-'+hashlib.sha256(canonical(label)).hexdigest()),label,count)
        x=[];y=[]
        for i in range(count):
            v=variants[i%len(variants)]
            public,truth=sampler(v,[namespace,label],i//len(variants),i)
            x.append(features(v,public));y.append(truth['y'])
            if i%64==0:pulse(phase='rival-training',family=family,labels=i,target_labels=count)
        return np.array(x),np.array(y)
    vx,vy=training('validation',4096);np.savez_compressed(root/'validation.npz',x=vx,y=vy)
    models={};metadata={}
    for labels in (2048,8192):
        x,y=training(['fixed',labels],labels);np.savez_compressed(root/f'training-{labels}.npz',x=x,y=y)
        for fit_index in (0,1):
            for fresh in ((False,True) if labels==8192 else (False,)):
                name=f'direct-{labels}-fit{fit_index}'+('-fresh' if fresh else '');draws=[]
                def provider(epoch,offset,size):
                    label=['fresh',fit_index,epoch,offset];xx,yy=training(label,size)
                    draws.append(dict(label=label,labels=size,x_sha256=hashlib.sha256(xx.tobytes()).hexdigest(),
                        y_sha256=hashlib.sha256(yy.tobytes()).hexdigest()))
                    return xx,yy
                model=Direct(x,y,vx,vy,classes,canonical([namespace,family]).decode(),fit_index,
                    fresh=provider if fresh else None,pulse=pulse)
                model.save(root/(name+'.npz'));models[name]=model;metadata[name]=model.metadata
                write(root/(name+'-cost.json'),dict(cpu_seconds=model.cpu_seconds))
                if fresh:write(root/(name+'-draws.json'),draws)
    write(root/'FITS.json',metadata)
    verify(root,models,metadata,vx,vy)
    return models,metadata
