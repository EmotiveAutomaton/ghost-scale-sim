"""Independently expressed G1 reconstruction from retained counts and labels.

Written from the packet, without importing the original reader or pre-flight.
Document loops and explicit library elimination replace its multi-axis einsums.
"""
import numpy as np
from scipy.special import logsumexp


def normalized_log(values):
    return values-logsumexp(values,axis=-1,keepdims=True)


def estimate_goals(histograms,habit,targets,rho):
    values=[]
    for target in targets:
        values.append(np.sum(histograms*np.log((1-rho)*habit+rho*target),axis=1))
    return np.exp(normalized_log(np.column_stack(values)))


def update_habit(documents,habit,targets,rho,pseudo,rounds):
    for _ in range(rounds):
        total=np.zeros_like(habit)
        for d in range(documents.shape[1]):
            counts=documents[:,d]
            goals=estimate_goals(counts,habit,targets,rho)
            responsibility=np.zeros_like(habit)
            for g,target in enumerate(targets):
                ratio=(1-rho)*habit/((1-rho)*habit+rho*target)
                responsibility+=goals[:,g,None]*ratio
            total+=counts*responsibility
        total+=pseudo
        habit=total/np.sum(total,axis=1)[:,None]
    return habit


def readers(raw,k,rho):
    library,targets=raw['h'],raw['goals'];documents=raw['counts'][:,:k+1].astype(float)
    w=len(documents);types=len(library);goals=len(targets)
    page=documents[:,0];loghistory=np.zeros((w,types));current=np.empty((w,types,goals))
    for i,habit in enumerate(library):
        logmixture=np.log((1-rho)*habit+rho*targets)
        current[:,i]=page@logmixture.T
        for d in range(1,k+1):
            loghistory[:,i]+=logsumexp(documents[:,d]@logmixture.T,axis=1)-np.log(goals)
    loghistory=normalized_log(loghistory)
    joint=np.exp(normalized_log((current+loghistory[:,:,None]).reshape(w,-1))).reshape(w,types,goals)
    results={'J':np.sum(joint,axis=1)}
    all_counts=np.sum(documents,axis=1)
    naive=normalized_log(all_counts@np.log(library).T)
    results['Ps']=np.exp(normalized_log(logsumexp(current+naive[:,:,None],axis=1)))
    log_ps=normalized_log(logsumexp(current+naive[:,:,None],axis=1))
    new_type=normalized_log(loghistory+logsumexp(current+log_ps[:,None,:],axis=2))
    results['I1']=np.exp(normalized_log(logsumexp(current+new_type[:,:,None],axis=1)))
    rankings=np.argsort(-(loghistory+page@np.log(library).T),axis=1,kind='stable')
    for b in (1,2):
        values=np.empty((w,goals))
        for j in range(w):
            chosen=rankings[j,:b]
            restricted=loghistory[j,chosen]-logsumexp(loghistory[j,chosen])
            values[j]=logsumexp(current[j,chosen]+restricted[:,None],axis=0)
        results['B'+str(b)]=np.exp(normalized_log(values))
    population=np.mean(library,axis=0)
    habit={'Pp':np.tile(population,(w,1))}
    a=all_counts+.5;habit['Pe']=a/a.sum(axis=1)[:,None]
    habit['PeL1']=update_habit(documents,habit['Pe'],targets,rho,.5,1)
    habit['PeL4']=update_habit(documents,habit['Pe'],targets,rho,.5,4)
    for strength in (32,128):
        a=all_counts+strength*population
        habit['PpS'+str(strength)]=a/a.sum(axis=1)[:,None]
        habit['PpSL'+str(strength)]=update_habit(documents,habit['PpS'+str(strength)],targets,rho,strength*population,4)
    for label,estimate in habit.items():results[label]=estimate_goals(page,estimate,targets,rho)
    # Independent exact-symmetry reconstruction. These signatures use integer
    # observed counts, not a tolerance on the inferred probabilities.
    for world in range(w):
        signatures=[]
        for target in targets:
            columns=documents[world].T[target>min(target)]
            signatures.append(sorted(map(tuple,columns)))
        visited=set()
        for goal in range(goals):
            if goal in visited:continue
            equal=[other for other in range(goals) if signatures[other]==signatures[goal]]
            visited.update(equal)
            if len(equal)>1:
                for label in ('Pe','PeL1','PeL4'):
                    average=sum(results[label][world,g] for g in equal)/len(equal)
                    for g in equal:results[label][world,g]=average
    joint_marks=np.zeros((w,library.shape[1]))
    for i,h in enumerate(library):
        for g,target in enumerate(targets):
            joint_marks+=joint[:,i,g,None]*(rho*target/((1-rho)*h+rho*target))
    marks={'J':np.take_along_axis(joint_marks,raw['tokens'],axis=1)}
    for label in ('Pp','Pe','PeL4'):
        target=targets[np.argmax(results[label],axis=1)]
        probability=rho*target/((1-rho)*habit[label]+rho*target)
        marks[label]=np.take_along_axis(probability,raw['tokens'],axis=1)
    return results,marks
