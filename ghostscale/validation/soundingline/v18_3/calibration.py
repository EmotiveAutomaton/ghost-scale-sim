"""Exact finite coverage and fixed-bin confidence diagnostics; no tuned bins."""
import numpy as np


def credible_coverage(reader_joint,true_joint,level=.95):
    reader_joint=np.asarray(reader_joint,float);true_joint=np.asarray(true_joint,float)
    if reader_joint.shape!=true_joint.shape or np.any(reader_joint<0) or np.any(true_joint<0):raise ValueError('invalid joint')
    if not np.isclose(reader_joint.sum(),1,atol=1e-10) or not np.isclose(true_joint.sum(),1,atol=1e-10):raise ValueError('unnormalized joint')
    normal=reader_joint.sum(0)
    if np.any(normal<=0):raise ValueError('reader support excludes an observation')
    posterior=reader_joint/normal;order=np.argsort(-posterior,axis=0,kind='stable')
    sorted_p=np.take_along_axis(posterior,order,axis=0)
    included=np.cumsum(sorted_p,axis=0)-sorted_p<level
    mask=np.zeros_like(included);np.put_along_axis(mask,order,included,axis=0)
    return dict(credible_level=level,coverage=float(np.sum(true_joint*mask)),
        expected_set_size=float(np.sum(mask.sum(0)*true_joint.sum(0))),outcomes=reader_joint.shape[1],states=reader_joint.shape[0])


def reliability(p,truth,bins=10):
    p=np.asarray(p,float);truth=np.asarray(truth,float)
    if p.shape!=truth.shape or np.any(p<0) or np.any(truth<0) or not np.allclose(p.sum(1),1,atol=1e-6) or not np.allclose(truth.sum(1),1,atol=1e-6):raise ValueError('bad calibration forecast')
    chosen=p.argmax(1);confidence=p[np.arange(len(p)),chosen];correct=truth[np.arange(len(p)),chosen]
    assignments=np.minimum((confidence*bins).astype(int),bins-1);rows=[];gap=0.
    for i in range(bins):
        mask=assignments==i;n=int(mask.sum())
        if not n:continue
        predicted=float(confidence[mask].mean());expected=float(correct[mask].mean())
        gap+=n/len(p)*abs(predicted-expected)
        rows.append(dict(bin=i,count=n,predicted_confidence=predicted,expected_accuracy=expected))
    return dict(expected_absolute_calibration_gap=float(gap),overconfidence=float((confidence-correct).mean()),bins=rows,
        scope='fixed ten-bin top-choice confidence against generator expected correctness; not sampled-label accuracy')


def native_coverage(world,condition):
    from . import world as W,active as A
    qs=(1,3);read_tables=[A.bank(world)[q] for q in qs];true_tables=[]
    truth=dict(world)
    if condition=='wrong-rule':truth['rule']='softmax' if world['rule']=='satisficing' else 'satisficing'
    for q in qs:
        base=W.matrix(truth,W.QUERIES[q]);table=np.zeros_like(read_tables[0])
        for state in range(len(W.STATES)):
            for profile in range(3):
                access=A.access(q,profile,state,condition=='hidden-state-access')
                table[state*3+profile,:-1]=access*base[state];table[state*3+profile,-1]=1-access
        true_tables.append(table)
    def joint(tables):
        full=tables[0][:,:,None]*tables[1][:,None,:]/(len(W.STATES)*3)
        return full.reshape(len(W.STATES),3,-1).sum(1)
    return credible_coverage(joint(read_tables),joint(true_tables))
