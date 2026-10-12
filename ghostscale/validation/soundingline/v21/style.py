"""E03 learned finite mechanism/rendering compatibility ruler.

Local archive selection is supplied, not inferred. Style is an invertible binary
rendering code; the decoded reader is given its parser. Two physical inversions
can cancel for a surface reader, so apparent compatibility must be decomposed.
"""
from itertools import product
import numpy as np
from . import world as W
from .studies import retain
from ..v18_3.io import write

LOCALITIES=('both','mechanism','style','neither')
READERS=tuple(f'{local}-{decoder}' for local,decoder in product(LOCALITIES,('surface','decoded')))


def tables(target,noise,reliability,alpha=.5):
    counts=np.full((2,2,2,2,2),alpha)
    for mechanism,style in product(range(2),repeat=2):
        canonical=target ^ mechanism ^ (noise>=reliability)
        surface=canonical ^ style
        for decoder,feature in enumerate((surface,canonical)):
            np.add.at(counts[mechanism,style,decoder],(feature,target),1)
    return counts/counts.sum(axis=-1,keepdims=True),counts


def predict(table,mechanism,style,token,locality,decoder):
    m=mechanism ^ int(locality in ('style','neither'))
    s=style ^ int(locality in ('mechanism','neither'))
    feature=token if decoder=='surface' else token ^ style
    return table[m,s,int(decoder=='decoded'),feature]


def controls():
    y=np.tile([0,1],20);u=np.zeros(len(y));table,_=tables(y,u,1.)
    m=np.array([0,0,1,1]);s=np.array([0,1,0,1]);truth=np.array([0,1,1,0]);token=truth^m^s
    mechanism=predict(table,m,s,token,'mechanism','decoded')[np.arange(4),truth]
    style=predict(table,m,s,token,'style','decoded')[np.arange(4),truth]
    double=predict(table,m,s,token,'neither','surface')[np.arange(4),truth]
    # Balanced noise makes every input/target table uninformative.
    null_y=np.tile([0,0,1,1],10);null_u=np.tile([0.,1.,0.,1.],10);null,_=tables(null_y,null_u,.5)
    return dict(live_mechanism_match=bool(np.all(mechanism>.9)),placebo_style_match=bool(np.all(style<.1)),
        positive_double_inversion=bool(np.all(double>.9)),placebo_uninformative=bool(np.all(null==.5)))


def run(root,d,pulse):
    from .style_check import fit_reference,predict_reference
    population=[];maximum=0.;casebook=[]
    for world in d['lineages']:
        reliability=float(np.random.default_rng(W.seed('v21-e03-law',world)).uniform(.7,.95))
        test=np.random.default_rng(W.seed('v21-e03-test',world));truth=test.integers(2,size=d['test_cases']);mechanism=test.integers(2,size=d['test_cases']);noise=test.random(d['test_cases']);style_draw=test.random(d['test_cases'])
        label_rng=np.random.default_rng(W.seed('v21-e03-labels',world));m_labels=label_rng.permutation([19,73]);s_labels=label_rng.permutation([31,89])
        for fit in d['fits']:
            train=np.random.default_rng(W.seed('v21-e03-train',world,fit));y=train.integers(2,size=d['labels']);u=train.random(d['labels'])
            table,counts=tables(y,u,reliability);independent=fit_reference(y,u,reliability)
            assert np.allclose(table,independent,atol=1e-12)
            er=root/'evaluator';er.mkdir(exist_ok=True);np.savez_compressed(er/f'{world}-{fit}-training_points.npz',target=y,noise=u,reliability=np.array(reliability),table=table,counts=counts,mechanism_labels=m_labels,style_labels=s_labels)
            for regime,correlation in (('native',.9),('independent',.5),('reversed',.1)):
                style=mechanism ^ (style_draw>=correlation);canonical=truth ^ mechanism ^ (noise>=reliability);token=canonical^style;key=f'{world}-{fit}-{regime}'
                np.savez_compressed(er/f'{key}_points.npz',truth=truth,mechanism=mechanism,style=style,canonical=canonical)
                rr=root/'reader';rr.mkdir(exist_ok=True);np.savez_compressed(rr/f'{key}_points.npz',token=token,mechanism_archive_label=m_labels[mechanism],style_code=style,style_archive_label=s_labels[style])
                for local,decoder in product(LOCALITIES,('surface','decoded')):
                    q=predict(table,mechanism,style,token,local,decoder)
                    reference=np.array([predict_reference(independent,int(m),int(s),int(t),local,decoder) for m,s,t in zip(mechanism,style,token)])
                    error=float(abs(q-reference).max());maximum=max(maximum,error);assert error<1e-12
                    # Arbitrary names cannot change selected physical archive or forecast.
                    recovered_m=np.array([int(np.flatnonzero(m_labels==v)[0]) for v in m_labels[mechanism]])
                    recovered_s=np.array([int(np.flatnonzero(s_labels==v)[0]) for v in s_labels[style]])
                    assert np.array_equal(q,predict(table,recovered_m,recovered_s,token,local,decoder))
                    lp=np.full((len(truth),128),-np.inf);lp[:,:2]=np.log(q);reader=f'{local}-{decoder}'
                    metrics=retain(root,f'{key}-{reader}',lp,truth)
                    population.append(dict(lineage=world,fit=fit,regime=regime,reader=reader,metrics=metrics))
                if fit==d['fits'][0] and world==d['lineages'][0]:casebook.append(dict(regime=regime,target=int(truth[0]),mechanism=int(mechanism[0]),style=int(style[0]),token=int(token[0]),role='evaluator only'))
                pulse(phase='mechanism-style',lineage=world,fit=fit,regime=regime)
    write(root/'CASEBOOK_EVALUATOR.json',casebook)
    return dict(card='G21-E03',population=population,controls=controls(),independent_max_probability_error=maximum,
        scope='Matched finite count learners; supplied mechanism locality and rendering parser; constructed-method ruler.',
        limitations=['The binary mechanism flips a target bit before noise; the rendering code independently flips its surface token. Two inversions can cancel.',
        'Each of four mechanism/style archives has the same label count, shared target/noise draws and one half-count per cell. Every reader fits the same four archives, selecting one per query.',
        'Mechanism identity for archive selection and the rendering parser are supplied privileges, not inferred authorship or learned family recognition.',
        'All decoders have the same finite table capacity and exact count fit; no validation convergence or B/C superiority gate is claimed.',
        'Correlation between mechanism and rendering changes from .9 through .5 to .1; target, mechanism and noise draws remain paired. This is a rendering-distribution shift, not a change to execution noise.',
        'Arbitrary archive-label renaming is an invariance control, not label-shuffling of training targets.',
        'No factor-product reader; duplicate-removal contrast unavailable. No human or actual LLM-family conclusion.',
        'Two fits nest within eight reused development worlds; no confirmation or multiplicity-adjusted promotion.',
        'Purpose-built miniature — architecture untested; unknown parsers and unequal real family priors remain untested.'])
