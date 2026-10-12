"""Independent scalar posterior, expansion and event scoring for finite omission."""
import math

def likelihood(identity,cue,reliability=(.8,.6)):
    family,parameter=divmod(identity,10)
    return math.prod(r if actual==observed else (1-r)/9 for actual,observed,r in zip((family,parameter),cue,reliability))

def posterior(prior,cue,reliability=(.8,.6)):
    masses=[float(prior[i])*likelihood(i,cue,reliability) for i in range(100)]
    total=math.fsum(masses)
    return [v/total for v in masses]

def predict(prior,cue,keep,budget,reader,uniform_order,truth=None):
    exact=posterior(prior,cue);included=list(range(keep));missing=list(range(keep,100))
    if reader=='uniform-expansion':included += [i for i in uniform_order if i>=keep][:budget]
    elif reader=='evidence-ranked-expansion':included += sorted(missing,key=lambda i:(-likelihood(i,cue),i))[:budget]
    elif reader=='truth-supplied' and truth not in included:included.append(int(truth))
    remaining=[i for i in range(100) if i not in included];mass=math.fsum(exact[i] for i in remaining)
    if reader=='forced-choice':
        total=math.fsum(exact[i] for i in included);q=[exact[i]/total if i in included else 0. for i in range(100)]
    else:q=[exact[i] if i in included else mass/len(remaining) for i in range(100)]
    return q,included,mass

def event_loss(q,included,truth):
    mass=q[truth] if truth in included else math.fsum(q[i] for i in range(100) if i not in included)
    return -math.log(mass) if mass else math.inf
