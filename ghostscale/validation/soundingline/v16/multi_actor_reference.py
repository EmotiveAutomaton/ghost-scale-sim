"""Independent factorized M04 posterior and exhaustive selected-style tables."""
from itertools import product
import math
from .audit_selection import selected_probability,style_of,style_probability
from .graphic_reference import interpret

def predict(public):
    world=public["world"];p=world["permutation"];b=public["brief"];n=public["produced_count"]
    def read_artifact(artifact):
        style=style_of(world,{"artifact":artifact})
        topic=0 if (artifact//(2**p[2]))%2 else 1
        return style,topic
    history=[read_artifact(value) for value in public["history"]]
    raw=None;order=None;acted=None
    if public["producer_view"] is not None:
        raw,topic=read_artifact(public["producer_view"]["artifact"])
        order=p[:2].index(public["producer_view"]["first_action"])
        if topic!=history[0][1]:
            raise ValueError("producer view has another topic")
    if public["revision_view"] is not None:
        view=public["revision_view"]
        revision_raw,topic=read_artifact(view["before"])
        result=interpret(view["program"],initial=view["before"])
        if not result["legal"] or result["artifact"]!=view["after"] or view["after"]!=public["history"][0]:
            raise ValueError("independent revision interpreter rejected the view")
        if raw is not None and raw!=revision_raw:
            raise ValueError("duplicate views disagree")
        raw=revision_raw
        acted=len(view["program"])>0
    unselected=read_artifact(public["unselected_view"]["artifact"]) if public["unselected_view"] is not None else None
    changed=read_artifact(public["brief_view"]["artifact"]) if public["brief_view"] is not None else None
    tables={(acquired,target,visible):selected_probability(world,visible,acquired,target,n)
            for acquired,target,visible in product([0,1],repeat=3)}
    core_weights=[0.5*(1 if order is None else world["core_reuse"] if order==core else 1-world["core_reuse"]) for core in [0,1]]
    core_total=sum(core_weights)
    core_weights=[weight/core_total for weight in core_weights]
    topics=[]
    for shared,purpose in product([0,1],repeat=2):
        topic=b if shared else purpose
        consistent=all(value==topic for _,value in history)
        if unselected is not None:
            consistent=consistent and unselected[1]==topic
        if changed is not None:
            consistent=consistent and changed[1]==((1-b) if shared else purpose)
        topics.append((shared,purpose,float(consistent)/4))
    topic_total=sum(weight for _,_,weight in topics)
    if topic_total<=0:
        raise ValueError("topic evidence outside the declared fixed-purpose catalog")
    topics=[(shared,purpose,weight/topic_total) for shared,purpose,weight in topics]
    craft=[]
    for revision,selector,style,other,target in product(range(3),[0,1],[0,1],[0,1],[0,1]):
        base=other if revision==2 else style
        weight=1/48
        for visible,_ in history:
            weight*=tables[(base,target,visible)] if selector else style_probability(world,visible,base)
        if raw is not None:
            weight*=int(raw==history[0][0]) if revision==0 else style_probability(world,raw,style)
        if acted is not None:
            weight*=int(acted==(revision!=0))
        for extra in [unselected,changed]:
            if extra is not None:
                weight*=style_probability(world,extra[0],base)
        craft.append((revision,selector,style,other,target,weight))
    total=sum(entry[-1] for entry in craft)
    if total<=0:
        raise ValueError("role evidence has no finite factorized model support")
    craft=[(*entry[:-1],entry[-1]/total) for entry in craft]
    output={key:[0.0,0.0] for key in ["producer_core","producer_style","release","brief","selector","shared_brief"]}
    output["revision"]=[0.0]*3;output["revision_relation"]=[0.0]*3;output["topology"]=[0.0]*12
    for bit in [0,1]:
        output["producer_core"][bit]=sum(weight*(world["core_reuse"] if bit==core else 1-world["core_reuse"]) for core,weight in enumerate(core_weights))
    for shared,purpose,weight in topics:
        output["shared_brief"][shared]+=weight
        output["brief"][(1-b) if shared else purpose]+=weight
    for revision,selector,style,other,target,weight in craft:
        base=other if revision==2 else style
        output["revision_relation"][revision]+=weight
        output["selector"][selector]+=weight
        for shared,_,topic_weight in topics:
            output["topology"][revision*4+selector*2+shared]+=weight*topic_weight
        if revision==0:
            output["revision"][0]+=weight
        for bit in [0,1]:
            output["producer_style"][bit]+=weight*style_probability(world,bit,style)
            if revision!=0:
                output["revision"][1+bit]+=weight*style_probability(world,bit,base)
            output["release"][bit]+=weight*(tables[(base,target,bit)] if selector else style_probability(world,bit,base))
    output["historical_core"]=list(output["producer_core"]) if order is None else [float(bit==order) for bit in [0,1]]
    output["topology_entropy"]=-sum(weight*math.log(weight) for weight in output["topology"] if weight>0)
    output["compatible_topologies"]=sum(weight>0 for weight in output["topology"])
    output["costs"]={"likelihood_terms":384*(6+int(raw is not None)+int(acted is not None)+int(order is not None)+2*int(unselected is not None)+2*int(changed is not None)),
                     "hypotheses":384,"prediction_terms":4864,"entropy_terms":output["compatible_topologies"]}
    return output
