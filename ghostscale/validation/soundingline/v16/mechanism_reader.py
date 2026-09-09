"""P04 readers over fixed production hypotheses, with explicit off-model abstention."""
import json
import numpy as np
from .mechanism_models import prior,distribution,families
from .assembly import World,plan as assembly_plan
from .assembly_inference import decode,routes
from .craft import construct
from .world import histories,execute

PUBLIC_KEYS={"schema_version","task_id","world","training","current","prior_works",
             "future_goal","production_budget","max_steps","beta","length_cost","reader_budget"}


def state_for(world,artifact):
    if world["family"]=="W1":
        if type(artifact) is not int or artifact not in range(16):
            raise ValueError("invalid board artifact")
        return artifact
    return decode(World(tuple(world["parents"]),tuple(world["defaults"])),artifact)


def public_reader(payload:bytes,strategy="mixture"):
    public=json.loads(payload)
    if set(public)!=PUBLIC_KEYS or public["schema_version"]!="v16.mechanism-reader.1":
        raise ValueError("production-mechanism reader schema violation")
    allowed={"softmax-maker","softmax-generic","mixture","direct-mixture","bounded-model","habit-model","options-model"}
    if strategy not in allowed:
        raise ValueError("unknown production-mechanism reader")
    world=public["world"]
    expected_world={"family","option_training"} if world.get("family")=="W1" else {"family","parents","defaults"}
    if world.get("family") not in {"W1","W2"} or set(world)!=expected_world:
        raise ValueError("undeclared public physical-world fields")
    expected_training={"attempts"} if world["family"]=="W1" else {"attempts","reliability","topic_probability"}
    if set(public["training"])!=expected_training or any(set(item)!={"goal","artifact"} for item in [public["current"],*public["prior_works"]]):
        raise ValueError("undeclared training or observation fields")
    library_values,library_mass=prior(world,public["training"])
    catalog=families(world)
    if strategy in {"mixture","direct-mixture"}:
        hypotheses=[(family,index) for family in catalog for index in range(len(library_values))]
        initial=np.array([library_mass[index]/len(catalog) for family,index in hypotheses],dtype=float)
    else:
        family={"softmax-maker":"softmax","softmax-generic":"softmax","bounded-model":"bounded",
                "habit-model":"habit","options-model":"options"}[strategy]
        if family not in catalog:
            return {"model_mismatch":True,"capability_state":"not_admitted","reason":"reader family unadmitted for this physical world"}
        hypotheses=[(family,index) for index in range(len(library_values))]
        initial=np.array(library_mass,dtype=float)
    observations=[public["current"]] if strategy=="softmax-generic" else [*public["prior_works"],public["current"]]
    weights=initial.copy()
    likelihoods=[]
    evaluations=0
    def model(family,index,goal):
        return distribution(world,library_values[index],goal,family,budget=public["production_budget"],
                            max_steps=public["max_steps"],beta=public["beta"],length_cost=public["length_cost"])
    for observation in observations:
        state=state_for(world,observation["artifact"])
        likelihood=[]
        for family,index in hypotheses:
            support,probabilities,programs,route_weights=model(family,index,observation["goal"])
            likelihood.append(probabilities[support.index(state)] if state in support else 0.0)
            evaluations+=len(programs)
        likelihoods.append(likelihood)
        weights*=likelihood
        if float(weights.sum())<=0:
            return {"model_mismatch":True,"reason":"observed artifact outside assumed mechanism support",
                    "conditioning_events":len(likelihoods),"likelihood_evaluations":evaluations}
        weights/=weights.sum()
    future=[model(family,index,public["future_goal"]) for family,index in hypotheses]
    support=future[0][0]
    table=np.array([item[1] for item in future],dtype=float)
    if strategy=="direct-mixture":
        # Contract the full evidence/future table; no MAP family or biography is needed.
        joint=initial.copy()
        for row in likelihoods:
            joint=np.multiply(joint,np.array(row))
        prediction=(joint[:,None]*table).sum(axis=0)/float(joint.sum())
    else:
        prediction=weights@table
    future_evaluations=sum(len(item[2]) for item in future)
    library_posterior=np.zeros(len(library_values))
    family_posterior={family:0.0 for family in catalog}
    for weight,(family,index) in zip(weights,hypotheses):
        library_posterior[index]+=weight
        family_posterior[family]+=float(weight)
    chosen=library_values[int(np.argmax(library_posterior))]
    target=state_for(world,public["current"]["artifact"])
    if world["family"]=="W1":
        reconstruction=construct(target,chosen,primitive_budget=public["reader_budget"])
        collisions=sum(execute(program).artifact==target for program in histories())
    else:
        physical=World(tuple(world["parents"]),tuple(world["defaults"]))
        reconstruction=assembly_plan(physical,target,library=chosen,budget=public["reader_budget"],max_steps=public["max_steps"])
        collisions=sum(candidate==target for candidate in routes(physical,public["max_steps"])[1])
    return {"model_mismatch":False,"future_support":[list(state) if isinstance(state,tuple) else state for state in support],
            "future_probabilities":[float(value) for value in prediction],"family_posterior":family_posterior,
            "library_posterior":[float(value) for value in library_posterior],"reconstruction":reconstruction,
            "inferred_library":[list(fragment) for fragment in chosen],
            "grammatically_compatible_routes":collisions,
            "costs":{"likelihood_route_evaluations":evaluations,"future_route_evaluations":future_evaluations,
                     "prior_work_queries":len(observations)-1,"joint_table_cells":int(table.size)},
            "scope":"finite declared family catalog; global route collisions are not a claim of equal posterior history weights"}
