"""Frozen-before-scout competence-changing inquiry comparisons."""
POLICIES=["recognition","surprise","eig","signed-progress","absolute-progress",
          "uniform","decline","value-learning"]

DESIGNS={
    "R01":{"question":"Can familiarity guide useful inquiry when visible cue strength is matched?",
           "conditions":[{"id":f"familiar-{familiar}-noise-{noise}","familiarity":[familiar,0],
                          "noisy_domains":[0] if noise else []}
                         for familiar in [0,4] for noise in [False,True]],
           "primary":[("recognition","surprise","gain","success_fraction",0.05),
                      ("recognition","eig","gain","success_fraction",0.05)]},
    "R02":{"question":"How does acquired competence change inquiry on the same tasks?",
           "conditions":[{"id":f"initial-{n}","initial_examples":n} for n in [0,2,8]],
           "primary":[("value-learning","uniform","utility","utility_per_packet",0.01),
                      ("value-learning","surprise","gain","success_fraction",0.05)]},
    "R03":{"question":"Which inquiry policies survive learnable, noisy, forgotten and delayed feedback?",
           "conditions":[{"id":"stable"},{"id":"noise","noisy_domains":[0]},
                         {"id":"both-noise","noisy_domains":[0,1]},
                         {"id":"delayed","feedback_batches":[2,1]},
                         {"id":"skill-loss","forget_steps":[3,6]},
                         {"id":"saturated","initial_examples":8}],
           "primary":[("value-learning",rival,"utility","utility_per_packet",0.01)
                      for rival in POLICIES if rival!="value-learning"]},
    "R04":{"question":"Can equal initial abilities support different interests under different objectives?",
           "conditions":[{"id":f"weights-{weight}-cost-{cost}","future_weights":[weight,1-weight],
                          "opportunity_cost":cost,"initial_examples":2}
                         for weight in [0.1,0.9] for cost in [0.0,0.06]],
           "primary":[("value-learning","uniform","utility","utility_per_packet",0.01),
                      ("value-learning","eig","utility","utility_per_packet",0.01)],
           "cross_condition_pairing":"same constructor/history/offers and independent reader ties; no pooling of unlike utility targets"},
    "R05":{"question":"Does enacted production improve new-maker reading beyond equal observed examples?",
           "conditions":[{"id":f"practice-{n}","practice_examples":n,"future_weights":[1.0,0.0]}
                         for n in [0,2,4]],
           "arms":["enact","observe","unrelated"],
           "primary":[("enact","observe","future_log_score","nats_per_event",0.02),
                      ("enact","unrelated","future_log_score","nats_per_event",0.02),
                      ("enact","observe","success","success_fraction",0.05),
                      ("enact","unrelated","success","success_fraction",0.05)],
           "scope_limit":"enactment and observation convey identical sufficient information; any execution premium needs evidence, not a label"}
}
for card,design in DESIGNS.items():
    design.setdefault("arms",POLICIES)
    design.update(card_id=card,generator_families=["W1 command-map acquisition with two-cell composition"],
                  unit="one acquired reader history with paired policy episodes and nested held-out tasks",
                  conditions_visible="cue exposure counts, offers, delivery contract, objectives and costs; no hidden learnability labels",
                  constructors="independent hidden bijections AND varying curriculum target probabilities",
                  sample_rule={"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,
                               "final_expansion":1024},
                  horizon=8,lookahead_cap=2,mechanism="executed examples update a posterior used to compile new compositions",
                  learning_feedback="returned command/output examples only; held-out task success never enters policy input",
                  computation_cost="hypothesis terms in uncached reference computation (lookup caches allowed), utility weight 0.0000001; actual time separate",
                  practice_cost="one declared opportunity cost per enacted interaction; demonstrations and primitive actions separately retained",
                  dependencies=["inquiry-learning","inquiry-noise","inquiry-value","public-process","independent-inquiry-scoring"],
                  adversaries=["X01","X02","X03","X04","X05","X07","X08"],repair_budget=1,
                  continuation="expand only a named policy/learning boundary; retain nulls and cost frontiers")


def condition_values(condition):
    return {"initial_examples":0,"noisy_domains":[],"feedback_batches":[1,1],
            "familiarity":[0,0],"future_weights":[0.5,0.5],"opportunity_cost":0.01,
            "forget_steps":[],**condition}
