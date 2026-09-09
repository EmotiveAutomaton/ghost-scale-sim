"""Frozen contrasts for causal opportunities and self-reconstruction."""
DESIGNS = {
 "O01":{"question":"Which diagnostic action distinguishes inability, ignorance, omission and purpose?",
        "family":"opportunity","conditions":[
            {"id":"before-probes","observed_probes":[],"target_probe":"reminder"},
            {"id":"after-reminder","observed_probes":["reminder"],"target_probe":"demonstration"},
            {"id":"after-demonstration","observed_probes":["demonstration"],"target_probe":"tool"},
            {"id":"after-two-probes","observed_probes":["reminder","demonstration"],"target_probe":"retarget"}],
        "arms":["latent-menu","without-probe","fixed-menu"],
        "primary":[("latent-menu","without-probe","future_log_score","nats_per_event",0.02)],
        "mechanism":"Same initial choice under four distinct opportunity components; actual interventions change the corresponding component.",
        "target_realization":"Independent primitive execution verifies each intervention response; private causes are never observation fields."},
 "O02":{"question":"How much of a critic's improvement survives matching the maker's constraints and knowledge?",
        "family":"critic","conditions":[{"id":cause,"force_cause":cause} for cause in ["physical","knowledge","consideration"]],
        "arms":["own-resources","actual-constraints","known-repertoire"],
        "primary":[("own-resources","actual-constraints","success","success_fraction",0.05),
                   ("actual-constraints","known-repertoire","success","success_fraction",0.05)],
        "mechanism":"The same requested goal is attempted under the critic's tools, the maker's physical tools and the maker's believed repertoire.",
        "target_realization":"Each critic receives its explicitly declared resource view and executes in that corresponding environment.",
        "scope_limit":"Supplied-state resource comparison, excluded from learned-reader capability promotion."},
 "O03":{"question":"Does uncertainty about consideration outperform fixed menus and budget-only inference?",
        "family":"opportunity","conditions":[
            {"id":"reminder","observed_probes":["reminder"],"target_probe":"demonstration"},
            {"id":"demonstration","observed_probes":["demonstration"],"target_probe":"reminder"},
            {"id":"extra-search","observed_probes":["search"],"target_probe":"reminder"}],
        "arms":["latent-menu","fixed-menu","budget-only"],
        "primary":[("latent-menu","fixed-menu","future_log_score","nats_per_event",0.02),
                   ("latent-menu","budget-only","future_log_score","nats_per_event",0.02)],
        "mechanism":"The maker's considered options and number of evaluated candidates differ; readers represent different uncertainty sets.",
        "target_realization":"A reminder changes consideration; an extra-search intervention changes effort while the considered set stays fixed."},
 "O04":{"question":"Can artifact evidence diagnose false beliefs about feasibility?",
        "family":"opportunity","conditions":[
            {"id":cause+"-"+probe,"force_cause":cause,"observed_probes":[] if probe=="none" else [probe],
             "target_probe":"tool","causes":["physical","knowledge","consideration","purpose","search","false-affordance"]}
            for cause in ["physical","false-affordance"] for probe in ["none","demonstration"]],
        "arms":["latent-menu","accurate-belief","without-probe"],
        "primary":[("latent-menu","accurate-belief","future_log_score","nats_per_event",0.02)],
        "mechanism":"A believed and considered action can be physically impossible; actual demonstrations can correct the belief.",
        "target_realization":"Private records retain attempted illegal actions and changed beliefs separately from final artifacts.",
        "scope_limit":"Truth-stratified cases; the hidden stratum is not passed to readers. Artifact-equivalent belief states remain ambiguous."},
 "S01":{"question":"Can one's actions reveal a controller missing from accessible self-knowledge?",
        "family":"self","conditions":[{"id":memory,"memory":memory} for memory in ["intact","partial"]],
        "arms":["self-model","memory-only","bayes-error"],
        "primary":[("self-model","memory-only","controller_log_score","nats_per_event",0.02)],
        "mechanism":"Eight actual training attempts compile a controller; accessible goal memory and controller memory are separate.",
        "target_realization":"Intact control memory supplies no extra controller information through action; partial memory can gain it."},
 "S02":{"question":"Does explicit self-reconstruction improve repair over ordinary goal-error monitoring?",
        "family":"self","conditions":[{"id":memory,"memory":memory} for memory in ["partial","intact"]],
        "arms":["self-model","direct-error","bayes-error"],
        "primary":[("self-model","bayes-error","net_repair","net_repair_fraction",0.05),
                   ("self-model","direct-error","net_repair","net_repair_fraction",0.05)],
        "mechanism":"A risky executable reset can fix a wrong controller or damage a correct one; all monitors share evidence and repair opportunities.",
        "target_realization":"Separate independently written Bayesian error monitor and signal comparator; actual resets determine useful and harmful changes.",
        "scope_limit":"In this binary finite model, the complete expected-error statistic can be sufficient for the same decision as self inference."},
 "S04":{"question":"Does an unfinished artifact recover a forgotten goal or support a new goal?",
        "family":"self","conditions":[
            {"id":"absent-with-artifact","memory":"absent","action_evidence":1},
            {"id":"absent-no-artifact","memory":"absent","action_evidence":0},
            {"id":"notes","memory":"absent","notes":True},
            {"id":"new-information","memory":"absent","action_evidence":4},
            {"id":"retarget","memory":"absent","retarget":True},
            {"id":"retarget-with-notes","memory":"absent","retarget":True,"notes":True}],
        "arms":["self-model","memory-only","direct-completion","bayes-error"],
        "primary":[("self-model","memory-only","original_goal_recovery","success_fraction",0.05)],
        "mechanism":"Memory loss, preserved notes, extra work evidence and independently assigned goals affect different decisions.",
        "target_realization":"Original-goal posterior accuracy, continuing the old goal, and attaining the adopted goal are scored separately."},
 "S05":{"question":"Can misleading self-evidence cause confident but harmful continuation?",
        "family":"self-trajectory","conditions":[
            {"id":"accurate","memory":"partial","action_evidence":1},
            {"id":"misleading-memory","memory":"misleading","action_evidence":1},
            {"id":"misleading-with-correction","memory":"misleading","action_evidence":4},
            {"id":"other-actor-edit","memory":"partial","edited_artifact":True,"action_evidence":1}],
        "arms":["self-model","direct-completion","bayes-error","naive-self"],
        "primary":[("self-model","direct-completion","original_goal_success","success_fraction",0.05),
                   ("self-model","bayes-error","original_goal_success","success_fraction",0.05)],
        "mechanism":"A plausible wrong memory or externally edited artifact changes a goal/controller posterior and the chosen repair.",
        "target_realization":"Three actual reset/action feedback rounds; the valid reader conditions on its own resets, while a naive feedback reader is an attack diagnostic.",
        "scope_limit":"Misspecified-source diagnostics are not calibrated-source evidence."},
}
for card,design in DESIGNS.items():
    design.update({"card_id":card,
                   "strongest_rival":"complete Bayesian ordinary error monitor" if design["family"].startswith("self") else
                                     "same-evidence fixed-menu and bounded-search models" if design["family"]=="opportunity" else
                                     "critic with matched physical constraints and known repertoire",
                   "access_arms":"physically serialized public observations; memory-only/without-probe get an explicitly reduced observation file",
                   "secondary":["legality","original versus adopted goal quality","considered set versus search effort",
                                "posterior confidence","actual monitoring/query/training/reset costs","ambiguity"],
                   "generator_families":[design["family"]],
                   "paired_unit":"one independent maker/history with shared diagnostic or reset/future random draws across readers",
                   "sample_rule":{"scout":64,"scout_constructors":8,"expansion":256,"expansion_constructors":20,
                                  "further_expansion":1024},
                   "dependencies":["opportunity-realization","nonnested-capabilities","opportunity-inference"]
                                  if not design["family"].startswith("self") else
                                  ["self-information","ordinary-error-rival","executed-repair"],
                   "adversaries":["X01","X03","X04","X07","X08"] +
                                 (["X05"] if card in ["O04","S04","S05"] else []),
                   "repair_budget":1,
                   "continuation":"Retain informative ambiguity and ordinary-monitoring nulls; expand only named cost, evidence or intervention boundaries."})
DESIGNS["O03"]["cost_qualification"] = "Equal online finite-search budget; actual hypothesis-evaluation counts differ and are reported. No claim of equal total work."
DESIGNS["S05"]["dependencies"].append("self-feedback-causality")
DESIGNS["S05"]["dependencies"].extend(["S01.instrument","S04.instrument","source-reliability"])
DESIGNS["O03"]["dependencies"].append("O01.instrument")
for condition in DESIGNS["O01"]["conditions"]:
    condition["causes"] = ["physical","knowledge","consideration","purpose"]
