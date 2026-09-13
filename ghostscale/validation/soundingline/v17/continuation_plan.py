"""Finite, deterministic V17 study tree. Selection never trains on fresh outcomes."""
import math
from ..v16.records import digest

BRANCHES=[
 ("a_graphic","A2","graphic:familiar_combinations"),("a_assembly","A2","assembly:changed_constraint"),
 ("a_novel","A2","graphic:new_combinations"),("a_exception","A2","graphic:changed_constraint"),
 ("pooled_graphic","AP","graphic:familiar_combinations"),("pooled_assembly","AP","assembly:new_combinations"),
 ("b_changed","B","changed_recipient"),("b_wrong","B","wrong"),
 ("ba_correct","BA","correct"),("ba_wrong","BA","wrong"),("ba_changed","BA","changed_recipient"),
 ("c_accidental","C","accidentally_correct"),("c_misleading","C","misleading_old_cue"),("c_false","C","false_belief"),("c_noisy","C","noisy_evidence"),
 ("d_stale","D","stale"),("d_misleading","D","misleading_feedback"),("d_stable","D","stable"),
 ("e_a","E","A"),("e_b","E","B"),("e_c","E","C"),("e_d","E","D"),
 ("f_expert","F","expert_partner"),("f_unfamiliar","F","unfamiliar_partner")]

def design(branch,namespace):
    identifier,family,regime=branch
    result=dict(id=identifier,family=family,regime=regime,namespace=namespace,
        histories_per_constructor=8,constructors=32,claim_status="descriptive")
    if family in ("A2","AP"):
        result.update(memory_caps=[32],budgets=[128,512,2048],
            budgets_by_world={"graphic":[256,512,1024,2048],"assembly":[64,128,256,512]})
    return result

def selected_contrasts(discovery):
    # Choices made on the retained screens: practical audience gain, a model
    # misspecification reversal, and the value of feedback under stale purpose.
    choices=[
        dict(id="recipient_shift",family="B",regime="changed_recipient",namespace="v17-b-fresh-1",
             method="inferred_recipient",rival="retrieval",target="recipient_outcome",metric="brier",
             low=-2.,high=2.,minimum_gain=.02,planned_radius=.04,direction="inferred recipient improves forecast after recipient change"),
        dict(id="accidental_reversal",family="C",regime="accidentally_correct",namespace="v17-c-fresh-1",
             method="empirical",rival="fixed_inverse",target="recipient_after_12_observations",metric="brier",
             low=-2.,high=2.,minimum_gain=.02,planned_radius=.04,direction="empirical prediction beats misspecified fixed inverse model"),
        dict(id="stale_feedback",family="D",regime="stale",namespace="v17-d-fresh-1",
             method="recipient_feedback",rival="internal_reconsideration",target="next_revision",metric="success",
             low=-1.,high=1.,minimum_gain=.05,planned_radius=.04,direction="outside feedback improves actual revision success under stale purpose")]
    for contrast in choices:
        span=contrast["high"]-contrast["low"]
        contrast["constructors"]=math.ceil(span*span*math.log(3/.05)/(2*contrast["planned_radius"]**2))
        contrast["histories_per_constructor"]=4
        contrast["discovery_source_sha256"]=digest(discovery)
        contrast["planning"]="fixed distribution-free simultaneous precision; between-constructor screen variation retained separately"
    return choices

def rank_branches(surfaces):
    """Promote two contrasting regions per family; retain negative differences."""
    families={}
    for branch in surfaces:
        rows=branch["comparisons"]
        targets={}
        for r in rows: targets.setdefault(r["target"],[]).append(r)
        spread=max((max(x["mean_brier_or_failure"] for x in xs)-min(x["mean_brier_or_failure"] for x in xs)
                    for xs in targets.values()),default=0.)
        uncertainty=max((x["between_constructor_variance"]/max(1,x["constructor_clusters"]) for x in rows),default=0.)**.5
        entry=dict(id=branch["id"],family=branch["family"],priority=spread+2*uncertainty,
            score_separation=spread,constructor_uncertainty=uncertainty)
        families.setdefault(branch["family"],[]).append(entry)
    chosen=[]
    for family,entries in sorted(families.items()):
        chosen.extend(sorted(entries,key=lambda x:(-x["priority"],x["id"]))[:2])
    return dict(schema="v17.branch-selection.1",selected=chosen,
        rule="two regions per family maximizing unsigned method separation plus twice constructor standard error; ties by ID",
        confirmation_selection_independent=True,interpretation="exploratory branch allocation, never significance-based stopping")

def build_plan(discovery):
    jobs=[design(b,"v17-continuation-"+b[0]+"-1") for b in BRANCHES]
    return dict(schema="v17.continuation-plan.1",workers=1,gpu_used=False,run_hours=120,
        original_implementation_start="2026-09-12T18:16:41.531700+00:00",
        original_soft_target="2026-09-17T18:16:41.531700+00:00",
        timing_amendment="User commissioned a new 120-hour continuously serviced execution allocation after setup; earlier clocks and completed work are preserved.",
        initial_expansions=jobs,confirmation=selected_contrasts(discovery),
        robustness_max_constructor_index=10000000,robustness_block_constructors=32,
        robustness_cycle="three selected-region units then one coverage unit, deterministic round robin; fresh namespaces separated",
        max_database_bytes=256*1024**3,min_free_disk_bytes=64*1024**3,
        scientific_checkpoints=["expansion_selection_frozen","confirmation_complete","robustness_32768_constructor_units","run_complete"],
        state_transition_notifications_only=True,stop_hook_continuation=False,
        fresh_discovery_exposure=discovery,claim_status="ongoing, not campaign complete")
