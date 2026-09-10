"""Bounded B02 comparison; case discovery is distinct from population confirmation."""
from .selection import DESIGN as SELECTION

CONDITIONS = [dict(row) for row in SELECTION["conditions"] if row["count"] == 4]
CASE_IDS = ["direct-equivalence", "artifact-core-ambiguity", "process-narrows-core",
            "selection-changes-raw-prediction", "naive-confident-error", "rejected-work-corrects-account"]
DESIGN = {"card_id": "B02", "question": "Does bounded adaptive editing discover more independently validated selection and process distinctions than a fixed stratified archive at equal physical primitive budget?",
    "scope": "explanatory archive search in the existing M02 acquired-production/selection mechanism; no universal search superiority claim",
    "conditions": CONDITIONS, "editable_axes": ["retention", "process", "rejected"],
    "fixed_count": 4, "search_replicates": 8, "steps_per_replicate": 12,
    "physical_primitives_per_candidate": 113, "primitive_budget_per_method": 10848,
    "budget_derivation": "28 acquired-training primitives + 60 observed production + 25 fresh raw/released production; 96 candidate maker packets per method",
    "inference_cost": "every actual likelihood/prediction term and OS request resources reported separately; equal primitive budget is not equal total CPU",
    "fixed_policy": "one complete 12-condition cycle in each independent replicate, with frozen rotation",
    "adaptive_policy": "three retention anchors, then bounded one-axis edits around the best novelty witness; every fourth step explores the whole grid",
    "reward": "number of previously unseen semantic case IDs, never UUID or archive occupancy",
    "case_ids": CASE_IDS, "case_thresholds": {"equivalence": 1e-10, "core_confidence": 0.75,
        "prediction_change": 0.05, "naive_true_style_mass_below": 0.2, "aware_true_style_mass_above": 0.5},
    "followup": {"namespace": "v16-explanatory-archive-followup-1", "makers_per_condition": 24,
        "constructors_per_condition": 24, "required_case_recurrences": 18,
        "freeze": "all 12 conditions and 24 seed indices fixed before any archive search",
        "interpretation": "prespecified descriptive recurrence filter, not a significance test or confirmation claim"},
    "search_namespace": "v16-explanatory-archive-search-1",
    "sampling": "each method/replicate has its own independent history and constructor namespace; compare eight independent complete archive runs per method. Follow-up conditions share constructor/history draws and are not pooled as independent worlds",
    "dependencies": ["M02.instrument", "X01", "X03", "X04", "X06", "X07", "X08"],
    "archive_lenses": "full commissioned role catalogue additionally indexes existing diverse native cases; this bounded search comparison covers selection, process ambiguity, and their rivals",
    "repair_budget": 1, "expansion": "one finite 96-candidate allocation per method plus the frozen 288-maker-condition follow-up; no adaptive target or threshold changes"}
