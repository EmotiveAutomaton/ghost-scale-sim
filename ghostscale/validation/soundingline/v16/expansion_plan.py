"""Finite constructor expansion decisions made before opening fresh outcomes."""
from .expansion_adapter import CARDS, design

REASONS = {
    "K01": "Test acquired construction and personal-versus-pooled search savings over fresh acquisition histories and 32 constructor redraws, including the large-budget null.",
    "K02": "Test the boundary where generic prediction already explains nearly all prospective performance; retain the exact direct-table rival and the practically tiny personal increment.",
    "K03": "Separate useful old-craft reuse from purpose interference under aligned, partial and opposed tasks; resolve uncertain inhibition/relearning boundaries.",
    "K04": "Compare acquired motifs with observed-graph options at the same search budget across fresh constructors, preserving missing-graph and ample-budget boundaries.",
    "K05": "Separate actual focal acquisition from instruction and feedback; resolve the uncertain transfer condition and retain no-feedback failures.",
    "P01": "Locate the bounded-search region where useful reconstruction helps, ties or harms while historical routes remain nonunique.",
    "P04": "Separate production-law misspecification from direct predictive sufficiency in independent graphic and assembly mechanisms.",
    "O01": "Resolve which physical interventions separate artifact-equivalent opportunity causes and improve the named unseen response.",
    "O03": "Resolve consideration-versus-effort inference against both fixed-menu and budget-only rivals with matched public evidence.",
    "O04": "Separate actual physical impossibility from false affordance beliefs, retaining artifact ambiguity and demonstration boundaries.",
    "S01": "Test action evidence about inaccessible controller state over fresh constructors; intact memory remains the no-additional-information boundary.",
    "S02": "Challenge the observed equivalence of self reconstruction and ordinary error monitoring across 32 redraws before any useful-null promotion.",
    "S03": "Test the value and collateral cost of dependency inspection in independent assembly redraws; direct simulation remains the serious rival.",
    "S04": "Separate recovering an old goal from achieving a newly adopted goal under notes, absent memory and fresh action evidence.",
    "S05": "Test helpful self-correction and harmful self-explanation under false memory and other-actor edits; keep the ordinary Bayesian rival.",
    "R01": "Test recognition's inquiry advantage against surprise and the stronger same-information EIG policy across acquisition/familiarity conditions.",
    "R02": "Resolve learning, saturation and pure-noise stopping boundaries with actual costs and the corrected, independently checked inquiry reader.",
    "R03": "Resolve delayed-learning and noise utility boundaries against every registered rival; preserve regions where bounded value learning loses.",
    "R04": "Test the reversal caused by objective weights and query cost across new command-map curricula, without retuning the planning horizon.",
    "M01": "Separate maker recognition from process prediction across evidence dose, process access and misleading changed purpose.",
    "M02": "Resolve selective-publication distortion against release-naive prediction while retaining equal-input direct equivalence and all rejected work.",
    "M03": "Separate audience knowledge from executable rehearsal and historical production evidence over fresh maker pairs and task contexts.",
    "M04": "Resolve which role records improve producer, editor, release and brief predictions under self/other and changed-purpose interventions.",
    "V01": "Resolve conditional constructed tradeoff persistence against context-mixture and stable-profile rivals; this remains a simulator-defined shadow branch.",
    "V02": "Resolve inherited craft and redirected-purpose trajectory boundaries, preserving shared constructor variation and competing non-profile explanations.",
    "V03": "Resolve future-target versus cause-target probe selection under capability, audience and value collisions; retain direct predictive equivalence."}

CLOSED = {
    "O02": "The exact supplied-state resource comparison already realizes the named physical/known-resource bounds. It is excluded from learned-reader promotion, and adds no unresolved sampled discriminator requiring more units.",
    "P02": "All five prospective probe increments have narrow scout intervals far below the registered practical bar. Exact collision controls retain the information/history distinction; no additional prospective claim is promoted.",
    "P03": "All five evidence-dose increments are precisely below the practical bar. K02 receives the constructor replication of the personal/generic boundary; these dose curves remain descriptive and are not separate promoted mechanisms.",
    "R05": "Observed and enacted arms receive the same command/output feedback and return identical construction/prediction updates. Exact source and independent controls establish this information identity; action cost stays separate and no enactment-specific benefit is promoted."}


def plan():
    if set(REASONS)|set(CLOSED) != set(CARDS) or set(REASONS)&set(CLOSED):
        raise ValueError("finite expansion disposition omits or duplicates a commissioned native card")
    cards = [{"card_id": card, "state": "planned", "reason": REASONS[card],
              "n_per_condition": 256, "constructors": 32, "design": design(card),
              "namespace": "v16-constructor-expansion-1-"+card}
             for card in sorted(REASONS)]
    return {"schema_version": "v16.expansion-plan.1", "cards": cards,
        "closed_at_scout": [{"card_id": card, "state": "exhausted", "reason": CLOSED[card]} for card in sorted(CLOSED)],
        "planned_maker_condition_records": sum(256*len(row["design"]["conditions"]) for row in cards),
        "selection": "named ambiguity, serious rival, capability or constructor boundary; retain every registered condition of each expanded card",
        "scientific_changes": "none to generators, recipes, reader outputs, scoring or primary bars; early public-byte calls move to the guarded worker with exact-bytecode parity controls",
        "reader_cost_correction": "actual frozen invocation meter supplements the incomplete historical direct-table counter; primary scientific scores unchanged",
        "sampling": "32 independent constructor redraws, eight maker histories per constructor within each condition; conditions sharing draws are not independent when combined",
        "further_expansion": "At most one fresh 1024-maker expansion for a named contrast that remains informative but imprecise; decide once from the completed 256 packet, with explicit exhausted dispositions otherwise",
        "original_records": "all original scouts and the failed superseded inquiry lineage retained; fresh namespaces never pooled as a causal before/after repair experiment",
        "worker_cap": 1, "deadline": "inherited immutable campaign ceiling; resumptions never reset it",
        "confirmation": "not selected or frozen by this discovery plan"}
