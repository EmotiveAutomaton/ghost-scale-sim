"""Independent bounded-catalogue predicate, denominator and evidence recount."""
import math
from .records import read, file_digest, digest
from .aggregate_science import registered_designs
from .replay_plan import lookup


def predicates(row, public, truth, prediction):
    result = {}
    card, arms = row["card_id"], row["arms"]
    if card in {"P01", "P02"}:
        reader = arms["maker"]
        same_program = reader["reconstruction"]["program"] == truth["true_production_record"]["program"]
        routes = len(truth["equivalence_classes"])
        if reader["outcomes"]["success"] == 1:
            if same_program:
                result["enactable-history-match"] = {"program_matches_observed_truth": True, "compatible_routes": routes}
            if routes > 1:
                result["enactable-ambiguous"] = {"compatible_routes": routes, "program_matches_observed_truth": same_program}
        elif same_program and reader["reconstruction"]["search_timeout"]:
            result["accurate-unenactable"] = {"program_matches_observed_truth": True, "search_timeout": True}
        improvement = reader["outcomes"]["future_log_score"]-arms["without-query"]["outcomes"]["future_log_score"]
        if card == "P02" and routes > 1 and improvement > 1e-10:
            result["evidence-gain-history-ambiguous"] = {"paired_future_log_score_gain": improvement, "compatible_routes": routes}
    if card == "M01":
        gain = arms["craft"]["outcomes"]["future_core_log_score"]-arms["direct-table"]["outcomes"]["future_core_log_score"]
        if arms["craft"]["outcomes"]["identity_accuracy"] == 1 and gain <= 1e-10:
            result["recognition-without-process-gain"] = {"identity_accuracy": 1, "future_core_log_score_gain_over_direct_table": gain}
    if card == "S05":
        values = {name: arms[name]["outcomes"]["original_goal_success"] for name in ["self-model", "direct-completion", "naive-self"]}
        if values["self-model"] > values["direct-completion"]:
            result["self-correction-help"] = {"original_goal_success_gain_over_direct_completion": values["self-model"]-values["direct-completion"]}
        if values["self-model"] > values["naive-self"]:
            result["self-explanation-harm"] = {"naive_original_goal_success_loss": values["self-model"]-values["naive-self"]}
    if card in {"R02", "R03"}:
        value = arms["value-learning"]["outcomes"]
        if value["queries"] > 0 and value["gain"] > 0:
            result["inquiry-competence"] = {"queries": value["queries"], "realized_heldout_success_gain": value["gain"]}
        for name in ["absolute-progress", "surprise", "recognition", "value-learning"]:
            value = arms[name]["outcomes"]
            if value["queries"] > 0 and value["gain"] <= 0:
                result["inquiry-no-gain"] = {"reader": name, "queries": value["queries"], "realized_heldout_success_gain": value["gain"]}
                break
    if card == "M02":
        index = truth["maker"]["decoration"]["choice"]
        naive = prediction["arms"]["release-naive"]["acquired_style"][index]
        aware = prediction["arms"]["selection-aware"]["acquired_style"][index]
        if public["retention"] != "all" and naive < .2 and aware > .5:
            result["selection-looking-personal"] = {"naive_true_routine_probability": naive, "selection_aware_true_routine_probability": aware}
    return result


def first_loss(row, contrasts):
    for contrast in contrasts:
        spec = contrast["estimand"]
        left, right = [row["arms"][name]["outcomes"][spec["target"]] for name in [spec["arm"], spec["rival"]]]
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("catalogue source contains a nonfinite comparison")
        if left <= right:
            return {"estimand": spec, "arm_value": left, "rival_value": right, "paired_difference": left-right}
    return None


def check_fields(saved, calculated):
    for key, value in calculated.items():
        if saved.get(key) != value:
            raise ValueError("independent catalogue case measurement differs: "+key)


def check_files(base, mapping):
    for name, expected in mapping.items():
        path = (base/name).resolve()
        if not path.is_relative_to(base.resolve()) or file_digest(path) != expected:
            raise ValueError("catalogue replay evidence missing or changed")


def run(root):
    base = root/"case-catalogue-2"
    packet = read(root/"packets/case-catalogue-2.json")
    spec = packet["identity"]["design"]
    completion = read(base/"COMPLETION.json")
    selection = read(base/"SELECTION.json")
    if completion["execution_state"] != "completed" or completion["instrument_state"] != "valid" or completion["packet_hash"] != packet["packet_hash"] or completion["selection_sha256"] != file_digest(base/"SELECTION.json"):
        raise ValueError("catalogue audit requires completed unchanged evidence")
    sources = {}
    for name, expected in spec["source_hashes"].items():
        path = root/name
        if file_digest(path) != expected:
            raise ValueError("frozen catalogue source changed")
        if path.name not in {"SUMMARY.json", "AGGREGATE.json"}:
            continue
        summary = read(path)
        card = summary["card_id"]
        original_packet = read(root/"packets"/(name.split("/")[0]+".json"))
        design = registered_designs(original_packet["identity"]["design"])[card]
        if card in sources or digest(design) != spec["source_design_hashes"][card]:
            raise ValueError("catalogue native design inventory differs")
        sources[card] = (path.parent, summary, design)
    if len(sources) != 30 or spec["candidate_indices_per_condition"] != list(range(8)):
        raise ValueError("independent catalogue search denominator differs")
    selected, hits, losses, examined = {}, {}, {}, 0
    for card, (source, summary, design) in sorted(sources.items()):
        for condition in design["conditions"]:
            for index in range(8):
                path, row = lookup(source, condition["id"], index)
                uid = row["unit_id"]
                public, truth, prediction = [read(source/folder/(uid+".json")) for folder in ["public", "private", "predictions"]]
                flags = predicates(row, public, truth, prediction)
                new = {name: value for name, value in flags.items() if name not in hits}
                loss = first_loss(row, summary["conditions"][condition["id"]]["contrasts"])
                source_name = path.relative_to(root).as_posix()
                if new or (loss is not None and card not in losses):
                    selected.setdefault(source_name, {"lenses": {}, "failure": None})["lenses"].update(new)
                    hits.update({name: source_name for name in new})
                    if loss is not None and card not in losses:
                        selected[source_name]["failure"] = loss
                        losses[card] = source_name
                examined += 1
    if examined != selection["candidates_examined"] or list(selected) != [case["replay"]["source_unit"] for case in selection["selected"]] or len(selected) > spec["case_cap"] or len(selected) != completion["case_count"]:
        raise ValueError("independent first-match catalogue selection/count differs")
    for name in spec["lens_names"]:
        expected = {"state": "found" if name in hits else "not found in bounded search", "source": hits.get(name)}
        if completion["lens_coverage"][name] != expected or selection["lens_coverage"][name] != expected:
            raise ValueError("independent descriptive lens coverage differs")
    for card in sources:
        expected = {"state": "found" if card in losses else "not found in bounded search", "source": losses.get(card)}
        if completion["failure_coverage"][card] != expected or selection["failure_coverage"][card] != expected:
            raise ValueError("independent descriptive failure coverage differs")
    files_rechecked = 0
    for index, case in enumerate(selection["selected"]):
        key = f"case-{index:03d}"
        original = root/case["replay"]["source_unit"]
        if file_digest(original) != case["replay"]["source_unit_sha256"]:
            raise ValueError("catalogue selected source checksum differs")
        source, row = original.parent.parent, read(original)
        uid = row["unit_id"]
        values = selected[case["replay"]["source_unit"]]
        if set(case["lenses"]) != set(values["lenses"]) or (case["failure"] is None) != (values["failure"] is None):
            raise ValueError("catalogue recorded case predicate differs")
        for name, value in values["lenses"].items():
            check_fields(case["lenses"][name], value)
        if values["failure"] is not None:
            check_fields(case["failure"], values["failure"])
        public, truth, prediction = [read(source/folder/(uid+".json")) for folder in ["public", "private", "predictions"]]
        exported = read(base/"public"/(key+".json"))
        if set(exported) != {"case_id", "public_observation", "pre_reveal_reader_submission", "source_observation_sha256", "access_contract"} or exported["public_observation"] != public or exported["pre_reveal_reader_submission"] != prediction or exported["source_observation_sha256"] != file_digest(source/"public"/(uid+".json")):
            raise ValueError("catalogue public export changed its evidence boundary")
        private = read(base/"private"/(key+"_points.json"))
        if private["actual_truth_and_continuation"] != truth or private["evaluated_arms_and_all_recorded_costs"] != row["arms"] or private["classification"] != case:
            raise ValueError("catalogue evaluator truth/cost/classification join differs")
        check_path = base/"checks"/(key+".json")
        if completion["checks"][index]["check_sha256"] != file_digest(check_path):
            raise ValueError("catalogue whole-unit replay receipt changed")
        check = read(check_path)
        check_files(source, check["source_files"])
        check_files(base/"private/replay"/key, check["replay_files"])
        files_rechecked += len(check["source_files"])+len(check["replay_files"])
    if selection["new_explanatory_case_count"] != 0:
        raise ValueError("descriptive catalogue cannot inflate independent search discoveries")
    return {"instrument_state": "valid", "source_cards": len(sources), "bounded_candidates": examined,
        "selected_cases": len(selected), "found_lenses": sorted(hits), "missing_lenses_in_bounded_search": sorted(set(spec["lens_names"])-set(hits)),
        "cards_with_descriptive_failure_or_tie": sorted(losses), "source_and_replay_files_rechecked": files_rechecked,
        "source_completion_sha256": file_digest(base/"COMPLETION.json"), "selection_sha256": file_digest(base/"SELECTION.json"),
        "scope": "Independent descriptive predicate/selection/export/cost/replay-identity recount; no population or additional adaptive-search claims"}
