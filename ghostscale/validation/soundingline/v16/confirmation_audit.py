"""Independent confirmation arithmetic: analytic mean-test inversion and SciPy tails.

No confirmation reducer, grouping helper, binomial routine or Holm helper is
imported. Native physical reconstruction is checked by the separate packet audit.
"""
import math
import statistics
from collections import Counter
from scipy.stats import beta, binom
from .records import read, file_digest


def same(actual, expected):
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("independent confirmation vector differs")
        for left, right in zip(actual, expected):
            same(left, right)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError("independent confirmation numerical result differs")
    elif actual != expected:
        raise ValueError("independent confirmation result differs")


def primary(claim, rows, reported):
    n = claim["n_constructor_packets"]
    conditions = claim["condition_ids"]
    histories = 2 if claim["card_id"] == "M01" else len(conditions)
    if n < 2 or n*histories > 4096 or claim["total_fresh_acquisition_histories"] != n*histories:
        raise ValueError("independent confirmation history count differs")
    if len(rows) != n*len(conditions) or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("independent confirmation raw denominator differs")
    groups = {}
    for row in rows:
        if row["lineage"] != claim["namespace"] or row["packet_hash"] != claim["execution_packet_hash"] or row["card_id"] != claim["card_id"]:
            raise ValueError("independent confirmation source identity differs")
        index = row["seed_components"]["index"]
        if row["seed_components"]["constructors"] != n or index not in range(n):
            raise ValueError("independent confirmation constructor allocation differs")
        groups.setdefault(index, []).append(row)
    if set(groups) != set(range(n)):
        raise ValueError("independent confirmation seed coverage differs")
    values, constructors = [], []
    for index in range(n):
        group = groups[index]
        if Counter(row["condition"] for row in group) != Counter(conditions) or len({row["constructor_id"] for row in group}) != 1:
            raise ValueError("independent confirmation context pairing differs")
        constructors.append(group[0]["constructor_id"])
        by_condition = {row["condition"]: row for row in group}
        values.append([by_condition[condition]["arms"][spec["arm"]]["outcomes"][spec["target"]]
            - by_condition[condition]["arms"][spec["rival"]]["outcomes"][spec["target"]]
            for condition in conditions for spec in claim["estimands"]])
    if len(set(constructors)) != n:
        raise ValueError("independent confirmation constructors repeat")
    alpha = claim["alpha_planning"]
    expected = {"n_independent_constructors": n}
    if claim["kind"] == "equivalence":
        bound, tolerance = claim["external_difference_bound"], 1e-10
        if any(not math.isfinite(x) or abs(x) > bound for vector in values for x in vector):
            raise ValueError("independent equivalence range differs")
        count = sum(any(abs(x) > tolerance for x in vector) for vector in values)
        q = (claim["margin"]-tolerance)/bound
        p = float(binom.cdf(count, n, q))
        upper = 1. if count == n else float(beta.ppf(1-alpha, count+1, n-count))
        expected.update(n_independent_maker_packets=n, paired_components=len(values[0]),
            disagreement_events=count, exact_one_sided_p=p, event_probability_upper=upper,
            simultaneous_absolute_mean_bound=tolerance+bound*upper,
            paired_means=[statistics.fmean(vector[j] for vector in values) for j in range(len(values[0]))])
    elif claim["kind"] == "capability":
        if any(len(vector) != 1 or not math.isfinite(vector[0]) or not -1 <= vector[0] <= 1 for vector in values):
            raise ValueError("independent bounded primary layout or range differs")
        numbers = [vector[0] for vector in values]
        mean, variance = statistics.fmean(numbers), statistics.variance(numbers)
        null = claim["estimands"][0]["practical_bar"]
        log = math.log(2/alpha)
        lower = max(-1., mean-math.sqrt(2*variance*log/n)-14*log/(3*(n-1)))
        # Solve c*s^2+a*s=(mean-null), where s=sqrt(log(2/p)).
        # This analytically inverts the bound; the original reducer bisects p.
        difference, a, c = mean-null, math.sqrt(2*variance/n), 14/(3*(n-1))
        s = 0. if difference <= 0 else 2*difference/(math.sqrt(a*a+4*c*difference)+a)
        p = min(1., max(1e-15, 2*math.exp(-s*s)))
        expected.update(n_independent_makers=n, paired_mean=mean, paired_sample_variance=variance,
            null_mean=null, one_sided_lower_bound=lower, valid_one_sided_p=p,
            criterion_state="held" if lower > null else "not established")
    else:
        raise ValueError("unknown independent confirmation estimand")
    for key, value in expected.items():
        same(reported["primary_analysis"][key], value)
    same(reported["primary_p"], p)
    same(reported["n_independent_constructor_packets"], n)
    same(reported["n_native_condition_records"], len(rows))
    same(reported["total_fresh_acquisition_histories"], n*histories)
    return {"instrument_state": "valid", "independent_primary_p": p, "recalculated": expected,
        "constructor_packets": n, "condition_records": len(rows), "primary_histories": n*histories}


def run(root):
    packet = read(root/"packets/confirmation-1.json")
    completed = read(root/"confirmation/COMPLETION.json")
    if completed.get("execution_state") != "completed" or completed.get("instrument_state") != "valid" or completed["packet_hash"] != packet["packet_hash"]:
        raise ValueError("independent primary audit requires completed unchanged confirmation")
    reports, probabilities = {}, []
    for entry in packet["identity"]["design"]["cards"]:
        claim = {**entry["claim"], "execution_packet_hash": packet["packet_hash"]}
        base = root/"confirmation"/claim["card_id"]
        rows = [read(path) for path in sorted((base/"units").glob("*_points.json"))]
        reported = read(base/"PRIMARY.json")
        control = root/"confirmation-controls"/claim["card_id"]/"COMPLETION.json"
        if reported["actual_source_control_sha256"] != file_digest(control) or read(control)["instrument_state"] != "valid":
            raise ValueError("independent primary lacks its actual source controls")
        result = primary(claim, rows, reported)
        result["primary_sha256"] = file_digest(base/"PRIMARY.json")
        reports[claim["claim_id"]] = result
        probabilities.append((result["independent_primary_p"], claim["claim_id"]))
    ledger = read(root/"confirmation/MULTIPLICITY.json")
    if ledger["frozen_primary_count"] != len(reports) or len(ledger["claims"]) != len(reports) or len(reports) > 3:
        raise ValueError("independent multiplicity count differs")
    previous = 0.
    for index, (p, claim_id) in enumerate(sorted(probabilities)):
        record = ledger["claims"][index]
        if record["claim_id"] != claim_id:
            raise ValueError("independent Holm order differs")
        adjusted = min(1., max(previous, (len(probabilities)-index)*p))
        same(record["p"], p)
        same(record["holm_adjusted_p"], adjusted)
        same(record["familywise_rejected"], adjusted <= .05)
        previous = adjusted
    same(completed["claims"], ledger["claims"])
    same(completed["condition_records"], sum(row["condition_records"] for row in reports.values()))
    return {"instrument_state": "valid", "claims": reports, "multiplicity_sha256": file_digest(root/"confirmation/MULTIPLICITY.json"),
        "primary_count": len(reports), "empty_selection": not reports,
        "method": "Independent constructor grouping; analytic bounded-test inversion; SciPy binomial/beta calculations; independent Holm arithmetic"}
