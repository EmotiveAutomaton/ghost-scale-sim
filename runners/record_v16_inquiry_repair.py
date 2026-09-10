"""Retain the measured zero-boundary extension of the first numerical repair."""
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import write, canonical, file_digest, now
from ghostscale.validation.soundingline.v16.inquiry import choose as original
from ghostscale.validation.soundingline.v16.inquiry_stable import choose as amended


def main():
    root = REPO/"results/v16/repairs/inquiry-ties-1"
    public = {"schema_version": "v16.inquiry.1", "task_id": "zero-boundary-fixture",
        "beliefs": [], "offers": [0, 1], "histories": [[], []], "familiarity": [4, 0],
        "pending_commands": [[], []], "feedback_batches": [1, 1], "future_weights": [0.5, 0.5],
        "opportunity_cost": 0.0, "remaining_interactions": 8, "tie_draw": 0.25,
        "committed_domain": None,
        "offer_probabilities": [[0.1, 0.2, 0.6, 0.1], [0.4, 0.3, 0.2, 0.1]]}
    rows = []
    for mapping in range(24):
        for noise in (0.01, 0.1, 0.5):
            belief = [0.0]*25
            belief[mapping], belief[24] = 1-noise, noise
            public["beliefs"] = [belief, belief]
            for batch in (1, 2):
                public["feedback_batches"] = [batch, batch]
                old, new = original(canonical(public), "value-learning"), amended(canonical(public), "value-learning")
                rows.append({"mapping": mapping, "noise": noise, "batch": batch, "original": old, "amended": new})
    digest = write(root/"private/ZERO_BOUNDARY_points.json", rows)
    write(root/"ZERO_BOUNDARY.json", {"recorded_at": now(), "repair_id": "inquiry-ties-1",
        "relation_to_plan": "same strict-comparison rounding cause, found before the first bounded repair was finalized",
        "known_answer": "a mixture of one known bijection and uniform noise has the same best construction under every query answer; learning the mixture weight cannot improve expected construction",
        "n_fixture_cases": len(rows),
        "original_spurious_queries": sum(row["original"]["domain"] is not None for row in rows),
        "amended_spurious_queries": sum(row["amended"]["domain"] is not None for row in rows),
        "raw_sha256": digest, "correction": "the already declared binary64 tie resolution also applies at zero-value abstention",
        "numerical_resolution": 2.0**-46, "probabilities_and_cost_formulas_changed": False,
        "scientific_criterion_changed": False, "unknown_scout_means_inspected": False,
        "new_scientific_packet_started": False,
        "preserved_failure_hashes": {str(path.relative_to(root)).replace("\\", "/"): file_digest(path)
            for path in [root/"attempt-1.xml", root/"private/inquiry_stable_before_zero_boundary.py",
                         root/"private/reference_before_zero_boundary.py", root/"private/test_stable_before_zero_boundary.py"]}})
    print({"fixture_cases": len(rows), "original_spurious_queries": sum(row["original"]["domain"] is not None for row in rows),
           "amended_spurious_queries": sum(row["amended"]["domain"] is not None for row in rows)})


if __name__ == "__main__":
    main()
