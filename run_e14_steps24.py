#!/usr/bin/env python
"""E14 re-run at infer_steps = 24. A DIAGNOSTIC, and explicitly not a reportable configuration.

WHY IT IS RUN. The single-generation measurement says the per-generation contraction scales as
1/inference: 1 - r = 0.147 / 0.140 / 0.078 / 0.035 / 0.014 at 2 / 4 / 6 / 12 / 24 steps. V3 ran
at 6. If the CHAIN follows that measurement, the f=0 leak should fall from ~ +0.0055 toward
~ +0.001. If it does not, something beyond early-step misattribution is contributing to the
generational contraction, and that is worth knowing before any estimator work is trusted.

WHY ITS RESULT MAY NOT BE USED TO REPORT E8, even if N11 passes on it. Raising infer_steps does
not remove the bias; it divides it by the number of steps. The obvious referee question is why
24 and not 12, and the honest answer would be "because that is where it dropped under our
gate" — which is choosing the operating point to clear one's own threshold. That is precisely
the failure decision D1 was written to prevent, in a new costume.

    A number that passes a gate only at a hand-picked setting of a free parameter is a tuned
    result, not a result.

The reportable route is to fix the ESTIMATOR so the bias is absent at any ``infer_steps`` — see
``ghostscale/learning.py``, deferred commitment. This run exists to check the mechanism, and to
bound how much of the contraction early-step misattribution actually accounts for.

Writes to ``results_e14_steps24/`` so it cannot be mistaken for, or overwrite, the reportable
E14 cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ghostscale.config import load_config
from ghostscale.experiments import e14_engagement_floor as E14, _common as C

OUT = Path(__file__).resolve().parent / "results_e14_steps24"
OVERRIDES = {"experiments.e14.infer_steps": 24}
# Pre-registered before the run, from the single-generation scaling measurement.
PREDICTION = {
    "single_generation_contraction_at_6_steps": 0.078,
    "single_generation_contraction_at_24_steps": 0.014,
    "expected_factor": 5.7,
    "free_arm_leak_slope_at_6_steps": 0.00548,
    "forced_arm_leak_slope_at_6_steps": 0.00724,
    "predicted_leak_slope_at_24_steps_if_chain_follows": 0.0010,
    "n11_slope_ceiling": 0.001,
    "if_chain_does_not_follow": (
        "early-step misattribution is not the whole of the generational contraction; a second "
        "contributor exists and must be found before the estimator fix is credited with "
        "anything"),
}


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else C.default_workers()
    cfg = load_config(overrides=OVERRIDES)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prediction.json").write_text(json.dumps(PREDICTION, indent=2), encoding="utf-8")

    E14.run(cfg, out_dir=OUT, workers=workers)
    v = json.loads((OUT / "e14_verdict.json").read_text(encoding="utf-8"))

    print("\nE14 @ infer_steps=24 (DIAGNOSTIC — not a reportable configuration):")
    for arm, a in v["arms"].items():
        s = a["leak_slope"]
        print(f"  {arm:7s} DEEP {a['deep_per_genuine']:.2f}/24  "
              f"leak slope {s['slope']:+.5f} (t {s['t']:+.2f})   "
              f"column KL {a['creator_col_kl_mean']:.4f}  "
              f"gen0 {a['creator_col_kl_gen0']:.4f}")
    print(f"\n  at infer_steps=6 the same arms gave "
          f"free {PREDICTION['free_arm_leak_slope_at_6_steps']:+.5f}, "
          f"forced {PREDICTION['forced_arm_leak_slope_at_6_steps']:+.5f}")
    print(f"  predicted if the chain follows the single-generation scaling: "
          f"~{PREDICTION['predicted_leak_slope_at_24_steps_if_chain_follows']:+.5f}")
    print("\n  REMINDER: passing N11 here does NOT make E8 reportable. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
