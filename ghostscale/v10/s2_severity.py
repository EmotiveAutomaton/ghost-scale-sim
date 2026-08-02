"""S-2 — the severity pass on V10's headlines. Not optional, and here is why.

V9's minimal-model programme established that EVERY finding in this project rests on one structural
commitment: the reader modelling a maker. Which means any new finding produced by looking at this
model and asking "what about--" will also rest on that commitment, and will therefore reproduce.
Not because it is true, but because it is what the architecture does.

V10's findings were generated exactly that way -- by the author and the model in conversation. So
the severity check is the thing standing between "we kept finding things" and "we kept finding
things and here is how much of each was ever ours."

THE PROCEDURE, unchanged from V8's S-1. Keep the model's SHAPE. Throw its SETTINGS away and redraw
them at random -- which features mean which goals, what the synthetic distribution looks like, which
goals share values. Then count how often the finding still appears.

A high rate is not a refutation. It says the result comes from the structural commitment, which IS
the theory but is the part shared with any account built the same way. A low rate says the theory's
SPECIFIC commitments are doing the work. Both go in the table; only one of them is distinguishing
evidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config
from . import SEED_OFFSET, v10_dir


def _draw_cfg(cfg: Config, draw: int) -> Config:
    """One randomly reparameterised world of the same shape."""
    c = cfg.copy()
    c.set("inference.exact", True)
    c.set("artifact_model.noise_free_synth_seed", int(1000 + draw * 17))
    c.set("v4.foreign.draw_seed", int(2000 + draw * 29))
    return c


def _severity_e55(cfg: Config, n_draws: int, n_artifacts: int, n_learners: int,
                  infer_steps: int) -> dict:
    """Does 'intent-gating beats surface filtering on disguised grooming' survive redrawing?"""
    from . import e55_intent_gate as E55

    reproduced, rows = 0, []
    for d in range(int(n_draws)):
        try:
            v = E55.run(_draw_cfg(cfg, d), n_artifacts=n_artifacts, n_learners=n_learners,
                        infer_steps=infer_steps)
            g = pd.DataFrame(v["grid"])

            def _c(reader, col):
                r = g[(g.corpus == "disguised") & (g.reader == reader)]
                return float(r[col].iloc[0]) if len(r) else float("nan")

            intent = _c("intent_reconstructibility", "human_model_corrupted")
            surface = _c("surface_filter", "human_model_corrupted")
            none_ = _c("no_filter", "human_model_corrupted")
            hit = bool(np.isfinite(intent) and np.isfinite(surface)
                       and intent < surface and intent < none_)
            reproduced += int(hit)
            rows.append({"draw": d, "intent": intent, "surface": surface,
                         "no_filter": none_, "reproduced": int(hit)})
        except Exception as exc:                        # noqa: BLE001
            rows.append({"draw": d, "failed": repr(exc)})
    ok = [r for r in rows if "failed" not in r]
    return {"finding": "intent-gating beats surface filtering on disguised grooming",
            "draws": len(ok), "reproduced": reproduced,
            "false_positive_rate": float(reproduced / len(ok)) if ok else float("nan"),
            "rows": rows}


def _severity_e56(cfg: Config, n_draws: int, n_obs: int) -> dict:
    """Does 'the gate blocks purpose and passes method' survive redrawing?"""
    from . import e56_selective_gate as E56

    reproduced, rows = 0, []
    for d in range(int(n_draws)):
        try:
            v = E56.run(_draw_cfg(cfg, d), n_obs=n_obs, n_timesteps=16)
            r = v["H10.5"]["adversarial_over_sympathetic"]
            hit = bool(np.isfinite(r["process"]) and np.isfinite(r["goal"])
                       and r["process"] > r["goal"])
            reproduced += int(hit)
            rows.append({"draw": d, "process": r["process"], "goal": r["goal"],
                         "value": r["value"], "reproduced": int(hit)})
        except Exception as exc:                        # noqa: BLE001
            rows.append({"draw": d, "failed": repr(exc)})
    ok = [r for r in rows if "failed" not in r]
    return {"finding": "adversarial reading blocks purpose and passes method",
            "draws": len(ok), "reproduced": reproduced,
            "false_positive_rate": float(reproduced / len(ok)) if ok else float("nan"),
            "rows": rows}


def _severity_e57(cfg: Config, n_draws: int, n_obs: int) -> dict:
    """Does 'the stale detector fails asymmetrically' survive redrawing?"""
    from . import e57_arms_race as E57

    reproduced, rows = 0, []
    for d in range(int(n_draws)):
        try:
            v = E57.run(_draw_cfg(cfg, d), n_obs=n_obs, n_timesteps=16)
            h = v["H10.8"]
            hit = bool(abs(h["hit_rate_change"]) > 2.0 * abs(h["false_alarm_change"])
                       if abs(h["false_alarm_change"]) > 1e-9
                       else abs(h["hit_rate_change"]) > 0.05)
            reproduced += int(hit)
            rows.append({"draw": d, "d_hit": h["hit_rate_change"],
                         "d_fa": h["false_alarm_change"], "reproduced": int(hit)})
        except Exception as exc:                        # noqa: BLE001
            rows.append({"draw": d, "failed": repr(exc)})
    ok = [r for r in rows if "failed" not in r]
    return {"finding": "a stale detector fails asymmetrically",
            "draws": len(ok), "reproduced": reproduced,
            "false_positive_rate": float(reproduced / len(ok)) if ok else float("nan"),
            "rows": rows}


def run(cfg: Config, n_draws: int = 12, n_artifacts: int = 24, n_learners: int = 1,
        infer_steps: int = 6, n_obs: int = 16, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    rates = {
        "E55 intent beats surface on disguised grooming":
            _severity_e55(cfg, n_draws, n_artifacts, n_learners, infer_steps),
        "E56 the gate blocks purpose and passes method":
            _severity_e56(cfg, n_draws, n_obs),
        "E57 the stale detector fails asymmetrically":
            _severity_e57(cfg, n_draws, n_obs),
    }

    verdict = {
        "check": "S-2, the severity pass on V10",
        "why": ("V9 established that every finding in this project rests on the reader modelling "
                "a maker, so a finding generated by looking at this model will reproduce whether "
                "or not it is true. This is what separates the two cases."),
        "how_to_read": (
            "a HIGH rate means the finding comes from the structural commitment -- which is the "
            "theory, but the part shared with any account built the same way, so it does not "
            "distinguish this framework from a competitor. A LOW rate means the theory's specific "
            "commitments are doing the work. Neither throws a result out; both change the sentence "
            "you may write after it."),
        "rates": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                  for k, v in rates.items()},
        "detail": rates,
        "n_draws": int(n_draws),
    }
    (v10_dir() / "s2_severity.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
