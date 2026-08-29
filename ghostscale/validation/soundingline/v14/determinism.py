"""Determinism battery for card I02: build a fixed set of objects in every lane and hash their
scientific fields. Run in two fresh clones and in two process orders; the hashes must agree.

    python -m ghostscale.validation.soundingline.v14.determinism --order forward|reverse
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import common as C
from . import communication as CM
from . import foraging as F
from . import hierarchy as H
from . import joint as J
from . import world as W
from .cards import world_for


def battery(order: str = "forward") -> dict:
    lanes = [("discovery", 0), ("discovery", 5), ("confirmation", 1000), ("transfer", 2000), ("pilot", 9000)]
    if order == "reverse":
        lanes = lanes[::-1]
    out = {}
    for lane, wid in lanes:
        w = world_for({"wid": wid, "lane": lane, "cfg": {}, "smoke": False})
        r = np.random.default_rng(C.seed(f"det|{lane}|{wid}"))
        makers = [W.make_maker(w, f"m{i}", r, family=0, competence="mid") for i in range(6)]
        eps = [W.episode(w, makers[0], r, index=k) for k in range(3)]
        rd = J.Reader(w, 0, 0.75, 0.8)
        post = J.joint(J.uniform_prior(), rd.route_tables(eps, ("action", "semantic", "context")))
        src = CM.source(np.random.default_rng(C.seed(f"det|src|{lane}|{wid}")), "honest_warning")
        arts = [CM.speak(src, np.random.default_rng(C.seed(f"det|speak|{lane}|{wid}|{i}"))) for i in range(3)]
        cpost = CM.posterior(sum(CM.loglik_artifact(a) for a in arts), CM.region_prior())
        hr = np.random.default_rng(C.seed(f"det|h|{lane}|{wid}"))
        reward, potential = hr.normal(0, 1, H.N_PRIM), hr.normal(0, 1, H.N_PRIM)
        pol = H.policy_from_reward(reward)
        pp, ps = H.resolving_intervention(reward, potential, hr)
        items = [F.make_item(np.random.default_rng(C.seed(f"det|item|{lane}|{wid}|{k}")), k) for k in ("structured_learnable", "unlearnable_noise", "novel_explained")]
        out[f"{lane}:{wid}"] = {
            "plan_tables": C.obj_sha([f.plan for f in w.families]), "vocab": C.obj_sha([f.vocab for f in w.families]),
            "makers": C.obj_sha([(m.plan, m.pref, m.k_exec, m.k_obs, m.h_feat, m.h_strength) for m in makers]),
            "episodes": C.obj_sha([(e["action"], e["surface"], e["semantic"], e["goal"]) for e in eps]),
            "posterior": C.obj_sha(post), "artifacts": C.obj_sha([(a["assertion"], a["evidence"], a["intensity"]) for a in arts]),
            "source_posterior": C.obj_sha(cpost), "policies": C.obj_sha([pol, pp, ps]), "items": C.obj_sha([it["p"] for it in items]),
            "seed_world": C.world_seed(lane, wid)}
    ids = {lane: C.lane_ids(lane, {"discovery_worlds": 256, "transfer_worlds": 128, "confirmation_worlds": 128, "pilot_worlds": 4}) for lane in ("discovery", "transfer", "confirmation", "pilot")}
    out["lineage_disjoint"] = C.lineage_disjoint(ids)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="forward")
    a = ap.parse_args()
    json.dump(battery(a.order), sys.stdout, indent=1, sort_keys=True)
