"""Determinism battery for card I12: build a fixed set of objects in every lane and hash their
scientific fields. Run in two fresh clones and in two process orders; the hashes must agree.

    python -m ghostscale.validation.soundingline.v13.determinism --order forward|reverse
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import common as C
from . import exact as X, priors as P, world as W, costs as CO, hierarchy as H, goals_trust as GT


def battery(order: str = "forward") -> dict:
    lanes = [("discovery", 0), ("discovery", 5), ("confirmation", 1000), ("transfer", 2000), ("pilot", 9000)]
    if order == "reverse":
        lanes = lanes[::-1]
    out = {}
    for lane, wid in lanes:
        w = W.make_world(wid, lane, rng=np.random.default_rng(C.world_seed(lane, wid) + 1))
        r = np.random.default_rng(C.seed(f"det|{lane}|{wid}"))
        makers = W.population(w, 12, r)
        rd = makers[0]
        model = X.reader_model(w, rd, families=[rd.family])
        arts = W.stream(w, makers[1] if makers[1].family == rd.family else [m for m in makers if m.family == rd.family][1], 0, r, 4)
        post = model.posterior(X.uniform_prior(model), arts, ("surface", "goal_consequences"))
        sm = P.measure_self(w, rd, model, np.random.default_rng(C.seed(f"det|self|{lane}|{wid}")))
        fam = w.family(rd.family)
        actor = CO.Actor(fam.grid[1], motivation=1.2)
        recs = CO.stream(actor, np.random.default_rng(C.seed(f"det|cost|{lane}|{wid}")), 6, fam.ng)
        team = H.make_team(w, np.random.default_rng(C.seed(f"det|team|{lane}|{wid}")), "central", n_subs=3, family=rd.family)
        prod = H.produce_team(w, "central", team, np.random.default_rng(C.seed(f"det|prod|{lane}|{wid}")), n_parts=6, steps=8)
        d0, d1 = GT.kind_dists(np.random.default_rng(C.seed(f"det|kinds|{lane}|{wid}")))
        src = GT.Source("s", "persuasion", {0: 0.5})
        spoken = [a for a in (GT.speak(src, np.random.default_rng(C.seed(f"det|speak|{lane}|{wid}|{i}")), d0, d1, 6, t=i) for i in range(6)) if a]
        out[f"{lane}:{wid}"] = {
            "family_sig": C.obj_sha([f.sig for f in w.families]), "maker_w": C.obj_sha([m.w for m in makers]),
            "artifacts": C.obj_sha([a["features"] for a in arts]), "posterior": C.obj_sha(post),
            "self_w_hat": C.obj_sha(sm["w_hat"]), "choices": C.obj_sha([rr["choice"] for rr in recs]),
            "team_final_goals": C.obj_sha(prod["final_goals"]), "spoken": C.obj_sha([a["tokens"] for a in spoken]),
            "seed_world": C.world_seed(lane, wid)}
    ids = {lane: C.lane_ids(lane, {"discovery_worlds": 512, "transfer_worlds": 128, "confirmation_worlds": 96, "pilot_worlds": 4}) for lane in ("discovery", "transfer", "confirmation", "pilot")}
    out["lineage_disjoint"] = C.lineage_disjoint(ids)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="forward")
    a = ap.parse_args()
    json.dump(battery(a.order), sys.stdout, indent=1, sort_keys=True)
