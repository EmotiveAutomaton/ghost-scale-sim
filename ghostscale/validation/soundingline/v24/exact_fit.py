"""Exact grouped objective for the pinned certified choice fit.

This is a Ghost engineering adapter, not a new canonical engine version. Only
the two arithmetic closures in _fit are replaced in a private module. The
optimizer, Newton finish, acceptance certificate, chronology and model remain
the pinned implementation. No source file or already-imported module is edited.
"""
import ast
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax


def grouped_objective(rows, center, penalty, guard=lambda: None):
    """Group by option count without padding, dropping or reweighting choices."""
    center = np.asarray(center, dtype=float)
    if center.ndim != 1 or not center.size or not np.isfinite(center).all():
        raise ValueError("finite nonempty center required")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("positive finite ridge required")
    groups = {}
    for row in rows:
        x = np.asarray(row.features, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(center) or not len(x)
                or not np.isfinite(x).all() or not 0 <= row.taken < len(x)
                or not np.isfinite(row.weight) or not 0 <= row.weight <= 1):
            raise ValueError("invalid choice")
        groups.setdefault(len(x), []).append(row)
    packed = []
    for members in groups.values():
        x = np.stack([r.features for r in members])
        chosen = np.asarray([r.taken for r in members], dtype=int)
        weights = np.asarray([r.weight for r in members])
        packed.append((x, x[np.arange(len(x)), chosen], chosen, weights))

    def objective(offset):
        guard()
        beta = center + offset
        loss = penalty * (offset @ offset)
        gradient = 2 * penalty * offset.copy()
        for x, chosen_x, chosen, weights in packed:
            logits = x @ beta
            p = softmax(logits, axis=1)
            loss += weights @ (logsumexp(logits, axis=1) - logits[np.arange(len(x)), chosen])
            mean = np.einsum("nk,nkd->nd", p, x)
            gradient += weights @ (mean - chosen_x)
        return float(loss), gradient

    def hessian(offset):
        guard()
        matrix = 2 * penalty * np.eye(len(center))
        for x, _, _, weights in packed:
            p = softmax(x @ (center + offset), axis=1)
            centered = x - np.einsum("nk,nkd->nd", p, x)[:, None, :]
            # A matrix product uses the exact categorical covariance identity.
            flat = centered.reshape(-1, len(center))
            mass = (weights[:, None] * p).ravel()
            matrix += (flat.T * mass) @ flat
        return matrix

    return objective, hessian


def load_certified(source, expected_sha256, *, accelerated, guard=lambda: None):
    """Load pinned code privately; return explicit adaptation provenance."""
    source = Path(source)
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("certified source byte pin mismatch")
    tree = ast.parse(raw)
    fits = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fit"]
    if len(fits) != 1:
        raise ValueError("unknown certified source")
    closures = [n.name for n in fits[0].body if isinstance(n, ast.FunctionDef)]
    if closures != ["objective", "hessian"]:
        raise ValueError("unknown arithmetic closures")
    if accelerated:
        replacement = ast.parse(
            "objective, hessian = _ghost_grouped_objective(rows, center, penalty, _ghost_guard)"
        ).body[0]
        body = []
        for node in fits[0].body:
            if isinstance(node, ast.FunctionDef) and node.name == "objective":
                body.append(replacement)
            elif isinstance(node, ast.FunctionDef) and node.name == "hessian":
                continue
            else:
                body.append(node)
        fits[0].body = body
    tree = ast.fix_missing_locations(tree)
    # Dataclasses look up their module during compilation. Each load is isolated.
    name = "_ghost_v24_certified_" + actual[:16] + ("_grouped" if accelerated else "_scalar")
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module._ghost_grouped_objective = grouped_objective
    module._ghost_guard = guard
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    module.ghost_provenance = dict(
        upstream_sha256=actual, accelerated=bool(accelerated),
        canonical_engine_unchanged=True, canonical_engine_admission=False,
        replacement="grouped objective and Hessian" if accelerated else "none",
        transformed_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
    )
    return module
