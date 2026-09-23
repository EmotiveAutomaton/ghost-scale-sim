"""Independent marginal reconstruction; complete review handler remains unimplemented.

Integer memberships and accurate scalar sums replace the producer tensor axes.
Array gathering avoids Python generator iteration; math.fsum still sums every
scalar in each independently enumerated bucket, without approximation.
"""
import math
import numpy as np
from .joint_factorization_review import N, close

MEMBERS = np.array([(k//1944, k//648%3, k//216%3,
                     k//36%6, k//6%6, k%6) for k in range(N)])
PAIR_MEMBERS = MEMBERS[:, :3]*6 + MEMBERS[:, 3:]
BUCKETS = tuple(tuple(np.flatnonzero(MEMBERS[:, a] == v) for v in range(size))
                for a, size in enumerate((3, 3, 3, 6, 6, 6)))
PAIR_BUCKETS = tuple(tuple(np.flatnonzero(PAIR_MEMBERS[:, a] == v) for v in range(18))
                     for a in range(3))


def marginals(q):
    q = np.asarray(q)
    if q.shape != (N,) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('distribution')
    close(math.fsum(q.tolist()), 1., 1e-12)
    singles = [np.array([math.fsum(q[bucket].tolist()) for bucket in axis]) for axis in BUCKETS]
    pairs = np.array([[math.fsum(q[bucket].tolist()) for bucket in axis] for axis in PAIR_BUCKETS])
    return pairs, np.array(singles[:3]), np.array(singles[3:])


def reconstruct(q, saved_steps, saved_goals, saved_operations):
    steps, goals, operations = marginals(q)
    error = max(close(steps, saved_steps, 1e-12), close(goals, saved_goals, 1e-12),
                close(operations, saved_operations, 1e-12))
    def products(p, g, o):
        pair = np.ones(N); single = np.ones(N)
        for t in range(3):
            pair *= p[t, PAIR_MEMBERS[:, t]]
            single *= g[t, MEMBERS[:, t]] * o[t, MEMBERS[:, 3+t]]
        return pair, single
    expected = products(steps, goals, operations)
    scored = products(np.asarray(saved_steps), np.asarray(saved_goals), np.asarray(saved_operations))
    for a, b in zip(expected, scored):
        error = max(error, close(a, b, 1e-12))
        close(math.fsum(b.tolist()), 1., 6e-12)
    return scored[0], scored[1], error
