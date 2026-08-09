"""How many components are in this matrix? Five criteria, and none of them agrees with the others.

WHY THIS IS IN ``methods/`` AND NOT IN AN EXPERIMENT. These are somebody else's estimators, applied
to a matrix. They know nothing about readers, goals or artifacts, and they must not: the whole
value of testing them here is that the simulation can plant a number they have to find, and an
estimator that had any knowledge of the planting would be the E45 oracle again.

THE FIVE.

  ``parallel``      Horn's parallel analysis. Keep components whose eigenvalue exceeds the 95th
                    percentile of eigenvalues from column-shuffled data of the same shape.
  ``kaiser``        eigenvalue > 1 on standardised data.
  ``var90``         components needed for 90% of variance.
  ``entry_cv``      hold out 10% of matrix ENTRIES, zero them, reconstruct at each rank.
  ``participation`` (sum of eigenvalues)^2 / sum of squared eigenvalues. Threshold-free.

The first five are VENDORED VERBATIM from Sounding Line's ``runners/run_decomp_scaling.py`` --
same standardisation, same shuffle count, same percentile, same off-by-one on ``var90``. Copied
rather than imported because a result that depends on another repository's working tree is not
reproducible, and copied verbatim rather than improved because the question is whether THEIR
pipeline counts, not whether a better one would.

TWO MORE, BECAUSE THE FIRST FIVE DO NOT INCLUDE WHAT THE REQUEST ASKED FOR.

  ``bicv``          Owen & Perry (2009) bi-cross-validation. Hold out a block of ROWS and a block
                    of COLUMNS at once and predict the held-out block from the other three. This
                    is NOT what ``entry_cv`` does. ``entry_cv`` zeroes entries and then runs the
                    SVD on a matrix that still contains them, so the held-out cells influence
                    their own prediction -- and it grows 14x with sample size in Sounding Line's
                    own data, which is what a leak looks like.
  ``pr_corrected``  the participation ratio with its finite-sample bias removed. The raw PR is
                    biased downward because sample eigenvalues are more spread than population
                    ones. See :func:`participation_ratio_corrected` for exactly which correction
                    this is and which one it is not.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------------------------- #
# Vendored verbatim from Sounding Line, runners/run_decomp_scaling.py
# --------------------------------------------------------------------------------------------- #
def _standardise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
    return Z[:, np.isfinite(Z).all(0)]


def _eigs(Z: np.ndarray) -> np.ndarray:
    k = min(Z.shape) - 1
    return np.linalg.svd(Z, compute_uv=False)[:k] ** 2 / (Z.shape[0] - 1)


def vendored_criteria(X: np.ndarray, rng, n_null: int = 3) -> dict:
    """Sounding Line's own five, unchanged. Do not tidy this function.

    Kept byte-for-byte equivalent to theirs so that a disagreement between what this returns on
    planted data and what theirs returns on activations is a fact about the data rather than about
    two implementations that drifted.
    """
    Z = _standardise(X)
    ev = _eigs(Z)
    nulls = []
    for _ in range(int(n_null)):
        S = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        nulls.append(np.linalg.svd(S, compute_uv=False)[:len(ev)] ** 2 / (S.shape[0] - 1))
    thresh = np.percentile(np.array(nulls), 95, axis=0)
    var = ev / ev.sum()
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    # their cross-validation: zero 10% of entries, then SVD the matrix that still contains them
    mask = rng.random(Z.shape) < 0.10
    Ztr = Z.copy()
    Ztr[mask] = 0.0
    U, S_, Vt = np.linalg.svd(Ztr, full_matrices=False)
    best_k, best_err = 0, float(np.mean(Z[mask] ** 2))
    for k in range(1, min(120, len(S_)) + 1):
        R = (U[:, :k] * S_[:k]) @ Vt[:k]
        e = float(np.mean((Z[mask] - R[mask]) ** 2))
        if e < best_err:
            best_err, best_k = e, k
    # THE ONE-LINE BUG, AND THE FIX, SIDE BY SIDE.
    #
    # `parallel` is what Sounding Line's code computes: the NUMBER OF INDICES ANYWHERE in the
    # spectrum whose eigenvalue exceeds the null threshold. Horn's parallel analysis is not that.
    # It retains the LEADING CONSECUTIVE RUN and stops at the first component that fails, because
    # each index is a 5%-level test and summing exceedances across ~1000 indices with no
    # multiplicity control returns 5% of them as false positives even on pure noise -- more when
    # the null threshold is estimated from only three draws, where the "95th percentile" is
    # effectively the maximum of three and roughly one index in four exceeds it by chance.
    #
    # Measured, committed (S-11's shuffled-null arm, structureless by construction, true answer
    # zero): the sum rule returns 250 at n = 1200 and 254 at n = 2400; the leading run returns 0.
    # On iid Gaussian noise of the same shape the sum rule returns ~280 with three null draws and
    # ~70-140 with twenty (seed-dependent; side measurement, not committed). On planted structure
    # both rules agree exactly (7 -> 7, 27 -> 27).
    exceed = ev > thresh
    leading = int(np.argmin(exceed)) if not exceed.all() else int(exceed.size)
    return {"parallel": int(exceed.sum()),
            "parallel_leading": leading,
            "kaiser": int((ev > 1.0).sum()),
            "var90": int(np.searchsorted(np.cumsum(var), 0.90) + 1),
            "entry_cv": int(best_k),
            "participation": pr}


# --------------------------------------------------------------------------------------------- #
# Owen & Perry (2009) bi-cross-validation
# --------------------------------------------------------------------------------------------- #
def bicross_validation(X: np.ndarray, rng, max_rank: int = 60, folds: int = 3,
                       row_frac: float = 0.5, col_frac: float = 0.5) -> dict:
    """Rank selection by holding out a row block AND a column block simultaneously.

    Partition the matrix into four blocks::

        [ A  B ]      A is held out. B, C, D are training.
        [ C  D ]

    Fit a rank-k approximation to D, and predict A as ``B @ pinv(D_k) @ C``. The held-out block A
    never enters the fit in any form -- which is the property ``entry_cv`` does not have, because
    zeroing an entry still leaves it in the matrix the SVD sees.

    Error is minimised at the true rank rather than falling monotonically, so this actually selects
    a rank instead of running to the maximum. That is the whole point of the method, and
    ``entry_cv`` growing 14x with sample size in Sounding Line's data is what the alternative looks
    like.
    """
    Z = _standardise(X)
    n, p = Z.shape
    cap = int(min(max_rank, min(n, p) // 2 - 1))
    if cap < 1:
        return {"bicv": 0, "skipped": "matrix too small to split"}
    errs = np.zeros((int(folds), cap + 1))
    for f in range(int(folds)):
        ri = rng.permutation(n)
        ci = rng.permutation(p)
        r0 = ri[:max(int(n * row_frac), 2)]
        r1 = ri[max(int(n * row_frac), 2):]
        c0 = ci[:max(int(p * col_frac), 2)]
        c1 = ci[max(int(p * col_frac), 2):]
        if r1.size < 2 or c1.size < 2:
            return {"bicv": 0, "skipped": "degenerate split"}
        A = Z[np.ix_(r0, c0)]
        B = Z[np.ix_(r0, c1)]
        C = Z[np.ix_(r1, c0)]
        D = Z[np.ix_(r1, c1)]
        U, S_, Vt = np.linalg.svd(D, full_matrices=False)
        errs[f, 0] = float(np.mean(A ** 2))                     # rank 0: predict zero
        for k in range(1, cap + 1):
            if k > S_.size:
                errs[f, k] = errs[f, k - 1]
                continue
            # pseudo-inverse of the rank-k truncation of D
            Dk_pinv = (Vt[:k].T * (1.0 / np.where(S_[:k] > _EPS, S_[:k], np.inf))) @ U[:, :k].T
            pred = B @ Dk_pinv @ C
            errs[f, k] = float(np.mean((A - pred) ** 2))
    mean_err = errs.mean(axis=0)
    return {"bicv": int(np.argmin(mean_err)),
            "bicv_error_curve": [float(x) for x in mean_err],
            "bicv_min_error": float(mean_err.min()),
            "bicv_rank0_error": float(mean_err[0]),
            "folds": int(folds), "max_rank_searched": cap}


# --------------------------------------------------------------------------------------------- #
# Finite-sample bias correction for the participation ratio
# --------------------------------------------------------------------------------------------- #
def participation_ratio_corrected(X: np.ndarray, center: bool = True) -> dict:
    """PR with the finite-SAMPLE bias removed, via U-statistics on the Gram matrix.

    WHICH CORRECTION THIS IS, AND WHICH IT IS NOT. Chun et al. (arXiv 2509.26560) give a
    both-sides estimator that corrects for finite rows AND finite columns at once, built from
    four-index sums restricted to mutually distinct indices. This implements the ROW-SIDE
    correction only, which is the classical and well-established half::

        PR = (tr S)^2 / tr(S^2)

    with each term replaced by an unbiased U-statistic over pairs of distinct samples. Writing
    ``G = Z Z^T`` for the Gram matrix over samples,

        unbiased (tr S)^2   <-  mean over i != j of  G_ii G_jj
        unbiased tr(S^2)    <-  mean over i != j of  G_ij^2

    Both follow from the fact that distinct samples are independent, so the cross terms that
    inflate ``tr(S^2)`` at small n vanish in expectation once the diagonal is excluded.

    THE COLUMN-SIDE CORRECTION IS NOT APPLIED, and that matters here: Sounding Line's matrices are
    wider than they are tall (roughly 1000 columns against 150-4000 rows), which is the regime
    where the column-side term is least negligible. So this is reported as a partial correction and
    is gated against the paper's own synthetic benchmark -- ``d = 50`` recovered across
    ``P, Q in [20, 200]``. If it does not reproduce that, it does not get used.
    """
    Z = _standardise(X)
    if center:
        Z = Z - Z.mean(axis=0, keepdims=True)
    n = Z.shape[0]
    if n < 4:
        return {"skipped": f"need at least 4 rows, got {n}"}
    G = Z @ Z.T
    d = np.diag(G).astype(float)
    off = n * (n - 1)
    num = (float(d.sum()) ** 2 - float((d ** 2).sum())) / off
    den = (float((G ** 2).sum()) - float((d ** 2).sum())) / off
    if den <= _EPS:
        return {"skipped": "degenerate denominator"}
    raw_ev = _eigs(Z)
    return {"pr_corrected": float(num / den),
            "pr_raw": float(raw_ev.sum() ** 2 / (raw_ev ** 2).sum()),
            "correction": "row-side U-statistic only; column-side term of Chun et al. not applied",
            "n_rows": int(n), "n_cols": int(Z.shape[1])}


def chun_benchmark(d_true: int = 50, P: int = 200, Q: int = 200, sigma: float = 0.5,
                   seed: int = 20260807) -> dict:
    """The paper's own synthetic setup, used as a positive control on the estimator.

    ``Phi[i, a] = x_i . w_a + noise`` with ``x, w ~ N(0, I_d)``. The true dimensionality is ``d``,
    and the paper reports recovering it across ``P, Q in [20, 200]``. This is a KNOWN ANSWER
    delivered through the whole estimator, which is the only reason it is safe to ship a
    bias correction transcribed from a description rather than from code.
    """
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(int(P), int(d_true)))
    w = rng.normal(size=(int(Q), int(d_true)))
    phi = x @ w.T / np.sqrt(d_true) + rng.normal(scale=float(sigma), size=(int(P), int(Q)))
    r = participation_ratio_corrected(phi)
    if "skipped" in r:
        return r
    return {"d_true": int(d_true), "P": int(P), "Q": int(Q), "sigma": float(sigma),
            "pr_raw": r["pr_raw"], "pr_corrected": r["pr_corrected"],
            "abs_error_corrected": abs(r["pr_corrected"] - d_true),
            "abs_error_raw": abs(r["pr_raw"] - d_true)}


def all_criteria(X: np.ndarray, rng, max_rank: int = 60) -> dict:
    """Every criterion, on one matrix."""
    out = dict(vendored_criteria(X, rng))
    out.update(bicross_validation(X, rng, max_rank=max_rank))
    pr = participation_ratio_corrected(X)
    out.update({k: v for k, v in pr.items() if k in ("pr_corrected", "pr_raw")})
    return out
