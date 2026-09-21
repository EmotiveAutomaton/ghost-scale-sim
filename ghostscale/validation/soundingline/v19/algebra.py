"""Numerical identification diagnostics; tolerances are not exact null proofs."""
import numpy as np


def audit(atoms, target):
    atoms, target = np.asarray(atoms, float), np.asarray(target, float)
    if atoms.ndim != 2 or target.ndim != 2 or len(atoms) != len(target):
        raise ValueError('bank/target state axes differ')
    if not np.isfinite(atoms).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite law')
    augmented = np.column_stack((np.ones(len(atoms)), atoms))
    singular = np.linalg.svd(augmented, compute_uv=False)
    eps = np.finfo(float).eps * max(augmented.shape)
    rows = []
    for tolerance in (eps, 1e-14, 1e-12, 1e-10, 1e-3):
        mapping = np.linalg.pinv(augmented, rcond=tolerance) @ target
        rows.append(dict(relative_tolerance=tolerance, absolute_threshold=float(tolerance*singular[0]),
            rank=int(np.sum(singular > tolerance*singular[0])),
            target_span_residual=float(np.max(abs(augmented@mapping-target))),
            bank_to_target_operator_norm=float(np.linalg.norm(mapping[1:],2))))
    differences = atoms[1:]-atoms[0]
    affine_singular = np.linalg.svd(differences, compute_uv=False)
    tight = np.linalg.pinv(augmented, rcond=1e-14) @ target
    return dict(states=len(atoms), bank_entries=atoms.shape[1], target_entries=target.shape[1],
        singular_values=singular.tolist(), affine_singular_values=affine_singular.tolist(),
        full_numerical_rank=rows[0]['rank'], affine_numerical_dimension=int(np.linalg.matrix_rank(differences)),
        tolerance_sweep=rows, scale='absolute threshold = relative tolerance times largest augmented singular value',
        exact_rank='not asserted from floating point',
        nonidentification='not asserted without a separate valid simplex witness'), tight


def bank_error(atoms, target, posterior, bank, mapping):
    exact = posterior@atoms; truth = posterior@target
    error = np.asarray(bank).reshape(-1)-exact
    amplified = error@mapping[1:]
    exact_error = np.r_[1.,exact]@mapping-truth
    return dict(bank_squared_error=float(error@error),
        target_weighted_squared_error=float(amplified@amplified),
        observed_amplification=float(np.linalg.norm(amplified)/np.linalg.norm(error)) if np.linalg.norm(error)>0 else 0.,
        exact_bank_max_readout_error=float(np.max(abs(exact_error))))


def controls():
    positive, _ = audit(np.eye(3),np.eye(3))
    # Same bank for every posterior; targets distinguish the first two states.
    alias = np.ones((3,2))*.5; targets = np.eye(3)
    p, q = np.array([1.,0,0]),np.array([0.,1,0])
    # Mathematically full rank with a direction the coarse inverse discards.
    fragile, _ = audit(np.array([[.5,.5],[.50000001,.49999999]]),np.eye(2))
    return dict(live_known_identity=positive['full_numerical_rank']==3 and positive['tolerance_sweep'][1]['target_span_residual']<1e-10,
        placebo_exact_simplex_alias=bool(np.array_equal(p@alias,q@alias) and not np.array_equal(p@targets,q@targets)),
        truncated_is_not_exact_null=fragile['full_numerical_rank']==2 and fragile['tolerance_sweep'][-1]['rank']==1)
