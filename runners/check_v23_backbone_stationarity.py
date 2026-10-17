"""Independent high-precision diagnosis of a retained failed choice fit.

This computes no forecasts and never changes a solver disposition. Input is a
synthetic, retained NPZ containing the exact failed fit's rows and coefficients.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

import numpy as np


def decimal_objective(features, taken, weights, center, offset, penalty):
    """Evaluate ridge multinomial loss, gradient and Hessian with 50 digits."""
    with localcontext() as context:
        context.prec = 50
        number = lambda value: Decimal.from_float(float(value))
        delta = list(map(number, offset))
        beta = [number(a) + b for a, b in zip(center, delta, strict=True)]
        strength = number(penalty)
        dimension = len(beta)
        loss = strength * sum(x * x for x in delta)
        gradient = [2 * strength * x for x in delta]
        hessian = [[2 * strength if i == j else Decimal(0)
                    for j in range(dimension)] for i in range(dimension)]
        for raw, chosen, raw_weight in zip(features, taken, weights, strict=True):
            rows = [list(map(number, row)) for row in raw]
            weight = number(raw_weight)
            logits = [sum(a * b for a, b in zip(row, beta, strict=True)) for row in rows]
            top = max(logits)
            masses = [(value - top).exp() for value in logits]
            total = sum(masses)
            probability = [mass / total for mass in masses]
            loss += weight * (top + total.ln() - logits[int(chosen)])
            means = [sum(p * row[j] for p, row in zip(probability, rows, strict=True))
                     for j in range(dimension)]
            for j in range(dimension):
                gradient[j] += weight * (means[j] - rows[int(chosen)][j])
                for k in range(dimension):
                    hessian[j][k] += weight * sum(
                        p * (row[j] - means[j]) * (row[k] - means[k])
                        for p, row in zip(probability, rows, strict=True))
        return loss, gradient, hessian


def check(path):
    with np.load(path, allow_pickle=False) as data:
        taken = data['taken'].copy()
        features = [data[f'features_{i}'].copy() for i in range(len(taken))]
        weights, center, failed = (data[name].copy() for name in ('weights', 'center', 'failed'))
        penalty = float(data['penalty'])
    if not penalty > 0:
        raise ValueError('Positive ridge penalty required for the objective-gap bound')
    loss, gradient, hessian = decimal_objective(features, taken, weights, center, failed, penalty)
    g = np.array(list(map(float, gradient)))
    h = np.array([[float(value) for value in row] for row in hessian])
    step = np.linalg.solve(h, g)
    candidate = failed - step
    candidate_loss, candidate_gradient, _ = decimal_objective(features, taken, weights, center, candidate, penalty)
    with localcontext() as context:
        context.prec = 50
        gap_bound = sum(value * value for value in gradient) / (4 * Decimal.from_float(penalty))
        improvement = loss - candidate_loss
    return dict(state='DIAGNOSIS_ONLY',input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        decimal_precision=50,rows=len(taken),dimensions=len(failed),penalty=penalty,
        objective=str(loss),gradient=[str(value) for value in gradient],
        gradient_infinity_norm=float(max(abs(value) for value in gradient)),
        hessian_minimum_eigenvalue=float(np.linalg.eigvalsh(h).min()),
        objective_gap_upper_bound=str(gap_bound),bound_basis='Ridge Hessian at least 2 * penalty * identity; gradient norm squared / (4 * penalty)',
        diagnostic_newton_step_norm=float(np.linalg.norm(step)),
        diagnostic_newton_objective=str(candidate_loss),diagnostic_newton_objective_improvement=str(improvement),
        diagnostic_newton_gradient_infinity_norm=float(max(abs(value) for value in candidate_gradient)),
        original_failure_preserved=True,coefficients_used_for_prediction=False,repair_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(check(args.input), stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
