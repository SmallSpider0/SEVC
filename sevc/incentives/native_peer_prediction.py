"""It Takes Two v5, Eqs. (11)-(14) and Tables 5-9, native PoL game.

Mechanism-level four-observation reproduction; no SEVC replay or cost conversion.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import linprog


def constraint_system(spec):
    belief = np.asarray(spec["principal_belief_matrix"], dtype=float)
    costs = np.asarray(spec["costs"], dtype=float)
    blind = np.asarray(spec["principal_prior"]) @ np.asarray(spec["observation_matrix"])
    rows, bounds = [], []
    for observed, reported in itertools.product(range(4), repeat=2):
        row = np.zeros((4, 4))
        row[reported] = belief[observed]
        if observed == reported:
            rows.append(-row.ravel()); bounds.append(-costs[observed] - spec["delta"])
        else:
            rows.append(row.ravel()); bounds.append(costs[observed] - spec["delta"])
    for reported in range(4):
        row = np.zeros((4, 4)); row[reported] = blind
        rows.append(row.ravel()); bounds.append(-spec["delta"])
    return np.asarray(rows), np.asarray(bounds)


def solve_score(spec, *, simple_agreement=False):
    a, b = constraint_system(spec)
    # Affine agreement: T_xy = scale * 1[x=y] + shift, same amplitude bound.
    basis = np.stack((np.eye(4).ravel(), np.ones(16)), axis=1) if simple_agreement else np.eye(16)
    width = basis.shape[1]
    inequalities = np.vstack((np.column_stack((a @ basis, np.zeros(len(a)))),
                              np.column_stack((basis, -np.ones(16))),
                              np.column_stack((-basis, -np.ones(16)))))
    rhs = np.concatenate((b, np.zeros(32)))
    objective = np.zeros(width + 1); objective[-1] = 1
    result = linprog(objective, A_ub=inequalities, b_ub=rhs,
                     bounds=[(None, None)] * width + [(0, None)], method="highs")
    if result.status == 2:
        # A normalized nonnegative Farkas multiplier proves infeasibility,
        # including the explicit -K <= 0 constraint, without trusting status.
        aa = np.vstack((inequalities, -objective)); bb = np.r_[rhs, 0.]
        cert = linprog(bb, A_eq=np.vstack((aa.T, np.ones(len(bb)))),
                       b_eq=np.r_[np.zeros(width + 1), 1.], bounds=(0, None), method="highs")
        if not cert.success or cert.fun >= -1e-8:
            raise RuntimeError("solver infeasible without independently checkable certificate")
        return {"status": "INFEASIBLE", "score": None,
                "certificate": {"multipliers": cert.x.tolist(), "A": aa.tolist(), "b": bb.tolist(),
                                "rhs_dot": float(cert.fun)}}
    if not result.success:
        raise RuntimeError(f"native LP technical failure: {result.message}")
    score = (basis @ result.x[:width]).reshape(4, 4)
    violation = float(max(0., np.max(a @ score.ravel() - b), np.max(np.abs(score)) - result.x[-1]))
    if violation > spec["lp_feasibility_tolerance"]:
        raise ValueError("native LP residual exceeds frozen feasibility tolerance")
    return {"status": "OPTIMAL", "score": score.tolist(), "K": float(result.x[-1]),
            "max_violation": violation, "dual": result.ineqlin.marginals.tolist(),
            "primal": result.x.tolist(), "A": inequalities.tolist(), "b": rhs.tolist()}


def evaluate_score(spec, score, epsilon):
    score, observation = np.asarray(score), np.asarray(spec["observation_matrix"])
    prior = np.array([.5 - epsilon, .25, .25, epsilon])
    joint = np.einsum("t,tx,ty->xy", prior, observation, observation)
    marginal = joint.sum(axis=1)
    cost = float(marginal @ np.asarray(spec["costs"]))
    strategies = []
    for mapping in itertools.product(range(4), repeat=4):
        reward = float(sum(joint[x, y] * score[mapping[x], y] for x in range(4) for y in range(4)))
        strategies.append({"observed": True, "mapping": list(mapping), "reward": reward,
                           "cost": cost, "utility": reward - cost})
    for report in range(4):
        reward = float(marginal @ score[report])
        strategies.append({"observed": False, "mapping": [report] * 4, "reward": reward,
                           "cost": 0., "utility": reward})
    honest = next(r for r in strategies if r["observed"] and r["mapping"] == list(range(4)))
    deviations = [r for r in strategies if r is not honest]
    best = max(deviations, key=lambda r: r["utility"])
    # The zero-probability cheat observation retains its model-defined belief.
    belief = np.divide(joint, marginal[:, None], out=observation.copy(), where=marginal[:, None] != 0)
    conditional = belief @ score.T - np.asarray(spec["costs"])[:, None]
    return {"epsilon": epsilon, "joint": joint.tolist(), "strategies": strategies,
            "honest": honest, "best_deviation": best,
            "honest_minus_best_deviation": honest["utility"] - best["utility"],
            "conditional_utilities": conditional.tolist(), "K": float(np.abs(score).max())}


def native_cell(spec, method, epsilon, scores):
    """Shared native M8 assembly with reusable optimization certificate."""
    if method not in scores:
        scores[method] = ({'status':'PUBLISHED_ROUNDED','score':spec['published_score']}
            if 'published' in method else solve_score(spec, simple_agreement=method.startswith('simple-agreement')))
    solution=scores[method]
    return solution, (evaluate_score(spec,solution['score'],epsilon) if solution['score'] is not None else None)
