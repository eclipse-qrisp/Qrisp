import numpy as np

def qubo_cost(Q, P):
    expected_cost = 0.0
    for bitstring, prob in P.items():
        # Convert bitstring (e.g., "10110") to numpy array of ints
        x = np.array([int(b) for b in bitstring], dtype=float)
        # Compute quadratic form x^T Q x
        cost = x @ Q @ x
        # Weight by probability
        expected_cost += prob * cost
    return expected_cost

def success_prob(meas, solution):
    sp = 0
    for s in solution.keys():
        try:
            sp += meas[s]
        except KeyError:
            continue
    return sp

def approx_ratio(Q, meas, solution):
    cost = qubo_cost(Q, meas)
    opt_cost = list(solution.values())[0]
    ar = cost/opt_cost
    return ar

def most_likely_res(Q, meas, N):
    keys = list(meas.keys())[:N]
    N_most_likely = {k: float(qubo_cost(Q, {k: 1})) for k in keys}
    return N_most_likely
