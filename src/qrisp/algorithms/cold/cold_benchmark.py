
def avg_qubo_cost(res):
    """ Returns the average QUBO cost of the measurement. """

    expected_cost = 0.0
    for prob, cost in res.values():
        # Weight cost by probability
        expected_cost += prob * cost
    return expected_cost

def success_prob(meas, solution):
    """ Returns the success probability of the given measurement and solution. """

    sp = 0
    for s in solution.keys():
        try:
            prob, cost = meas[s]
            sp += prob
        except KeyError:
            continue
    return sp

def approx_ratio(meas, solution):
    """ Returns the approximation ratio of the given measurement and solution. """

    cost = avg_qubo_cost(meas)
    opt_cost = list(solution.values())[0]
    ar = cost/opt_cost
    return ar

def most_likely_cost_and_prob(meas, N):
    """ Get the N most likely QUBO costs and their probabilites.
    Returns two dictionaries of the form {bitstring: cost/prob}. """

    keys = list(meas.keys())[:N]
    most_likely_cost = {k: meas[k][1] for k in keys}
    most_likely_prob = {k: meas[k][0] for k in keys}
    
    return most_likely_cost, most_likely_prob
