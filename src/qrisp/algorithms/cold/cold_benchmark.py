# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Benchmark metrics (average QUBO cost, success probability) for COLD/LCD solver results."""


def _avg_qubo_cost(res):
    """Returns the average QUBO cost of the measurement."""
    expected_cost = 0.0
    for prob, cost in res.values():
        # Weight cost by probability
        expected_cost += prob * cost
    return expected_cost


def _success_prob(meas, solution):
    """Returns the success probability of the given measurement and solution."""
    sp = 0
    for s in solution.keys():
        try:
            prob, cost = meas[s]
            sp += prob
        except KeyError:
            continue
    return sp


def _approx_ratio(meas, solution):
    """Returns the approximation ratio of the given measurement and solution."""
    cost = _avg_qubo_cost(meas)
    opt_cost = list(solution.values())[0]
    ar = cost / opt_cost
    return ar


def _most_likely_cost_and_prob(meas, N):
    """Get the N most likely QUBO costs and their probabilites.
    Returns two dictionaries of the form {bitstring: cost/prob}.
    """
    keys = list(meas.keys())[:N]
    most_likely_cost = {k: meas[k][1] for k in keys}
    most_likely_prob = {k: meas[k][0] for k in keys}

    return most_likely_cost, most_likely_prob
