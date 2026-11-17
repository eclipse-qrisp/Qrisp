# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
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

"""Tests for the Heisenberg VQE problem in qrisp.vqe.problems.heisenberg."""

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from qrisp.vqe.problems.heisenberg import *

from qrisp import QuantumFloat, QuantumVariable
from qrisp.misc.exceptions import QrispDeprecationWarning
from qrisp.jasp import jaspify


def test_vqe_heisenberg():

    # Create a graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

    vqe = heisenberg_problem(G, 1, 1)

    results = []
    for i in range(5):
        res = vqe.run(QuantumVariable(G.number_of_nodes()), depth=2, max_iter=50, mes_kwargs={"atol": 0.01})
        results.append(res)

    assert np.abs(min(results) - (-8.0)) < 1e-1


def test_jasp_vqe_heisenberg():

    @jaspify(terminal_sampling=True)
    def main():
        # Create a graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

        vqe = heisenberg_problem(G, 1, 1)

        results = jnp.array([0.0] * 5)
        for i in range(5):
            res = vqe.run(QuantumFloat(G.number_of_nodes()), depth=1, max_iter=50, optimizer="SPSA")
            results = results.at[i].set(res)

        return results

    results = main()

    assert np.abs(min(results) - (-8.0)) < 2


@pytest.mark.parametrize(
    ("mes_kwargs", "message"),
    [
        ({"diagonalisation_method": "commuting"}, "use diagonalization_method instead"),
        ({"precision": 0.05}, "precision is deprecated; use atol"),
    ],
)
def test_vqe_accepts_deprecated_measurement_kwargs(mes_kwargs, message):
    vqe = heisenberg_problem(nx.path_graph(2), 1, 0)

    with pytest.warns(QrispDeprecationWarning, match=message):
        vqe.run(QuantumVariable(2), depth=1, max_iter=2, mes_kwargs=mes_kwargs)


def _benchmark(vqe, **kwargs):
    return vqe.benchmark(
        QuantumVariable(2), depth_range=[1], iter_range=[2], optimal_energy=-1.0, repetitions=2, **kwargs
    )


def test_vqe_benchmark_tolerances():
    vqe = heisenberg_problem(nx.path_graph(2), 1, 0)

    benchmark = _benchmark(vqe, atol_range=[0.1])
    assert benchmark.atol == [0.1, 0.1]
    assert benchmark.rtol == [None, None]
    cost_data, _ = benchmark.evaluate()
    assert all(c > 0 for c in cost_data)
    assert benchmark.rank(average_repetitions=True)[0]["atol"] == 0.1

    benchmark = _benchmark(vqe, rtol_range=[0.1])
    assert benchmark.atol == [None, None]
    assert benchmark.rtol == [0.1, 0.1]
    assert all(c > 0 for c in benchmark.evaluate()[0])

    with pytest.warns(QrispDeprecationWarning, match="VQEBenchmark.precision is deprecated"):
        assert benchmark.precision == [0.1, 0.1]
    with pytest.warns(QrispDeprecationWarning, match="run data is deprecated"):
        benchmark.evaluate(cost_metric=lambda run_data: run_data["precision"])


def test_vqe_benchmark_precision_range_deprecated():
    vqe = heisenberg_problem(nx.path_graph(2), 1, 0)

    with pytest.warns(QrispDeprecationWarning, match="precision_range is deprecated"):
        benchmark = _benchmark(vqe, precision_range=[0.1])
    assert benchmark.atol == [0.1, 0.1]

    for kwargs in ({}, {"atol_range": [0.1], "rtol_range": [0.1]}, {"precision_range": [0.1], "rtol_range": [0.1]}):
        with pytest.raises(ValueError, match="Specify"):
            _benchmark(vqe, **kwargs)
