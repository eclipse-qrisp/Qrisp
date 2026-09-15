"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from collections import Counter
import pytest

import numpy as np
import cudaq

from qrisp import (
    QuantumVariable,
    measure,
)
from qrisp.block_encodings import BlockEncoding
from qrisp.jasp import terminal_sampling
from qrisp.jasp.cudaq_interface import cudaq_kernel
from qrisp.operators import X, Y, Z

# ---------------------------------------------------------------------------
# Test BlockEncoding
# ---------------------------------------------------------------------------


def _post_selection(res_dict):
    # Post-selection on ancillas being in |0> state
    filtered_dict = {k[0]: p for k, p in res_dict.items() if all(x == 0 for x in k[1:])}
    success_prob = sum(filtered_dict.values())
    filtered_dict = {k: p / success_prob for k, p in filtered_dict.items()}
    return filtered_dict


@pytest.mark.parametrize("operator", [X(0), X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding(operator):
    """Test that we can create a BlockEncoding from a simple Hamiltonian and apply it to a quantum variable."""

    BE = BlockEncoding.from_operator(operator)

    @cudaq_kernel
    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return measure(operand)

    results = cudaq.run(main, shots_count=10)


@pytest.mark.parametrize("operator", [X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding_results(operator):
    """
    Test that applying a BlockEncoding of a simple Hamiltonian produces the expected
    measurement distribution on the operand qubits after post-selection on ancillas.
    """

    BE = BlockEncoding.from_operator(operator)

    @cudaq_kernel
    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return measure(operand), measure(ancs[0])

    results = cudaq.run(main, shots_count=500)

    counts = Counter(results)
    counts_dict = dict(counts)
    res_dict = {outcome: count / len(results) for outcome, count in counts.items()}
    filtered_dict = _post_selection(res_dict)

    # Now do the same thing with JASP's terminal_sampling to get the expected distribution for comparison
    @terminal_sampling
    def main():

        operand = QuantumVariable(2)
        ancs = BE.apply(operand)
        return operand, ancs[0]

    res_dict_jasp = main()
    filtered_dict_jasp = _post_selection(res_dict_jasp)

    all_keys = set(filtered_dict.keys()).union(set(filtered_dict_jasp.keys()))

    for key in all_keys:
        val1 = filtered_dict.get(key, 0.0)
        val2 = filtered_dict_jasp.get(key, 0.0)
        assert np.isclose(val1, val2, atol=1e-1)


@pytest.mark.parametrize("operator", [X(0) + Z(1), X(0) * X(1) + Z(0) * Z(1)])
def test_simple_block_encoding_poly(operator):
    """Test that we can create a polynomial from a BlockEncoding and apply it to a quantum variable."""

    BE = BlockEncoding.from_operator(operator)
    BE_poly = BE.poly(np.array([0.5, 0.5]))

    @cudaq_kernel
    def main():

        operand = QuantumVariable(2)
        ancs = BE_poly.apply(operand)
        return measure(operand)

    results = cudaq.run(main, shots_count=10)
