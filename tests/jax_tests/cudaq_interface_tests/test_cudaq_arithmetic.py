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

import operator
import pytest

import cudaq

from qrisp import (
    QuantumFloat,
    measure,
)
from qrisp.jasp.cudaq_interface import cudaq_kernel

# ---------------------------------------------------------------------------
# Test QuantumFloat and arithmetic operations
# ---------------------------------------------------------------------------


ops = [operator.add, operator.sub]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 4, 0, 1, id="QuantumFloat case 1"),
    pytest.param(3, 0, 3, 4, 0, 2, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_arithmetic(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """Arithmetic operations on QuantumFloat with classical or quantum RHS."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        c = op(a, b)
        return measure(c)

    expected = op(val1, val2)

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {results}"
    )


ops_inpl = [operator.iadd, operator.isub]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 3, 0, 1, id="QuantumFloat case 1"),
    pytest.param(3, 0, 3, 4, 0, 3, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops_inpl)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_arithmetic_inpl(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """In-place arithmetic operations on QuantumFloat."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        op(a, b)
        return measure(a)

    expected = op(val1, val2)

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {results}"
    )


ops_comp = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
rhs_type = ["classical", "quantum"]
instances = [
    # (size1, exp1, val1, size2, exp2, val2)
    pytest.param(3, 0, 2, 3, 0, 1, id="QuantumFloat case 1"),
    # pytest.param(3, 0, 3, 4, 0, 3, id="QuantumFloat case 2"),
]


@pytest.mark.parametrize("op", ops_comp)
@pytest.mark.parametrize("rhs_type", rhs_type)
@pytest.mark.parametrize("size1, exp1, val1, size2, exp2, val2", instances)
def test_quantum_float_comparison(op, rhs_type, size1, exp1, val1, size2, exp2, val2):
    """Comparison operations on QuantumFloat with classical or quantum RHS."""

    if op in (operator.eq, operator.ne):
        pytest.skip("Equality and inequality comparisons on QuantumFloat are not supported.")

    @cudaq_kernel
    def main():
        a = QuantumFloat(size1, exponent=exp1)
        a[:] = val1

        if rhs_type == "classical":
            b = val2
        else:
            b = QuantumFloat(size2, exponent=exp2)
            b[:] = val2

        c = op(a, b)
        return measure(c)

    expected = op(val1, val2)

    results = cudaq.run(main, shots_count=1)
    assert results == 1 * [expected], (
        f"Expected quantum-{rhs_type} {op.__name__} of {val1} and {val2} to yield {expected}, got {results}"
    )
