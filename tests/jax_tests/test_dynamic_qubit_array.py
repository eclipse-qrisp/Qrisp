"""
********************************************************************************
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

import numpy as np
import pytest
from qrisp import *
from qrisp.jasp import *


# ---------------------------------------------------------------------------
# Construction & size
# ---------------------------------------------------------------------------


def test_dqa_construction_and_size():
    """DynamicQubitArray wraps a QuantumVariable's register."""

    @boolean_simulation
    def main(n):
        qf = QuantumFloat(n)
        dqa = qf.reg
        return dqa.size, qf.size

    for n in [1, 2, 4, 8]:
        dqa_sz, qf_sz = main(n)
        assert dqa_sz == n
        assert qf_sz == n


def test_dqa_reg_self():
    """The .reg property returns the DynamicQubitArray itself."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(3)
        dqa = qf.reg
        return dqa.reg is dqa

    assert main()


# ---------------------------------------------------------------------------
# Integer indexing
# ---------------------------------------------------------------------------


def test_dqa_integer_indexing():
    """Integer indexing returns individual qubits."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        dqa = qf.reg
        # Flip qubit at index 1
        x(dqa[1])
        # Flip qubit at index 3
        x(dqa[3])
        return measure(qf)

    # Bits: 0=LSB, 1, 2, 3=MSB → 1010 in binary = 10 decimal
    assert main() == 10


def test_dqa_integer_indexing_edge():
    """Indexing the first and last positions of the DynamicQubitArray."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        dqa = qf.reg
        x(dqa[0])  # LSB
        x(dqa[-1])  # MSB (wraps to last)
        return measure(qf)

    assert main() == 9  # 1001


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def test_dqa_slicing_full():
    """Slicing with no arguments returns a view of the full array."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        full = qf.reg[:]
        x(full[0])
        x(full[3])
        return measure(qf)

    assert main() == 9


def test_dqa_slicing_with_start_stop():
    """Slicing with explicit start and stop."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(6)
        # Slice qubits [1:4] → indices 1, 2, 3
        sub = qf.reg[1:4]
        x(sub[0])  # original index 1
        x(sub[2])  # original index 3
        return measure(qf)

    assert main() == 10  # 001010


def test_dqa_slicing_with_start_only():
    """Slicing with start and no stop (stop defaults to size)."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        sub = qf.reg[2:]
        x(sub[0])  # original index 2
        x(sub[1])  # original index 3
        return measure(qf)

    assert main() == 12  # 1100


def test_dqa_slicing_with_stop_only():
    """Slicing with no start and explicit stop (start defaults to 0)."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        sub = qf.reg[:2]
        x(sub[0])  # original index 0
        x(sub[1])  # original index 1
        return measure(qf)

    assert main() == 3  # 0011


def test_dqa_slicing_step_not_one_raises():
    """Slicing with step != 1 raises NotImplementedError."""

    @jaspify
    def main():
        qf = QuantumFloat(4)
        _ = qf.reg[::2]

    with pytest.raises(
        NotImplementedError,
        match="Slicing with DynamicQubitArray only supports step=1",
    ):
        main()


# ---------------------------------------------------------------------------
# Concatenation (__add__)
# ---------------------------------------------------------------------------


def test_dqa_add_dqa():
    """Concatenating two DynamicQubitArrays."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        a[:] = 1  # 01
        b[:] = 2  # 10
        combined = a.reg + b.reg
        x(combined[0])  # flip a[0]: 01 → 00
        x(combined[3])  # flip b[1]: 10 → 00
        return measure(a), measure(b)

    a_res, b_res = main()
    assert a_res == 0  # was 01, flipped bit 0 → 00
    assert b_res == 0  # was 10, flipped bit 1 → 00


def test_dqa_add_list_of_qubits():
    """Concatenating a DynamicQubitArray with a list of individual qubits (append)."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c = QuantumBool()
        a[:] = 1  # 01
        # a.reg + [c[0]] appends c to the end
        plus = a.reg + [c[0]]
        # plus[0] = a[0], plus[1] = a[1], plus[2] = c[0]
        x(plus[0])  # flip a[0]
        x(plus[2])  # flip c[0]
        return measure(a), measure(c)

    a_res, c_res = main()
    assert a_res == 0  # 01 → 00
    assert c_res


def test_dqa_add_single_qubit():
    """Concatenating a DynamicQubitArray with a single qubit (not in a list)."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c = QuantumBool()
        a[:] = 1
        x(c[0])
        plus = a.reg + c[0]
        x(plus[0])  # flip a[0]: 01 → 00
        x(plus[2])  # flip c[0]: True → False
        return measure(a), measure(c)

    a_res, c_res = main()
    assert a_res == 0
    assert not c_res


# ---------------------------------------------------------------------------
# Reverse concatenation (__radd__)
# ---------------------------------------------------------------------------


def test_dqa_radd_list_of_qubits():
    """Prepending a list of qubits to a DynamicQubitArray via __radd__."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c = QuantumBool()
        a[:] = 1  # 01
        # [c[0]] + a.reg prepends c to the front
        radd = [c[0]] + a.reg
        # radd[0] = c[0], radd[1] = a[0], radd[2] = a[1]
        x(radd[0])  # flip c[0]
        x(radd[1])  # flip a[0]: 01 → 00
        return measure(a), measure(c)

    a_res, c_res = main()
    assert a_res == 0
    assert c_res


def test_dqa_radd_single_qubit():
    """Prepending a single qubit to a DynamicQubitArray via __radd__."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c = QuantumBool()
        a[:] = 1  # 01
        x(c[0])
        radd = c[0] + a.reg
        x(radd[0])  # flip c[0]: True → False
        x(radd[2])  # flip a[1]: 01 → 11
        return measure(a), measure(c)

    a_res, c_res = main()
    assert a_res == 3  # 01 → 11
    assert not c_res


def test_dqa_radd_multi_element_list():
    """Prepending a multi-element list preserves left-to-right order."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c0 = QuantumBool()
        c1 = QuantumBool()
        a[:] = 1  # 01
        x(c0[0])  # c0 = True
        # [c1[0], c0[0]] + a.reg → [c1, c0, a[0], a[1]]
        radd = [c1[0], c0[0]] + a.reg
        x(radd[0])  # flip c1: False → True
        x(radd[1])  # flip c0: True → False
        x(radd[3])  # flip a[1]: 01 → 11
        return measure(a), measure(c0), measure(c1)

    a_res, c0_res, c1_res = main()
    assert a_res == 3
    assert not c0_res
    assert c1_res


def test_dqa_radd_preserves_qubit_identity():
    """After __radd__, accessing qubits via the new DQA should reference the
    original qubits (not copies). Flipping via the concatenated DQA must
    affect the original QuantumVariable."""
    
    @boolean_simulation
    def main():
        a = QuantumFloat(3)
        c_in = QuantumBool()
        a[:] = 4     # 100
        x(c_in[0])   # c_in = True
        
        # Pattern used by gidney_adder: prepend carry-in qubit
        combined = [c_in[0]] + a.reg
        # combined layout: [c_in, a[0], a[1], a[2]]
        
        x(combined[0])  # flip c_in: True → False
        x(combined[3])  # flip a[2] (MSB): 100 → 000
        
        return measure(a), measure(c_in)
    
    a_res, c_res = main()
    assert a_res == 0     # 100 → 000
    assert not c_res       # True → False


# ---------------------------------------------------------------------------
# Chained concatenation
# ---------------------------------------------------------------------------


def test_dqa_chained_concat():
    """Multiple concatenations in sequence."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        c = QuantumBool()
        a[:] = 1  # 01
        b[:] = 2  # 10
        # Chained: a[:] + b[:] + [c[0]]
        combined = a.reg + b.reg + [c[0]]
        # Layout: a[0], a[1], b[0], b[1], c[0]
        x(combined[2])  # flip b[0]: 10 → 11
        x(combined[4])  # flip c[0]
        return measure(a), measure(b), measure(c)

    a_res, b_res, c_res = main()
    assert a_res == 1
    assert b_res == 3
    assert c_res


def test_dqa_chained_radd():
    """Multiple __radd__ operations: list + list + DQA."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        c0 = QuantumBool()
        c1 = QuantumBool()
        a[:] = 1  # 01
        x(c1[0])  # c1 = True
        # [c1[0]] + [c0[0]] + a.reg
        combined = c1[0] + c0[0] + a.reg
        # Layout: c1, c0, a[0], a[1]
        x(combined[0])  # flip c1: True → False
        x(combined[1])  # flip c0: False → True
        return measure(a), measure(c0), measure(c1)

    a_res, c0_res, c1_res = main()
    assert a_res == 1
    assert c0_res
    assert not c1_res


# ---------------------------------------------------------------------------
# Invalid concatenation
# ---------------------------------------------------------------------------


def test_dqa_add_invalid_element_raises():
    """Concatenating a non-AbstractQubit to a DQA raises ValueError."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        _ = a.reg + [42]

    with pytest.raises(
        ValueError,
        match="Can only concatenate type AbstractQubit or list\\[AbstractQubit\\]",
    ):
        main()


def test_dqa_radd_invalid_element_raises():
    """Prepending a non-AbstractQubit raises ValueError."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        _ = [42] + a.reg

    with pytest.raises(
        ValueError,
        match="Can only concatenate type AbstractQubit or list\\[AbstractQubit\\]",
    ):
        main()


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


def test_dqa_measure():
    """Measuring a DynamicQubitArray returns the integer value of its qubits."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        qf[:] = 11  # 1011
        return qf.reg.measure()

    assert main() == 11


def test_dqa_measure_after_concat():
    """Measuring a concatenated DynamicQubitArray."""

    @boolean_simulation
    def main():
        a = QuantumFloat(2)
        b = QuantumFloat(2)
        a[:] = 1  # 01
        b[:] = 2  # 10
        combined = a.reg + b.reg
        return combined.measure()

    assert main() == 1 + (2 << 2)  # a in LSBs, b in next two bits → 1001 = 9


# ---------------------------------------------------------------------------
# Slicing + concatenation combined
# ---------------------------------------------------------------------------


def test_dqa_slice_then_concat():
    """Slicing a DQA and then concatenating."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        qf[:] = 5  # 0101
        # Slice bits 1..2, then prepend bit 0
        middle = qf.reg[1:3]  # bits [1, 2]
        combined = qf.reg[0] + middle  # bits [0, 1, 2]
        # Now flip all: 101 → 010
        x(combined[0])
        x(combined[1])
        x(combined[2])
        return measure(qf)

    assert main() == 2  # 0010


# ---------------------------------------------------------------------------
# PyTree compatibility
# ---------------------------------------------------------------------------


def test_dqa_is_jax_pytree():
    """DynamicQubitArray is registered as a JAX pytree node."""

    @jaspify
    def make_dqa():
        qf = QuantumFloat(3)
        return qf.reg

    dqa = make_dqa()
    from jax import tree_util

    leaves, treedef = tree_util.tree_flatten(dqa)
    # Unflatten should reconstruct a DynamicQubitArray
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, DynamicQubitArray)


# ---------------------------------------------------------------------------
# Integration: real-world pattern from gidney_adder
# ---------------------------------------------------------------------------


def test_dqa_gidney_adder_pattern():
    """Simulate the gidney_adder prepend pattern: [c_in_qb] + b_qbs."""

    @boolean_simulation
    def main():
        b = QuantumFloat(4)
        c_in = QuantumBool()
        b[:] = 5  # 0101
        x(c_in[0])

        # Pattern: prepend c_in to b's register
        b_qbs = [c_in[0]] + b.reg
        # Layout: [c_in, b[0], b[1], b[2], b[3]]

        # Verify ordering by flipping individual positions
        x(b_qbs[0])  # flip c_in
        x(b_qbs[3])  # flip b[2] → 0101 → 0001

        return measure(b), measure(c_in)

    b_res, c_res = main()
    assert b_res == 1  # 0101 → 0001
    assert not c_res


def test_dqa_gidney_adder_append_pattern():
    """Simulate the gidney_adder append pattern: b_qbs + [c_out_qb]."""

    @boolean_simulation
    def main():
        b = QuantumFloat(4)
        c_out = QuantumBool()
        b[:] = 5  # 0101

        # Pattern: append c_out to b's register
        b_qbs = b.reg + [c_out[0]]
        # Layout: [b[0], b[1], b[2], b[3], c_out]

        x(b_qbs[0])  # flip b[0] → 0101 → 0100
        x(b_qbs[4])  # flip c_out
        return measure(b), measure(c_out)

    b_res, c_res = main()
    assert b_res == 4
    assert c_res


# ---------------------------------------------------------------------------
# Measure on sliced / empty DQA
# ---------------------------------------------------------------------------


def test_dqa_measure_sliced():
    """Measuring a sliced DynamicQubitArray returns the integer value of the
    qubits in the slice only."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        qf[:] = 11  # 1011
        # Slice bits [1:3] → indices 1, 2 → binary 01 = 1
        return qf.reg[1:3].measure()

    assert main() == 1


def test_dqa_measure_empty():
    """Measuring an empty DynamicQubitArray (slice start == stop) returns 0."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(4)
        empty = qf.reg[2:2]
        return empty.measure()

    assert main() == 0


# ---------------------------------------------------------------------------
# Concatenation with empty lists
# ---------------------------------------------------------------------------


def test_dqa_concat_empty_list():
    """Concatenating an empty list (dqa + []) returns the same DQA."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(3)
        qf[:] = 5  # 101
        result = qf.reg + []
        return result.measure()

    assert main() == 5


def test_dqa_radd_empty_list():
    """Prepending an empty list ([] + dqa) returns the same DQA."""

    @boolean_simulation
    def main():
        qf = QuantumFloat(3)
        qf[:] = 5  # 101
        result = [] + qf.reg
        return result.measure()

    assert main() == 5