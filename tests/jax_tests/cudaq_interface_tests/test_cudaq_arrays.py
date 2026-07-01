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

import cudaq
import numpy as np
import jax.numpy as jnp

from qrisp import (
    QuantumVariable,
    rx,
    rz,
    measure,
)
from qrisp.jasp import qache
from qrisp.jasp.cudaq_interface import cudaq_kernel

# ---------------------------------------------------------------------------
# Test arrays
# ---------------------------------------------------------------------------


def test_array():
    """Test that we can create classical (traced) arrays and access them in the quantum program."""

    @cudaq_kernel
    def main():
        """Static indexing."""

        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = QuantumVariable(5)
        rz(arr[0], a[0])

        return measure(a[0])

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]

    @cudaq_kernel
    def main():
        """Dynamic indexing."""

        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = QuantumVariable(5)
        b = QuantumVariable(1)
        ind = jnp.int32(measure(b[0]))
        rz(arr[ind], a[0])

        return measure(a[0])

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]


def test_array_in_qache():
    """Test that we can pass classical arrays as arguments to a @qache function and use them in the quantum program."""

    @qache
    def test(arr, qv):
        rz(arr[0], qv[0])
        return measure(qv[0])

    @cudaq_kernel
    def main():
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        a = QuantumVariable(5)
        res = test(arr, a)
        return res

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]


# ---------------------------------------------------------------------------
# Test dynamic classical array indexing
# ---------------------------------------------------------------------------


def test_array_dynamic_index_return():
    """Dynamic index into a classical array used as the return value."""

    @cudaq_kernel
    def main():
        arr = jnp.array([1.57, 0.78, 0.39, 0.25, 0.12])
        qv = QuantumVariable(1)
        ind = jnp.int32(measure(qv[0]))
        return arr[ind]

    results = cudaq.run(main, shots_count=5)
    assert np.allclose(results, 1.57)


def test_array_dynamic_index_gate_angle():
    """Dynamic index into a classical array used as a gate rotation angle."""

    @cudaq_kernel
    def main():
        arr = jnp.array([0.0, 3.14159265])
        qv = QuantumVariable(2)
        # qubit 0 is |0⟩ → measure returns 0 → arr[0] = 0.0 → rx(0) leaves |0⟩
        ind = jnp.int32(measure(qv[0]))
        rx(arr[ind], qv[1])
        return measure(qv[1])

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]


def test_array_dynamic_index_in_qache():
    """Dynamic index into an array inside a @qache-decorated subroutine."""

    @qache
    def apply_angle(arr, qv):
        ind = jnp.int32(measure(qv[0]))
        rx(arr[ind], qv[1])
        return measure(qv[1])

    @cudaq_kernel
    def main():
        arr = jnp.array([0.0, 3.14159265])
        qv = QuantumVariable(2)
        return apply_angle(arr, qv)

    results = cudaq.run(main, shots_count=10)
    assert results == 10 * [0]
