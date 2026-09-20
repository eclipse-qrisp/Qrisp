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

"""Tests for CUDA-Q quantum memory management."""

import pytest
import cudaq

from qrisp import QuantumFloat, x, reset, measure
from qrisp.jasp.cudaq_interface import cudaq_kernel


@pytest.mark.timeout(30)
def test_cudaq_memory_management():
    """Test for CUDA-Q quantum memory management."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(10)
        x(a[0])
        x(a[0])
        reset(a)
        a.delete()

        b = QuantumFloat(10)
        x(b[0])
        x(b[0])
        reset(b)
        b.delete()

        c = QuantumFloat(10)
        x(c[0])
        return measure(c[0])

    cudaq.run(main, shots_count=10)


@pytest.mark.timeout(30)
def test_cudaq_memory_management_arithmetic():
    """Test for CUDA-Q quantum memory management using Qrisp's QuantumFloat arithmetic.
    Each inplace addition allocates auxiliary qubits, which are uncomputed, and must be deallocated properly.
    Otherwise, the total number of allocated qubits grows, leading to prohibitive simulation costs."""

    @cudaq_kernel
    def main():
        a = QuantumFloat(6)
        a[:] = 4

        a += 4
        a += 4
        a += 4
        a += 6

        return measure(a)

    cudaq.run(main, shots_count=10)
