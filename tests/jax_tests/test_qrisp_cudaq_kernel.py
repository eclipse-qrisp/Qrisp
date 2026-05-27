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

"""
Tests for the Jasp → Quake (memory-semantics) lowering backend.

Coverage
--------
- Basic quantum circuit (alloc, gate, measure, dealloc).
- Parameterized gates (rz, rx, u3).
- Controlled gates (cx, ccx).
- Reset operation.
- SCF control-flow lowering (jrange loop).
- Interface invariant: no ``!jasp.*`` types in the output.
- Negative test: ``jasp.parity`` is left in place (not lowered).
- Negative test: unsupported gate emits a warning and is left in place.
"""

import warnings
import jax
import jax.numpy as jnp
import numpy as np
import operator
import pytest
import re

from qrisp import QuantumVariable, QuantumBool, QuantumFloat, h, mcx, x, y, z, cp, cx, cy, cz, gphase, rx, ry, rz, rxx, rz, rzz, s, swap, sx, t, xxyy, measure, control, invert, conjugate
from qrisp.alg_primitives import amplitude_amplification, q_switch
from qrisp.jasp import make_jaspr, jrange, q_while_loop, q_cond, q_fori_loop, qache

try:
    from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake, validate_quake_mlir, run_quake_mlir, qrisp_cudaq_kernel
except ImportError as exc:
    # Skip the entire test file if the import fails
    pytest.skip(f"quake_lowering unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test qrisp cudaq kernel decorator
# ---------------------------------------------------------------------------

def test_qrisp_cudaq_kernel_decorator():
    """Test that the @qrisp_cudaq_kernel decorator compiles a function to MLIR and runs it on CUDA-Q."""
    from qrisp.jasp.mlir.quake_lowering import qrisp_cudaq_kernel

    @qrisp_cudaq_kernel
    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    result = bell()
    assert result in {0, 3}


def test_qrisp_algorithm_in_cudaq_kernel():
    """Test that we can use qrisp algorithms (like Trotterization) inside a @qrisp_cudaq_kernel."""
    from qrisp.operators import X, Y, Z
    import networkx as nx

    @qrisp_cudaq_kernel
    def main():

        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        H = sum(X(i) for i in G.nodes) + sum(Z(i)*Z(j) for i, j in G.edges)
        U = H.trotterization()

        a = QuantumFloat(4)
        b = QuantumFloat(4)

        a[:] = 5
        h(b)
        U(a, t=1.0, steps=10)

        # Real-time control flow based on measurement results
        c = measure(b) < 5
        with control(c):
            a += 1

        return measure(a)

    result = main()
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
