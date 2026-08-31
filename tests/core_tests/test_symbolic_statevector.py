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

import numpy as np
import sympy
from sympy import symbols

from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import cx, h, rx


class TestSymbolicStatevector:
    """Tests for QuantumSession.statevector on circuits with unbound parameters.

    These exercise the same simulator path as the QuantumCircuit-level tests in
    ``tests/circuit_tests/test_quantum_circuit.py``, but through the
    user-facing :meth:`QuantumSession.statevector` entry point documented in
    ``documentation/source/reference/Examples/AbstractParameter.rst``.
    """

    def test_statevector_sympy_retains_free_symbols(self):
        """The ket expression keeps the unbound parameters.

        ``free_symbols`` also contains the basis-state kets, so the parameters
        are checked for as a subset.
        """
        phi, theta = symbols("phi theta")
        qv = QuantumVariable(2)
        rx(phi, qv[0])
        rx(theta, qv[1])
        state = qv.qs.statevector()
        assert {phi, theta}.issubset(state.free_symbols)

    def test_statevector_sympy_matches_documented_example(self):
        """Reproduces the ket from the abstract-parameter documentation page.

        ``rx(phi) (x) rx(theta)`` populates all four basis states of the two
        active qubits, and the amplitudes carry the expected trigonometric
        factors.
        """
        phi, theta = symbols("phi theta")
        qv = QuantumVariable(4)
        rx(phi, qv[0])
        rx(theta, qv[1])
        state = qv.qs.statevector()

        assert len(state.args) == 4
        assert {phi, theta}.issubset(state.free_symbols)

        # rx(0) (x) rx(0) is the identity, so the state collapses to |0000>
        # with amplitude 1 and only the ket symbol survives.
        collapsed = sympy.simplify(state.subs({phi: 0, theta: 0}))
        (ground_ket,) = collapsed.free_symbols
        assert sympy.simplify(collapsed / ground_ket) == 1

        # rx(pi) maps |0> to -I|1>, so only the |1000> term survives.
        flipped = sympy.simplify(state.subs({phi: sympy.pi, theta: 0}))
        (flipped_ket,) = flipped.free_symbols
        assert flipped_ket != ground_ket
        assert sympy.simplify(flipped / flipped_ket) == -sympy.I

    def test_statevector_array_symbolic_matches_numeric_after_substitution(self):
        """Binding the symbols reproduces the numeric session's statevector."""
        phi, theta = symbols("phi theta")
        values = {phi: 0.3, theta: 1.1}

        symbolic_qv = QuantumVariable(4)
        rx(phi, symbolic_qv[0])
        rx(theta, symbolic_qv[1])
        symbolic_sv = symbolic_qv.qs.statevector(return_type="array")
        bound = np.array([complex(sympy.sympify(entry).subs(values)) for entry in symbolic_sv])

        numeric_qv = QuantumVariable(4)
        rx(values[phi], numeric_qv[0])
        rx(values[theta], numeric_qv[1])
        reference = np.asarray(numeric_qv.qs.statevector(return_type="array"))

        assert np.allclose(bound, reference, atol=1e-5)

    def test_statevector_array_symbolic_is_object_dtype(self):
        """``return_type="array"`` yields an object array for symbolic circuits."""
        phi = symbols("phi")
        qv = QuantumVariable(2)
        h(qv[0])
        rx(phi, qv[1])
        sv = qv.qs.statevector(return_type="array")
        assert sv.dtype == np.dtype("O")

    def test_statevector_array_symbolic_after_entangling_gate(self):
        """A symbolic gate after an entangling numeric gate does not raise.

        Regression test: ``h``/``cx`` leave the state in ``complex64``, and the
        symbolic ``rx`` then hands the simulator an ``object`` gate matrix.
        Casting that matrix to the state dtype raises
        ``TypeError: Cannot convert expression to float``.
        """
        phi = symbols("phi")
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        rx(phi, qv[1])

        sv = qv.qs.statevector(return_type="array")
        assert sv.dtype == np.dtype("O")

        bound = np.array([complex(sympy.sympify(entry).subs({phi: 0.9})) for entry in sv])
        assert np.isclose(np.linalg.norm(bound), 1.0, atol=1e-5)

    def test_get_measurement_with_subs_dic_matches_numeric(self):
        """Binding a symbol via ``subs_dic`` reproduces the numeric measurement."""
        phi = symbols("phi")
        symbolic_qv = QuantumVariable(1)
        rx(phi, symbolic_qv[0])
        symbolic_result = symbolic_qv.get_measurement(subs_dic={phi: np.pi / 3})

        numeric_qv = QuantumVariable(1)
        rx(np.pi / 3, numeric_qv[0])
        numeric_result = numeric_qv.get_measurement()

        assert symbolic_result.keys() == numeric_result.keys()
        for outcome, probability in numeric_result.items():
            assert np.isclose(symbolic_result[outcome], probability, atol=1e-4)
