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
import sympy
from numpy.linalg import norm

from qrisp.circuit import (
    ControlledOperation,
    PTControlledOperation,
    QuantumCircuit,
    U3Gate,
    transpile,
)
from qrisp.circuit.operation import PauliGate
from qrisp.circuit.standard_operations import (
    GPhaseGate,
    HGate,
    PGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    TGate,
    SXGate,
)


# ---------------------------------------------------------------------------
# Helper: analytical U3 unitary
# ---------------------------------------------------------------------------


def _u3_unitary(theta, phi, lam, global_phase=0.0):
    """Compute the analytical 2x2 unitary of a U3 gate.

    U3(theta, phi, lambda) =
        [[cos(theta/2),           -exp(i*lambda)*sin(theta/2)],
         [exp(i*phi)*sin(theta/2), exp(i*(phi+lambda))*cos(theta/2)]]

    Multiplied by exp(i * global_phase).
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    mat = np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )
    return np.exp(1j * global_phase) * mat


def _controlled_unitary(u, ctrl_state="1"):
    """Build the controlled-unitary matrix for given control qubits and ctrl_state.

    Parameters
    ----------
    u : ndarray
        The target unitary (2x2 for a single target qubit).
    ctrl_state : str
        String of '0's and '1's indicating the state of each control qubit.
        Default "1" gives a single control qubit active on |1⟩ (I ⊕ u).

    Returns
    -------
    ndarray
        The controlled unitary matrix of shape (2**(k+1), 2**(k+1))
        where k = len(ctrl_state).
    """
    num_ctrl = len(ctrl_state)
    n = u.shape[0]
    full_dim = (2**num_ctrl) * n
    result = np.eye(full_dim, dtype=complex)

    ctrl_val = int(ctrl_state, 2)
    start = ctrl_val * n
    result[start : start + n, start : start + n] = u

    return result


# =============================================================================
# U3Gate – initialization & basic properties
# =============================================================================


class TestU3GateInitialization:
    """Tests for U3Gate construction and basic attributes."""

    def test_default_name(self):
        """U3Gate with default name is 'u3'."""
        gate = U3Gate(1.0, 2.0, 3.0)
        assert gate.name == "u3"
        assert gate.num_qubits == 1
        assert gate.num_clbits == 0

    def test_params_stored(self):
        """Theta, phi, lam are stored as .params and as attributes."""
        gate = U3Gate(0.5, 1.5, 2.5)
        assert gate.params == [0.5, 1.5, 2.5]
        assert gate.theta == 0.5
        assert gate.phi == 1.5
        assert gate.lam == 2.5

    def test_default_global_phase(self):
        """Global phase defaults to 0."""
        gate = U3Gate(1.0, 2.0, 3.0)
        assert gate.global_phase == 0

    def test_custom_global_phase(self):
        """Global phase can be set explicitly."""
        gate = U3Gate(1.0, 2.0, 3.0, global_phase=1.5)
        assert gate.global_phase == 1.5

    def test_zero_params(self):
        """All parameters can be zero."""
        gate = U3Gate(0.0, 0.0, 0.0, global_phase=0.0)
        assert gate.theta == 0.0
        assert gate.phi == 0.0
        assert gate.lam == 0.0
        assert gate.global_phase == 0.0

    def test_negative_params(self):
        """Negative parameters are accepted."""
        gate = U3Gate(-1.0, -2.0, -3.0, global_phase=-0.5)
        assert gate.theta == -1.0
        assert gate.phi == -2.0
        assert gate.lam == -3.0
        assert gate.global_phase == -0.5


# =============================================================================
# U3Gate – unitary matrix correctness
# =============================================================================


class TestU3GateUnitary:
    """Tests for U3Gate.get_unitary()."""

    @pytest.mark.parametrize(
        "theta, phi, lam, gp",
        [
            (0.0, 0.0, 0.0, 0.0),
            (np.pi, 0.0, 0.0, 0.0),
            (np.pi / 2, 0.0, 0.0, 0.0),
            (0.0, np.pi, 0.0, 0.0),
            (0.0, 0.0, np.pi, 0.0),
            (1.0, 2.0, 3.0, 0.0),
            (2.5, -1.3, 0.7, 0.0),
            (1.0, 2.0, 3.0, 4.0),
            (2.5740044828568522, 3.141592653589793, 3.141592653589793, 3.141592653589793),
            (2.5740044828568522, 3.141592653589793, 3.141592653589793, 0.0),
            (2.5740044828568522, 0.0, 0.0, 0.0),
            (2.5740044828568522, 0.0, 0.0, 1.0),
        ],
    )
    def test_unitary_matches_analytical(self, theta, phi, lam, gp):
        """get_unitary() matches the analytical U3 formula."""
        gate = U3Gate(theta, phi, lam, global_phase=gp)
        actual = gate.get_unitary()
        expected = _u3_unitary(theta, phi, lam, gp)
        assert np.allclose(actual, expected, atol=1e-10)

    def test_unitary_is_cached(self):
        """Subsequent calls to get_unitary() return the cached matrix."""
        gate = U3Gate(1.0, 2.0, 3.0)
        u1 = gate.get_unitary()
        u2 = gate.get_unitary()
        assert u1 is u2  # same object, cached

    def test_unitary_decimals_rounding(self):
        """get_unitary(decimals=N) returns a rounded matrix."""
        gate = U3Gate(1.23456789, 2.3456789, 3.456789)
        u = gate.get_unitary(decimals=3)
        assert np.allclose(u, np.around(gate.get_unitary(), 3))


# =============================================================================
# U3Gate – inverse
# =============================================================================


class TestU3GateInverse:
    """Tests for U3Gate.inverse()."""

    @pytest.mark.parametrize(
        "theta, phi, lam, gp",
        [
            (1.0, 2.0, 3.0, 0.0),
            (2.5, -1.3, 0.7, 0.5),
            (np.pi, np.pi / 2, -np.pi / 3, 1.0),
            (0.0, 0.0, 0.0, 0.0),
        ],
    )
    def test_inverse_unitary_is_dagger(self, theta, phi, lam, gp):
        """U.inverse().get_unitary() == U.get_unitary().conj().T."""
        gate = U3Gate(theta, phi, lam, global_phase=gp)
        inv_gate = gate.inverse()
        expected = gate.get_unitary().conj().T
        assert np.allclose(inv_gate.get_unitary(), expected, atol=1e-10)

    def test_inverse_name_u3_unchanged(self):
        """Inverse of a generic u3 gate keeps the name 'u3'."""
        gate = U3Gate(1.0, 2.0, 3.0)
        inv = gate.inverse()
        assert inv.name == "u3"

    def test_inverse_params_negated(self):
        """Inverse has negated theta, lam, phi and global_phase."""
        gate = U3Gate(1.0, 2.0, 3.0, global_phase=4.0)
        inv = gate.inverse()
        assert inv.theta == -1.0
        assert inv.phi == -3.0  # note: phi ← -lam, lam ← -phi
        assert inv.lam == -2.0
        assert inv.global_phase == -4.0

    def test_inverse_preserves_permeability(self):
        """Inverse preserves the permeability dict."""
        gate = U3Gate(1.0, 2.0, 3.0)
        gate.permeability = {0: True}
        inv = gate.inverse()
        assert inv.permeability == {0: True}

    def test_inverse_preserves_is_qfree(self):
        """Inverse preserves is_qfree."""
        gate = U3Gate(1.0, 2.0, 3.0)
        gate.is_qfree = True
        inv = gate.inverse()
        assert inv.is_qfree is True

    def test_double_inverse_is_identity(self):
        """Two successive inverses return the original unitary."""
        gate = U3Gate(1.5, -0.7, 2.1, global_phase=0.3)
        inv_inv = gate.inverse().inverse()
        assert np.allclose(inv_inv.get_unitary(), gate.get_unitary(), atol=1e-10)


# =============================================================================
# U3Gate – controlled gate (1 ctrl qubit)
# =============================================================================


class TestU3GateControl:
    """Tests for U3Gate.control()."""

    def test_control_returns_controlled_operation(self):
        """control() returns a ControlledOperation for default method."""
        gate = U3Gate(1.0, 2.0, 3.0)
        cgate = gate.control(1)
        assert isinstance(cgate, ControlledOperation)
        assert cgate.name == "cu3"

    def test_pt_control_returns_pt_controlled_operation(self):
        """control(method='gray_pt') on 2+ ctrls returns PTControlledOperation.

        For a single control qubit, gray_pt and gray behave identically
        because no phase tolerance is needed. The result is a plain
        ControlledOperation.
        """
        gate = U3Gate(1.0, 2.0, 3.0)
        # Single-ctrl: gray_pt is equivalent to gray -> ControlledOperation
        cgate_1 = gate.control(1, method="gray_pt")
        assert isinstance(cgate_1, ControlledOperation)
        assert cgate_1.name == "cu3"

        # Multi-ctrl: gray_pt returns a PTControlledOperation
        cgate_2 = gate.control(2, method="gray_pt")
        assert isinstance(cgate_2, PTControlledOperation)

    @pytest.mark.parametrize(
        "theta, phi, lam, gp",
        [
            # Exact reproduction of the reported bug
            (2.5740044828568522, 3.141592653589793, 3.141592653589793, 3.141592653589793),
            (2.5740044828568522, 3.141592653589793, 3.141592653589793, 0.0),
            (2.5740044828568522, 0.0, 0.0, 0.0),
            (2.5740044828568522, 0.0, 0.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
            # Additional test points
            (0.5, 1.2, 0.8, 2.1),
            (np.pi, np.pi / 2, np.pi / 4, np.pi / 3),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.5),
            (1.5, 0.0, 0.0, 0.0),
            (0.0, 1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5, 0.0),
        ],
    )
    def test_controlled_unitary_vs_transpiled_definition(self, theta, phi, lam, gp):
        """The unitary of a controlled U3Gate must match its transpiled definition.

        This is the core regression test for the global-phase bug.
        """
        gate = U3Gate(theta, phi, lam, global_phase=gp)
        cgate = gate.control(1)
        unitary_direct = cgate.get_unitary()
        unitary_from_def = cgate.definition.transpile().get_unitary()
        assert norm(unitary_direct - unitary_from_def) < 1e-4

    def test_controlled_unitary_matches_analytical(self):
        """Controlled U3 unitary matches analytical controlled-unitary formula."""

        for gate in [
            TGate(),
            SGate(),
            TGate().inverse(),
            SGate().inverse(),
            SXGate(),
            SXGate().inverse(),
            U3Gate(1, 2, 3),
        ]:
            for i in range(1, 3):
                ctrl_state = "1" * i
                expected_unitary = _controlled_unitary(gate.get_unitary(), ctrl_state)
                compiled_unitary = gate.control(i, ctrl_state=ctrl_state).definition.get_unitary()
                assert np.linalg.norm(expected_unitary - compiled_unitary) < 1e-4

    def test_non_trivial_control_state(self):

        for gate in [
            TGate(),
            SGate(),
            TGate().inverse(),
            SGate().inverse(),
            SXGate(),
            SXGate().inverse(),
            U3Gate(1, 2, 3),
        ]:
            for i in range(1, 3):
                ctrl_state = "0" * i
                expected_unitary = _controlled_unitary(gate.get_unitary(), ctrl_state)
                compiled_unitary = gate.control(i, ctrl_state=ctrl_state).definition.get_unitary()

                assert np.linalg.norm(expected_unitary - compiled_unitary) < 1e-4


# =============================================================================
# U3Gate – multi-controlled gates (2+ ctrl qubits)
# =============================================================================


class TestU3GateMultiControl:
    """Tests for U3Gate with multiple control qubits."""

    @pytest.mark.parametrize("num_ctrl", [2, 3])
    @pytest.mark.parametrize(
        "theta, phi, lam, gp",
        [
            (1.0, 2.0, 3.0, 0.0),
            (2.5, -1.3, 0.7, 1.2),
            (0.0, 0.0, 0.0, 0.0),
        ],
    )
    def test_multi_controlled_unitary_vs_transpiled(self, num_ctrl, theta, phi, lam, gp):
        """Multi-controlled U3 gate unitary matches transpiled definition."""
        gate = U3Gate(theta, phi, lam, global_phase=gp)
        cgate = gate.control(num_ctrl)
        unitary_direct = cgate.get_unitary()
        unitary_from_def = cgate.definition.transpile().get_unitary()
        # Multi-controlled definitions are more complex; relax tolerance
        assert norm(unitary_direct - unitary_from_def) < 1e-3


# =============================================================================
# U3Gate – named variants (p, rx, ry, rz, h, s, t, …)
# =============================================================================


class TestU3GateNamedVariants:
    """Tests for U3Gate when constructed with special names."""

    def test_p_gate(self):
        """P gate via PGate."""
        lam = 1.5
        gate = PGate(lam)
        assert gate.name == "p"
        assert gate.params == [lam]
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

        expected = np.diag([1.0, np.exp(1j * lam)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_rz_gate(self):
        """RZ gate via RZGate.

        Standard convention: RZ(lam) = diag(exp(-i*lam/2), exp(i*lam/2)).
        """
        lam = 2.0
        gate = RZGate(lam)
        assert gate.name == "rz"
        assert gate.params == [lam]
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

        expected = np.diag([np.exp(-1j * lam / 2), np.exp(1j * lam / 2)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_rx_gate(self):
        """RX gate via RXGate."""
        theta = 1.2
        gate = RXGate(theta)
        assert gate.name == "rx"
        assert gate.params == [theta]
        assert gate.permeability[0] is False
        assert gate.is_qfree is False

        # RX(theta) = exp(-i * theta/2 * X)
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        expected = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_ry_gate(self):
        """RY gate via RYGate."""
        theta = 0.8
        gate = RYGate(theta)
        assert gate.name == "ry"
        assert gate.params == [theta]
        assert gate.permeability[0] is False
        assert gate.is_qfree is False

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        expected = np.array([[c, -s], [s, c]], dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_gphase_gate(self):
        """Global phase gate via GPhaseGate."""
        gp = 0.75
        gate = GPhaseGate(gp)
        assert gate.name == "gphase"
        assert gate.params == [gp]
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

        expected = np.exp(1j * gp) * np.eye(2, dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_h_gate(self):
        """H gate via HGate."""
        gate = HGate()
        assert gate.name == "h"
        assert gate.params == []
        assert gate.permeability[0] is False
        assert gate.is_qfree is False

        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_s_gate(self):
        """S gate via SGate."""
        gate = SGate()
        assert gate.name == "s"
        assert gate.params == []
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

        expected = np.diag([1.0, 1j])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_s_dg_gate(self):
        """S† gate via SGate().inverse()."""
        gate = SGate().inverse()
        assert gate.name == "s_dg"

        expected = np.diag([1.0, -1j])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_t_gate(self):
        """T gate via TGate."""
        gate = TGate()
        assert gate.name == "t"
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

        expected = np.diag([1.0, np.exp(1j * np.pi / 4)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_t_dg_gate(self):
        """T† gate via TGate().inverse()."""
        gate = TGate().inverse()
        assert gate.name == "t_dg"

        expected = np.diag([1.0, np.exp(-1j * np.pi / 4)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_inverse_named_gate_name_convention(self):
        """Inverse of s_dg is s, inverse of t_dg is t, inverse of h is h."""
        assert SGate().inverse().inverse().name == "s"
        assert TGate().inverse().inverse().name == "t"
        assert HGate().inverse().name == "h"


# =============================================================================
# U3Gate – controlled named variants
# =============================================================================


class TestU3GateNamedVariantControl:
    """Tests for controlled versions of named U3 variant gates."""

    @pytest.mark.parametrize(
        "name, gate_factory, args",
        [
            ("p", PGate, (1.5,)),
            ("rz", RZGate, (2.0,)),
            ("rx", RXGate, (1.2,)),
            ("ry", RYGate, (0.8,)),
        ],
    )
    def test_named_variant_controlled_unitary_vs_transpiled(self, name, gate_factory, args):
        """Controlled named-gate unitary must match its transpiled definition."""
        gate = gate_factory(*args)
        cgate = gate.control(1)
        unitary_direct = cgate.get_unitary()
        unitary_from_def = cgate.definition.transpile().get_unitary()
        assert norm(unitary_direct - unitary_from_def) < 1e-4


# =============================================================================
# U3Gate – bind_parameters / abstract params
# =============================================================================


class TestU3GateAbstractParams:
    """Tests for U3Gate with sympy abstract parameters."""

    def test_abstract_params_from_sympy(self):
        """Sympy expressions in params are added to abstract_params."""
        a = sympy.Symbol("a")
        gate = U3Gate(a, 0.0, 0.0)
        assert a in gate.abstract_params
        assert gate.abstract_params == {a}

    def test_abstract_params_from_global_phase(self):
        """Sympy global_phase adds to abstract_params."""
        d = sympy.Symbol("d")
        gate = U3Gate(1.0, 2.0, 3.0, global_phase=d)
        assert d in gate.abstract_params

    def test_constant_sympy_global_phase_no_abstract(self):
        """A sympy expression with no free symbols is treated as numeric."""
        gate = U3Gate(1.0, 2.0, 3.0, global_phase=sympy.Float(2.5))
        assert len(gate.abstract_params) == 0
        assert isinstance(gate.global_phase, float)
        assert gate.global_phase == 2.5

    def test_bind_parameters_numeric(self):
        """bind_parameters substitutes numeric values into sympy params."""
        a = sympy.Symbol("a")
        gate = U3Gate(a, 2.0, 3.0, global_phase=0.0)
        bound = gate.bind_parameters({a: 1.5})
        assert bound.theta == 1.5
        assert bound.phi == 2.0
        assert bound.lam == 3.0
        assert bound.global_phase == 0.0

    def test_bind_parameters_global_phase(self):
        """bind_parameters works when global_phase is also symbolic."""
        a, d = sympy.Symbol("a"), sympy.Symbol("d")
        gate = U3Gate(a, 2.0, 3.0, global_phase=d)
        bound = gate.bind_parameters({a: 1.5, d: 0.75})
        assert bound.theta == 1.5
        assert bound.global_phase == 0.75


# =============================================================================
# U3Gate – PauliGate subclass
# =============================================================================


class TestPauliGate:
    """Tests for the PauliGate subclass of U3Gate."""

    def test_x_gate(self):
        """Pauli X gate matches expected unitary."""
        gate = PauliGate("x")
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)
        assert gate.name == "x"

    def test_y_gate(self):
        """Pauli Y gate matches expected unitary."""
        gate = PauliGate("y")
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_z_gate(self):
        """Pauli Z gate matches expected unitary."""
        gate = PauliGate("z")
        expected = np.diag([1.0, -1.0])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_pauli_inverse_is_self_up_to_phase(self):
        """X, Y, Z are self-inverse up to a global phase."""
        for name in ["x", "y", "z"]:
            gate = PauliGate(name)
            inv = gate.inverse()
            # Pauli gates square to I (up to global phase)
            prod = gate.get_unitary() @ inv.get_unitary()
            # Should be proportional to identity
            assert np.allclose(np.abs(prod[0, 0]), 1.0, atol=1e-10)
            assert np.allclose(np.abs(prod[0, 1]), 0.0, atol=1e-10)

    def test_x_controlled_is_cx(self):
        """Controlled X gate is named 'cx' and has correct unitary."""
        cx = PauliGate("x").control(1)
        assert cx.name == "cx"
        expected_cx = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        assert np.allclose(cx.get_unitary(), expected_cx, atol=1e-10)

    def test_z_controlled_is_cz(self):
        """Controlled Z gate is named 'cz' and has correct unitary."""
        cz = PauliGate("z").control(1)
        assert cz.name == "cz"
        expected_cz = np.diag([1.0, 1.0, 1.0, -1.0])
        assert np.allclose(cz.get_unitary(), expected_cz, atol=1e-10)

    def test_invalid_pauli_raises(self):
        """PauliGate('w') raises Exception."""
        with pytest.raises(Exception, match="is not a Pauli gate"):
            PauliGate("w")


# =============================================================================
# U3Gate – transpile roundtrip
# =============================================================================


class TestU3GateTranspile:
    """Tests for transpiling circuits containing U3 gates."""

    def test_single_u3_transpile_preserves_unitary(self):
        """Transpiling a circuit with one U3Gate preserves the unitary."""
        theta, phi, lam, gp = 1.0, 2.0, 3.0, 0.5
        qc = QuantumCircuit(1)
        qc.append(U3Gate(theta, phi, lam, global_phase=gp), qc.qubits)
        transpiled = transpile(qc)
        assert np.allclose(qc.get_unitary(), transpiled.get_unitary(), atol=1e-10)

    def test_two_u3_transpile_preserves_unitary(self):
        """Transpiling two consecutive U3 gates preserves the product unitary."""
        qc = QuantumCircuit(1)
        qc.append(U3Gate(1.0, 0.5, 0.3, global_phase=0.1), [qc.qubits[0]])
        qc.append(U3Gate(0.7, -0.2, 1.1, global_phase=0.3), [qc.qubits[0]])
        transpiled = transpile(qc)
        assert np.allclose(qc.get_unitary(), transpiled.get_unitary(), atol=1e-10)


# =============================================================================
# U3Gate – edge cases
# =============================================================================


class TestU3GateEdgeCases:
    """Edge cases for U3Gate."""

    def test_identity_gate(self):
        """U3Gate(0, 0, 0, global_phase=0) is identity."""
        gate = U3Gate(0.0, 0.0, 0.0)
        assert np.allclose(gate.get_unitary(), np.eye(2), atol=1e-10)

    def test_all_zero_params_with_phase(self):
        """U3Gate(0, 0, 0, global_phase=d) is exp(i*d)*I."""
        d = 2.5
        gate = U3Gate(0.0, 0.0, 0.0, global_phase=d)
        expected = np.exp(1j * d) * np.eye(2)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_large_theta(self):
        """Large theta value should work correctly (periodic)."""
        gate = U3Gate(100.0, 0.0, 0.0)
        # Should be equivalent modulo 4*pi
        expected = U3Gate(100.0 % (4 * np.pi), 0.0, 0.0)
        assert np.allclose(gate.get_unitary(), expected.get_unitary(), atol=1e-10)

    def test_negative_theta(self):
        """Negative theta is handled correctly."""
        gate = U3Gate(-1.5, 0.0, 0.0)
        inv = U3Gate(1.5, 0.0, 0.0).inverse()
        assert np.allclose(gate.get_unitary(), inv.get_unitary(), atol=1e-10)

    @pytest.mark.parametrize("gp", [0.0, 1.0, -1.0, np.pi, -np.pi, 10.0])
    def test_global_phase_only_changes_overall_phase(self, gp):
        """Changing global_phase only multiplies the unitary by exp(i*gp)."""
        theta, phi, lam = 1.0, 2.0, 3.0
        gate_no_gp = U3Gate(theta, phi, lam, global_phase=0.0)
        gate_gp = U3Gate(theta, phi, lam, global_phase=gp)
        ratio = gate_gp.get_unitary() / gate_no_gp.get_unitary()
        # All entries of the ratio should be the same (exp(i*gp))
        assert np.allclose(ratio, np.exp(1j * gp) * np.ones((2, 2)), atol=1e-10)
