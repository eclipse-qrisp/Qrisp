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

"""Tests for standard operations defined in qrisp.circuit.standard_operations."""

import numpy as np
import pytest
import sympy

from qrisp.circuit import U3Gate
from qrisp.circuit.standard_operations import (
    Barrier,
    CPGate,
    CXGate,
    CYGate,
    CZGate,
    IDGate,
    MCRXGate,
    MCXGate,
    Measurement,
    QubitAlloc,
    QubitDealloc,
    Reset,
    RGate,
    RXGate,
    RXXGate,
    RZZGate,
    SwapGate,
    SXDGGate,
    SXGate,
    U1Gate,
    XXYYGate,
    u3Gate,
)


class TestSXGate:
    """Tests for SXGate (sqrt-X) and SXDGGate (its adjoint)."""

    def test_sx_name_and_params(self):
        """SXGate has name 'sx' and no parameters."""
        gate = SXGate()
        assert gate.name == "sx"
        assert gate.params == []

    def test_sx_permeability_and_qfree(self):
        """SXGate is not permeable and not qfree."""
        gate = SXGate()
        assert gate.permeability[0] is False
        assert gate.is_qfree is False

    def test_sx_unitary(self):
        """SXGate unitary matches RX(π/2) = (1/√2)*[[1, -i], [-i, 1]]."""
        gate = SXGate()
        s = 1 / np.sqrt(2)
        expected = np.array([[s, -1j * s], [-1j * s, s]], dtype=complex)
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_sx_squared_is_x_up_to_phase(self):
        """SX @ SX = -i·X (RX(π/2) composed twice = RX(π) = -i·X)."""
        sx = SXGate().get_unitary().astype(complex)
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(sx @ sx, -1j * x, atol=1e-6)

    def test_sxdg_name_and_params(self):
        """SXDGGate has name 'sx_dg' and no parameters."""
        gate = SXDGGate()
        assert gate.name == "sx_dg"
        assert gate.params == []

    def test_sxdg_unitary(self):
        """SXDGGate unitary is the conjugate transpose of SXGate."""
        sx = SXGate().get_unitary()
        sxdg = SXDGGate().get_unitary()
        assert np.allclose(sxdg, sx.conj().T, atol=1e-10)

    def test_sxdg_permeability_and_qfree(self):
        """SXDGGate is not permeable and not qfree."""
        gate = SXDGGate()
        assert gate.permeability[0] is False
        assert gate.is_qfree is False

    def test_sx_sxdg_compose_to_identity(self):
        """SX @ SX† = I."""
        sx = SXGate().get_unitary().astype(complex)
        sxdg = SXDGGate().get_unitary().astype(complex)
        assert np.allclose(sx @ sxdg, np.eye(2), atol=1e-6)


class TestIDGate:
    """Tests for the single-qubit identity gate."""

    def test_id_name(self):
        """IDGate has name 'id'."""
        assert IDGate().name == "id"

    def test_id_unitary_is_identity(self):
        """IDGate unitary is the 2x2 identity matrix."""
        assert np.allclose(IDGate().get_unitary(), np.eye(2), atol=1e-10)


class TestU1Gate:
    """Tests for U1Gate (equivalent to RZGate up to global phase)."""

    def test_u1_name_and_params(self):
        """U1Gate has name 'u1' and stores phi in params."""
        phi = 1.5
        gate = U1Gate(phi)
        assert gate.name == "u1"
        assert gate.params == [phi]

    def test_u1_permeability_and_qfree(self):
        """U1Gate is permeable and qfree."""
        gate = U1Gate(1.0)
        assert gate.permeability[0] is True
        assert gate.is_qfree is True

    @pytest.mark.parametrize("phi", [0.0, np.pi / 4, np.pi / 2, np.pi, 2.5])
    def test_u1_unitary(self, phi):
        """U1Gate(phi) unitary matches diag(exp(-i*phi/2), exp(i*phi/2))."""
        gate = U1Gate(phi)
        expected = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)


class TestRGate:
    """Tests for the R gate (rotation around cos(phi)*X + sin(phi)*Y axis)."""

    def test_r_name_and_params(self):
        """RGate stores [theta, phi] in params and has name 'r'."""
        theta, phi = 1.2, 0.8
        gate = RGate(theta, phi)
        assert gate.name == "r"
        assert gate.params == [theta, phi]

    @pytest.mark.parametrize(
        "theta, phi",
        [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi / 2, 0.0),
            (np.pi, np.pi / 2),
            (1.2, 0.7),
            (2.5, -1.3),
        ],
    )
    def test_r_unitary(self, theta, phi):
        """R(theta, phi) unitary matches the analytical formula."""
        gate = RGate(theta, phi)
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        expected = np.array(
            [[c, np.exp(-1j * phi) * s], [-np.exp(1j * phi) * s, c]],
            dtype=complex,
        )
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_r_zero_theta_is_identity(self):
        """R(0, phi) is identity for any phi."""
        for phi in [0.0, 1.0, np.pi]:
            gate = RGate(0.0, phi)
            assert np.allclose(gate.get_unitary(), np.eye(2), atol=1e-10)


class TestCPGate:
    """Tests for CPGate and its special-case short-circuits."""

    def test_cp_near_pi_returns_cz(self):
        """CPGate(phi ≈ π) returns a CZ gate."""
        gate = CPGate(np.pi)
        assert gate.name == "cz"

    def test_cp_near_pi_tolerance(self):
        """CPGate returns CZ for values within 1e-8 of π."""
        gate = CPGate(np.pi + 5e-9)
        assert gate.name == "cz"

    def test_cp_near_zero_returns_empty_circuit(self):
        """CPGate(phi ≈ 0) returns an empty 2-qubit gate with params=[0]."""
        gate = CPGate(0.0)
        assert gate.name == "cp"
        assert gate.params == [0]
        assert gate.num_qubits == 2

    def test_cp_near_two_pi_returns_empty_circuit(self):
        """CPGate(2π) is equivalent to 0 and returns the empty gate."""
        gate = CPGate(2 * np.pi)
        assert gate.name == "cp"
        assert gate.params == [0]

    @pytest.mark.parametrize("phi", [np.pi / 4, np.pi / 2, 1.5, 2.0])
    def test_cp_generic_phase_unitary(self, phi):
        """CPGate(phi) unitary matches diag(1,1,1,exp(i*phi))."""
        gate = CPGate(phi)
        expected = np.diag([1.0, 1.0, 1.0, np.exp(1j * phi)])
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_cp_symbolic_returns_controlled_p(self):
        """CPGate with a sympy symbol returns a controlled PGate."""
        phi = sympy.Symbol("phi")
        gate = CPGate(phi)
        assert gate.num_qubits == 2


class TestMCXGate:
    """Tests for MCXGate (multi-controlled X)."""

    def test_mcx_default_is_cx(self):
        """MCXGate() with default control_amount=1 matches CXGate."""
        mcx = MCXGate(control_amount=1)
        cx = CXGate()
        assert np.allclose(mcx.get_unitary(), cx.get_unitary(), atol=1e-10)

    @pytest.mark.parametrize("n_ctrl", [1, 2, 3])
    def test_mcx_num_qubits(self, n_ctrl):
        """MCXGate has n_ctrl + 1 qubits."""
        gate = MCXGate(control_amount=n_ctrl)
        assert gate.num_qubits == n_ctrl + 1

    def test_mcx_2ctrl_unitary(self):
        """MCXGate(2) (Toffoli) unitary flips the target only when both controls are |1⟩."""
        gate = MCXGate(control_amount=2)
        u = gate.get_unitary()
        expected = np.eye(8, dtype=complex)
        expected[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        assert np.allclose(u, expected, atol=1e-10)


class TestMCRXGate:
    """Tests for MCRXGate (multi-controlled RX)."""

    def test_mcrx_name(self):
        """MCRXGate has name 'mcrx'."""
        assert MCRXGate(1.0, control_amount=1).name == "mcrx"

    def test_mcrx_1ctrl_unitary(self):
        """MCRXGate with 1 control applies RX only when control is |1⟩."""
        phi = 1.0
        gate = MCRXGate(phi, control_amount=1)
        u = gate.get_unitary()
        rx_u = RXGate(phi).get_unitary()
        expected = np.eye(4, dtype=complex)
        expected[2:4, 2:4] = rx_u
        assert np.allclose(u, expected, atol=1e-10)


class TestSwapGate:
    """Tests for SwapGate."""

    def test_swap_permeability_and_qfree(self):
        """SwapGate has permeability False on both qubits but is qfree."""
        gate = SwapGate()
        assert gate.permeability[0] is False
        assert gate.permeability[1] is False
        assert gate.is_qfree is True

    def test_swap_unitary(self):
        """SwapGate unitary exchanges the two qubit states."""
        expected = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=complex,
        )
        assert np.allclose(SwapGate().get_unitary(), expected, atol=1e-10)

    def test_swap_is_self_inverse(self):
        """SWAP @ SWAP = I."""
        u = SwapGate().get_unitary()
        assert np.allclose(u @ u, np.eye(4), atol=1e-10)

    def test_swap_inverse_is_self(self):
        """SwapGate.inverse() returns a gate with the same unitary."""
        gate = SwapGate()
        inv = gate.inverse()
        assert np.allclose(inv.get_unitary(), gate.get_unitary(), atol=1e-10)


class TestRXXGate:
    """Tests for RXXGate (Ising XX-coupling)."""

    def test_rxx_name_and_params(self):
        """RXXGate has name 'rxx' and stores phi in params."""
        phi = 1.2
        gate = RXXGate(phi)
        assert gate.name == "rxx"
        assert gate.params == [phi]

    def test_rxx_permeability_and_qfree(self):
        """RXXGate is not permeable on either qubit and not qfree."""
        gate = RXXGate(1.0)
        assert gate.permeability[0] is False
        assert gate.permeability[1] is False
        assert gate.is_qfree is False

    @pytest.mark.parametrize("phi", [0.0, np.pi / 4, np.pi / 2, np.pi, 1.5])
    def test_rxx_unitary(self, phi):
        """RXX(phi) = exp(-i*phi/2 * X⊗X) matches the analytical formula."""
        gate = RXXGate(phi)
        c = np.cos(phi / 2)
        s = np.sin(phi / 2)
        expected = np.array(
            [
                [c, 0, 0, -1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [-1j * s, 0, 0, c],
            ],
            dtype=complex,
        )
        # Uses atol=1e-6 because the unitary is computed from a circuit decomposition.
        assert np.allclose(gate.get_unitary(), expected, atol=1e-6)

    def test_rxx_is_unitary(self):
        """RXXGate unitary satisfies U @ U† = I."""
        u = RXXGate(1.3).get_unitary()
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-6)


class TestRZZGate:
    """Tests for RZZGate (Ising ZZ-coupling)."""

    def test_rzz_name_and_params(self):
        """RZZGate has name 'rzz' and stores phi in params."""
        phi = 0.9
        gate = RZZGate(phi)
        assert gate.name == "rzz"
        assert gate.params == [phi]

    def test_rzz_permeability_and_qfree(self):
        """RZZGate is permeable on both qubits and qfree."""
        gate = RZZGate(1.0)
        assert gate.permeability[0] is True
        assert gate.permeability[1] is True
        assert gate.is_qfree is True

    @pytest.mark.parametrize("phi", [0.0, np.pi / 4, np.pi / 2, np.pi, 1.5])
    def test_rzz_unitary(self, phi):
        """RZZ(phi) = exp(-i*phi/2 * Z⊗Z) matches the analytical formula."""
        gate = RZZGate(phi)
        expected = np.diag(
            [
                np.exp(-1j * phi / 2),
                np.exp(1j * phi / 2),
                np.exp(1j * phi / 2),
                np.exp(-1j * phi / 2),
            ]
        )
        assert np.allclose(gate.get_unitary(), expected, atol=1e-10)

    def test_rzz_is_unitary(self):
        """RZZGate unitary satisfies U @ U† = I."""
        u = RZZGate(2.1).get_unitary()
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-10)


class TestXXYYGate:
    """Tests for XXYYGate (XX+YY interaction)."""

    def test_xxyy_name_and_params(self):
        """XXYYGate has name 'xxyy' and stores [phi, beta] in params."""
        phi, beta = 1.2, 0.5
        gate = XXYYGate(phi, beta)
        assert gate.name == "xxyy"
        assert gate.params == [phi, beta]

    def test_xxyy_is_unitary(self):
        """XXYYGate unitary satisfies U @ U† = I."""
        u = XXYYGate(1.2, 0.5).get_unitary()
        # Uses atol=1e-6 because the unitary is computed from a circuit decomposition
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-6)

    def test_xxyy_zero_phi_is_identity(self):
        """XXYYGate(0, beta) is the identity for any beta."""
        for beta in [0.0, 1.0, np.pi]:
            gate = XXYYGate(0.0, beta)
            # Uses atol=1e-6 because the unitary is computed from a circuit decomposition
            assert np.allclose(gate.get_unitary(), np.eye(4), atol=1e-6)

    def test_xxyy_unitary_matches_definition(self):
        """XXYYGate unitary matches its decomposed definition circuit."""
        phi, beta = 1.2, 0.5
        gate = XXYYGate(phi, beta)
        u_direct = gate.get_unitary()
        u_from_def = gate.definition.transpile().get_unitary()
        assert np.allclose(u_direct, u_from_def, atol=1e-10)


class TestMeasurement:
    """Tests for the Measurement operation."""

    def test_measurement_name_and_shape(self):
        """Measurement has name 'measure', 1 qubit and 1 classical bit."""
        op = Measurement()
        assert op.name == "measure"
        assert op.num_qubits == 1
        assert op.num_clbits == 1


class TestReset:
    """Tests for the Reset operation."""

    def test_reset_name_and_shape(self):
        """Reset has name 'reset' and acts on 1 qubit."""
        op = Reset()
        assert op.name == "reset"
        assert op.num_qubits == 1


class TestBarrier:
    """Tests for the Barrier operation."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_barrier_num_qubits(self, n):
        """Barrier spans the requested number of qubits."""
        assert Barrier(n).num_qubits == n

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_barrier_permeability_all_false(self, n):
        """Barrier permeability is False on every qubit it spans."""
        gate = Barrier(n)
        for i in range(n):
            assert gate.permeability[i] is False


class TestQubitAllocDealloc:
    """Tests for internal QubitAlloc and QubitDealloc markers."""

    def test_qubit_alloc_name_and_unitary(self):
        """QubitAlloc has name 'qb_alloc' and an identity unitary."""
        op = QubitAlloc()
        assert op.name == "qb_alloc"
        assert np.allclose(op.unitary, np.eye(2), atol=1e-10)

    def test_qubit_dealloc_name_and_unitary(self):
        """QubitDealloc has name 'qb_dealloc' and an identity unitary."""
        op = QubitDealloc()
        assert op.name == "qb_dealloc"
        assert np.allclose(op.unitary, np.eye(2), atol=1e-10)


class TestU3GateAlias:
    """Tests for the u3Gate factory (thin alias for U3Gate)."""

    def test_u3gate_alias_returns_u3gate(self):
        """u3Gate returns an instance of U3Gate."""
        assert isinstance(u3Gate(1.0, 2.0, 3.0), U3Gate)

    def test_u3gate_alias_same_unitary(self):
        """u3Gate unitary matches U3Gate with the same parameters."""
        theta, phi, lam = 1.0, 2.0, 3.0
        assert np.allclose(
            u3Gate(theta, phi, lam).get_unitary(),
            U3Gate(theta, phi, lam).get_unitary(),
            atol=1e-10,
        )


class TestControlledPauliGates:
    """Tests for CXGate, CYGate, CZGate."""

    def test_cx_unitary(self):
        """CXGate unitary matches the standard CNOT matrix."""
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        assert np.allclose(CXGate().get_unitary(), expected, atol=1e-10)

    def test_cy_unitary(self):
        """CYGate unitary applies Y on the target when control is |1⟩."""
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            dtype=complex,
        )
        assert np.allclose(CYGate().get_unitary(), expected, atol=1e-10)

    def test_cz_unitary(self):
        """CZGate unitary matches diag(1,1,1,-1)."""
        expected = np.diag([1.0, 1.0, 1.0, -1.0])
        assert np.allclose(CZGate().get_unitary(), expected, atol=1e-10)
