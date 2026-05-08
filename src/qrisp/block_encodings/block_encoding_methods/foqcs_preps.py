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
from qrisp.core import QuantumVariable, Qubit
from qrisp.core.gate_application_functions import x, cx, ry
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_W_state
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from qrisp.environments import control
from collections.abc import Sequence
from functools import partial
from typing import Any
from qrisp.operators import QubitOperator

def foqcs_prep_heisenberg_1D(
    prep_qv: QuantumVariable | Sequence[Qubit],
    L: int,
    g: dict,
    J: dict,
    conjugate: bool = False
) -> None:
    """
    Parameters
    ----------
    prep_qv : QuantumVariable | Sequence[Qubit]
        Ancillae qubits register for PREP.

    L : int
        Number of system qubits in the Heisenberg chain.

    g : dict (length 3)
        Dictionary of local field coefficients. {"X": gx, "Y": gy, "Z": gz}

    J : dict (length 3)
        Dictionary of coupling coefficients for the Heisenberg interaction. {"X": Jx, "Y": Jy, "Z": Jz}

    conjugate : bool = False
        Indicates whether the prep is PREP_R or PREP_L.
        In case of PREP_L, the conjugates of g and J are used.
        The default is False, which indicates it is PREP_R.

    Raises
    ------
    ValueError
        If ``g`` or ``J`` has length .

    """
    # Check that received g and J dictionaries exactly contain the expected entries.
    req_keys = {"X", "Y", "Z"}

    g_keys = set(g.keys())
    J_keys = set(J.keys())

    if g_keys != req_keys:
        missing = req_keys - g_keys
        extra = g_keys - req_keys
        raise ValueError(
            f"g must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    if J_keys != req_keys:
        missing = req_keys - J_keys
        extra = J_keys - req_keys
        raise ValueError(
            f"J must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    extra_anc = 6 # Depends on the method and can be potentially decreased

    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")

    _g[0] = np.sqrt(g["X"] * L)
    _g[1] = np.sqrt(g["Y"] * L * -1j)
    _g[2] = np.sqrt(g["Z"] * L)
    _J[0] = np.sqrt(J["X"] * (L - 1))
    _J[1] = np.sqrt(J["Y"] * -(L - 1))
    _J[2] = np.sqrt(J["Z"] * (L - 1))

    # Normalization for state preparation
    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    if conjugate:
        _g = np.conj(_g)
        _J = np.conj(_J)

    # SUBPREP
    unbalanced_W_state(prep_qv[:extra_anc], np.block([_g, _J]), extra_anc)
    
    # PREP
    fh1 = extra_anc                # First qubit first half
    lh1 = extra_anc + L - 1        # Last qubit first half
    fh2 = extra_anc + L            # First qubit second half
    lh2 = extra_anc + (L * 2) - 1  # Last qubit second half

    # 3 specific CNOTs
    cx(prep_qv[0], prep_qv[lh1]) # from gx
    cx(prep_qv[1], prep_qv[lh1]) # from gy
    cx(prep_qv[2], prep_qv[lh2]) # from gz
    # 2 parallel Gamma gates
    theta = 2 * np.acos(np.sqrt(1 / (L - 1 + 1)))
    _gamma_gate(prep_qv[lh1 - 1], prep_qv[lh1], theta)
    _gamma_gate(prep_qv[lh2 - 1], prep_qv[lh2], theta)
    # 3 specific CNOTs
    cx(prep_qv[3], prep_qv[lh1 - 1]) # from Jx
    cx(prep_qv[4], prep_qv[lh1 - 1]) # from Jy
    cx(prep_qv[5], prep_qv[lh2 - 1]) # from Jz
    # 2 parallel delta gates
    theta_coeffs = []
    for i in range(1, L - 1):
        theta_coeffs.append(2 * np.acos(np.sqrt(1 / (i + 1))))
    _delta_gate(prep_qv[fh1:lh1], theta_coeffs)
    _delta_gate(prep_qv[fh2:lh2], theta_coeffs)
    # One spec CNOT
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # controlled CNOT ladder
    with control(prep_qv[4]): # from Jy
        _cx_ladder(prep_qv[fh1:lh1 + 1], L)
    # One spec CNOT
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # controlled CNOT ladder
    with control(prep_qv[5]): # from Jz
        _cx_ladder(prep_qv[fh2:], L)
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy
    # controlled Element-wise CNOT
    with control(prep_qv[4]): # from Jy
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:])
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy

def foqcs_prep_placeholder( qv: QuantumVariable, coeffs: list, some_param: float ) -> None:
    """
    TO-DO DOC
    """
    print("I am doing nothing")

###################################
############# Helpers #############
###################################
    
def get_foqcs_lcu_prep_num_of_ancillae(prep: partial, num_ops: int = 1) -> int:
    r"""
        Constructs a BlockEncoding using the Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) protocol.

        Parameters
        ----------
        prep : partial
            Partially initialised FOQCS-LCU PREP method.
        
        num_ops : int
            Number of operand qubits (L argument for FOQCS-LCU PREP routines).
            The default is 1.

        Returns
        -------
        int
            An integer with number of ancillae required by the received FOQCS-LCU PREP method
    """
    if prep.func == foqcs_prep_heisenberg_1D:
        return num_ops * 2 + 6
    elif prep.func == foqcs_prep_placeholder:
        return 0
    else:
        raise ValueError(f"Received unknown FOQCS-LCU PREP routine: {prep}")

def foqcs_analyze_operator(
        O: QubitOperator,
        L: int = -1,
        tol: float = 1e-12,
        raise_errors: bool = True
) -> dict:
    """
    TODO: DOC
    O: Qubit operator of form: O = X(0) + X(1) + 0.5 * Y(0) + 0.5 * Y(1) + 0.2 * Z(0) * Z(1)
    L: Number of interacting qubits. If not specified / set to -1, will infer the number of interacting qubits from the operator
    tol: Tolerance for considering the entry zero
    raise_errors: Flag deternmining whether to raise ValueError in case if operator is not compatible with FOQCS-LCU.

    Returns:
    dict {
        "method": str ("heisenberg" || "spin_glass")
        "L": int (number of intracting qubits)
        "g": dict ("X": np.arr, "Z": np.arr, "Y": np.arr)
        "J": dict ("X": np.arr, "Z": np.arr, "Y": np.arr) (In case of spin_glass, numpy arrays are 2-dim)
        is_hermitian: bool (Whether the operator is Hermitian - no imaginary coeffs)
    }
    """
    terms = O.to_pauli_coeff_dict()

    # When analyzys fails, it either throws a ValueError with failure reason,
    # or silently returns None if raise_errors flag is set to False.
    def fail(arg: str):
        if raise_errors:
            raise ValueError(arg)
        else:
            return None

    # Verify that operator exists and checks its length
    max_ind = -1
    for pauli_str in terms:
        for ind, _ in pauli_str:
            max_ind = max(max_ind, ind)
    if max_ind < 0:
        return fail(f"Received empty or constant operator: {O}")

    # Infer the length or sanity check the passed length.
    if L == -1:
        L = max_ind + 1
    elif L < max_ind + 1:
        return fail(f"Received L = {L}, while operator acts on {max_ind + 1} qubits.")

    pauli_to_ind = {"X": 0, "Z": 1, "Y": 2}

    g_dict = {"X": np.zeros(L, dtype="complex"), "Y": np.zeros(L, dtype="complex"), "Z": np.zeros(L, dtype="complex")}
    J_dict = {"X": np.zeros((L, L), dtype="complex"), "Y": np.zeros((L, L), dtype="complex"), "Z": np.zeros((L, L), dtype="complex")}

    is_hermitian = True # Assume hermitian until check

    # Verify operator against the spin-glass model.
        # Every non-zero term is Xi, Yi, Zi, XiXj, YiYj, ZiZj.
        # Fails if:
            # Has const cI
            # Mixed terms: XiZj, XiYj, YiZj
            # Three-or-more-body interaction: XiXjXk
    for pauli_str, coeff in terms.items():
        # All coeffs in arrays are zeroes, so just skip entry
        if np.isclose(coeff, 0, atol=tol):
            continue

        # Fail on a constant: cI
        if len(pauli_str) == 0:
            return fail(f"FOQCS-LCU does not support constant/identity terms, but received {(pauli_str, coeff)}")

        # Check for imaginary coeffs. Finding any means that the operator is not hermitian.
        if not np.isclose(np.imag(coeff), 0, atol = tol):
            is_hermitian = False

        # Inspect one-body terms Xi, Yi, Zi:
        if len(pauli_str) == 1:
            (i, pauli), = pauli_str
            g_dict[pauli][i] += coeff
            continue

        # Inspect two-body terms XiXj, YiYj, ZiZj:
        if len(pauli_str) == 2:
            (i, pauli_i), (j, pauli_j) = sorted(pauli_str)

            if pauli_i != pauli_j:
                return fail(f"FOQCS-LCU supports only same-axis couplings, but received: {pauli_i}({i}) * {pauli_j}({j})")

            # Should not happen.
            #if i == j:
            #    return fail(f"Coupling: {pauli_i}({i}) * {pauli_j}({j}) has the same index. This was not supposed to trigger.")

            J_dict[pauli_i][i, j] += coeff
            J_dict[pauli_i][j, i] += coeff
            continue

        if len(pauli_str) > 2:
            return fail(f"FOQCS-LCU supports only one and two-body interactions.")

    # Spin-glass has passed by this point. YAY! :D


    # Verify ourselves against more specific heisenberg-model.
        # Spin-glass-compatible
        # Must have form: Xi, Yi, Zi, XiXi+1, YiYi+1, ZiZi+1; local fields must be uniform.
        # Fails if:
            # Local field is position dependent: g_0X != g_1X, 0.5*X(0) + 0.7*X(1)
            # Has long-range couplings: X_0*X_2
            # Has NN couplings with different sterngths: 0.5*X(0)*X(1) + 0.9*X(1)*X(2) # Zeroes matter, don't pass 0.5*X(0)*X(1) + 0*X(1)*X(2) , so depends on L.
    is_heisenberg = True # Assume that is heisenberg before the check
    g_heis_dict = {}
    J_heis_dict = {}
 
    # Verify that all one-body (local) coeffs are the same (uniform).
    for pauli in {"X", "Z", "Y"}:
        values = g_dict[pauli][:]

        if not np.allclose(values, values[0], atol=tol):
            is_heisenberg = False
            #print("Heisenberg Rejected: non-uniform local interactions")
            break

        g_heis_dict[pauli] = values[0]
    
    # Verify couplings
    if is_heisenberg:
        for pauli in {"X", "Z", "Y"}:
            if L == 1:
                J_heis_dict[pauli] = 0
                continue
            
            nn_val = np.array(
                [J_dict[pauli][i, i + 1] for i in range(L - 1)],
                dtype = complex,
            )
            # All same-axis nearest-neighbours interactions must be uniform
            if not np.allclose(nn_val, nn_val[0], atol=tol):
                is_heisenberg = False
                #print("Heisenberg Rejected: non-uniform NN interactions")
                break

            J_heis_dict[pauli] = nn_val[0]

            # Reject non-nearest-neighbour couplings
            for i in range(L):
                for j in range(i + 1, L):
                    if j == i + 1:
                        continue
                    if not np.isclose(J_dict[pauli][i, j], 0, atol=tol):
                        is_heisenberg = False
                        #print("Heisenberg Rejected: not NN interactions present")
                        break
                if not is_heisenberg:
                    break
            if not is_heisenberg:
                break

    # Heisenberg check passed, build simpler structure
    if is_heisenberg:
        return {
            "method": "heisenberg",
            "L": L,
            "g": g_heis_dict,
            "J": J_heis_dict,
            "is_hermitian": is_hermitian,
        }
    # General spin-glass / position-dependent XYZ spin model.
    else:
        return {
            "method": "spin_glass",
            "L": L,
            "g": g_dict,
            "J": J_dict,
            "is_hermitian": is_hermitian,
        }

def is_operator_foqcs_compatible(
        O: QubitOperator,
        L: int = -1,
        tol: float = 1e-12
) -> tuple: #(bool, dict || None)
    """
    TODO: DOC
    Returns tuple of boolean with the check result and the output of foqcs_analyze_operator
    """
    res = foqcs_analyze_operator(O, L = L, tol = tol, raise_errors = False)
    return (res != None, res)

def _cx_ladder(qv: QuantumVariable | Sequence[Qubit], n: int, k: int = 1) -> None:

    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])

def _gamma_gate(qv_rot: Any, qv_x: Any, theta: float):
    with control(qv_x):
        ry(theta, qv_rot)
    cx(qv_rot, qv_x)

def _delta_gate(qv: QuantumVariable | Sequence[Qubit], theta_coeffs):
    if len(theta_coeffs) == 1:
        _gamma_gate(qv[0], qv[1], theta_coeffs[0])
    else:
        _gamma_gate(qv[-2], qv[-1], theta_coeffs[-1])
        _delta_gate(qv[:-1], theta_coeffs[:-1])
